//! The engine thread's idle ticks and its shutdown: the running request stops at a token and is told to retry,
//! queued and later requests are refused, and the engine saves before the acknowledgement.
use dsx_core::server::engine::{Engine, EngineHandle, Event, Finish, GenParams};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct Probe {
    idle: Arc<AtomicUsize>,
    saved: Arc<AtomicBool>,
    generating: Arc<AtomicBool>,
}

impl Engine for Probe {
    fn generate(&mut self, _prompt: &[u32], _p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> anyhow::Result<Finish> {
        self.generating.store(true, Ordering::SeqCst);
        for t in 0..10_000 {
            std::thread::sleep(Duration::from_millis(5));
            if !emit(t) {
                return Ok(Finish::Cancelled);
            }
        }
        Ok(Finish::Stop)
    }
    fn idle(&mut self, _idle: Duration) -> bool {
        self.idle.fetch_add(1, Ordering::SeqCst);
        false
    }
    fn shutdown(&mut self) {
        assert!(!self.saved.swap(true, Ordering::SeqCst), "shutdown runs once");
    }
}

fn params() -> GenParams {
    GenParams {
        max_new: 100_000,
        temperature: 0.0,
        top_p: 1.0,
        min_p: 0.0,
        seed: None,
        stop_token_ids: vec![],
        hesitation: None,
        logprobs: None,
        think: None,
        images: Default::default(),
        client: "owner".into(),
        priority: false,
    }
}

/// tokens until the end, and how it ended
fn drain(mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>) -> (usize, Option<String>) {
    let mut n = 0;
    while let Some(e) = rx.blocking_recv() {
        match e {
            Event::Token(_) | Event::TokenWithStats(_, _) => n += 1,
            Event::Done { .. } => return (n, None),
            Event::Error(e) => return (n, Some(e)),
        }
    }
    (n, None)
}

#[test]
fn idle_ticks_then_shutdown_stops_refuses_and_saves() {
    let (idle, saved, generating) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (i, s, g) = (idle.clone(), saved.clone(), generating.clone());
    let h = EngineHandle::spawn(move || Ok(Box::new(Probe { idle: i, saved: s, generating: g }) as Box<dyn Engine>)).unwrap();
    let t0 = Instant::now();
    while idle.load(Ordering::SeqCst) == 0 {
        assert!(t0.elapsed() < Duration::from_secs(5), "an idle engine gets ticks");
        std::thread::sleep(Duration::from_millis(50));
    }
    let running = h.submit("a".into(), vec![1, 2, 3], params()).unwrap();
    let queued = h.submit("b".into(), vec![4, 5, 6], params()).unwrap();
    while !generating.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(50));
    h.shutdown_and_wait();
    assert!(saved.load(Ordering::SeqCst), "the engine saved before the acknowledgement");
    let (n, err) = drain(running);
    assert!(n > 0, "the running request had produced tokens");
    assert!(err.as_deref().is_some_and(|e| e.contains("restarting")), "and is told to retry: {err:?}");
    let (n, err) = drain(queued);
    assert_eq!(n, 0);
    assert!(err.as_deref().is_some_and(|e| e.contains("restarting")));
    let late = h.submit("c".into(), vec![7], params()).unwrap();
    assert!(drain(late).1.is_some_and(|e| e.contains("restarting")), "requests after the shutdown are refused too");
    assert_eq!(h.pending(), 0);
}

#[test]
fn panic_marks_the_engine_dead_instead_of_reporting_healthy() {
    struct Crash;
    impl Engine for Crash {
        fn generate(&mut self, _: &[u32], _: &GenParams, _: &mut dyn FnMut(u32) -> bool) -> anyhow::Result<Finish> {
            panic!("simulated selected-session panic");
        }
    }
    let h = EngineHandle::spawn(|| Ok(Box::new(Crash))).unwrap();
    assert!(h.is_alive());
    let mut rx = h.submit("crash".into(), vec![1], params()).unwrap();
    assert!(rx.blocking_recv().is_none());
    let deadline = Instant::now() + Duration::from_secs(2);
    while h.is_alive() && Instant::now() < deadline { std::thread::yield_now(); }
    assert!(!h.is_alive());
    assert!(h.submit("after-panic".into(), vec![1], params()).is_err());
}
