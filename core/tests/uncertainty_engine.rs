//! Both worker paths forward opt-in statistics and filter stop ids before reporting them.
use dsx_core::server::uncertainty::{Distribution, TokenStats};
use dsx_core::server::{BatchEngine, EngineHandle, Event, Finish, GenParams};

#[derive(Default)]
struct Batch {
    slot: Option<(u64, u32)>,
    statistics: bool,
}
impl BatchEngine for Batch {
    fn has_room(&self) -> bool {
        self.slot.is_none()
    }
    fn start(&mut self, id: u64, _: &[u32], p: &GenParams) -> anyhow::Result<()> {
        self.slot = Some((id, 0));
        self.statistics = p.logprobs.is_some();
        Ok(())
    }
    fn step(&mut self) -> anyhow::Result<Vec<(u64, u32)>> {
        let (id, tok) = self.slot.as_mut().unwrap();
        *tok += 1;
        Ok(vec![(*id, *tok)])
    }
    fn token_stats(&mut self, _: u64) -> anyhow::Result<TokenStats> {
        assert!(self.statistics, "off requests must never ask the engine for statistics");
        Ok(Distribution::new(&[0.0, 1.0], 2).chosen(1.0))
    }
    fn finish(&mut self, _: u64) {
        self.slot = None;
    }
}

#[test]
fn batch_opt_in_and_stop_filtering() {
    let engine = EngineHandle::spawn_batch(|| Ok(Box::new(Batch::default()))).unwrap();
    for logprobs in [None, Some(2)] {
        let p = GenParams {
            max_new: 4,
            temperature: 0.0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(1),
            logprobs,
            think: None,
            stop_token_ids: vec![2],
            hesitation: None,
            images: Default::default(),
            client: "default".into(),
            priority: false,
        };
        let mut rx = engine.submit("batch-test".into(), vec![0], p).unwrap();
        match rx.blocking_recv().unwrap() {
            Event::Token(1) => assert!(logprobs.is_none()),
            Event::TokenWithStats(1, s) => {
                assert!(logprobs.is_some());
                assert_eq!(s.top.len(), 2);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(rx.blocking_recv().unwrap(), Event::Done { finish: Finish::Stop }));
        assert!(rx.blocking_recv().is_none());
    }
    engine.shutdown_and_wait();
}
