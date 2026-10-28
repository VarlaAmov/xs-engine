//! The engine abstraction and the single OS thread that runs it (GPU contexts are thread-bound).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use super::uncertainty::{Channels, TokenStats};

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new: usize,
    /// 0 = greedy
    pub temperature: f32,
    pub top_p: f32,
    /// keep only tokens whose probability is at least this fraction of the top token's (0 = off)
    pub min_p: f32,
    pub seed: Option<u64>,
    /// Raw model statistics: None is free; Some(k) includes up to 20 alternatives (0 = no alternatives).
    pub logprobs: Option<u8>,
    /// Thinking delimiters, used only when collecting uncertainty.
    pub think: Option<(u32, u32)>,
    /// EOS and friends. The worker enforces these and `max_new` too, so an engine may ignore them.
    pub stop_token_ids: Vec<u32>,
    /// what to do to the logits while the model is reasoning: hesitation penalty and/or thinking budget (None = nothing)
    pub hesitation: Option<Hesitation>,
    /// images in the prompt, in order (their spans already expanded in the prompt ids)
    pub images: std::sync::Arc<Vec<crate::vision::ImageInput>>,
    /// who asked (`X-DSX-Client`, default "default"): owns the conversation's saved sessions
    pub client: Arc<str>,
    /// served before every queued request without priority (never interrupts the running one)
    pub priority: bool,
}

/// Reasoning control, active until `close` (`</think>`) is generated: subtract `lambda` from the logits of `ids`
/// (the hesitation words; 0 = no penalty) and, with a `budget`, land the reasoning softly by about that many tokens.
#[derive(Debug, Clone)]
pub struct Hesitation {
    pub lambda: f32,
    pub ids: std::sync::Arc<Vec<u32>>,
    pub close: u32,
    pub budget: Option<ThinkBudget>,
}

/// Thinking budget with a soft landing. The reasoning may have `tokens` tokens; during the last `window` of them the
/// logits of `</think>` and of the paragraph-break tokens get a bonus growing linearly to `boost`, so the model tends to
/// finish a paragraph and close; the token after the budget is `</think>` whatever the model wanted.
#[derive(Debug, Clone)]
pub struct ThinkBudget {
    pub tokens: usize,
    pub window: usize,
    pub boost: f32,
    /// ids of the paragraph-break tokens ("\n\n")
    pub breaks: std::sync::Arc<Vec<u32>>,
}

impl ThinkBudget {
    /// Landing window: a quarter of the budget, between 8 and 256 tokens.
    pub fn new(tokens: usize, boost: f32, breaks: std::sync::Arc<Vec<u32>>) -> Self {
        Self { tokens, window: (tokens / 4).clamp(8, 256), boost, breaks }
    }

    /// The ids whose logits `apply` may change.
    pub fn touched(&self, close: u32) -> impl Iterator<Item = u32> + '_ {
        std::iter::once(close).chain(self.breaks.iter().copied())
    }

    /// `n`: 1-based index of the reasoning token about to be sampled. Past the budget the token is forced to be `close`.
    pub fn apply(&self, n: usize, close: u32, logits: &mut [f32]) {
        if n > self.tokens {
            for l in logits.iter_mut() {
                *l = f32::NEG_INFINITY;
            }
            if let Some(l) = logits.get_mut(close as usize) {
                *l = 0.0;
            }
        } else if n + self.window > self.tokens {
            let bonus = self.boost * (n + self.window - self.tokens) as f32 / self.window as f32;
            for id in self.touched(close) {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += bonus;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

/// What an engine knows about its last `generate` call. Unset timings fall back to the worker's wall clock
/// (prefill = until the first token, decode = the rest).
#[derive(Debug, Clone, Copy, Default)]
pub struct GenStats {
    pub prefill: Option<Duration>,
    pub decode: Option<Duration>,
    /// prompt tokens served from a KV/prefix cache
    pub cached_prompt_tokens: usize,
}

pub trait Engine: Send {
    /// Generate after `prompt`. Call `emit(token)` per generated token; stop early when it returns false.
    fn generate(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish>;
    /// Opt-in statistics callback. Legacy engines remain usable without statistics.
    fn generate_with_stats(&mut self, _prompt: &[u32], _p: &GenParams, _emit: &mut dyn FnMut(u32, Option<TokenStats>) -> bool) -> Result<Finish> {
        anyhow::bail!("this engine does not support uncertainty statistics")
    }
    /// Timings/cache stats of the last `generate` call.
    fn last_stats(&self) -> GenStats {
        GenStats::default()
    }
    /// Client disconnected or the server stopped a generation: persist a safe retry point.
    fn cancelled(&mut self) {}
    /// Background work between requests, on the engine thread; `idle` is the time since the last request ended.
    /// Return true to be called again right away (more to do), false to wait for the next tick or request.
    fn idle(&mut self, _idle: Duration) -> bool {
        false
    }
    /// The server is stopping: save what should survive a restart. Called once, after the last request.
    fn shutdown(&mut self) {}
}

/// An engine that runs many sequences at once (continuous batching). The worker thread admits queued requests while
/// `has_room`, then calls `step` in a loop; each step advances every running sequence (prefill chunks and/or one
/// decoded token) and returns the tokens it produced. The worker enforces stop tokens, `max_new` and disconnects,
/// and calls `finish` when a sequence ends.
pub trait BatchEngine: Send {
    /// Can another sequence start now?
    fn has_room(&self) -> bool;
    /// Start sequence `id` after `prompt` (its prefill may run over the following steps).
    fn start(&mut self, id: u64, prompt: &[u32], p: &GenParams) -> Result<()>;
    /// Advance every running sequence by one step: (id, token) for each token produced.
    fn step(&mut self) -> Result<Vec<(u64, u32)>>;
    /// Raw statistics for the token just returned by `step`; called only for requests that opted in.
    fn token_stats(&mut self, _id: u64) -> Result<TokenStats> {
        anyhow::bail!("this batched engine does not support uncertainty statistics")
    }
    /// Sequence `id` is done (stop, length, cancelled or failed): free its slot.
    fn finish(&mut self, id: u64);
    /// Prompt tokens of sequence `id` served from a cache (for the log).
    fn cached_prompt_tokens(&self, _id: u64) -> usize {
        0
    }
}

/// Engine thread -> request handler.
#[derive(Debug)]
pub enum Event {
    Token(u32),
    TokenWithStats(u32, TokenStats),
    Done { finish: Finish },
    Error(String),
}

struct Job {
    id: String,
    prompt: Vec<u32>,
    params: GenParams,
    tx: mpsc::UnboundedSender<Event>,
    queued: Instant,
}

/// Jobs waiting for the engine thread: priority clients first, FIFO inside each class.
#[derive(Default)]
struct JobQueue {
    high: std::collections::VecDeque<Job>,
    normal: std::collections::VecDeque<Job>,
}

impl JobQueue {
    fn push(&mut self, j: Job) {
        if j.params.priority { self.high.push_back(j) } else { self.normal.push_back(j) }
    }
    fn pop(&mut self) -> Option<Job> {
        self.high.pop_front().or_else(|| self.normal.pop_front())
    }
    fn is_empty(&self) -> bool {
        self.high.is_empty() && self.normal.is_empty()
    }
}

/// Request handlers -> engine thread.
enum Msg {
    Job(Job),
    /// stop: refuse what is queued, let the engine save its state, then acknowledge
    Shutdown(std_mpsc::Sender<()>),
}

/// what a request that cannot run any more is told
const RESTARTING: &str = "the model server is restarting: send the request again in a minute";

struct Liveness(Arc<AtomicBool>);
impl Drop for Liveness {
    fn drop(&mut self) { self.0.store(false, Ordering::SeqCst); }
}

/// Cloneable handle to the engine thread: jobs one at a time, priority clients ahead of the others.
#[derive(Clone)]
pub struct EngineHandle {
    alive: Arc<AtomicBool>,
    tx: std_mpsc::Sender<Msg>,
    pending: Arc<AtomicUsize>,
    /// set at shutdown: the running generation stops at its next token
    stop: Arc<AtomicBool>,
}

impl EngineHandle {
    /// Construct the engine on a dedicated thread (`make` runs there) and start serving jobs.
    pub fn spawn(make: impl FnOnce() -> Result<Box<dyn Engine>> + Send + 'static) -> Result<Self> {
        let (tx, rx) = std_mpsc::channel::<Msg>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pend = pending.clone();
        let alive = Arc::new(AtomicBool::new(true));
        let thread_alive = alive.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        std::thread::Builder::new()
            .name("dsx-engine".into())
            .spawn(move || {
                let _liveness = Liveness(thread_alive);
                let mut engine = match make() {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // everything that arrived while the last job ran is ordered before the next one starts
                let mut queue = JobQueue::default();
                let mut shutdown: Option<std_mpsc::Sender<()>> = None;
                let mut last_end = Instant::now();
                // the engine has more background work: poll the queue instead of waiting a tick
                let mut busy = false;
                let refuse = |j: Job| {
                    tracing::info!(id = %j.id, "refused: shutting down");
                    let _ = j.tx.send(Event::Error(RESTARTING.into()));
                    pend.fetch_sub(1, Ordering::SeqCst);
                };
                loop {
                    if queue.is_empty() && shutdown.is_none() {
                        let m = if busy {
                            rx.try_recv().map_err(|e| matches!(e, std_mpsc::TryRecvError::Disconnected))
                        } else {
                            rx.recv_timeout(Duration::from_secs(1)).map_err(|e| matches!(e, std_mpsc::RecvTimeoutError::Disconnected))
                        };
                        match m {
                            Ok(Msg::Job(j)) => queue.push(j),
                            Ok(Msg::Shutdown(ack)) => shutdown = Some(ack),
                            Err(true) => {
                                engine.shutdown();
                                return;
                            }
                            Err(false) => {
                                busy = engine.idle(last_end.elapsed());
                                continue;
                            }
                        }
                    }
                    while let Ok(m) = rx.try_recv() {
                        match m {
                            Msg::Job(j) => queue.push(j),
                            Msg::Shutdown(ack) => shutdown = Some(ack),
                        }
                    }
                    if let Some(ack) = shutdown.take() {
                        while let Some(j) = queue.pop() {
                            refuse(j);
                        }
                        engine.shutdown();
                        let _ = ack.send(());
                        // until the process exits
                        for m in rx.iter() {
                            match m {
                                Msg::Job(j) => refuse(j),
                                Msg::Shutdown(ack) => {
                                    let _ = ack.send(());
                                }
                            }
                        }
                        return;
                    }
                    let job = queue.pop().expect("a queued job");
                    busy = false;
                    run_job(engine.as_mut(), job, &stopping);
                    last_end = Instant::now();
                    pend.fetch_sub(1, Ordering::SeqCst);
                }
            })?;
        ready_rx.recv().context("engine thread died during construction")??;
        Ok(Self { tx, pending, stop, alive })
    }

    /// Stop serving: the running generation ends at its next token (its client is told to retry), queued requests
    /// are refused, and the engine saves what should survive a restart. Blocks until that is done.
    pub fn shutdown_and_wait(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let (tx, rx) = std_mpsc::channel();
        if self.tx.send(Msg::Shutdown(tx)).is_ok() {
            let _ = rx.recv();
        }
    }

    /// Construct a batched engine on a dedicated thread and serve jobs concurrently (continuous batching).
    pub fn spawn_batch(make: impl FnOnce() -> Result<Box<dyn BatchEngine>> + Send + 'static) -> Result<Self> {
        let (tx, rx) = std_mpsc::channel::<Msg>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pend = pending.clone();
        let alive = Arc::new(AtomicBool::new(true));
        let thread_alive = alive.clone();
        std::thread::Builder::new()
            .name("dsx-engine".into())
            .spawn(move || {
                let _liveness = Liveness(thread_alive);
                let engine = match make() {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                run_batch(engine, rx, pend);
            })?;
        ready_rx.recv().context("engine thread died during construction")??;
        Ok(Self { tx, pending, stop: Arc::new(AtomicBool::new(false)), alive })
    }

    /// Queue a generation; tokens and the final event arrive on the returned channel.
    /// Dropping the receiver cancels the job (queued or running).
    pub fn submit(&self, id: String, prompt: Vec<u32>, params: GenParams) -> Result<mpsc::UnboundedReceiver<Event>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let ahead = self.pending.fetch_add(1, Ordering::SeqCst);
        tracing::info!(%id, prompt_tokens = prompt.len(), max_new = params.max_new, queue_position = ahead, client = &*params.client, priority = params.priority, "queued");
        let job = Job { id, prompt, params, tx, queued: Instant::now() };
        if self.tx.send(Msg::Job(job)).is_err() {
            self.pending.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!("engine thread is gone");
        }
        Ok(rx)
    }

    pub fn is_alive(&self) -> bool { self.alive.load(Ordering::SeqCst) }

    pub fn is_stopping(&self) -> bool { self.stop.load(Ordering::SeqCst) }

    /// Jobs queued or running.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }
}

fn run_job(engine: &mut dyn Engine, job: Job, stop: &AtomicBool) {
    let Job { id, prompt, params, tx, queued } = job;
    if tx.is_closed() {
        tracing::info!(%id, "cancelled while queued");
        return;
    }
    let mut stopped = false;
    let wait = queued.elapsed();
    let t0 = Instant::now();
    let mut first: Option<Instant> = None;
    let mut generated = 0usize;
    let (mut hit_stop, mut hit_len, mut gone) = (false, false, false);
    let mut uncertainty = params.logprobs.map(|_| Channels::new(params.think, &prompt));
    let r = {
        let mut emit = |tok: u32, stats: Option<TokenStats>| -> bool {
            if hit_stop || hit_len || gone || stopped {
                return false;
            }
            if stop.load(Ordering::Relaxed) {
                stopped = true;
                return false;
            }
            first.get_or_insert_with(Instant::now);
            if params.stop_token_ids.contains(&tok) {
                hit_stop = true;
                return false;
            }
            generated += 1;
            let event = match stats {
                Some(s) => {
                    if let Some(u) = &mut uncertainty { u.token(tok, &s); }
                    Event::TokenWithStats(tok, s)
                }
                None => Event::Token(tok),
            };
            if tx.send(event).is_err() {
                gone = true;
                return false;
            }
            if generated >= params.max_new {
                hit_len = true;
                return false;
            }
            true
        };
        if params.max_new == 0 { Ok(Finish::Length) }
        else if params.logprobs.is_some() { engine.generate_with_stats(&prompt, &params, &mut emit) }
        else { engine.generate(&prompt, &params, &mut |tok| emit(tok, None)) }
    };
    if gone || stopped { engine.cancelled(); }
    let total = t0.elapsed();
    let st = engine.last_stats();
    let ttft = first.map(|f| f - t0).unwrap_or(total);
    let prefill = st.prefill.unwrap_or(ttft);
    let decode = st.decode.unwrap_or(total.saturating_sub(ttft));
    let finish = match r {
        Ok(_) if stopped => {
            // cut by a shutdown: an error, so the client sends it again instead of keeping half an answer
            let _ = tx.send(Event::Error(RESTARTING.into()));
            "Restarting".into()
        }
        Ok(f) => {
            // the worker's own stop/length/disconnect checks win over what the engine says
            let f = if hit_stop { Finish::Stop } else if hit_len { Finish::Length } else if gone { Finish::Cancelled } else { f };
            let _ = tx.send(Event::Done { finish: f });
            format!("{f:?}")
        }
        Err(e) => {
            tracing::error!(%id, "engine error: {e:#}");
            let _ = tx.send(Event::Error(format!("{e:#}")));
            "Error".into()
        }
    };
    let decode_tps = if generated > 1 { (generated - 1) as f64 / decode.as_secs_f64().max(1e-9) } else { 0.0 };
    tracing::info!(
        %id,
        prompt_tokens = prompt.len(),
        cached_prompt_tokens = st.cached_prompt_tokens,
        generated,
        queue_ms = wait.as_millis() as u64,
        prefill_ms = prefill.as_millis() as u64,
        decode_ms = decode.as_millis() as u64,
        prefill_tps = format!("{:.1}", (prompt.len() - st.cached_prompt_tokens.min(prompt.len())) as f64 / prefill.as_secs_f64().max(1e-9)),
        decode_tps = format!("{decode_tps:.2}"),
        finish,
        content_mean_entropy = uncertainty.as_ref().map(|u| u.content_stats.mean_entropy),
        content_min_logprob = uncertainty.as_ref().and_then(|u| u.content_stats.min_logprob),
        "done"
    );
}

struct Running {
    job_id: String,
    prompt_tokens: usize,
    params: GenParams,
    tx: mpsc::UnboundedSender<Event>,
    queued: Instant,
    started: Instant,
    first: Option<Instant>,
    generated: usize,
    uncertainty: Option<Channels>,
}

/// The batched engine thread: admit, step, route tokens, end sequences.
fn run_batch(mut engine: Box<dyn BatchEngine>, rx: std_mpsc::Receiver<Msg>, pend: Arc<AtomicUsize>) {
    let mut queue = JobQueue::default();
    let mut running: std::collections::HashMap<u64, Running> = Default::default();
    let mut next_id = 1u64;
    let done = |engine: &mut dyn BatchEngine, running: &mut std::collections::HashMap<u64, Running>, id: u64, f: Option<Finish>, err: Option<String>| {
        engine.finish(id);
        pend.fetch_sub(1, Ordering::SeqCst);
        if let Some(r) = running.remove(&id) {
            let total = r.started.elapsed();
            let ttft = r.first.map(|t| t - r.started).unwrap_or(total);
            let fin = match (&f, &err) {
                (_, Some(e)) => {
                    let _ = r.tx.send(Event::Error(e.clone()));
                    "Error".to_string()
                }
                (Some(f), None) => {
                    let _ = r.tx.send(Event::Done { finish: *f });
                    format!("{f:?}")
                }
                _ => "?".into(),
            };
            let decode = total.saturating_sub(ttft);
            let tps = if r.generated > 1 { (r.generated - 1) as f64 / decode.as_secs_f64().max(1e-9) } else { 0.0 };
            tracing::info!(id = %r.job_id, prompt_tokens = r.prompt_tokens, cached_prompt_tokens = engine.cached_prompt_tokens(id), generated = r.generated,
                queue_ms = (r.started - r.queued).as_millis() as u64, prefill_ms = ttft.as_millis() as u64, decode_ms = decode.as_millis() as u64,
                decode_tps = format!("{tps:.2}"), finish = fin, running = running.len(),
                content_mean_entropy = r.uncertainty.as_ref().map(|u| u.content_stats.mean_entropy),
                content_min_logprob = r.uncertainty.as_ref().and_then(|u| u.content_stats.min_logprob), "done");
        }
    };
    loop {
        // intake: block when there is nothing to do
        if running.is_empty() && queue.is_empty() {
            match rx.recv() {
                Ok(Msg::Job(j)) => queue.push(j),
                // the batched engines keep no state worth saving
                Ok(Msg::Shutdown(ack)) => {
                    let _ = ack.send(());
                    return;
                }
                Err(_) => return,
            }
        }
        while let Ok(m) = rx.try_recv() {
            match m {
                Msg::Job(j) => queue.push(j),
                Msg::Shutdown(ack) => {
                    let _ = ack.send(());
                    return;
                }
            }
        }
        // admit
        while engine.has_room() {
            let Some(j) = queue.pop() else { break };
            if j.tx.is_closed() {
                tracing::info!(id = %j.id, "cancelled while queued");
                pend.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            if j.params.max_new == 0 {
                let _ = j.tx.send(Event::Done { finish: Finish::Length });
                pend.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            let id = next_id;
            next_id += 1;
            let r = Running { job_id: j.id.clone(), prompt_tokens: j.prompt.len(), params: j.params.clone(), tx: j.tx, queued: j.queued, started: Instant::now(), first: None, generated: 0, uncertainty: j.params.logprobs.map(|_| Channels::new(j.params.think, &j.prompt)) };
            running.insert(id, r);
            if let Err(e) = engine.start(id, &j.prompt, &j.params) {
                tracing::error!(id = %j.id, "start failed: {e:#}");
                done(engine.as_mut(), &mut running, id, None, Some(format!("{e:#}")));
            }
        }
        if running.is_empty() {
            continue;
        }
        // disconnects
        let gone: Vec<u64> = running.iter().filter(|(_, r)| r.tx.is_closed()).map(|(&id, _)| id).collect();
        for id in gone {
            done(engine.as_mut(), &mut running, id, Some(Finish::Cancelled), None);
        }
        if running.is_empty() {
            continue;
        }
        match engine.step() {
            Ok(out) => {
                for (id, tok) in out {
                    let Some(r) = running.get_mut(&id) else { continue };
                    r.first.get_or_insert_with(Instant::now);
                    if r.params.stop_token_ids.contains(&tok) {
                        done(engine.as_mut(), &mut running, id, Some(Finish::Stop), None);
                        continue;
                    }
                    r.generated += 1;
                    let event = if r.params.logprobs.is_some() {
                        let stats = match engine.token_stats(id) {
                            Ok(s) => s,
                            Err(e) => {
                                done(engine.as_mut(), &mut running, id, None, Some(e.to_string()));
                                continue;
                            }
                        };
                        if let Some(u) = &mut r.uncertainty { u.token(tok, &stats); }
                        Event::TokenWithStats(tok, stats)
                    } else { Event::Token(tok) };
                    if r.tx.send(event).is_err() {
                        done(engine.as_mut(), &mut running, id, Some(Finish::Cancelled), None);
                        continue;
                    }
                    if r.generated >= r.params.max_new {
                        done(engine.as_mut(), &mut running, id, Some(Finish::Length), None);
                    }
                }
            }
            Err(e) => {
                tracing::error!("engine step failed: {e:#}");
                let ids: Vec<u64> = running.keys().copied().collect();
                for id in ids {
                    done(engine.as_mut(), &mut running, id, None, Some(format!("{e:#}")));
                }
            }
        }
    }
}

static REQ: AtomicU64 = AtomicU64::new(0);

/// Unique-enough request id: prefix + unix seconds + counter.
pub fn request_id(prefix: &str) -> String {
    let n = REQ.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{prefix}{t:x}{n:04x}")
}
