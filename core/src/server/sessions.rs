//! GPU-independent prefix selection and bounded pools for inactive conversations: host memory (the shelf) and,
//! optionally, disk (the archive). `sessions` includes the live conversation; one preserves the original server path.
//!
//! Without quotas every saved conversation shares one LRU (the original policy, kept bit for bit). With
//! `--session-quota CLIENT=SESSIONS:GB` each named client (the `X-DSX-Client` request header) gets a pool of its own:
//! only that client's requests save into it, so only they can evict from it, and the pools' budgets add up to at
//! most `--session-gb`. Clients without a quota share the `default` pool, which gets what the quotas leave.
//! Inside a quota pool the conversation that is cheapest to read again goes first (size classes by powers of two,
//! the least recently used inside a class), a conversation used in the last `recent` is skipped while any other can
//! go, and one idle longer than `stale` goes before everything else. A conversation is not saved at all when that
//! would push out a costlier one that is not stale: short side conversations never displace a long main one.
//!
//! With an archive (`--session-dir`) nothing that fits it is thrown away. A conversation that leaves the shelf is
//! written to disk first, unless an up-to-date copy is already there; one the shelf will not admit goes straight to
//! disk; and while the engine is idle it copies shelf conversations to disk, so that a later eviction or a restart
//! costs nothing. A conversation loaded from disk keeps its file as a fallback until a newer unit of the same
//! conversation reaches the disk. The archive has per-client quotas of its own (conversations, bytes, age) and the
//! same isolation; inside a pool the superseded units go first, then the expired ones, then the cheapest to read
//! again, with the same admission rule. Files the cache lets go are handed to the engine through `drain_trash`.

use anyhow::{Result, bail, ensure};
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// the pool of requests without an `X-DSX-Client` header, and of clients without a quota
pub const DEFAULT_CLIENT: &str = "default";

/// `CLIENT=SESSIONS:GB`: up to SESSIONS saved conversations and GB (decimal) of host memory for CLIENT.
#[derive(Clone, Debug, PartialEq)]
pub struct Quota {
    pub client: String,
    pub sessions: usize,
    pub max_bytes: usize,
}

impl Quota {
    pub fn parse(s: &str) -> Result<Self> {
        let Some((client, rest)) = s.split_once('=') else { bail!("--session-quota {s:?}: expected CLIENT=SESSIONS:GB") };
        let Some((n, gb)) = rest.split_once(':') else { bail!("--session-quota {s:?}: expected CLIENT=SESSIONS:GB") };
        let client = client.trim().to_ascii_lowercase();
        ensure!(valid_client(&client), "--session-quota {s:?}: client names are 1-32 of [a-z0-9._-]");
        let sessions: usize = n.trim().parse().map_err(|_| anyhow::anyhow!("--session-quota {s:?}: SESSIONS is not a number"))?;
        let gb: f64 = gb.trim().parse().map_err(|_| anyhow::anyhow!("--session-quota {s:?}: GB is not a number"))?;
        ensure!(gb.is_finite() && gb >= 0.0 && gb * 1e9 < usize::MAX as f64, "--session-quota {s:?}: GB must be finite and nonnegative");
        Ok(Self { client, sessions, max_bytes: (gb * 1e9) as usize })
    }
}

/// `CLIENT=SESSIONS:GB[:DAYS]`: up to SESSIONS conversations and GB (decimal) of disk for CLIENT, each kept at most
/// DAYS after its last use (default `--session-disk-days`).
#[derive(Clone, Debug, PartialEq)]
pub struct DiskQuota {
    pub client: String,
    pub sessions: usize,
    pub max_bytes: usize,
    pub ttl: Option<Duration>,
}

impl DiskQuota {
    pub fn parse(s: &str) -> Result<Self> {
        let flag = "--session-disk-quota";
        let Some((client, rest)) = s.split_once('=') else { bail!("{flag} {s:?}: expected CLIENT=SESSIONS:GB[:DAYS]") };
        let parts: Vec<&str> = rest.split(':').collect();
        ensure!(parts.len() == 2 || parts.len() == 3, "{flag} {s:?}: expected CLIENT=SESSIONS:GB[:DAYS]");
        let client = client.trim().to_ascii_lowercase();
        ensure!(valid_client(&client), "{flag} {s:?}: client names are 1-32 of [a-z0-9._-]");
        let sessions: usize = parts[0].trim().parse().map_err(|_| anyhow::anyhow!("{flag} {s:?}: SESSIONS is not a number"))?;
        let gb = positive(parts[1], flag, s, "GB")?;
        let ttl = parts.get(2).map(|d| positive(d, flag, s, "DAYS").map(days)).transpose()?;
        Ok(Self { client, sessions, max_bytes: (gb * 1e9) as usize, ttl })
    }
}

fn positive(v: &str, flag: &str, s: &str, what: &str) -> Result<f64> {
    let x: f64 = v.trim().parse().map_err(|_| anyhow::anyhow!("{flag} {s:?}: {what} is not a number"))?;
    ensure!(x.is_finite() && x >= 0.0 && x * 1e9 < usize::MAX as f64, "{flag} {s:?}: {what} must be finite and nonnegative");
    Ok(x)
}

fn days(d: f64) -> Duration {
    Duration::from_secs_f64(d * 86400.0)
}

/// A client label as it may appear in `X-DSX-Client` and `--session-quota`.
pub fn valid_client(s: &str) -> bool {
    (1..=32).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
}

/// The archive: where, how much, how long, and when the engine copies to it.
#[derive(Clone, Debug)]
pub struct DiskConfig {
    pub dir: PathBuf,
    pub sessions: usize,
    pub max_bytes: usize,
    pub quotas: Vec<DiskQuota>,
    /// default age limit (since the last use)
    pub ttl: Duration,
    /// the engine starts copying the shelf (and checkpointing the live conversation) after this long without requests
    pub idle: Duration,
    /// keep this much free on the filesystem
    pub reserve_bytes: u64,
}

impl DiskConfig {
    pub fn new(dir: PathBuf, sessions: usize, gb: f64, ttl_days: f64, idle: Duration, reserve_gb: f64) -> Result<Self> {
        ensure!(sessions >= 1, "--session-disk-sessions must be at least 1");
        ensure!(gb.is_finite() && gb > 0.0 && gb * 1e9 < usize::MAX as f64, "--session-disk-gb must be positive");
        ensure!(ttl_days.is_finite() && ttl_days > 0.0, "--session-disk-days must be positive");
        ensure!(reserve_gb.is_finite() && reserve_gb >= 0.0, "--session-disk-reserve-gb must be nonnegative");
        Ok(Self { dir, sessions, max_bytes: (gb * 1e9) as usize, quotas: vec![], ttl: days(ttl_days), idle, reserve_bytes: (reserve_gb * 1e9) as u64 })
    }

    pub fn with_quotas(mut self, quotas: Vec<DiskQuota>) -> Result<Self> {
        for (i, q) in quotas.iter().enumerate() {
            ensure!(!quotas[..i].iter().any(|p| p.client == q.client), "--session-disk-quota: {} is given twice", q.client);
        }
        let n: usize = quotas.iter().map(|q| q.sessions).sum();
        let b: usize = quotas.iter().map(|q| q.max_bytes).sum();
        ensure!(n <= self.sessions, "--session-disk-quota: {n} conversations in quotas exceed --session-disk-sessions {}", self.sessions);
        ensure!(b <= self.max_bytes, "--session-disk-quota: {:.1} GB in quotas exceeds --session-disk-gb {:.1}", b as f64 / 1e9, self.max_bytes as f64 / 1e9);
        self.quotas = quotas;
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub sessions: usize,
    pub max_bytes: usize,
    pub min_tokens: usize,
    /// host-only shared prompt snapshots per pool (0 disables capture)
    pub prefix_max: usize,
    /// per-client pools; empty = the original single LRU
    pub quotas: Vec<Quota>,
    /// quota pools: a conversation used this recently goes only when nothing else in its pool can
    pub recent: Duration,
    /// quota pools: a conversation idle this long goes first, whatever its size
    pub stale: Duration,
    /// the archive on disk (None: host memory only)
    pub disk: Option<DiskConfig>,
}

impl SessionConfig {
    pub fn new(sessions: usize, gb: f64, min_tokens: usize) -> Result<Self> {
        ensure!(sessions >= 1, "--sessions must be at least 1");
        ensure!(
            gb.is_finite() && gb >= 0.0 && gb * 1e9 < usize::MAX as f64,
            "--session-gb must be finite, nonnegative and fit in usize"
        );
        Ok(Self {
            sessions,
            max_bytes: (gb * 1e9) as usize,
            min_tokens,
            prefix_max: 4,
            quotas: vec![],
            recent: Duration::from_secs(600),
            stale: Duration::from_secs(3 * 3600),
            disk: None,
        })
    }

    /// Per-client pools. Their sums must fit `--sessions` (minus the live slot) and `--session-gb`.
    pub fn with_quotas(mut self, quotas: Vec<Quota>, recent: Duration, stale: Duration) -> Result<Self> {
        for (i, q) in quotas.iter().enumerate() {
            ensure!(!quotas[..i].iter().any(|p| p.client == q.client), "--session-quota: {} is given twice", q.client);
        }
        if !quotas.is_empty() {
            ensure!(self.enabled(), "--session-quota needs --sessions 2 or more and a nonzero --session-gb");
        }
        let n: usize = quotas.iter().map(|q| q.sessions).sum();
        let b: usize = quotas.iter().map(|q| q.max_bytes).sum();
        ensure!(n <= self.sessions - 1, "--session-quota: {n} saved conversations in quotas, --sessions {} leaves room for {}", self.sessions, self.sessions - 1);
        ensure!(b <= self.max_bytes, "--session-quota: {:.2} GB in quotas exceeds --session-gb {:.2}", b as f64 / 1e9, self.max_bytes as f64 / 1e9);
        ensure!(stale > recent, "--session-stale-min must be longer than --session-recent-min");
        self.quotas = quotas;
        self.recent = recent;
        self.stale = stale;
        Ok(self)
    }

    /// The archive. Needs the shelf (`--sessions` 2 or more, nonzero `--session-gb`): conversations reach the disk
    /// through it or at its door.
    pub fn with_disk(mut self, disk: DiskConfig) -> Result<Self> {
        ensure!(self.enabled(), "--session-dir needs --sessions 2 or more and a nonzero --session-gb");
        self.disk = Some(disk);
        Ok(self)
    }

    pub fn enabled(&self) -> bool {
        self.sessions > 1 && self.max_bytes > 0
    }

    /// pools by client: (client, saved conversations, bytes, cost-aware)
    fn pools(&self) -> Vec<(Arc<str>, usize, usize, bool)> {
        if self.quotas.is_empty() {
            return vec![(DEFAULT_CLIENT.into(), self.sessions.saturating_sub(1), self.max_bytes, false)];
        }
        let mut p: Vec<_> = self.quotas.iter().map(|q| (Arc::<str>::from(q.client.as_str()), q.sessions, q.max_bytes, true)).collect();
        if !self.quotas.iter().any(|q| q.client == DEFAULT_CLIENT) {
            let n = self.sessions - 1 - self.quotas.iter().map(|q| q.sessions).sum::<usize>();
            let b = self.max_bytes - self.quotas.iter().map(|q| q.max_bytes).sum::<usize>();
            p.push((DEFAULT_CLIENT.into(), n, b, true));
        }
        p
    }

    /// archive pools by client: (client, conversations, bytes, age limit). Without shelf quotas everyone shares one.
    fn disk_pools(&self) -> Vec<(Arc<str>, usize, usize, Duration)> {
        let Some(d) = &self.disk else { return vec![] };
        if self.quotas.is_empty() && d.quotas.is_empty() {
            return vec![(DEFAULT_CLIENT.into(), d.sessions, d.max_bytes, d.ttl)];
        }
        let mut p: Vec<_> = d.quotas.iter().map(|q| (Arc::<str>::from(q.client.as_str()), q.sessions, q.max_bytes, q.ttl.unwrap_or(d.ttl))).collect();
        if !d.quotas.iter().any(|q| q.client == DEFAULT_CLIENT) {
            let n = d.sessions - d.quotas.iter().map(|q| q.sessions).sum::<usize>();
            let b = d.max_bytes - d.quotas.iter().map(|q| q.max_bytes).sum::<usize>();
            p.push((DEFAULT_CLIENT.into(), n, b, d.ttl));
        }
        p
    }
}

/// The original reuse rule, including the mandatory token that regenerates logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixReuse {
    pub lcp: usize,
    pub start: usize,
}

impl PrefixReuse {
    pub fn new(prompt: &[u32], cached: &[u32], pos: usize, rollback: Option<usize>) -> Self {
        let lcp = prompt
            .iter()
            .zip(cached)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len().saturating_sub(1));
        let start = if lcp == cached.len() && pos == cached.len() {
            lcp
        } else {
            rollback.filter(|&p| p <= lcp).unwrap_or(0)
        };
        Self { lcp, start }
    }
}

/// A conversation's file in the archive.
#[derive(Clone, Debug, PartialEq)]
pub struct DiskCopy {
    pub path: PathBuf,
    pub bytes: u64,
}

pub struct Session<S> {
    pub id: u64,
    pub client: Arc<str>,
    pub key: Box<[u32]>,
    pub pos: usize,
    pub rollback: Option<usize>,
    /// a reusable shared prompt, loaded by copy and never archived or superseded
    pub prefix: bool,
    /// the snapshot in host memory; None when the conversation is only on disk
    pub state: Option<S>,
    /// its file in the archive, when it has one (the same unit as `state`)
    pub disk: Option<DiskCopy>,
    /// host-memory footprint while `state` is there
    bytes: usize,
    pool: usize,
    dpool: usize,
    /// when it stopped being the live conversation
    used: Instant,
    /// a newer unit of the same conversation exists: its file goes first
    superseded: bool,
    /// the archive refused it once: the idle copier does not retry
    refused: bool,
}

impl<S> Session<S> {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn on_disk(&self) -> bool {
        self.disk.is_some()
    }
    /// when it was last the live conversation
    pub fn used(&self) -> Instant {
        self.used
    }
}

/// What the cache asks of the engine when a conversation changes tier. The engine writes the files; the cache only
/// keeps their names and sizes.
pub trait Tiers<S> {
    /// the live sequence into host memory
    fn capture(&mut self) -> Result<S>;
    /// the live sequence straight to a file, without a host copy
    fn write_live(&mut self, client: &str, key: &[u32], pos: usize, rollback: Option<usize>) -> Result<DiskCopy>;
    /// a shelf snapshot to a file
    fn write_saved(&mut self, s: &Session<S>) -> Result<DiskCopy>;
}

/// `stash` without an archive: a capture closure.
struct CaptureOnly<F>(Option<F>);

impl<S, F: FnOnce() -> Result<S>> Tiers<S> for CaptureOnly<F> {
    fn capture(&mut self) -> Result<S> {
        (self.0.take().expect("one capture per stash"))()
    }
    fn write_live(&mut self, _: &str, _: &[u32], _: usize, _: Option<usize>) -> Result<DiskCopy> {
        bail!("no archive")
    }
    fn write_saved(&mut self, _: &Session<S>) -> Result<DiskCopy> {
        bail!("no archive")
    }
}

struct Pool {
    client: Arc<str>,
    sessions: usize,
    max_bytes: usize,
    cost_aware: bool,
    /// archive pools: the age limit
    ttl: Duration,
    len: usize,
    bytes: usize,
}

// Individually allocated nodes make the budget independent of a Vec/VecDeque's spare capacity and reallocations.
// Oldest first. The expected number of conversations is small; prefix comparisons dominate list traversal.
struct Node<S> {
    session: Session<S>,
    next: Option<Box<Node<S>>>,
}

pub struct SessionCache<S> {
    cfg: SessionConfig,
    pools: Vec<Pool>,
    dpools: Vec<Pool>,
    head: Option<Box<Node<S>>>,
    len: usize,
    bytes: usize,
    disk_bytes: usize,
    next_id: u64,
    peak_bytes: usize,
    /// files no entry owns any more: the engine deletes them
    trash: Vec<DiskCopy>,
    /// during a stash: the conversation about to be loaded, kept in its tier
    guard: Option<u64>,
    /// tests: a fixed clock
    now: Option<Instant>,
}

/// bytes a file is reserved before it is written: the host footprint plus header and padding pages
fn file_estimate(host_bytes: usize) -> usize {
    host_bytes + 4 * 4096
}

impl<S> SessionCache<S> {
    pub fn new(cfg: SessionConfig) -> Self {
        let pools = cfg
            .pools()
            .into_iter()
            .map(|(client, sessions, max_bytes, cost_aware)| Pool { client, sessions, max_bytes, cost_aware, ttl: Duration::MAX, len: 0, bytes: 0 })
            .collect();
        let dpools = cfg
            .disk_pools()
            .into_iter()
            .map(|(client, sessions, max_bytes, ttl)| Pool { client, sessions, max_bytes, cost_aware: true, ttl, len: 0, bytes: 0 })
            .collect();
        Self {
            cfg,
            pools,
            dpools,
            head: None,
            len: 0,
            bytes: 0,
            disk_bytes: 0,
            next_id: 1,
            peak_bytes: 0,
            trash: vec![],
            guard: None,
            now: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.cfg.enabled()
    }
    /// clients own their saved conversations (`--session-quota` given)
    pub fn isolated(&self) -> bool {
        !self.cfg.quotas.is_empty()
    }
    /// an archive on disk is configured
    pub fn archived(&self) -> bool {
        self.cfg.disk.is_some() && self.enabled()
    }
    pub fn disk_config(&self) -> Option<&DiskConfig> {
        self.cfg.disk.as_ref()
    }
    pub fn min_tokens(&self) -> usize {
        self.cfg.min_tokens
    }
    /// saved conversations, on the shelf or in the archive
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }
    /// host memory held by shelf snapshots
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn max_bytes(&self) -> usize {
        self.cfg.max_bytes
    }
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }
    /// disk held by archive files
    pub fn disk_bytes(&self) -> usize {
        self.disk_bytes
    }
    /// (saved conversations, bytes) on `client`'s shelf
    pub fn pool_usage(&self, client: &str) -> (usize, usize) {
        let p = &self.pools[self.pool_of(client)];
        (p.len, p.bytes)
    }
    /// (conversations, bytes) in `client`'s archive pool
    pub fn archive_usage(&self, client: &str) -> (usize, usize) {
        if self.dpools.is_empty() {
            return (0, 0);
        }
        let p = &self.dpools[self.dpool_of(client)];
        (p.len, p.bytes)
    }

    #[doc(hidden)]
    pub fn set_now(&mut self, t: Instant) {
        self.now = Some(t);
    }
    fn now(&self) -> Instant {
        self.now.unwrap_or_else(Instant::now)
    }

    fn pool_of(&self, client: &str) -> usize {
        if !self.isolated() {
            return 0;
        }
        self.pools
            .iter()
            .position(|p| &*p.client == client)
            .or_else(|| self.pools.iter().position(|p| &*p.client == DEFAULT_CLIENT))
            .expect("a default pool")
    }

    fn dpool_of(&self, client: &str) -> usize {
        if self.dpools.len() <= 1 {
            return 0;
        }
        self.dpools
            .iter()
            .position(|p| &*p.client == client)
            .or_else(|| self.dpools.iter().position(|p| &*p.client == DEFAULT_CLIENT))
            .expect("a default archive pool")
    }

    /// the label a conversation is filed under: its client's pool with quotas (clients without a quota share
    /// `default`), the label as given without
    fn owner(&self, client: &str) -> Arc<str> {
        if self.isolated() { self.pools[self.pool_of(client)].client.clone() } else { client.into() }
    }

    /// The label the engine should keep for the live conversation and compare with: the pool's owner.
    pub fn owner_of(&self, client: &str) -> Arc<str> {
        self.owner(client)
    }

    /// Reserve the live slot before starting an unrelated conversation. A failed restore can leave both
    /// snapshots in RAM while there is no valid live sequence; keep them available for a retry until now.
    pub fn begin_session(&mut self) -> u64 {
        for p in 0..self.pools.len() {
            while self.pools[p].len > self.pools[p].sessions {
                let (victim, why, _) = self.eviction_order(p, None)[0];
                self.evict(victim, why);
            }
        }
        self.fresh_id()
    }

    fn fresh_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn prefixes_enabled(&self) -> bool {
        self.enabled() && self.cfg.prefix_max > 0
    }

    /// A prefix stays on the shelf after a borrowed load. Move it to the end of the LRU as well as touching time.
    pub fn touch_prefix(&mut self, client: &str, key: &[u32]) -> bool {
        let p = self.pool_of(client);
        let id = self.iter().find(|s| s.prefix && s.pool == p && &*s.key == key).map(|s| s.id);
        let Some(id) = id else { return false };
        let mut s = self.take(id).unwrap();
        s.used = self.now();
        self.push(s);
        true
    }

    /// Capture a shared prompt at a completed prefill boundary, without a rewind point. Refusal never archives it.
    /// Eviction precedes capture, so the byte budget includes the allocation in transit.
    pub fn capture_prefix(&mut self, client: &str, key: &[u32], payload_bytes: usize, t: &mut dyn Tiers<S>) -> Result<bool> {
        if !self.prefixes_enabled() || key.len() < self.cfg.min_tokens || self.touch_prefix(client, key) {
            return Ok(false);
        }
        let p = self.pool_of(client);
        let bytes = Self::entry_bytes(key.len(), payload_bytes)?;
        if self.pools[p].sessions == 0 || bytes > self.pools[p].max_bytes {
            tracing::warn!(client, tokens = key.len(), bytes, "prefix not captured: no room in its pool");
            return Ok(false);
        }
        // prefixes may hold at most half of the pool's slots (one in a one-slot pool): the rest stays for conversations
        let cap = self.cfg.prefix_max.min((self.pools[p].sessions / 2).max(1));
        if cap == 0 {
            return Ok(false);
        }
        let prefixes: Vec<_> = self.iter().filter(|s| s.prefix && s.pool == p).map(|s| s.id).collect();
        if prefixes.len() >= cap {
            self.evict(prefixes[0], "prefix-lru");
        }
        // With an archive an evicted conversation goes to disk (back in ~0.3 s), while the prefix saves every new
        // conversation the shared prompt (10+ s): the size-class admission rule does not apply. Without one it does.
        let admit = if self.archived() { u32::MAX } else { class(key.len()) };
        let Some(victims) = self.shelf_victims(p, bytes, admit, None) else {
            tracing::warn!(client, tokens = key.len(), bytes, "prefix not captured: its pool holds only costlier conversations");
            return Ok(false);
        };
        for (id, why) in victims {
            self.demote(id, why, t);
        }
        let t0 = Instant::now();
        let state = t.capture()?;
        let id = self.fresh_id();
        let session = Session {
            id,
            client: self.owner(client),
            key: key.into(),
            pos: key.len(),
            rollback: None,
            prefix: true,
            state: Some(state),
            disk: None,
            bytes,
            pool: p,
            dpool: self.dpool_of(client),
            used: self.now(),
            superseded: false,
            refused: false,
        };
        self.push(session);
        tracing::info!(client, tokens = key.len(), bytes, ms = t0.elapsed().as_secs_f64() * 1e3, "prefix captured");
        Ok(true)
    }

    /// Payload bytes are its heap allocations; the entire LRU node (including S) and key are added here.
    pub fn entry_bytes(key_len: usize, payload_bytes: usize) -> Result<usize> {
        key_len
            .checked_mul(size_of::<u32>())
            .and_then(|b| b.checked_add(payload_bytes))
            .and_then(|b| b.checked_add(size_of::<Node<S>>()))
            .ok_or_else(|| anyhow::anyhow!("session size overflow"))
    }

    /// The saved conversation of `client` that continues `prompt` best, if it beats the live one by `min_tokens`.
    /// Without quotas any client's conversation qualifies (the original rule). Shelf and archive alike.
    pub fn best(
        &self,
        prompt: &[u32],
        client: &str,
        live: &[u32],
        live_pos: usize,
        live_rollback: Option<usize>,
    ) -> Option<u64> {
        if !self.enabled() {
            return None;
        }
        let live = PrefixReuse::new(prompt, live, live_pos, live_rollback);
        let isolated = self.isolated();
        let client = self.owner(client);
        self.iter()
            .filter(|s| !isolated || s.client == client)
            .filter_map(|s| {
                let r = PrefixReuse::new(prompt, &s.key, s.pos, s.rollback);
                (r.start > live.start && r.start - live.start >= self.cfg.min_tokens)
                    .then_some((s.id, r.start, s.state.is_some()))
            })
            // the longest usable prefix; between equal ones the copy already in host memory
            .max_by_key(|&(_, start, ram)| (start, ram))
            .map(|(id, _, _)| id)
    }

    pub fn get(&self, id: u64) -> Option<&Session<S>> {
        self.iter().find(|s| s.id == id)
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut Session<S>> {
        let mut link = self.head.as_deref_mut();
        while let Some(node) = link {
            if node.session.id == id {
                return Some(&mut node.session);
            }
            link = node.next.as_deref_mut();
        }
        None
    }

    fn iter(&self) -> impl Iterator<Item = &Session<S>> {
        std::iter::successors(self.head.as_deref(), |node| node.next.as_deref())
            .map(|node| &node.session)
    }

    /// Remove a conversation from both tiers. Its file, if any, becomes the caller's.
    pub fn take(&mut self, id: u64) -> Option<Session<S>> {
        let mut link = &mut self.head;
        while let Some(node) = link.as_ref() {
            if node.session.id == id {
                let mut node = link.take().unwrap();
                *link = node.next.take();
                self.unaccount(&node.session);
                self.len -= 1;
                return Some(node.session);
            }
            link = &mut link.as_mut().unwrap().next;
        }
        None
    }

    /// Hand a conversation over to become the live one, after the engine loaded it (from `state`, or from the file
    /// when there is no host snapshot). Its host snapshot comes along; its file stays in the archive as a fallback,
    /// under a new id, until a newer unit of the conversation is written.
    pub fn take_for_load(&mut self, id: u64) -> Option<Session<S>> {
        let mut s = self.take(id)?;
        if let Some(copy) = s.disk.take() {
            let fid = self.fresh_id();
            let now = self.now();
            let fallback = Session {
                id: fid,
                client: s.client.clone(),
                key: s.key.clone(),
                pos: s.pos,
                rollback: s.rollback,
                prefix: false,
                state: None,
                disk: Some(copy),
                bytes: s.bytes,
                pool: s.pool,
                dpool: s.dpool,
                used: now,
                superseded: false,
                refused: false,
            };
            self.push(fallback);
        }
        Some(s)
    }

    fn push(&mut self, s: Session<S>) {
        self.account(&s);
        self.len += 1;
        let mut link = &mut self.head;
        while link.is_some() {
            link = &mut link.as_mut().unwrap().next;
        }
        *link = Some(Box::new(Node { session: s, next: None }));
    }

    fn account(&mut self, s: &Session<S>) {
        if s.state.is_some() {
            self.bytes += s.bytes;
            let p = &mut self.pools[s.pool];
            p.bytes += s.bytes;
            p.len += 1;
            self.peak_bytes = self.peak_bytes.max(self.bytes);
        }
        if let Some(d) = &s.disk {
            self.disk_bytes += d.bytes as usize;
            let p = &mut self.dpools[s.dpool];
            p.bytes += d.bytes as usize;
            p.len += 1;
        }
    }

    fn unaccount(&mut self, s: &Session<S>) {
        if s.state.is_some() {
            self.bytes -= s.bytes;
            let p = &mut self.pools[s.pool];
            p.bytes -= s.bytes;
            p.len -= 1;
        }
        if let Some(d) = &s.disk {
            self.disk_bytes -= d.bytes as usize;
            let p = &mut self.dpools[s.dpool];
            p.bytes -= d.bytes as usize;
            p.len -= 1;
        }
    }

    /// Drop the host snapshot of `id`; a conversation without a file is gone.
    fn evict(&mut self, id: u64, why: &'static str) {
        let now = self.now();
        let Some(s) = self.get(id) else { return };
        let (on_disk, idle_s, tokens, bytes, client) = (s.disk.is_some(), now.saturating_duration_since(s.used).as_secs(), s.key.len(), s.bytes, s.client.clone());
        if on_disk {
            let s = self.get_mut(id).unwrap();
            let state = s.state.take();
            if state.is_some() {
                let pool = s.pool;
                self.bytes -= bytes;
                self.pools[pool].bytes -= bytes;
                self.pools[pool].len -= 1;
            }
            tracing::info!(session = id, client = &*client, tokens, bytes, idle_s, reason = why, "session left the shelf (its file stays)");
        } else if let Some(s) = self.take(id) {
            tracing::info!(session = s.id, client = &*s.client, tokens = s.key.len(), bytes = s.bytes, idle_s, reason = why, "session evicted");
        }
    }

    /// A shelf conversation that has to go: to disk when there is an archive (unless its file is current), else away.
    fn demote(&mut self, id: u64, why: &'static str, t: &mut dyn Tiers<S>) {
        let Some(s) = self.get(id) else { return };
        if s.prefix || s.disk.is_some() || !self.archived() || s.state.is_none() {
            return self.evict(id, why);
        }
        let (dpool, need, class) = (s.dpool, file_estimate(s.bytes), class(s.key.len()));
        if !self.archive_room(dpool, need, class, &[id]) {
            return self.evict(id, why);
        }
        let t0 = Instant::now();
        let r = t.write_saved(self.get(id).unwrap());
        match r {
            Ok(copy) => {
                let (bytes, tokens) = (copy.bytes, self.get(id).unwrap().key.len());
                self.attach(id, copy);
                tracing::info!(session = id, tokens, file_bytes = bytes, write_ms = t0.elapsed().as_secs_f64() * 1e3, reason = why, "session spilled to disk");
                self.evict(id, why);
                self.supersede_older(id);
            }
            Err(e) => {
                tracing::warn!(session = id, error = %format!("{e:#}"), "session not written to disk");
                self.evict(id, why);
            }
        }
    }

    /// Give `id` its file and account it; a pool pushed over its budget by a larger-than-estimated file is trimmed.
    fn attach(&mut self, id: u64, copy: DiskCopy) {
        let Some(s) = self.get_mut(id) else {
            self.trash.push(copy);
            return;
        };
        if let Some(old) = s.disk.replace(copy) {
            // cannot happen through the public paths (a current file is never rewritten); keep accounting exact
            let dp = s.dpool;
            self.disk_bytes -= old.bytes as usize;
            self.dpools[dp].bytes -= old.bytes as usize;
            self.dpools[dp].len -= 1;
            self.trash.push(old);
        }
        let s = self.get(id).unwrap();
        let (dp, b) = (s.dpool, s.disk.as_ref().unwrap().bytes as usize);
        self.disk_bytes += b;
        self.dpools[dp].bytes += b;
        self.dpools[dp].len += 1;
        self.trim_archive(dp, &[id]);
    }

    /// Drop `id`'s file; a conversation that is not on the shelf is gone.
    fn drop_file(&mut self, id: u64, why: &'static str) {
        let now = self.now();
        let Some(s) = self.get_mut(id) else { return };
        let Some(copy) = s.disk.take() else { return };
        let (dp, ram, tokens, client, idle_s) = (s.dpool, s.state.is_some(), s.key.len(), s.client.clone(), now.saturating_duration_since(s.used).as_secs());
        self.disk_bytes -= copy.bytes as usize;
        self.dpools[dp].bytes -= copy.bytes as usize;
        self.dpools[dp].len -= 1;
        tracing::info!(session = id, client = &*client, tokens, file_bytes = copy.bytes, idle_s, reason = why, "session file dropped");
        self.trash.push(copy);
        if !ram {
            self.take(id);
        }
    }

    /// Pool `p`'s conversations in the order they leave, with the reason and size class. LRU without quotas; with
    /// quotas the stale ones (oldest first), then those not used recently, then the recent ones, each group by size
    /// class (powers of two of tokens) and least recently used inside a class.
    fn eviction_order(&self, p: usize, protect: Option<u64>) -> Vec<(u64, &'static str, u32)> {
        // the list is oldest first, and the sort is stable
        let mut v: Vec<&Session<S>> = self.iter().filter(|s| s.state.is_some() && s.pool == p && Some(s.id) != protect).collect();
        if !self.pools[p].cost_aware {
            return v.into_iter().map(|s| (s.id, "lru", class(s.key.len()))).collect();
        }
        let now = self.now();
        let group = |s: &Session<S>| match now.saturating_duration_since(s.used) {
            d if d >= self.cfg.stale => 0,
            d if d >= self.cfg.recent => 1,
            _ => 2,
        };
        // with an archive a conversation leaves the shelf for disk (back in ~0.3 s) but a prefix is lost (its prompt is
        // read again, 10+ s, by every new conversation): prefixes go last; their own LRU cap bounds them
        let archived = self.archived();
        v.sort_by_key(|s| (archived && s.prefix, group(s), if group(s) == 0 { 0 } else { class(s.key.len()) }));
        v.into_iter().map(|s| (s.id, ["stale", "cheapest", "cheapest-recent"][group(s)], class(s.key.len()))).collect()
    }

    /// Archive pool `d`'s files in the order they leave: superseded units, expired ones (oldest first), then the
    /// cheapest to read again among those not used recently, then the recent ones.
    fn archive_order(&self, d: usize, protect: &[u64]) -> Vec<(u64, &'static str, u32)> {
        let now = self.now();
        let ttl = self.dpools[d].ttl;
        let group = |s: &Session<S>| {
            let age = now.saturating_duration_since(s.used);
            if s.superseded {
                0
            } else if age >= ttl {
                1
            } else if age >= self.cfg.recent {
                2
            } else {
                3
            }
        };
        let mut v: Vec<&Session<S>> =
            self.iter().filter(|s| s.disk.is_some() && s.dpool == d && !protect.contains(&s.id) && Some(s.id) != self.guard).collect();
        v.sort_by_key(|s| (group(s), if group(s) < 2 { 0 } else { class(s.key.len()) }));
        v.into_iter().map(|s| (s.id, ["superseded", "expired", "cheapest", "cheapest-recent"][group(s)], class(s.key.len()))).collect()
    }

    /// Make room for a file of `bytes` in archive pool `d` without pushing out a costlier conversation that is
    /// neither superseded nor expired. False (nothing dropped) when it does not fit.
    fn archive_room(&mut self, d: usize, bytes: usize, class: u32, protect: &[u64]) -> bool {
        let (cap_n, cap_b) = (self.dpools[d].sessions, self.dpools[d].max_bytes);
        if cap_n == 0 || bytes > cap_b {
            return false;
        }
        let (mut b, mut n) = (self.dpools[d].bytes, self.dpools[d].len);
        let mut victims = vec![];
        for (victim, why, vc) in self.archive_order(d, protect) {
            if b + bytes <= cap_b && n < cap_n {
                break;
            }
            if !matches!(why, "superseded" | "expired") && vc > class {
                return false;
            }
            b -= self.get(victim).unwrap().disk.as_ref().unwrap().bytes as usize;
            n -= 1;
            victims.push((victim, why));
        }
        if b + bytes > cap_b || n >= cap_n {
            return false;
        }
        for (victim, why) in victims {
            self.drop_file(victim, why);
        }
        true
    }

    /// After a file landed bigger than reserved: drop files (archive order, no admission rule) until the pool fits.
    fn trim_archive(&mut self, d: usize, protect: &[u64]) {
        while self.dpools[d].bytes > self.dpools[d].max_bytes || self.dpools[d].len > self.dpools[d].sessions {
            let Some(&(victim, why, _)) = self.archive_order(d, protect).first() else { break };
            self.drop_file(victim, why);
        }
    }

    /// Two prompt boundaries on the same token path represent one conversation (`related`).
    fn related(a_key: &[u32], a_rb: Option<usize>, b_key: &[u32], b_rb: Option<usize>) -> bool {
        // The rewind point precedes the final prompt token; include that token in the conversation boundary.
        let n = a_rb.map_or(a_key.len(), |r| r.saturating_add(1).min(a_key.len()))
            .min(b_rb.map_or(b_key.len(), |r| r.saturating_add(1).min(b_key.len())));
        n > 0 && a_key.len() >= n && b_key.len() >= n && a_key[..n] == b_key[..n]
    }

    /// Two prompt boundaries on the same token path represent one conversation, even if a client re-renders
    /// generated assistant tokens. A shared system prompt alone is not enough. With quotas only `client`'s.
    /// Shelf copies are dropped (the live sequence supersedes them); files stay as fallbacks, marked superseded.
    pub fn discard_related(&mut self, client: &str, key: &[u32], rollback: Option<usize>) {
        let isolated = self.isolated();
        let client = self.owner(client);
        let ids: Vec<(u64, bool)> = self
            .iter()
            .filter(|s| !s.prefix && (!isolated || s.client == client) && Self::related(key, rollback, &s.key, s.rollback))
            .map(|s| (s.id, s.disk.is_some()))
            .collect();
        for (id, on_disk) in ids {
            if on_disk {
                let s = self.get_mut(id).unwrap();
                s.superseded = true;
                if s.state.is_some() {
                    let (pool, bytes) = (s.pool, s.bytes);
                    s.state = None;
                    self.bytes -= bytes;
                    self.pools[pool].bytes -= bytes;
                    self.pools[pool].len -= 1;
                }
            } else {
                self.take(id);
            }
        }
    }

    /// `id` reached the disk: older units of the same conversation (files only, not on the shelf) are deleted.
    /// Older by last use, not by length: a regenerated answer or a retried turn is a shorter but newer branch.
    fn supersede_older(&mut self, id: u64) {
        let Some(s) = self.get(id) else { return };
        if s.prefix { return; }
        let (client, key, rb, used, isolated) = (s.client.clone(), s.key.clone(), s.rollback, s.used, self.isolated());
        let old: Vec<u64> = self
            .iter()
            .filter(|o| {
                !o.prefix && o.id != id && Some(o.id) != self.guard
                    && o.state.is_none()
                    && o.disk.is_some()
                    && (!isolated || o.client == client)
                    && o.used <= used
                    && Self::related(&key, rb, &o.key, o.rollback)
            })
            .map(|o| o.id)
            .collect();
        for o in old {
            self.drop_file(o, "superseded");
        }
    }

    /// Evict BEFORE allocating. During a swap the protected target stays charged until restoration finishes;
    /// an outgoing snapshot that cannot coexist with it is skipped. The byte limit also covers this transient.
    /// `capture` is never called when caching is disabled, the sequence is short, or it cannot fit.
    /// The conversation goes into `client`'s pool and evicts only from there.
    #[allow(clippy::too_many_arguments)]
    pub fn stash(
        &mut self,
        id: u64,
        client: &str,
        key: &[u32],
        pos: usize,
        rollback: Option<usize>,
        payload_bytes: usize,
        protect: Option<u64>,
        capture: impl FnOnce() -> Result<S>,
    ) -> Result<bool> {
        self.stash_tiered(id, client, key, pos, rollback, payload_bytes, protect, &mut CaptureOnly(Some(capture)))
    }

    /// `stash` with an archive: conversations pushed off the shelf are written to disk (`Tiers::write_saved`), and
    /// one the shelf will not admit goes straight to disk (`Tiers::write_live`).
    #[allow(clippy::too_many_arguments)]
    pub fn stash_tiered(
        &mut self,
        id: u64,
        client: &str,
        key: &[u32],
        pos: usize,
        rollback: Option<usize>,
        payload_bytes: usize,
        protect: Option<u64>,
        t: &mut dyn Tiers<S>,
    ) -> Result<bool> {
        if !self.enabled() || key.len() < self.cfg.min_tokens || pos != key.len() {
            return Ok(false);
        }
        let bytes = Self::entry_bytes(key.len(), payload_bytes)?;
        // the conversation about to be loaded stays in whichever tier it is in until the swap is over
        self.guard = protect;
        let r = self.stash_inner(id, client, key, pos, rollback, bytes, protect, t);
        self.guard = None;
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn stash_inner(
        &mut self,
        id: u64,
        client: &str,
        key: &[u32],
        pos: usize,
        rollback: Option<usize>,
        bytes: usize,
        target: Option<u64>,
        t: &mut dyn Tiers<S>,
    ) -> Result<bool> {
        let p = self.pool_of(client);
        let (cap_n, cap_b) = (self.pools[p].sessions, self.pools[p].max_bytes);
        let class = class(key.len());
        // Do not evict anything if the two snapshots alone cannot coexist.
        let protect = target.filter(|&t| self.get(t).is_some_and(|s| s.pool == p && s.state.is_some()));
        let protected = protect.and_then(|id| self.get(id)).map_or(0, |s| s.bytes);
        if cap_n == 0 || bytes > cap_b || bytes > cap_b - protected {
            if !self.archived() {
                return Ok(false);
            }
            // too big for the shelf: the archive may still take it
            if let Some(old) = self.take(id) {
                self.trash.extend(old.disk);
            }
            return self.archive_live(id, client, key, pos, rollback, bytes, t);
        }
        // Same id is always replaced, never duplicated.
        if let Some(old) = self.take(id) {
            self.trash.extend(old.disk);
        }
        let Some(victims) = self.shelf_victims(p, bytes, class, protect) else {
            if self.archived() && self.archive_live(id, client, key, pos, rollback, bytes, t)? {
                return Ok(true);
            }
            tracing::info!(session = id, client, tokens = key.len(), "session not saved: its pool holds only costlier conversations");
            return Ok(false);
        };
        for (victim, why) in victims {
            self.demote(victim, why, t);
        }
        let extra = usize::from(protect.and_then(|id| self.get(id)).is_some_and(|s| !s.prefix));
        assert!(self.pools[p].bytes <= cap_b - bytes && self.pools[p].len < cap_n + extra, "protected target fits");
        let state = t.capture()?;
        let session = Session {
            id,
            client: self.owner(client),
            key: key.into(),
            pos,
            rollback,
            prefix: false,
            state: Some(state),
            disk: None,
            bytes,
            pool: p,
            dpool: self.dpool_of(client),
            used: self.now(),
            superseded: false,
            refused: false,
        };
        self.push(session);
        debug_assert!(self.bytes <= self.cfg.max_bytes);
        Ok(true)
    }

    /// The shelf conversations that must go to admit `bytes` of size `class` into pool `p`, or None when the
    /// admission rule refuses (only costlier conversations that are not stale could make room).
    fn shelf_victims(&self, p: usize, bytes: usize, class: u32, protect: Option<u64>) -> Option<Vec<(u64, &'static str)>> {
        let (cap_n, cap_b) = (self.pools[p].sessions, self.pools[p].max_bytes);
        // A conversation target leaves the shelf after loading, so its slot can be borrowed during the swap.
        // A prefix target stays: the outgoing conversation must fit alongside it or go straight to disk.
        let extra = usize::from(protect.and_then(|id| self.get(id)).is_some_and(|s| !s.prefix));
        let (mut b, mut n) = (self.pools[p].bytes, self.pools[p].len);
        let mut victims = vec![];
        for (victim, why, vc) in self.eviction_order(p, protect) {
            if b <= cap_b - bytes && n < cap_n + extra {
                break;
            }
            if self.pools[p].cost_aware && why != "stale" && vc > class {
                // admission: keeping the costlier conversation beats saving this one
                return None;
            }
            b -= self.get(victim).unwrap().bytes;
            n -= 1;
            victims.push((victim, why));
        }
        (b <= cap_b - bytes && n < cap_n + extra).then_some(victims)
    }

    /// The live sequence straight into the archive (the shelf refused it).
    #[allow(clippy::too_many_arguments)]
    fn archive_live(&mut self, id: u64, client: &str, key: &[u32], pos: usize, rollback: Option<usize>, bytes: usize, t: &mut dyn Tiers<S>) -> Result<bool> {
        let d = self.dpool_of(client);
        if !self.archive_room(d, file_estimate(bytes), class(key.len()), &[]) {
            tracing::info!(session = id, client, tokens = key.len(), "session not saved: its archive pool holds only costlier conversations");
            return Ok(false);
        }
        let t0 = Instant::now();
        let copy = match t.write_live(client, key, pos, rollback) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(session = id, client, tokens = key.len(), error = %format!("{e:#}"), "session not written to disk");
                return Ok(false);
            }
        };
        tracing::info!(session = id, client, tokens = key.len(), file_bytes = copy.bytes, write_ms = t0.elapsed().as_secs_f64() * 1e3, "session saved straight to disk");
        let session = Session {
            id,
            client: self.owner(client),
            key: key.into(),
            pos,
            rollback,
            prefix: false,
            state: None,
            disk: None,
            bytes,
            pool: self.pool_of(client),
            dpool: d,
            used: self.now(),
            superseded: false,
            refused: false,
        };
        self.push(session);
        self.attach(id, copy);
        self.supersede_older(id);
        Ok(true)
    }

    /// Idle work: copy the costliest shelf conversation that has no file yet. True when one was written.
    pub fn persist_one(&mut self, t: &mut dyn Tiers<S>) -> bool {
        if !self.archived() {
            return false;
        }
        let Some((id, d, need, class)) = self
            .iter()
            .filter(|s| !s.prefix && s.state.is_some() && s.disk.is_none() && !s.refused)
            .max_by_key(|s| s.key.len())
            .map(|s| (s.id, s.dpool, file_estimate(s.bytes), class(s.key.len())))
        else {
            return false;
        };
        if !self.archive_room(d, need, class, &[id]) {
            self.get_mut(id).unwrap().refused = true;
            return false;
        }
        let t0 = Instant::now();
        let r = t.write_saved(self.get(id).unwrap());
        match r {
            Ok(copy) => {
                let (bytes, tokens) = (copy.bytes, self.get(id).unwrap().key.len());
                self.attach(id, copy);
                self.supersede_older(id);
                tracing::info!(session = id, tokens, file_bytes = bytes, write_ms = t0.elapsed().as_secs_f64() * 1e3, "session copied to disk");
                true
            }
            Err(e) => {
                tracing::warn!(session = id, error = %format!("{e:#}"), "session not written to disk");
                self.get_mut(id).unwrap().refused = true;
                false
            }
        }
    }

    /// Shelf conversations that have no file and could still get one (idle work left).
    pub fn unpersisted(&self) -> usize {
        self.iter().filter(|s| !s.prefix && s.state.is_some() && s.disk.is_none() && !s.refused).count()
    }

    /// The live sequence to disk as a unit of its own (idle checkpoint, shutdown); the live conversation goes on.
    /// Older files of the same conversation are deleted. Returns the new unit's id.
    pub fn checkpoint_live(&mut self, client: &str, key: &[u32], pos: usize, rollback: Option<usize>, payload_bytes: usize, t: &mut dyn Tiers<S>) -> Option<u64> {
        if !self.archived() || key.len() < self.cfg.min_tokens || pos != key.len() {
            return None;
        }
        let bytes = Self::entry_bytes(key.len(), payload_bytes).ok()?;
        let d = self.dpool_of(client);
        if !self.archive_room(d, file_estimate(bytes), class(key.len()), &[]) {
            return None;
        }
        let t0 = Instant::now();
        let copy = match t.write_live(client, key, pos, rollback) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(client, tokens = key.len(), error = %format!("{e:#}"), "live conversation not written to disk");
                return None;
            }
        };
        let id = self.fresh_id();
        tracing::info!(session = id, client, tokens = key.len(), file_bytes = copy.bytes, write_ms = t0.elapsed().as_secs_f64() * 1e3, "live conversation checkpointed to disk");
        let session = Session {
            id,
            client: self.owner(client),
            key: key.into(),
            pos,
            rollback,
            prefix: false,
            state: None,
            disk: None,
            bytes,
            pool: self.pool_of(client),
            dpool: d,
            used: self.now(),
            superseded: false,
            refused: false,
        };
        self.push(session);
        self.attach(id, copy);
        self.supersede_older(id);
        Some(id)
    }

    /// A file found at startup. Adopt in the order of last use (oldest first): the list order is the LRU order.
    #[allow(clippy::too_many_arguments)]
    pub fn adopt(&mut self, client: &str, key: Vec<u32>, pos: usize, rollback: Option<usize>, copy: DiskCopy, payload_bytes: usize, used: Instant) -> Option<u64> {
        if !self.archived() || key.len() < self.cfg.min_tokens || pos != key.len() {
            self.trash.push(copy);
            return None;
        }
        let id = self.fresh_id();
        let bytes = Self::entry_bytes(key.len(), payload_bytes).unwrap_or(usize::MAX / 4);
        let session = Session {
            id,
            client: self.owner(client),
            key: key.into(),
            pos,
            rollback,
            prefix: false,
            state: None,
            disk: Some(copy),
            bytes,
            pool: self.pool_of(client),
            dpool: self.dpool_of(client),
            used,
            superseded: false,
            refused: false,
        };
        self.push(session);
        Some(id)
    }

    /// After adopting: drop what is expired or over a quota (the limits may have changed since the files were
    /// written), and the older units of conversations that have a newer file.
    pub fn settle_archive(&mut self) {
        let mut ids: Vec<(Instant, u64)> = self.iter().filter(|s| s.disk.is_some()).map(|s| (s.used, s.id)).collect();
        ids.sort();
        for &(_, id) in ids.iter().rev() {
            self.supersede_older(id);
        }
        self.expire();
        for d in 0..self.dpools.len() {
            self.trim_archive(d, &[]);
        }
    }

    /// Drop files not used for longer than their pool's age limit.
    pub fn expire(&mut self) {
        let now = self.now();
        let old: Vec<u64> = self
            .iter()
            .filter(|s| s.disk.is_some() && now.saturating_duration_since(s.used) >= self.dpools[s.dpool].ttl)
            .map(|s| s.id)
            .collect();
        for id in old {
            self.drop_file(id, "expired");
        }
    }

    /// Files no conversation owns any more; the engine deletes them.
    pub fn drain_trash(&mut self) -> Vec<DiskCopy> {
        std::mem::take(&mut self.trash)
    }
}

/// size class: powers of two of tokens
fn class(tokens: usize) -> u32 {
    usize::BITS - tokens.leading_zeros()
}

impl<S> Drop for SessionCache<S> {
    fn drop(&mut self) {
        // Avoid recursive destruction for a deliberately large --sessions setting.
        while let Some(mut node) = self.head.take() {
            self.head = node.next.take();
        }
    }
}
