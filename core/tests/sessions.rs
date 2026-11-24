//! The production selector/LRU exercised with a CPU-only, token-state mock.
use dsx_core::server::sessions::{PrefixReuse, Quota, SessionCache, SessionConfig, DEFAULT_CLIENT};
use std::time::{Duration, Instant};

type Cache = SessionCache<Vec<u32>>;

fn cfg(n: usize, bytes: usize, min: usize) -> SessionConfig {
    SessionConfig {
        sessions: n,
        max_bytes: bytes,
        min_tokens: min,
        ..SessionConfig::new(1, 0.0, 0).unwrap()
    }
}

fn stash(c: &mut Cache, id: u64, key: &[u32], rollback: usize, protect: Option<u64>) -> bool {
    stash_as(c, DEFAULT_CLIENT, id, key, rollback, protect)
}

fn stash_as(c: &mut Cache, client: &str, id: u64, key: &[u32], rollback: usize, protect: Option<u64>) -> bool {
    c.stash(
        id,
        client,
        key,
        key.len(),
        Some(rollback),
        key.len() * 4,
        protect,
        || Ok(key.to_vec()),
    )
    .unwrap()
}

/// A mock live sequence goes through the same selection, save-before-overwrite and take-after-load as Real,
/// including who owns the live conversation.
struct Mock {
    cache: Cache,
    key: Vec<u32>,
    rollback: Option<usize>,
    id: u64,
    client: String,
}

impl Mock {
    fn new(n: usize, bytes: usize) -> Self {
        Self::with(cfg(n, bytes, 2))
    }
    fn with(cfg: SessionConfig) -> Self {
        Self {
            cache: Cache::new(cfg),
            key: vec![],
            rollback: None,
            id: 0,
            client: DEFAULT_CLIENT.into(),
        }
    }
    fn request(&mut self, prompt: &[u32]) -> usize {
        self.request_as(DEFAULT_CLIENT, prompt)
    }
    fn request_as(&mut self, client: &str, prompt: &[u32]) -> usize {
        let foreign = self.cache.isolated() && !self.key.is_empty() && self.client != client;
        let live: &[u32] = if foreign { &[] } else { &self.key };
        if let Some(id) = self.cache.best(prompt, client, live, live.len(), self.rollback) {
            if !self.key.is_empty() {
                stash_as(
                    &mut self.cache,
                    &self.client.clone(),
                    self.id,
                    &self.key,
                    self.rollback.unwrap(),
                    Some(id),
                );
            }
            let s = self.cache.get(id).unwrap();
            assert_eq!(&*s.key, s.state.as_ref().unwrap(), "the right sequence must be loaded");
            assert!(!self.cache.isolated() || &*s.client == client, "only the client's own conversations load");
            self.key = s.state.as_ref().unwrap().clone();
            self.rollback = s.rollback;
            if s.prefix {
                self.cache.touch_prefix(client, &self.key);
                self.id = self.cache.begin_session();
                self.client = client.into();
            } else {
                let s = self.cache.take_for_load(id).unwrap();
                self.id = id;
                self.client = s.client.to_string();
            }
        } else if foreign || PrefixReuse::new(prompt, &self.key, self.key.len(), self.rollback).start == 0 {
            if !self.key.is_empty() {
                stash_as(
                    &mut self.cache,
                    &self.client.clone(),
                    self.id,
                    &self.key,
                    self.rollback.unwrap(),
                    None,
                );
            }
            self.id = self.cache.begin_session();
            self.client = client.into();
        }
        let start = PrefixReuse::new(prompt, &self.key, self.key.len(), self.rollback).start;
        self.key = prompt.to_vec();
        self.rollback = Some(prompt.len() - 1);
        self.cache.discard_related(client, prompt, self.rollback);
        // An accepted generated token is part of the live sequence, as in Real::generate.
        self.key.push(99);
        assert!(self.cache.bytes() <= self.cache.max_bytes());
        start
    }
}

#[test]
fn alternating_a_b_a_b_restores_the_right_sequence() {
    let mut m = Mock::new(2, 1_000_000);
    let (mut a, mut b) = (vec![1; 30], vec![2; 40]);
    assert_eq!(m.request(&a), 0);
    assert_eq!(m.request(&b), 0);
    for i in 0..20 {
        let p = if i % 2 == 0 { &mut a } else { &mut b };
        p.extend_from_slice(&[99, 5, 6]);
        assert_eq!(m.request(p), p.len() - 2);
        assert_eq!(m.cache.len(), 1);
    }
}

#[test]
fn continuation_replaces_old_entry_including_rerendered_generation() {
    let mut c = Cache::new(cfg(3, 100_000, 2));
    assert!(stash(&mut c, 1, &[1, 2, 3, 99], 2, None));
    c.discard_related(DEFAULT_CLIENT, &[1, 2, 3, 88, 7], Some(4));
    assert!(stash(&mut c, 1, &[1, 2, 3, 88, 7, 99], 4, None));
    assert_eq!(c.len(), 1);
    assert_eq!(&*c.get(1).unwrap().key, &[1, 2, 3, 88, 7, 99]);
    // A common system prefix does not merge different user prompts.
    c.discard_related(DEFAULT_CLIENT, &[1, 2, 4, 8], Some(3));
    assert_eq!(c.len(), 1);
}

#[test]
fn byte_eviction_happens_before_capture_and_respects_lru() {
    let one = Cache::entry_bytes(4, 16).unwrap();
    let mut c = Cache::new(cfg(5, 2 * one, 2));
    assert!(stash(&mut c, 1, &[1; 4], 3, None));
    assert!(stash(&mut c, 2, &[2; 4], 3, None));
    assert!(stash(&mut c, 3, &[3; 4], 3, None));
    assert!(c.get(1).is_none());
    assert_eq!(c.bytes(), 2 * one);
    // Restoring and restashing updates recency.
    let s = c.take(2).unwrap();
    assert!(stash(&mut c, s.id, &s.key, 3, None));
    drop(s);
    assert!(stash(&mut c, 4, &[4; 4], 3, None));
    assert!(c.get(3).is_none());
    assert!(c.get(2).is_some());
}

#[test]
fn slot_limit_counts_the_live_session() {
    let mut c = Cache::new(cfg(2, 100_000, 2));
    for id in 1..5 {
        assert!(stash(&mut c, id, &[id as u32; 4], 3, None));
    }
    assert_eq!(c.len(), 1);
    assert!(c.get(4).is_some());
}

#[test]
fn swap_never_overcommits_budget_or_evicts_the_selected_target() {
    let one = Cache::entry_bytes(4, 16).unwrap();
    let mut c = Cache::new(cfg(2, one, 2));
    assert!(stash(&mut c, 1, &[1; 4], 3, None));
    assert!(
        !c.stash(2, DEFAULT_CLIENT, &[2; 4], 4, Some(3), 16, Some(1), || panic!(
            "would exceed hard budget"
        ))
        .unwrap()
    );
    assert!(c.get(1).is_some());
    let mut c = Cache::new(cfg(2, one * 2, 2));
    assert!(stash(&mut c, 1, &[1; 4], 3, None));
    assert!(stash(&mut c, 2, &[2; 4], 3, Some(1)));
    assert_eq!(c.bytes(), one * 2);
    drop(c.take(1).unwrap());
    assert_eq!(c.len(), 1);
}

#[test]
fn oversized_and_short_sequences_do_not_allocate() {
    let mut c = Cache::new(cfg(3, 100, 2));
    assert!(
        !c.stash(1, DEFAULT_CLIENT, &[1; 4], 4, Some(3), 1000, None, || panic!("too big"))
            .unwrap()
    );
    assert!(
        !c.stash(1, DEFAULT_CLIENT, &[1], 1, Some(1), 1, None, || panic!("too short"))
            .unwrap()
    );
}

#[test]
fn single_session_is_the_original_reuse_rule_and_never_captures() {
    let mut c = Cache::new(cfg(1, 10_000, 2));
    assert!(
        !c.stash(1, DEFAULT_CLIENT, &[1; 4], 4, Some(3), 16, None, || panic!(
            "single-session capture"
        ))
        .unwrap()
    );
    assert!(c.best(&[1; 8], DEFAULT_CLIENT, &[2; 4], 4, Some(3)).is_none());
    let old = |prompt: &[u32], cached: &[u32], pos: usize, snap: Option<usize>| {
        let lcp = prompt
            .iter()
            .zip(cached)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len() - 1);
        if lcp == cached.len() && pos == cached.len() {
            lcp
        } else {
            snap.filter(|&p| p <= lcp).unwrap_or(0)
        }
    };
    for prompt in [
        &[1, 2, 3, 4, 5, 6][..],
        &[1, 2, 3, 8],
        &[1, 8],
        &[1],
        &[1, 2, 3, 4],
    ] {
        for snap in [None, Some(1), Some(3), Some(4)] {
            let cached = [1, 2, 3, 4];
            assert_eq!(
                PrefixReuse::new(prompt, &cached, 4, snap).start,
                old(prompt, &cached, 4, snap)
            );
        }
    }
    assert!(c.is_empty());
}

#[test]
fn prefix_advantage_threshold_and_live_ties() {
    let mut c = Cache::new(cfg(3, 100_000, 3));
    assert!(stash(&mut c, 1, &[1, 2, 3, 4, 5, 6], 3, None));
    assert_eq!(
        c.best(&[1, 2, 3, 4, 5, 6, 7], DEFAULT_CLIENT, &[1, 2, 9], 3, Some(2)),
        Some(1)
    );
    assert_eq!(
        c.best(&[1, 2, 3, 4, 5, 6, 7], DEFAULT_CLIENT, &[1, 2, 3, 4, 9], 5, Some(4)),
        None
    );
    assert_eq!(
        c.best(&[1, 2, 3, 4, 5, 6, 7], DEFAULT_CLIENT, &[1, 2, 3, 4, 5, 6], 6, Some(3)),
        None
    );
}

#[test]
fn a_long_lcp_without_a_reachable_rewind_is_not_selected() {
    let mut c = Cache::new(cfg(3, 100_000, 2));
    assert!(stash(&mut c, 1, &[1, 2, 3, 4, 5, 6], 6, None));
    assert_eq!(c.best(&[1, 2, 3, 4, 9], DEFAULT_CLIENT, &[2], 1, None), None);
    // Identical prompt still needs a token to regenerate logits; rollback at its end is too late.
    assert_eq!(c.best(&[1, 2, 3, 4, 5, 6], DEFAULT_CLIENT, &[2], 1, None), None);
}

#[test]
fn invalid_limits_are_rejected() {
    for gb in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(SessionConfig::new(2, gb, 2).is_err());
    }
    assert!(SessionConfig::new(0, 2.0, 2).is_err());
    assert!(!SessionConfig::new(2, 0.0, 2).unwrap().enabled());
}

#[test]
fn restored_session_rewinds_when_assistant_tokens_are_rerendered() {
    let mut m = Mock::new(2, 1_000_000);
    assert_eq!(m.request(&[1, 2, 3, 4]), 0);
    assert_eq!(m.request(&[5, 6, 7, 8]), 0);
    // 99 was generated, but this client has encoded the assistant response as 88.
    assert_eq!(m.request(&[1, 2, 3, 4, 88, 9]), 3);
    assert_eq!(m.cache.len(), 1);
}

#[test]
fn failed_capture_keeps_accounting_consistent() {
    let mut c = Cache::new(cfg(2, 10_000, 2));
    assert!(
        c.stash(1, DEFAULT_CLIENT, &[1; 4], 4, Some(3), 16, None, || anyhow::bail!(
            "allocation failure"
        ))
        .is_err()
    );
    assert_eq!(c.bytes(), 0);
    assert!(c.is_empty());
    assert!(stash(&mut c, 2, &[2; 4], 3, None));
}

#[test]
fn failed_restore_then_new_conversation_reserves_the_live_slot() {
    let mut m = failed_restore();
    let newest = m.cache.get(2).unwrap().bytes();
    assert_eq!(m.request(&[3; 40]), 0);
    assert_eq!(m.cache.len(), 1);
    assert!(
        m.cache.get(1).is_none(),
        "evict the oldest saved conversation"
    );
    assert_eq!(m.cache.bytes(), newest);
    let mut b = vec![2; 40];
    b.extend_from_slice(&[99, 5, 6]);
    assert_eq!(m.request(&b), 41, "the newer saved conversation survives");
    assert_eq!(m.cache.len(), 1);
}

#[test]
fn failed_restore_can_retry_without_losing_the_other_snapshot() {
    let mut m = failed_restore();
    let mut a = vec![1; 30];
    a.extend_from_slice(&[99, 5, 6]);
    assert_eq!(m.request(&a), 31);
    assert_eq!(m.cache.len(), 1);
    assert!(m.cache.get(2).is_some());
}

fn failed_restore() -> Mock {
    let mut m = Mock::new(2, 1_000_000);
    m.request(&[1; 30]);
    m.request(&[2; 40]);
    // A is saved, B is live. Save B with A protected, then fail before taking A out of the cache.
    assert!(stash(
        &mut m.cache,
        m.id,
        &m.key,
        m.rollback.unwrap(),
        Some(1)
    ));
    m.key.clear();
    m.rollback = None;
    m.id = 0;
    assert_eq!(m.cache.len(), 2);
    m
}

// ---- per-client pools (`--session-quota`)

const KB: usize = 1000;

fn quotas(sessions: usize, bytes: usize, q: &[(&str, usize, usize)]) -> SessionConfig {
    cfg(sessions, bytes, 2)
        .with_quotas(
            q.iter().map(|&(c, n, b)| Quota { client: c.into(), sessions: n, max_bytes: b }).collect(),
            Duration::from_secs(600),
            Duration::from_secs(3 * 3600),
        )
        .unwrap()
}

/// a conversation of `n` tokens unique to `tag`
fn conv(tag: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| tag * 1_000_000 + i).collect()
}

/// the next turn: the generated token, then a new user message
fn turn(p: &mut Vec<u32>, tag: u32) {
    p.extend_from_slice(&[99, tag, tag + 1, tag + 2]);
}

#[test]
fn fifty_friend_conversations_never_evict_the_owner() {
    let mut m = Mock::with(quotas(8, 200 * KB, &[("owner", 3, 120 * KB), ("friend", 2, 40 * KB)]));
    let mut main = conv(1, 4000);
    assert_eq!(m.request_as("owner", &main), 0);
    for f in 0..50 {
        // each friend conversation has two turns, so its session is saved and reloaded too
        let mut c = conv(100 + f, 300 + 37 * f as usize);
        m.request_as("friend", &c);
        turn(&mut c, 7);
        m.request_as("friend", &c);
        let (n, b) = m.cache.pool_usage("friend");
        assert!(n <= 2 && b <= 40 * KB);
    }
    assert_eq!(m.cache.pool_usage("owner").0, 1, "the owner's main session was saved and stays");
    turn(&mut main, 5);
    assert_eq!(m.request_as("owner", &main), main.len() - 3, "the owner continues from the shelf");
}

#[test]
fn ten_subagents_do_not_evict_the_long_main_session() {
    // bytes for the main session and one subagent at a time: a tight pool, the admission rule keeps main
    let mut m = Mock::with(quotas(5, 300 * KB, &[("owner", 3, 170 * KB)]));
    let t0 = Instant::now();
    m.cache.set_now(t0);
    let mut main = conv(1, 15_000);
    assert_eq!(m.request_as("owner", &main), 0);
    for a in 0..10u32 {
        // a subagent: its own system prompt, a few growing turns; time passes beyond the recent window
        m.cache.set_now(t0 + Duration::from_secs(60 * (20 + 5 * a as u64)));
        let mut s = conv(10 + a, 1500 + 400 * a as usize);
        for t in 0..3 {
            assert_eq!(m.request_as("owner", &s), if t == 0 { 0 } else { s.len() - 3 });
            turn(&mut s, 7);
        }
        let (n, b) = m.cache.pool_usage("owner");
        assert!(n <= 3 && b <= 170 * KB);
    }
    m.cache.set_now(t0 + Duration::from_secs(60 * 80));
    turn(&mut main, 5);
    assert_eq!(m.request_as("owner", &main), main.len() - 3, "main is still on the shelf after 10 subagents");
}

#[test]
fn smallest_goes_first_recent_is_protected_stale_goes_before_all() {
    let t0 = Instant::now();
    let mut c = Cache::new(quotas(4, 10_000 * KB, &[("owner", 2, 1000 * KB)]));
    c.set_now(t0);
    assert!(stash_as(&mut c, "owner", 1, &conv(1, 5000), 4999, None)); // big, old
    c.set_now(t0 + Duration::from_secs(1200));
    assert!(stash_as(&mut c, "owner", 2, &conv(2, 300), 299, None)); // small, old
    c.set_now(t0 + Duration::from_secs(2400));
    assert!(stash_as(&mut c, "owner", 3, &conv(3, 800), 799, None));
    assert!(c.get(2).is_none() && c.get(1).is_some(), "the cheaper conversation went, not the older one");
    // a smaller newcomer would push out the costlier 1: it is not saved (admission)
    c.set_now(t0 + Duration::from_secs(2500));
    assert!(!stash_as(&mut c, "owner", 4, &conv(4, 900), 899, None));
    assert!(c.get(1).is_some() && c.get(3).is_some());
    // recently used conversations stay while another of the same cost can go
    let mut c = Cache::new(quotas(4, 10_000 * KB, &[("owner", 2, 1000 * KB)]));
    c.set_now(t0);
    assert!(stash_as(&mut c, "owner", 1, &conv(1, 800), 799, None)); // class 10, will be old
    c.set_now(t0 + Duration::from_secs(1200));
    assert!(stash_as(&mut c, "owner", 2, &conv(2, 300), 299, None)); // class 9, recent
    c.set_now(t0 + Duration::from_secs(1300));
    assert!(stash_as(&mut c, "owner", 3, &conv(3, 900), 899, None));
    assert!(c.get(1).is_none() && c.get(2).is_some(), "the recent one stays although it is smaller");
    // everyone recent: the cheapest goes
    assert!(stash_as(&mut c, "owner", 4, &conv(4, 1000), 999, None));
    assert!(c.get(2).is_none() && c.get(3).is_some());
    // a stale conversation goes first, whatever its size
    let mut c = Cache::new(quotas(4, 10_000 * KB, &[("owner", 2, 1000 * KB)]));
    c.set_now(t0);
    assert!(stash_as(&mut c, "owner", 1, &conv(1, 9000), 8999, None));
    c.set_now(t0 + Duration::from_secs(4 * 3600));
    assert!(stash_as(&mut c, "owner", 2, &conv(2, 300), 299, None));
    assert!(stash_as(&mut c, "owner", 3, &conv(3, 300), 299, None));
    assert!(c.get(1).is_none() && c.get(2).is_some());
    // admission: a small conversation is not saved when only a costlier one could make room
    let mut c = Cache::new(quotas(4, 10_000 * KB, &[("owner", 1, 1000 * KB)]));
    c.set_now(t0);
    assert!(stash_as(&mut c, "owner", 1, &conv(1, 9000), 8999, None));
    c.set_now(t0 + Duration::from_secs(1800));
    assert!(!c.stash(2, "owner", &conv(2, 300), 300, Some(299), 1200, None, || panic!("not admitted")).unwrap());
    assert!(c.get(1).is_some());
    // an equally costly one replaces it (least recently used inside the class)
    assert!(stash_as(&mut c, "owner", 3, &conv(3, 9500), 9499, None));
    assert!(c.get(1).is_none() && c.get(3).is_some());
}

#[test]
fn byte_quota_is_per_client_and_unknown_clients_share_default() {
    let mut c = Cache::new(quotas(6, 30 * KB, &[("owner", 2, 20 * KB)]));
    // default gets the rest: 6 - 1 - 2 = 3 conversations, 10 KB
    assert!(stash_as(&mut c, "owner", 1, &conv(1, 2000), 1999, None)); // ~16 KB
    assert!(stash_as(&mut c, "stranger", 2, &conv(2, 500), 499, None)); // ~4 KB
    assert!(stash_as(&mut c, DEFAULT_CLIENT, 3, &conv(3, 500), 499, None));
    assert!(stash_as(&mut c, "other", 4, &conv(4, 500), 499, None));
    assert_eq!(c.pool_usage("owner").0, 1);
    assert_eq!(c.pool_usage(DEFAULT_CLIENT).0, 2, "10 KB holds two ~4 KB conversations");
    assert!(c.get(1).is_some());
    // too big for its own pool: not saved, nobody evicted
    assert!(!c.stash(9, "stranger", &conv(9, 4000), 4000, Some(3999), 16_000, None, || panic!("does not fit")).unwrap());
    assert_eq!(c.len(), 3);
}

#[test]
fn foreign_request_saves_the_live_conversation_into_its_owners_pool() {
    let mut m = Mock::with(quotas(4, 200 * KB, &[("owner", 1, 100 * KB), ("friend", 1, 50 * KB)]));
    let mut main = conv(1, 3000);
    m.request_as("owner", &main);
    // the friend sends the same prefix: the live conversation still goes to the owner's pool
    let mut f = main.clone();
    turn(&mut f, 50);
    m.request_as("friend", &f);
    assert_eq!(m.cache.pool_usage("owner").0, 1);
    assert_eq!(m.client, "friend");
    // and the friend cannot load it
    let mut f2 = main.clone();
    turn(&mut f2, 60);
    assert_eq!(m.cache.best(&f2, "friend", &[], 0, None), None);
    turn(&mut main, 5);
    assert_eq!(m.request_as("owner", &main), main.len() - 3);
}

#[test]
fn without_quotas_client_labels_change_nothing() {
    let mut a = Mock::new(3, 1_000_000);
    let mut b = Mock::new(3, 1_000_000);
    let mut p = vec![conv(1, 40), conv(2, 50), conv(3, 60)];
    for i in 0..12 {
        let k = i % 3;
        let (ra, rb) = (a.request_as(["owner", "friend", "x"][k], &p[k]), b.request(&p[k]));
        assert_eq!(ra, rb);
        assert_eq!(a.cache.len(), b.cache.len());
        turn(&mut p[k], 7);
    }
}

#[test]
fn quota_flags_are_checked() {
    assert_eq!(Quota::parse("Owner=3:2.5").unwrap(), Quota { client: "owner".into(), sessions: 3, max_bytes: 2_500_000_000 });
    for bad in ["owner", "owner=3", "owner=x:1", "owner=3:-1", "=3:1", "a b=1:1", "owner=3:nan"] {
        assert!(Quota::parse(bad).is_err(), "{bad}");
    }
    let q = |c: &str, n, gb: f64| Quota { client: c.into(), sessions: n, max_bytes: (gb * 1e9) as usize };
    let base = || SessionConfig::new(6, 4.0, 2048).unwrap();
    let (r, s) = (Duration::from_secs(600), Duration::from_secs(3600));
    assert!(base().with_quotas(vec![q("owner", 3, 3.0), q("friend", 2, 1.0)], r, s).is_ok());
    assert!(base().with_quotas(vec![q("owner", 4, 3.0), q("friend", 2, 0.5)], r, s).is_err(), "6 slots in quotas, 5 saved");
    assert!(base().with_quotas(vec![q("owner", 2, 3.5), q("friend", 2, 1.0)], r, s).is_err(), "4.5 GB > 4");
    assert!(base().with_quotas(vec![q("owner", 1, 1.0), q("owner", 1, 1.0)], r, s).is_err(), "duplicate");
    assert!(SessionConfig::new(1, 4.0, 2048).unwrap().with_quotas(vec![q("owner", 0, 1.0)], r, s).is_err(), "needs sessions");
    assert!(base().with_quotas(vec![], r, r).is_err(), "stale must exceed recent");
}

// ---- the archive on disk (`--session-dir`)

use dsx_core::server::sessions::{DiskConfig, DiskCopy, DiskQuota, Session, Tiers};
use std::collections::HashMap;
use std::path::PathBuf;

/// What a file holds in this mock: the conversation (the token state stands in for the KV) and its labels.
#[derive(Clone)]
struct File {
    client: String,
    key: Vec<u32>,
    rollback: Option<usize>,
    seq: u64,
}

#[derive(Default)]
struct Disk {
    files: HashMap<PathBuf, File>,
    seq: u64,
    writes: usize,
}

impl Disk {
    fn put(&mut self, client: &str, key: &[u32], rollback: Option<usize>) -> anyhow::Result<DiskCopy> {
        self.seq += 1;
        self.writes += 1;
        let path = PathBuf::from(format!("mem/{}", self.seq));
        self.files.insert(path.clone(), File { client: client.into(), key: key.to_vec(), rollback, seq: self.seq });
        Ok(DiskCopy { path, bytes: key.len() as u64 * 8 })
    }
}

/// The engine side of a tier change: the live sequence is `live`.
struct Engine<'a> {
    disk: &'a mut Disk,
    live: &'a [u32],
}

impl Tiers<Vec<u32>> for Engine<'_> {
    fn capture(&mut self) -> anyhow::Result<Vec<u32>> {
        Ok(self.live.to_vec())
    }
    fn write_live(&mut self, client: &str, key: &[u32], _pos: usize, rollback: Option<usize>) -> anyhow::Result<DiskCopy> {
        assert_eq!(key, self.live);
        self.disk.put(client, key, rollback)
    }
    fn write_saved(&mut self, s: &Session<Vec<u32>>) -> anyhow::Result<DiskCopy> {
        assert_eq!(&*s.key, &s.state.as_ref().unwrap()[..]);
        self.disk.put(&s.client, &s.key, s.rollback)
    }
}

/// Real's request flow with both tiers: select, save the live one, load from host memory or from a file.
struct Arch {
    cache: Cache,
    disk: Disk,
    key: Vec<u32>,
    rollback: Option<usize>,
    id: u64,
    client: String,
    from_disk: usize,
}

impl Arch {
    fn new(cfg: SessionConfig) -> Self {
        Self { cache: Cache::new(cfg), disk: Disk::default(), key: vec![], rollback: None, id: 0, client: DEFAULT_CLIENT.into(), from_disk: 0 }
    }

    fn stash_live(&mut self, protect: Option<u64>) {
        if self.key.is_empty() {
            return;
        }
        let (key, client) = (self.key.clone(), self.client.clone());
        let mut e = Engine { disk: &mut self.disk, live: &key };
        self.cache.stash_tiered(self.id, &client, &key, key.len(), self.rollback, key.len() * 4, protect, &mut e).unwrap();
    }

    fn request(&mut self, client: &str, prompt: &[u32]) -> usize {
        // as Real: clients without a quota are filed under (and compared as) their pool's label
        let owner = self.cache.owner_of(client);
        let client: &str = &owner;
        let foreign = self.cache.isolated() && !self.key.is_empty() && self.client != client;
        let live: Vec<u32> = if foreign { vec![] } else { self.key.clone() };
        if let Some(id) = self.cache.best(prompt, client, &live, live.len(), self.rollback) {
            self.stash_live(Some(id));
            let s = self.cache.get(id).unwrap();
            let state = match &s.state {
                Some(v) => v.clone(),
                None => {
                    self.from_disk += 1;
                    self.disk.files[&s.disk.as_ref().unwrap().path].key.clone()
                }
            };
            assert_eq!(&*s.key, &state[..], "the right sequence must be loaded");
            assert!(!self.cache.isolated() || &*s.client == client, "only the client's own conversations load");
            self.rollback = s.rollback;
            self.key = state;
            if s.prefix {
                self.cache.touch_prefix(client, &self.key);
                self.id = self.cache.begin_session();
                self.client = client.into();
            } else {
                let s = self.cache.take_for_load(id).unwrap();
                self.id = id;
                self.client = s.client.to_string();
            }
        } else if foreign || PrefixReuse::new(prompt, &self.key, self.key.len(), self.rollback).start == 0 {
            self.stash_live(None);
            self.id = self.cache.begin_session();
            self.client = client.into();
        }
        let start = PrefixReuse::new(prompt, &self.key, self.key.len(), self.rollback).start;
        self.key = prompt.to_vec();
        self.rollback = Some(prompt.len() - 1);
        self.cache.discard_related(client, prompt, self.rollback);
        self.key.push(99);
        self.check();
        start
    }

    /// idle time: copy the shelf to disk
    fn idle(&mut self) {
        loop {
            let mut e = Engine { disk: &mut self.disk, live: &[] };
            if !self.cache.persist_one(&mut e) {
                break;
            }
        }
        self.check();
    }

    /// idle checkpoint (or shutdown) of the live conversation
    fn checkpoint(&mut self) -> Option<u64> {
        let (key, client) = (self.key.clone(), self.client.clone());
        let mut e = Engine { disk: &mut self.disk, live: &key };
        let id = self.cache.checkpoint_live(&client, &key, key.len(), self.rollback, key.len() * 4, &mut e);
        self.check();
        id
    }

    /// the process restarts: host memory and the GPU are gone, the files are adopted in the order they were written
    fn restart(&mut self, cfg: SessionConfig, now: Option<Instant>) {
        self.cache = Cache::new(cfg);
        if let Some(t) = now {
            self.cache.set_now(t);
        }
        let mut files: Vec<(PathBuf, File)> = self.disk.files.iter().map(|(p, f)| (p.clone(), f.clone())).collect();
        files.sort_by_key(|(_, f)| f.seq);
        for (path, f) in files {
            let bytes = f.key.len() as u64 * 8;
            let used = now.unwrap_or_else(Instant::now);
            self.cache.adopt(&f.client, f.key.clone(), f.key.len(), f.rollback, DiskCopy { path, bytes }, f.key.len() * 4, used);
        }
        self.cache.settle_archive();
        self.key.clear();
        self.rollback = None;
        self.id = 0;
        self.check();
    }

    /// Files and entries agree exactly (no orphan file, no entry without its file); budgets hold.
    fn check(&mut self) {
        for f in self.cache.drain_trash() {
            assert!(self.disk.files.remove(&f.path).is_some(), "trash names a file that does not exist");
        }
        assert!(self.cache.bytes() <= self.cache.max_bytes());
        let d = self.cache.disk_config().map_or(0, |d| d.max_bytes);
        assert!(self.cache.disk_bytes() <= d, "archive over budget: {} > {d}", self.cache.disk_bytes());
        let on_disk: usize = self.disk.files.values().map(|f| f.key.len() * 8).sum();
        assert_eq!(on_disk, self.cache.disk_bytes(), "accounted bytes = bytes of the files that exist");
    }
}

fn archive(shelf: &[(&str, usize, usize)], disk_sessions: usize, disk_bytes: usize, dq: &[(&str, usize, usize)]) -> SessionConfig {
    let mut d = DiskConfig::new(PathBuf::from("mem"), disk_sessions, 1.0, 7.0, Duration::from_secs(20), 0.0).unwrap();
    d.max_bytes = disk_bytes;
    let d = d.with_quotas(dq.iter().map(|&(c, n, b)| DiskQuota { client: c.into(), sessions: n, max_bytes: b, ttl: None }).collect()).unwrap();
    let total_n = 1 + shelf.iter().map(|q| q.1).sum::<usize>() + 1;
    let total_b = shelf.iter().map(|q| q.2).sum::<usize>() + 10 * KB;
    quotas(total_n, total_b, shelf).with_disk(d).unwrap()
}

#[test]
fn a_conversation_pushed_off_the_shelf_comes_back_from_disk() {
    // the shelf holds one conversation: every switch pushes the other one out
    let mut m = Arch::new(archive(&[("owner", 1, 40 * KB)], 10, 1000 * KB, &[("owner", 8, 900 * KB)]));
    let mut a = conv(1, 4000);
    let mut b = conv(2, 3000);
    let c = conv(3, 2000);
    assert_eq!(m.request("owner", &a), 0);
    assert_eq!(m.request("owner", &b), 0);
    assert_eq!(m.request("owner", &c), 0);
    assert_eq!(m.cache.pool_usage("owner").0, 1, "one on the shelf");
    turn(&mut a, 5);
    assert_eq!(m.request("owner", &a), a.len() - 3, "a continues from disk, nothing is read again");
    turn(&mut b, 6);
    assert_eq!(m.request("owner", &b), b.len() - 3);
    assert!(m.from_disk >= 1);
}

#[test]
fn a_subagent_storm_never_costs_the_main_conversation() {
    // a tight shelf (main + one subagent), a roomy archive: 40 subagents, four at a time, three turns each
    let mut m = Arch::new(archive(&[("owner", 2, 200 * KB)], 64, 4000 * KB, &[("owner", 48, 3500 * KB)]));
    let t0 = Instant::now();
    m.cache.set_now(t0);
    let mut main = conv(1, 15_000);
    assert_eq!(m.request("owner", &main), 0);
    for wave in 0..10u32 {
        m.cache.set_now(t0 + Duration::from_secs(60 * (5 + 7 * wave as u64)));
        let mut subs: Vec<Vec<u32>> = (0..4).map(|i| conv(100 + wave * 4 + i, 1800 + 300 * i as usize)).collect();
        for t in 0..3 {
            for s in subs.iter_mut() {
                let start = m.request("owner", s);
                assert_eq!(start, if t == 0 { 0 } else { s.len() - 3 }, "a subagent's own turns are never read again");
                turn(s, 7);
            }
        }
        if wave % 3 == 2 {
            m.idle();
        }
        // the main agent gets a tool result between waves
        turn(&mut main, 5);
        assert_eq!(m.request("owner", &main), main.len() - 3, "main survives wave {wave}");
    }
    let (n, b) = m.cache.archive_usage("owner");
    assert!(n <= 48 && b <= 3500 * KB);
}

#[test]
fn fifty_friend_conversations_cannot_touch_the_owners_archive() {
    let mut m = Arch::new(archive(&[("owner", 1, 60 * KB), ("friend", 1, 20 * KB)], 20, 2000 * KB, &[("owner", 10, 1500 * KB), ("friend", 4, 200 * KB)]));
    let mut main = conv(1, 6000);
    m.request("owner", &main);
    let side = conv(2, 4000);
    m.request("owner", &side); // main leaves the shelf
    m.idle();
    for f in 0..50 {
        let mut c = conv(1000 + f, 800 + 53 * f as usize);
        m.request("friend", &c);
        turn(&mut c, 9);
        m.request("friend", &c);
        let (n, b) = m.cache.archive_usage("friend");
        assert!(n <= 4 && b <= 200 * KB);
    }
    turn(&mut main, 5);
    assert_eq!(m.request("owner", &main), main.len() - 3, "the owner's main conversation is still there");
}

#[test]
fn a_restart_keeps_every_conversation() {
    let cfg = || archive(&[("owner", 2, 120 * KB), ("friend", 1, 30 * KB)], 20, 3000 * KB, &[("owner", 10, 2000 * KB), ("friend", 4, 500 * KB)]);
    let mut m = Arch::new(cfg());
    let mut a = conv(1, 5000);
    let mut b = conv(2, 3000);
    let mut f = conv(3, 2500);
    m.request("owner", &a);
    m.request("owner", &b);
    m.request("friend", &f);
    m.idle(); // the shelf reaches the disk
    turn(&mut f, 4);
    m.request("friend", &f);
    assert!(m.checkpoint().is_some(), "shutdown: the live conversation too");
    m.restart(cfg(), None);
    turn(&mut a, 5);
    assert_eq!(m.request("owner", &a), a.len() - 3, "a after the restart: from disk");
    turn(&mut b, 6);
    assert_eq!(m.request("owner", &b), b.len() - 3);
    turn(&mut f, 7);
    assert_eq!(m.request("friend", &f), f.len() - 3, "the friend's live conversation came back too");
    assert_eq!(m.from_disk, 3);
}

#[test]
fn one_file_per_conversation_older_units_are_deleted() {
    let mut m = Arch::new(archive(&[("owner", 2, 200 * KB)], 20, 3000 * KB, &[("owner", 10, 2500 * KB)]));
    let mut a = conv(1, 3000);
    for i in 0..12u32 {
        assert_eq!(m.request("owner", &a), if i == 0 { 0 } else { a.len() - 3 });
        m.checkpoint();
        turn(&mut a, 10 + i);
    }
    let a_files = m.disk.files.values().filter(|f| f.key[..3000] == a[..3000]).count();
    assert_eq!(a_files, 1, "only the newest unit of a stays on disk");
    // a crash now: the newest unit is what comes back
    m.restart(archive(&[("owner", 2, 200 * KB)], 20, 3000 * KB, &[("owner", 10, 2500 * KB)]), None);
    assert_eq!(m.request("owner", &a), a.len() - 3, "from the last checkpoint (it holds the generated token): only the new message is read");
}

#[test]
fn a_restored_file_stays_until_a_newer_unit_is_written() {
    let cfg = || archive(&[("owner", 1, 40 * KB)], 10, 1000 * KB, &[("owner", 8, 900 * KB)]);
    let mut m = Arch::new(cfg());
    let mut a = conv(1, 4000);
    m.request("owner", &a);
    m.checkpoint();
    m.restart(cfg(), None);
    turn(&mut a, 5);
    assert_eq!(m.request("owner", &a), a.len() - 3);
    // crash before anything newer reached the disk: the old file still serves
    m.restart(cfg(), None);
    turn(&mut a, 6);
    assert_eq!(m.request("owner", &a), a.len() - 7, "back to the checkpoint, two turns read again");
    m.checkpoint();
    assert_eq!(m.disk.files.len(), 1, "the newer checkpoint replaced the old file");
}

#[test]
fn expired_files_go_and_admission_keeps_costlier_ones() {
    let t0 = Instant::now();
    let cfg = archive(&[("owner", 1, 40 * KB)], 10, 100 * KB, &[("owner", 3, 90 * KB)]);
    let mut m = Arch::new(cfg);
    m.cache.set_now(t0);
    // two big conversations fill most of the archive
    let a = conv(1, 4000);
    let b = conv(2, 4000);
    m.request("owner", &a);
    m.request("owner", &b);
    m.request("owner", &conv(3, 900)); // a leaves the shelf for disk, b follows at idle
    m.idle();
    assert_eq!(m.cache.archive_usage("owner").0, 2);
    // small newcomers (a lower size class) would need a costlier file to go: refused, both stay
    m.cache.set_now(t0 + Duration::from_secs(3600));
    m.request("owner", &conv(4, 900));
    m.request("owner", &conv(5, 900));
    m.idle();
    assert!(m.disk.files.values().filter(|f| f.key.len() >= 4000).count() == 2, "the costly ones stay");
    // eight days later every file has expired
    m.cache.set_now(t0 + Duration::from_secs(8 * 86400));
    m.cache.expire();
    m.check();
    assert_eq!(m.cache.archive_usage("owner").0, 0);
    assert!(m.disk.files.is_empty());
}

#[test]
fn disk_quota_flags_are_checked() {
    assert_eq!(
        DiskQuota::parse("Owner=200:60:14").unwrap(),
        DiskQuota { client: "owner".into(), sessions: 200, max_bytes: 60_000_000_000, ttl: Some(Duration::from_secs(14 * 86400)) }
    );
    assert_eq!(DiskQuota::parse("friend=50:15").unwrap().ttl, None);
    for bad in ["owner", "owner=3", "owner=x:1", "owner=3:-1", "=3:1", "owner=3:1:x", "owner=3:1:2:3"] {
        assert!(DiskQuota::parse(bad).is_err(), "{bad}");
    }
    let d = || DiskConfig::new(PathBuf::from("x"), 100, 50.0, 7.0, Duration::from_secs(20), 20.0).unwrap();
    let q = |c: &str, n, gb: f64| DiskQuota { client: c.into(), sessions: n, max_bytes: (gb * 1e9) as usize, ttl: None };
    assert!(d().with_quotas(vec![q("owner", 60, 30.0), q("friend", 40, 20.0)]).is_ok());
    assert!(d().with_quotas(vec![q("owner", 80, 30.0), q("friend", 40, 10.0)]).is_err(), "120 > 100 conversations");
    assert!(d().with_quotas(vec![q("owner", 10, 40.0), q("friend", 10, 20.0)]).is_err(), "60 > 50 GB");
    assert!(DiskConfig::new(PathBuf::from("x"), 0, 50.0, 7.0, Duration::from_secs(20), 20.0).is_err());
    assert!(SessionConfig::new(1, 2.0, 2048).unwrap().with_disk(d()).is_err(), "the archive needs the shelf");
}

#[test]
fn clients_without_a_quota_get_their_own_conversations_back() {
    // "stranger" has no quota: it lives in `default`, and must still find its conversation there (and on disk)
    let mut m = Arch::new(archive(&[("owner", 1, 60 * KB)], 10, 1000 * KB, &[("owner", 4, 400 * KB)]));
    let mut a = conv(1, 3000);
    let b = conv(2, 2500);
    m.request("stranger", &a);
    m.request("owner", &b);
    turn(&mut a, 5);
    assert_eq!(m.request("stranger", &a), a.len() - 3, "from the default shelf");
    m.checkpoint();
    m.restart(archive(&[("owner", 1, 60 * KB)], 10, 1000 * KB, &[("owner", 4, 400 * KB)]), None);
    turn(&mut a, 6);
    assert_eq!(m.request("stranger", &a), a.len() - 3, "from the default archive after a restart");
    // and the owner still cannot load it
    let mut o = a.clone();
    turn(&mut o, 7);
    assert_eq!(m.cache.best(&o, "owner", &[], 0, None), None);
}

#[test]
fn a_regenerated_answer_is_newer_though_shorter() {
    // the same turn answered twice (regenerate, or a retry after a restart): the later, shorter branch must win
    let t0 = Instant::now();
    let cfg = || archive(&[("owner", 1, 60 * KB)], 10, 1000 * KB, &[("owner", 4, 400 * KB)]);
    let mut m = Arch::new(cfg());
    m.cache.set_now(t0);
    let p = conv(1, 3000);
    m.request("owner", &p);
    // a long first answer
    for t in 0..50 {
        m.key.push(500 + t);
    }
    m.checkpoint();
    // regenerate: the same prompt again, a short answer, checkpointed later
    m.cache.set_now(t0 + Duration::from_secs(60));
    let mut q = p.clone();
    q.push(7); // the request differs in one token so the prompt is read from the rewind point
    m.request("owner", &q);
    m.checkpoint();
    m.restart(cfg(), Some(t0 + Duration::from_secs(120)));
    assert_eq!(m.disk.files.len(), 1, "one file per conversation");
    let kept = m.disk.files.values().next().unwrap().key.clone();
    assert_eq!(kept.len(), q.len() + 1, "the newer, shorter branch stayed");
    let mut q2 = q.clone();
    turn(&mut q2, 9);
    assert_eq!(m.request("owner", &q2), q2.len() - 3);
}

fn prefix(c: &mut Cache, client: &str, key: &[u32]) -> bool {
    let mut disk = Disk::default();
    c.capture_prefix(client, key, key.len() * 4, &mut Engine { disk: &mut disk, live: key }).unwrap()
}

#[test]
fn prefix_lifecycle_loads_by_copy_and_outlives_related_conversations() {
    let mut m = Mock::new(5, 100_000);
    let a = [1, 2, 3, 4, 10, 11];
    let b = [1, 2, 3, 4, 20, 21];
    assert_eq!(m.request(&a), 0);
    assert!(prefix(&mut m.cache, DEFAULT_CLIENT, &a[..4]));
    let id = m.cache.best(&b, DEFAULT_CLIENT, &m.key, m.key.len(), m.rollback).unwrap();
    let s = m.cache.get(id).unwrap();
    let ptr = s.state.as_ref().unwrap().as_ptr();
    assert!(s.prefix);
    assert_eq!((s.pos, s.rollback), (4, None));
    assert_eq!(m.request(&b), 4, "the live LCP cannot block a usable prefix");
    assert_ne!(m.id, id, "a prefix starts a new conversation");
    assert_eq!(m.cache.get(id).unwrap().state.as_ref().unwrap().as_ptr(), ptr);
    m.cache.discard_related(DEFAULT_CLIENT, &a, Some(a.len() - 1));
    assert!(m.cache.get(id).is_some());
    assert_eq!(m.cache.pool_usage(DEFAULT_CLIENT).0, 1);
    assert_eq!(m.cache.bytes(), Cache::entry_bytes(4, 16).unwrap());
}

#[test]
fn prefixes_are_never_archived_even_on_eviction_checkpoint_shutdown_or_restart() {
    let cfg = || archive(&[("owner", 2, 100 * KB)], 10, 1000 * KB, &[("owner", 8, 900 * KB)]);
    let mut m = Arch::new(cfg());
    let s = [1, 2, 3, 4];
    assert!(prefix(&mut m.cache, "owner", &s));
    let id = m.cache.best(&[1, 2, 3, 4, 10, 11], "owner", &[], 0, None).unwrap();
    assert_eq!(m.request("owner", &[1, 2, 3, 4, 10, 11]), 4);
    m.checkpoint();
    m.idle();
    assert!(m.cache.get(id).is_some(), "discard_related and supersede_older skip prefixes");
    assert_eq!(m.cache.unpersisted(), 0, "shutdown has no prefix work left");
    assert!(m.disk.files.values().all(|f| f.key != s));
    m.request("owner", &[2; 8]);
    m.request("owner", &[3; 8]);
    m.idle();
    assert!(m.cache.get(id).is_some(), "with an archive the conversations go to disk first, the prefix stays");
    assert!(m.disk.files.values().all(|f| f.key != s), "and it is never spilled");
    m.restart(cfg(), None);
    assert!(m.cache.best(&[1, 2, 3, 4, 30, 31], "owner", &[], 0, None).is_none(), "no prefix was adopted");
}

#[test]
fn prefix_limit_is_lru_per_pool_and_duplicates_only_touch() {
    let mut config = quotas(10, 100_000, &[("owner", 4, 40_000), ("friend", 2, 20_000)]);
    config.prefix_max = 2;
    let mut c = Cache::new(config);
    let t = Instant::now();
    c.set_now(t);
    assert!(prefix(&mut c, "owner", &[1; 4]));
    let first = c.best(&[1; 5], "owner", &[], 0, None).unwrap();
    assert!(prefix(&mut c, "owner", &[2; 4]));
    let second = c.best(&[2; 5], "owner", &[], 0, None).unwrap();
    let bytes = c.bytes();
    c.set_now(t + Duration::from_secs(1));
    let mut no_capture = CaptureForbidden;
    assert!(!c.capture_prefix("owner", &[1; 4], 16, &mut no_capture).unwrap());
    assert_eq!(c.get(first).unwrap().used(), t + Duration::from_secs(1));
    assert_eq!(c.bytes(), bytes);
    assert!(prefix(&mut c, "friend", &[7; 4]));
    assert!(prefix(&mut c, "owner", &[3; 4]));
    assert!(c.get(first).is_some() && c.get(second).is_none());
    assert_eq!(c.pool_usage("owner").0, 2);
    assert_eq!(c.pool_usage("friend").0, 1);
    assert!(c.best(&[1; 5], "friend", &[], 0, None).is_none());
    assert!(prefix(&mut c, "unknown", &[8; 4]));
    assert!(!prefix(&mut c, "another", &[8; 4]), "unknown clients share default");
    assert!(c.best(&[8; 5], "another", &[], 0, None).is_some());
}

struct CaptureForbidden;
impl Tiers<Vec<u32>> for CaptureForbidden {
    fn capture(&mut self) -> anyhow::Result<Vec<u32>> { panic!("must not capture") }
    fn write_live(&mut self, _: &str, _: &[u32], _: usize, _: Option<usize>) -> anyhow::Result<DiskCopy> { panic!("must not archive") }
    fn write_saved(&mut self, _: &Session<Vec<u32>>) -> anyhow::Result<DiskCopy> { panic!("must not archive") }
}

#[test]
fn disabled_or_unaffordable_prefixes_do_not_capture_or_change_the_cache() {
    for (n, bytes, limit) in [(4, 100_000, 0), (1, 100_000, 4), (4, 1, 4)] {
        let mut config = cfg(n, bytes, 2);
        config.prefix_max = limit;
        let mut c = Cache::new(config);
        let before = (c.len(), c.bytes(), c.disk_bytes());
        assert!(!c.capture_prefix(DEFAULT_CLIENT, &[1; 4], 16, &mut CaptureForbidden).unwrap());
        assert_eq!((c.len(), c.bytes(), c.disk_bytes()), before);
    }
    let mut config = cfg(3, 100_000, 2);
    config.prefix_max = 0;
    let mut baseline = Mock::with(config.clone());
    let mut disabled = Mock::with(config);
    for prompt in [&[1; 10][..], &[2; 10], &[1; 12], &[2; 12]] {
        assert!(!prefix(&mut disabled.cache, DEFAULT_CLIENT, &prompt[..4]));
        assert_eq!(baseline.request(prompt), disabled.request(prompt));
        assert_eq!(baseline.key, disabled.key);
        assert_eq!(baseline.cache.bytes(), disabled.cache.bytes());
        assert_eq!(baseline.cache.len(), disabled.cache.len());
    }
}

#[test]
fn failed_prefix_capture_leaves_a_usable_live_sequence_and_consistent_accounting() {
    struct Failed;
    impl Tiers<Vec<u32>> for Failed {
        fn capture(&mut self) -> anyhow::Result<Vec<u32>> { anyhow::bail!("host allocation failed") }
        fn write_live(&mut self, _: &str, _: &[u32], _: usize, _: Option<usize>) -> anyhow::Result<DiskCopy> { unreachable!() }
        fn write_saved(&mut self, _: &Session<Vec<u32>>) -> anyhow::Result<DiskCopy> { unreachable!() }
    }
    let mut m = Mock::new(4, 100_000);
    m.request(&[1; 10]);
    let live = m.key.clone();
    assert!(m.cache.capture_prefix(DEFAULT_CLIENT, &[1; 4], 16, &mut Failed).is_err());
    assert_eq!(m.key, live);
    assert_eq!((m.cache.len(), m.cache.bytes()), (0, 0));
    assert_eq!(m.request(&[1; 10]), 9);
}

#[test]
fn best_compares_usable_positions_including_ranking_and_ram_ties() {
    let mut c = Cache::new(cfg(5, 100_000, 2));
    let prompt = [1, 2, 3, 4, 5, 6, 7, 8];
    assert!(stash(&mut c, 1, &prompt[..5], 4, None));
    // The live LCP is longer, but the live state cannot rewind to it. The old rule returned None.
    assert_eq!(c.best(&prompt, DEFAULT_CLIENT, &[1, 2, 3, 4, 5, 6, 99], 7, Some(7)), Some(1));
    // A longer saved LCP with an earlier rollback must not beat the usable continuation.
    assert!(stash(&mut c, 2, &[1, 2, 3, 4, 5, 6, 99], 2, None));
    assert_eq!(c.best(&prompt, DEFAULT_CLIENT, &[], 0, None), Some(1));
    assert_eq!(c.best(&prompt, DEFAULT_CLIENT, &prompt[..4], 4, Some(3)), None, "less than min_tokens gained");
    assert_eq!(c.best(&prompt, DEFAULT_CLIENT, &prompt[..5], 5, Some(4)), None, "live wins a tie");
    assert_eq!(c.best(&prompt, DEFAULT_CLIENT, &prompt[..3], 3, Some(2)), Some(1));
}

#[test]
fn identical_prompt_rewinds_one_token_on_gpu_shelf_and_disk() {
    let config = || archive(&[("owner", 2, 100 * KB)], 10, 1000 * KB, &[("owner", 8, 900 * KB)]);
    let mut m = Arch::new(config());
    let p = [1, 2, 3, 4, 5];
    assert_eq!(m.request("owner", &p), 0);
    assert_eq!(m.request("owner", &p), p.len() - 1);
    m.request("owner", &[2; 8]);
    assert_eq!(m.request("owner", &p), p.len() - 1);
    m.checkpoint();
    m.restart(config(), None);
    assert_eq!(m.request("owner", &p), p.len() - 1);
    assert_eq!(m.from_disk, 1);
    let mut one = Mock::new(2, 100_000);
    assert_eq!(one.request(&[7]), 0);
    assert_eq!(one.request(&[7]), 0);
}

#[test]
fn prefix_switch_cannot_borrow_the_targets_slot_and_archives_the_outgoing_conversation() {
    let mut c = Cache::new(archive(&[("owner", 1, 100 * KB)], 10, 1000 * KB, &[("owner", 8, 900 * KB)]));
    assert!(prefix(&mut c, "owner", &[1; 4]));
    let target = c.best(&[1; 5], "owner", &[], 0, None).unwrap();
    let mut disk = Disk::default();
    let live = [2; 8];
    let live_id = c.begin_session();
    let mut tier = Engine { disk: &mut disk, live: &live };
    assert!(c.stash_tiered(live_id, "owner", &live, live.len(), Some(7), 32, Some(target), &mut tier).unwrap());
    assert!(c.get(target).unwrap().state.is_some());
    assert!(c.get(live_id).unwrap().state.is_none());
    assert!(c.get(live_id).unwrap().on_disk());
    assert_eq!(c.pool_usage("owner").0, 1);
    c.begin_session();
    assert!(c.get(target).is_some() && c.get(live_id).is_some());
    assert_eq!(disk.writes, 1);
}

#[test]
fn a_ram_candidate_wins_a_tie_with_a_disk_candidate() {
    let mut c = Cache::new(archive(&[("owner", 2, 100 * KB)], 10, 1000 * KB, &[("owner", 8, 900 * KB)]));
    let key = [1, 2, 3, 4, 5];
    let ram = c.begin_session();
    assert!(stash_as(&mut c, "owner", ram, &key, 4, None));
    let copy = DiskCopy { path: "mem/tie".into(), bytes: 40 };
    c.adopt("owner", key.to_vec(), 5, Some(4), copy, 20, Instant::now());
    assert_eq!(c.best(&[1, 2, 3, 4, 5, 6], "owner", &[], 0, None), Some(ram));
}

#[test]
fn the_final_prompt_token_still_separates_unrelated_conversations() {
    let mut c = Cache::new(cfg(3, 100_000, 2));
    assert!(stash(&mut c, 1, &[1, 2, 3, 99], 2, None));
    c.discard_related(DEFAULT_CLIENT, &[1, 2, 4], Some(2));
    assert!(c.get(1).is_some(), "moving rewind back must not merge prompts differing in their last token");
    c.discard_related(DEFAULT_CLIENT, &[1, 2, 3], Some(2));
    assert!(c.get(1).is_none());
}

#[test]
fn with_an_archive_a_prefix_spills_a_costlier_conversation_and_keeps_half_the_slots() {
    // pool of 4 slots, all taken by long conversations: the prefix still gets in (one of them goes to disk),
    // and at most 2 prefixes (half the slots) stay on the shelf
    let mut c = Cache::new(archive(&[("owner", 4, 1000 * KB)], 30, 10_000 * KB, &[("owner", 20, 9000 * KB)]));
    let mut disk = Disk::default();
    let mut ids = vec![];
    for i in 0..4u32 {
        let key: Vec<u32> = std::iter::repeat(100 + i).take(64).collect();
        let id = c.begin_session();
        let mut tier = Engine { disk: &mut disk, live: &key };
        assert!(c.stash_tiered(id, "owner", &key, key.len(), Some(63), 32, None, &mut tier).unwrap());
        ids.push(id);
    }
    assert_eq!(c.pool_usage("owner").0, 4);
    assert!(prefix(&mut c, "owner", &[1; 4]));
    assert!(ids.iter().any(|&id| c.get(id).is_some_and(|s| s.state.is_none() && s.on_disk())), "a long conversation went to disk");
    assert!(prefix(&mut c, "owner", &[2; 4]));
    assert!(prefix(&mut c, "owner", &[3; 4]));
    let prefixes = (1..=3u32).filter(|&k| c.best(&[k, k, k, k, 9], "owner", &[], 0, None).is_some()).count();
    assert_eq!(prefixes, 2, "half of 4 slots");
}

#[test]
fn selected_disk_session_survives_superseding_archive_of_live_branch() {
    // Incident 2026-10-28: a smaller rollback on disk is selected, the outgoing live
    // branch cannot fit the shelf, and archive_live used to delete the selected file.
    let mut c = Cache::new(archive(&[("owner", 0, 0)], 10, 1000 * KB, &[("owner", 8, 900 * KB)]));
    let key = vec![7u32; 64];
    c.adopt("owner", key.clone(), 64, Some(47), DiskCopy { path: "mem/selected".into(), bytes: 1000 },
        256, Instant::now() - Duration::from_secs(60));
    let selected = c.best(&key, "owner", &[], 0, None).unwrap();
    let outgoing = vec![7u32; 80];
    let mut disk = Disk::default();
    let mut tier = Engine { disk: &mut disk, live: &outgoing };
    let id = c.begin_session();
    assert!(c.stash_tiered(id, "owner", &outgoing, 80, Some(63), 320, Some(selected), &mut tier).unwrap());
    assert!(c.get(selected).is_some(), "selected file must stay until its load finishes");
    assert!(c.drain_trash().iter().all(|f| f.path != std::path::Path::new("mem/selected")));
    assert!(c.take_for_load(selected).is_some());
}
