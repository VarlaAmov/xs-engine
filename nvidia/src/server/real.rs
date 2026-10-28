//! The GPU engine behind the server: GpuModel with prefix reuse.
//!
//! The KV state of the last conversation (prompt + generated ids) stays on the GPU. A request reuses the longest
//! common token prefix: if the new prompt extends everything cached, it just continues; if it diverges inside the
//! generated part (clients often re-send the assistant turn differently), it rewinds to the snapshot taken at the
//! end of the previous prompt; otherwise it starts over.
//!
//! Other conversations wait on the host-memory shelf and, with `--session-dir`, in the archive on disk
//! (`sessions.rs`): a conversation loads from whichever tier holds it, streaming from its file straight into the
//! GPU. While idle the engine copies the shelf to disk and checkpoints the live conversation; on shutdown it saves
//! both, and at startup it adopts the files, so a restart costs a file read instead of reading the prompt again.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::engine::{Engine, Finish, GenParams, GenStats};
use super::sample::Sampler;
use super::uncertainty::TokenStats;
use super::session_store::{DiskStore, Fingerprint, file_identity, fnv64, instant_to_unix_ms, unix_ms, unix_ms_to_instant};
use super::sessions::{DEFAULT_CLIENT, DiskCopy, PrefixReuse, Session, SessionCache, SessionConfig, Tiers};
use crate::gpumodel::{GpuModel, Placement, SessionMeta, SessionSnapshot, Snapshot, Staging, VisionWeights};
use crate::model::Model;

/// pinned host memory a session streams through between the GPU and its file
const STAGING_BYTES: usize = 32 << 20;

/// Mirrors the `gen` command's flags.
#[derive(Debug, Clone)]
pub struct RealEngineArgs {
    pub model: PathBuf,
    pub engram: Option<PathBuf>,
    pub ctx: usize,
    pub chunk: usize,
    pub gpu0_gb: f64,
    pub gpu1_gb: f64,
    pub arena_gb: f64,
    pub rank: Option<PathBuf>,
    /// vision weights: `vision.safetensors` or the official checkpoint directory (None = text only)
    pub vision: Option<PathBuf>,
    pub sessions: SessionConfig,
    /// first user message boundary, resolved once from the server tokenizer
    pub user_token: u32,
    pub assistant_token: Option<u32>,
}

/// Tokens kept between the rollback point and the `<｜Assistant｜>` header (XS_REWIND_MARGIN, default 8).
fn rewind_margin() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("XS_REWIND_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(8))
}

struct Real {
    gm: GpuModel<'static>,
    cached: Vec<u32>,
    snap: Option<Snapshot>,
    stats: GenStats,
    vision: Option<VisionWeights>,
    sessions: SessionCache<SessionSnapshot>,
    live_id: u64,
    user_token: u32,
    assistant_token: Option<u32>,
    /// whose request made the live conversation: it goes back to that client's pool
    live_client: std::sync::Arc<str>,
    /// the archive's files and the staging buffer to stream them (None: host memory only)
    store: Option<DiskStore>,
    staging: Option<Staging>,
    /// tokens of the newest file of the live conversation (0: none): the idle checkpoint waits for growth
    live_disk: usize,
    decode_heat: std::collections::HashMap<u64, Vec<u32>>,
    last_expire: Instant,
    #[cfg(test)]
    top1_logits: Vec<f32>,
}

// GPU state is created and used only on the engine thread (EngineHandle::spawn builds us there).
unsafe impl Send for Real {}

/// Runs on the engine thread (`EngineHandle::spawn`), so GPU contexts are created there.
pub fn real_engine(args: &RealEngineArgs) -> Result<Box<dyn Engine>> {
    Ok(Box::new(load_real(args)?))
}

fn load_real(args: &RealEngineArgs) -> Result<Real> {
    let m: &'static Model = Box::leak(Box::new(if args.model.is_dir() {
        Model::load_hf(&args.model)?
    } else {
        Model::load(&args.model, args.engram.as_deref())?
    }));
    let pl = Placement { gpu0_gb: args.gpu0_gb, aux_gb: args.gpu1_gb, arena_gb: args.arena_gb, prefill_chunk: args.chunk, rank: args.rank.clone() };
    let vision = args.vision.as_deref().map(VisionWeights::load).transpose()?;
    let gm = GpuModel::with_placement(m, -1, args.ctx, pl)?;
    let mut r = Real {
        gm,
        cached: vec![],
        snap: None,
        stats: GenStats::default(),
        vision,
        sessions: SessionCache::new(args.sessions.clone()),
        live_id: 0,
        user_token: args.user_token,
        assistant_token: args.assistant_token,
        live_client: DEFAULT_CLIENT.into(),
        store: None,
        staging: None,
        live_disk: 0,
        decode_heat: Default::default(),
        last_expire: Instant::now(),
        #[cfg(test)]
        top1_logits: vec![],
    };
    if let Some(d) = args.sessions.disk.as_ref().filter(|_| r.sessions.archived()) {
        let engram = args.engram.as_deref().map(file_identity).transpose()?.unwrap_or(0);
        let fp = Fingerprint { weights: fnv64(file_identity(&args.model)?, &engram.to_le_bytes()), layout: r.gm.session_layout_id() };
        r.store = Some(DiskStore::open(&d.dir, fp, d.reserve_bytes)?);
        r.staging = Some(Staging::new(STAGING_BYTES)?);
        r.adopt_archive();
    }
    Ok(r)
}

/// The engine side of a tier change (`sessions::Tiers`): host snapshots and files of the live sequence.
struct Tier<'a> {
    gm: &'a GpuModel<'static>,
    snap: Option<&'a Snapshot>,
    store: Option<&'a DiskStore>,
    staging: Option<&'a mut Staging>,
}

impl Tiers<SessionSnapshot> for Tier<'_> {
    fn capture(&mut self) -> Result<SessionSnapshot> {
        let bytes = self.gm.session_snapshot_bytes(self.snap);
        let s = self.gm.save_session(self.snap)?;
        anyhow::ensure!(s.heap_bytes() == bytes, "session footprint changed during capture");
        Ok(s)
    }

    fn write_live(&mut self, client: &str, key: &[u32], pos: usize, rollback: Option<usize>) -> Result<DiskCopy> {
        let store = self.store.context("no archive")?;
        let st = self.staging.as_deref_mut().context("no staging buffer")?;
        anyhow::ensure!(pos == self.gm.pos && rollback == self.snap.map(|s| s.pos), "the live sequence moved");
        let meta = self.gm.live_meta(self.snap);
        let mut w = store.create(client, key, pos, rollback, unix_ms(), &meta.encode(), meta.data_len)?;
        self.gm.write_session_stream(self.snap, &mut w, st)?;
        let (path, bytes) = w.finish()?;
        Ok(DiskCopy { path, bytes })
    }

    fn write_saved(&mut self, s: &Session<SessionSnapshot>) -> Result<DiskCopy> {
        let store = self.store.context("no archive")?;
        let snap = s.state.as_ref().context("not in host memory")?;
        let mut w = store.create(&s.client, &s.key, s.pos, s.rollback, instant_to_unix_ms(s.used()), &snap.meta().encode(), snap.meta().data_len)?;
        w.write(snap.data())?;
        let (path, bytes) = w.finish()?;
        Ok(DiskCopy { path, bytes })
    }
}

impl Real {
    fn tier(&mut self) -> (Tier<'_>, &mut SessionCache<SessionSnapshot>) {
        (Tier { gm: &self.gm, snap: self.snap.as_ref(), store: self.store.as_ref(), staging: self.staging.as_mut() }, &mut self.sessions)
    }

    /// Delete the files the cache let go.
    fn sweep(&mut self) {
        let trash = self.sessions.drain_trash();
        if let Some(store) = &self.store {
            for f in trash {
                store.remove(&f.path);
            }
        }
    }

    /// Startup: every file of these weights and this layout joins the archive, oldest use first.
    fn adopt_archive(&mut self) {
        let Some(store) = &self.store else { return };
        let t0 = Instant::now();
        let mut files = store.scan();
        files.sort_by_key(|f| f.used_ms);
        for f in files {
            let copy = DiskCopy { path: f.path.clone(), bytes: f.file_bytes };
            let used = unix_ms_to_instant(f.used_ms);
            self.sessions.adopt(&f.client, f.key, f.pos, f.rollback, copy, f.meta_len + f.data_len, used);
        }
        self.sessions.settle_archive();
        self.sweep();
        let summary: Vec<String> = ["owner", "friend", DEFAULT_CLIENT]
            .iter()
            .map(|c| {
                let (n, b) = self.sessions.archive_usage(c);
                format!("{c} {n} ({:.2} GB)", b as f64 / 1e9)
            })
            .collect();
        tracing::info!(dir = %store_dir(&self.store).display(), conversations = self.sessions.len(), gb = self.sessions.disk_bytes() as f64 / 1e9,
            ms = t0.elapsed().as_millis() as u64, pools = %summary.join(", "), "session archive");
    }

    fn stash_live(&mut self, protect: Option<u64>) -> Result<bool> {
        if self.cached.is_empty() { return Ok(false); }
        if std::env::var("XS_SESSION_HEAT").is_ok_and(|v| v == "1") {
            self.decode_heat.retain(|id, _| self.sessions.get(*id).is_some());
            self.decode_heat.insert(self.live_id, self.gm.decode_heat());
        }
        let bytes = self.gm.session_snapshot_bytes(self.snap.as_ref());
        let (id, client, key, pos, rollback) = (self.live_id, self.live_client.clone(), std::mem::take(&mut self.cached), self.gm.pos, self.snap.as_ref().map(|s| s.pos));
        let (mut t, sessions) = self.tier();
        let r = sessions.stash_tiered(id, &client, &key, pos, rollback, bytes, protect, &mut t);
        self.cached = key;
        self.sweep();
        r
    }

    /// Stream a session file into the live sequence.
    fn load_file(&mut self, path: &Path) -> Result<()> {
        let store = self.store.as_ref().context("no archive")?;
        let r = store.open_reader(path)?;
        let meta = SessionMeta::decode(&r.meta)?;
        let st = self.staging.as_mut().context("no staging buffer")?;
        self.gm.read_session_stream(&meta, r, st, &mut self.snap)
    }

    fn select_session(&mut self, key: &[u32], client: &std::sync::Arc<str>) -> Result<()> {
        if !self.sessions.enabled() { return Ok(()); }
        // with quotas the live conversation belongs to one client: another client's request saves it into its owner's
        // pool first (the GPU may still reuse a common prefix, which is the same tokens' KV)
        let foreign = self.sessions.isolated() && !self.cached.is_empty() && self.live_client != *client;
        let live: &[u32] = if foreign { &[] } else { &self.cached };
        if let Some(id) = self.sessions.best(key, client, live, self.gm.pos, self.snap.as_ref().map(|s| s.pos)) {
            let from = self.live_id;
            let from_client = self.live_client.clone();
            let save = Instant::now();
            let saved = self.stash_live(Some(id))?;
            let save_ms = save.elapsed().as_secs_f64() * 1e3;
            let peak_host_bytes = self.sessions.bytes();
            let saved_bytes = self.sessions.get(from).map_or(0, |s| s.bytes());
            let load = Instant::now();
            let s = self.sessions.get(id).context("selected session was removed during save")?;
            let tokens = s.key.len();
            let prefix = s.prefix;
            let loaded_bytes = s.disk.as_ref().map_or(s.bytes(), |d| if s.state.is_some() { s.bytes() } else { d.bytes as usize });
            let has_file = s.disk.is_some();
            let (memory, r) = match &s.state {
                Some(state) => (state.memory_kind(), self.gm.load_session(state, &mut self.snap)),
                None => {
                    let path = s.disk.as_ref().expect("a saved conversation is in one tier at least").path.clone();
                    ("disk", self.load_file(&path))
                }
            };
            if let Err(e) = r {
                // A partially restored GPU sequence cannot be used as a prefix on the next request.
                self.gm.reset(); self.cached.clear(); self.snap = None; self.live_id = 0; self.live_disk = 0;
                if memory != "disk" {
                    return Err(e);
                }
                // an unreadable or damaged file: forget it and read the prompt instead
                tracing::warn!(session = id, client = &**client, tokens, error = %format!("{e:#}"), "session file failed to load; reading the prompt");
                if let Some(s) = self.sessions.take(id) {
                    if let (Some(store), Some(d)) = (&self.store, s.disk) {
                        store.remove(&d.path);
                    }
                }
                self.live_id = self.sessions.begin_session();
                self.live_client = client.clone();
                return Ok(());
            }
            if !prefix {
                if let Some(heat) = self.decode_heat.get(&id) { self.gm.restore_decode_heat(heat); }
            }
            let load_ms = load.elapsed().as_secs_f64() * 1e3;
            if prefix {
                // The borrowed snapshot remains charged and alive throughout H2D and subsequent requests.
                self.cached = self.sessions.get(id).unwrap().key.to_vec();
                self.sessions.touch_prefix(client, &self.cached);
                self.live_id = self.sessions.begin_session();
                self.live_client = client.clone();
                self.live_disk = 0;
            } else {
                let s = self.sessions.take_for_load(id).unwrap();
                self.cached = s.key.into_vec();
                self.live_id = id;
                self.live_client = s.client.clone();
                self.live_disk = if has_file { self.cached.len() } else { 0 };
                drop(s.state); // target stayed charged and alive until all H2D copies completed
            }
            let (pool_sessions, pool_bytes) = self.sessions.pool_usage(client);
            tracing::info!(from, from_client = &*from_client, session = id, client = &**client, tokens, saved, saved_bytes, loaded_bytes, peak_host_bytes, memory, prefix, save_ms, load_ms,
                pool_sessions, pool_bytes, host_bytes = self.sessions.bytes(), limit_bytes = self.sessions.max_bytes(),
                disk_bytes = self.sessions.disk_bytes(), "session switch");
        } else {
            let reuse = PrefixReuse::new(key, &self.cached, self.gm.pos, self.snap.as_ref().map(|s| s.pos));
            if reuse.start == 0 || foreign {
                let (from, from_client) = (self.live_id, self.live_client.clone());
                let tokens = self.cached.len();
                let t = Instant::now();
                let saved = self.stash_live(None)?;
                self.live_id = self.sessions.begin_session();
                self.live_client = client.clone();
                self.live_disk = 0;
                if tokens > 0 {
                    let memory = self.sessions.get(from).map(|s| if s.state.is_some() { s.state.as_ref().unwrap().memory_kind() } else { "disk" }).unwrap_or("none");
                    let saved_bytes = self.sessions.get(from).map_or(0, |s| s.bytes());
                    let (pool_sessions, pool_bytes) = self.sessions.pool_usage(&from_client);
                    tracing::info!(from, from_client = &*from_client, session = self.live_id, client = &**client, tokens, saved, memory,
                        saved_bytes,
                        save_ms = t.elapsed().as_secs_f64() * 1e3, load_ms = 0.0,
                        pool_sessions, pool_bytes, host_bytes = self.sessions.bytes(), limit_bytes = self.sessions.max_bytes(),
                        disk_bytes = self.sessions.disk_bytes(), "session switch");
                }
            }
        }
        Ok(())
    }

    /// The live conversation as a file of its own, when it is long enough and has grown since its newest file.
    fn checkpoint_live(&mut self, force: bool) -> bool {
        let n = self.cached.len();
        let min = self.sessions.min_tokens();
        let grown = n >= self.live_disk + min.max(self.live_disk / 4);
        if !self.sessions.archived() || n < min || self.gm.pos != n || n <= self.live_disk || !(force || grown) {
            return false;
        }
        let bytes = self.gm.session_snapshot_bytes(self.snap.as_ref());
        let (client, key, rollback) = (self.live_client.clone(), std::mem::take(&mut self.cached), self.snap.as_ref().map(|s| s.pos));
        let pos = self.gm.pos;
        let (mut t, sessions) = self.tier();
        sessions.checkpoint_live(&client, &key, pos, rollback, bytes, &mut t);
        self.cached = key;
        // also when the archive said no: the next try waits for more growth
        self.live_disk = n;
        self.sweep();
        true
    }
}

fn store_dir(s: &Option<DiskStore>) -> PathBuf {
    s.as_ref().map_or_else(PathBuf::new, |s| s.dir().to_path_buf())
}

#[cfg(test)]
mod tests;

impl Real {
    fn generate_inner(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32, Option<TokenStats>) -> bool) -> Result<Finish> {
        let t0 = Instant::now();
        #[cfg(test)]
        self.top1_logits.clear();
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        // the prefix-cache key: every image position carries the same token id, so key those by image content
        let mut key = prompt.to_vec();
        for im in p.images.iter() {
            for (i, k) in key[im.start..im.start + im.len()].iter_mut().enumerate() {
                *k = 0xf000_0000 | ((im.hash as u32 ^ (im.hash >> 32) as u32 ^ i as u32) & 0x0fff_ffff);
            }
        }
        // clients without a quota share the `default` pool: keep and compare the pool's label
        let client = self.sessions.owner_of(&p.client);
        self.select_session(&key, &client)?;
        // at least one prompt token must be fed to get the next-token logits
        let lcp = key.iter().zip(&self.cached).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
        let start = if lcp == self.cached.len() && self.gm.pos == self.cached.len() {
            lcp
        } else if let Some(s) = self.snap.as_ref().filter(|s| s.pos <= lcp) {
            self.gm.restore(s)?;
            // the newest file covers at most the rewind point of what the live sequence now is
            self.live_disk = self.live_disk.min(s.pos);
            s.pos
        } else {
            self.gm.reset();
            self.live_disk = 0;
            0
        };
        tracing::info!(client = &*client, prompt_tokens = prompt.len(), cached_tokens = self.cached.len(), lcp,
            reused_tokens = start, rollback = self.snap.as_ref().map(|s| s.pos), "prefix reuse decision");
        self.gm.ensure_context_capacity(prompt.len(), true)?;
        // images not already in the KV cache go through the vision tower; their span rows replace the embeddings
        self.gm.images.clear();
        for im in p.images.iter().filter(|im| im.start + im.len() > start) {
            let vw = self.vision.as_ref().ok_or_else(|| anyhow::anyhow!("image input needs the vision weights (--vision)"))?;
            let tv = Instant::now();
            let rows = self.gm.encode_image(vw, im)?;
            tracing::info!(patches = im.n_vit_h * im.n_vit_w, tokens = im.len(), ms = tv.elapsed().as_millis() as u64, "image encoded");
            self.gm.images.push((im.start, rows));
        }
        let boundary = prompt.iter().position(|&t| t == self.user_token).filter(|&b| {
            self.sessions.prefixes_enabled() && start < b && b >= self.sessions.min_tokens()
                && b < prompt.len() && !p.images.iter().any(|im| im.start < b)
        });
        let pb = self.gm.pb.take().expect("prefill buffers");
        let r = (|| {
            let mut next = start;
            if let Some(b) = boundary {
                self.gm.prefill(&pb, &prompt[next..b])?;
                let bytes = self.gm.session_snapshot_bytes(None);
                let mut tier = Tier { gm: &self.gm, snap: None, store: self.store.as_ref(), staging: self.staging.as_mut() };
                if let Err(e) = self.sessions.capture_prefix(&client, &key[..b], bytes, &mut tier) {
                    tracing::warn!(client = &*client, tokens = b, error = %format!("{e:#}"), "prefix capture failed; continuing the prompt");
                }
                next = b;
            }
            // Complete the decoder cone before taking the rewind point. Use prefill for the final token too:
            // step uses decode expert/attention kernels with different rounding and transport.
            // A cancelled assistant is often omitted when the client appends a new user turn, and the encoder then
            // MERGES the two user messages ("...u1\n\nu2"): BPE re-tokenises the end of u1 (its last token fuses with
            // the newline), so the real common prefix ends a token or two BEFORE the <｜Assistant｜> header (28.10:
            // lcp = header - 1 on the live server, the rollback point one token too late, 9 s of re-reading 11K tokens).
            // Keep the rewind point `rewind_margin()` tokens before the header, ahead of any such re-tokenisation.
            let margin = rewind_margin();
            let rewind = self.assistant_token.and_then(|id| prompt.iter().rposition(|&t| t == id))
                .map(|n| n.saturating_sub(margin).max(next)).unwrap_or(prompt.len() - 1).min(prompt.len() - 1);
            if next < rewind {
                self.gm.prefill(&pb, &prompt[next..rewind])?;
            }
            self.snap = Some(self.gm.snapshot()?);
            self.gm.prefill(&pb, &prompt[rewind..])
        })();
        self.gm.pb = Some(pb);
        self.gm.images.clear();
        let mut logits = r?;
        if self.sessions.enabled() {
            self.sessions.discard_related(&client, &key, self.snap.as_ref().map(|s| s.pos));
        }
        self.cached = key;
        let prefill = t0.elapsed();
        let td = Instant::now();
        let mut sampler = Sampler::from_params(p);
        let mut fin = Finish::Length;
        let mut hes = p.hesitation.clone();
        for _ in 0..p.max_new {
            let (tok, stats) = sampler.sample_with_penalty(&mut logits, hes.as_ref(), p.logprobs);
            #[cfg(test)]
            self.top1_logits.push(logits[tok as usize]);
            if hes.as_ref().is_some_and(|h| tok == h.close) {
                hes = None;
            }
            if !emit(tok, stats) {
                fin = Finish::Cancelled;
                break;
            }
            if self.gm.pos + 1 >= self.gm.max_ctx() {
                break;
            }
            logits = self.gm.step(tok)?;
            self.cached.push(tok);
        }
        self.stats = GenStats { prefill: Some(prefill), decode: Some(td.elapsed()), cached_prompt_tokens: start };
        let _ = Duration::ZERO;
        Ok(fin)
    }
}

impl Engine for Real {
    fn generate(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish> {
        self.generate_with_stats(prompt, p, &mut |tok, _| emit(tok))
    }

    fn generate_with_stats(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32, Option<TokenStats>) -> bool) -> Result<Finish> {
        let result = self.generate_inner(prompt, p, emit);
        if result.is_err() && self.sessions.enabled() {
            // Failed prefill/decode may already have overwritten sequence buffers. Only host snapshots remain
            // valid; never let a stale live key win a tie against them on the next request.
            self.gm.reset(); self.cached.clear(); self.snap = None; self.live_id = 0; self.live_disk = 0;
        }
        self.sweep();
        result
    }

    fn cancelled(&mut self) { self.checkpoint_live(true); }

    fn last_stats(&self) -> GenStats {
        GenStats { prefill: self.stats.prefill, decode: self.stats.decode, cached_prompt_tokens: self.stats.cached_prompt_tokens }
    }

    /// Between requests: expire old files now and then; after `--session-idle-s` without requests copy the shelf to
    /// disk (costliest first, one per call) and then checkpoint the live conversation.
    fn idle(&mut self, idle: Duration) -> bool {
        let Some(wait) = self.sessions.disk_config().map(|d| d.idle) else { return false };
        if self.last_expire.elapsed() >= Duration::from_secs(60) {
            self.sessions.expire();
            self.sweep();
            self.last_expire = Instant::now();
        }
        if idle < wait {
            return false;
        }
        let (mut t, sessions) = self.tier();
        if sessions.persist_one(&mut t) {
            self.sweep();
            return true;
        }
        self.checkpoint_live(false)
    }

    /// Before exit: the live conversation (unless its newest file is current) and the whole shelf go to disk.
    fn shutdown(&mut self) {
        if !self.sessions.archived() {
            return;
        }
        let t0 = Instant::now();
        self.checkpoint_live(true);
        loop {
            let (mut t, sessions) = self.tier();
            if !sessions.persist_one(&mut t) {
                break;
            }
        }
        self.sweep();
        tracing::info!(conversations = self.sessions.len(), gb = self.sessions.disk_bytes() as f64 / 1e9, ms = t0.elapsed().as_millis() as u64,
            left_in_memory = self.sessions.unpersisted(), "shutdown: conversations saved to disk");
    }
}
