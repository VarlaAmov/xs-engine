//! Lossless, host-resident sequence snapshots. No tensor is requantized and no expert state is changed.
//! The same bulk state streams to and from session files (`dsx_core::server::session_store`) through a pinned
//! staging buffer, without a host copy of the whole sequence.

use super::{GpuModel, Snapshot};
use crate::gpu::check;
use anyhow::{Result, ensure};
use cudarc::driver::sys as cu;
use dsx_core::server::session_store::{FNV0, Reader, Writer, fnv64};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::NonNull;

const PAGE: usize = 4096;
/// bump when the bulk state's layout or meaning changes: older session files are then rejected
const SESSION_FORMAT: u64 = 1;
const META_VERSION: u64 = 1;

/// Kept separate from the expert arena's HostBuf: exact page accounting, fallible pinning, and no hugepage
/// compaction or parallel fault-in of gigabytes on the inference thread. Failed registration owns no CUDA resource.
struct SessionBuffer {
    ptr: NonNull<u8>,
    bytes: usize,
    pinned: bool,
}

impl SessionBuffer {
    fn allocation_bytes(bytes: usize) -> usize {
        bytes.div_ceil(PAGE) * PAGE
    }

    fn new(bytes: usize) -> Result<Self> {
        let bytes = Self::allocation_bytes(bytes).max(PAGE);
        let layout = Layout::from_size_align(bytes, PAGE)?;
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| anyhow::anyhow!("session allocation of {bytes} bytes failed"))?;
        let mut b = Self {
            ptr,
            bytes,
            pinned: false,
        };
        if !std::env::var("DSX_SESSION_PAGEABLE").is_ok_and(|v| v == "1") {
            // PORTABLE: the same host pages can be copied in every CUDA context. No device mapping is needed.
            let r = unsafe { cu::cuMemHostRegister_v2(ptr.as_ptr().cast(), bytes, 1) };
            match check(r, "session cuMemHostRegister") {
                Ok(()) => b.pinned = true,
                Err(e) => tracing::warn!(bytes, error = %e, "session memory: pageable fallback"),
            }
        }
        Ok(b)
    }

    fn slice(&self, off: usize, bytes: usize) -> &[u8] {
        assert!(off <= self.bytes && bytes <= self.bytes - off);
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(off), bytes) }
    }

    fn slice_mut(&mut self, off: usize, bytes: usize) -> &mut [u8] {
        assert!(off <= self.bytes && bytes <= self.bytes - off);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(off), bytes) }
    }
}

impl Drop for SessionBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.pinned {
                cu::cuMemHostUnregister(self.ptr.as_ptr().cast::<c_void>());
            }
            dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align(self.bytes, PAGE).unwrap(),
            );
        }
    }
}

/// Pinned, page-aligned host memory through which a session streams between the GPU and a file.
pub struct Staging(SessionBuffer);

impl Staging {
    pub fn new(bytes: usize) -> Result<Self> {
        Ok(Self(SessionBuffer::new(bytes)?))
    }
    pub fn bytes(&self) -> usize {
        self.0.bytes
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Rollback {
    pos: usize,
    counters: Vec<(usize, usize)>,
    engram_len: usize,
}

/// Everything about a sequence except its bulk state: what a session file's metadata region holds.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionMeta {
    pub pos: usize,
    counters: Vec<(usize, usize)>,
    engram_hist: Vec<i64>,
    topk_rows: Vec<i32>,
    candidates: Vec<bool>,
    topk_k: usize,
    comp_src: usize,
    idxk_src: usize,
    rollback: Option<Rollback>,
    /// bytes of the bulk state (window rings, compressor carry, compressed KV and index keys, rewind point)
    pub data_len: usize,
}

impl SessionMeta {
    pub fn encode(&self) -> Vec<u8> {
        fn u(b: &mut Vec<u8>, x: u64) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fn pairs(b: &mut Vec<u8>, v: &[(usize, usize)]) {
            u(b, v.len() as u64);
            for &(x, y) in v {
                u(b, x as u64);
                u(b, y as u64);
            }
        }
        let mut b = vec![];
        for x in [META_VERSION, self.pos as u64, self.data_len as u64, self.topk_k as u64, self.comp_src as u64, self.idxk_src as u64] {
            u(&mut b, x);
        }
        pairs(&mut b, &self.counters);
        u(&mut b, self.engram_hist.len() as u64);
        for &x in &self.engram_hist {
            b.extend_from_slice(&x.to_le_bytes());
        }
        u(&mut b, self.topk_rows.len() as u64);
        for &x in &self.topk_rows {
            b.extend_from_slice(&x.to_le_bytes());
        }
        u(&mut b, self.candidates.len() as u64);
        b.extend(self.candidates.iter().map(|&c| c as u8));
        match &self.rollback {
            None => u(&mut b, 0),
            Some(r) => {
                u(&mut b, 1);
                u(&mut b, r.pos as u64);
                u(&mut b, r.engram_len as u64);
                pairs(&mut b, &r.counters);
            }
        }
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self> {
        struct C<'a>(&'a [u8], usize);
        impl C<'_> {
            fn take(&mut self, n: usize) -> Result<&[u8]> {
                ensure!(self.1.checked_add(n).is_some_and(|e| e <= self.0.len()), "session metadata truncated");
                let s = &self.0[self.1..self.1 + n];
                self.1 += n;
                Ok(s)
            }
            fn u(&mut self) -> Result<u64> {
                Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            }
            fn n(&mut self, elem: usize) -> Result<usize> {
                let n = self.u()? as usize;
                ensure!(n.checked_mul(elem).is_some_and(|b| b <= self.0.len()), "session metadata: bad length");
                Ok(n)
            }
            fn pairs(&mut self) -> Result<Vec<(usize, usize)>> {
                let n = self.n(16)?;
                (0..n).map(|_| Ok((self.u()? as usize, self.u()? as usize))).collect()
            }
        }
        let mut c = C(b, 0);
        ensure!(c.u()? == META_VERSION, "session metadata version");
        let (pos, data_len, topk_k, comp_src, idxk_src) = (c.u()? as usize, c.u()? as usize, c.u()? as usize, c.u()? as usize, c.u()? as usize);
        let counters = c.pairs()?;
        let n = c.n(8)?;
        let engram_hist = (0..n).map(|_| Ok(i64::from_le_bytes(c.take(8)?.try_into().unwrap()))).collect::<Result<_>>()?;
        let n = c.n(4)?;
        let topk_rows = (0..n).map(|_| Ok(i32::from_le_bytes(c.take(4)?.try_into().unwrap()))).collect::<Result<_>>()?;
        let n = c.n(1)?;
        let candidates = c.take(n)?.iter().map(|&x| x != 0).collect();
        let rollback = match c.u()? {
            0 => None,
            1 => Some(Rollback { pos: c.u()? as usize, engram_len: c.u()? as usize, counters: c.pairs()? }),
            _ => anyhow::bail!("session metadata: bad rollback flag"),
        };
        ensure!(c.1 == b.len(), "session metadata: trailing bytes");
        Ok(Self { pos, counters, engram_hist, topk_rows, candidates, topk_k, comp_src, idxk_src, rollback, data_len })
    }
}

pub struct SessionSnapshot {
    meta: SessionMeta,
    data: SessionBuffer,
}

impl SessionSnapshot {
    pub fn memory_kind(&self) -> &'static str {
        if self.data.pinned {
            "pinned"
        } else {
            "pageable"
        }
    }
    pub fn heap_bytes(&self) -> usize {
        let m = &self.meta;
        self.data.bytes
            + std::mem::size_of_val(&*m.counters)
            + std::mem::size_of_val(&*m.engram_hist)
            + std::mem::size_of_val(&*m.topk_rows)
            + std::mem::size_of_val(&*m.candidates)
            + m.rollback.as_ref().map_or(0, |r| std::mem::size_of_val(&*r.counters))
    }
    pub fn pos(&self) -> usize {
        self.meta.pos
    }
    pub fn meta(&self) -> &SessionMeta {
        &self.meta
    }
    /// the bulk state, as a session file stores it
    pub fn data(&self) -> &[u8] {
        self.data.slice(0, self.meta.data_len)
    }
}

impl GpuModel<'_> {
    // Archives remain in the original BF16 wire layout, independent of live KV representation
    // and capacity. This keeps existing files and rollback releases readable without rewriting archives.
    fn session_dtoh(&self, dst: &mut [u8], src: u64, offset: usize) -> Result<()> {
        if src & 3 == 0 { return self.g.dtoh(dst, src + offset as u64); }
        ensure!(offset % 64 == 0 && dst.len() % 64 == 0, "unaligned packed cache export");
        let temp = self.g.alloc(dst.len().min(32 << 20))?;
        for (i, chunk) in dst.chunks_mut(temp.bytes).enumerate() {
            let first = (offset + i * temp.bytes) / 2;
            self.g.launch(self.k["cache_unpack"], ((chunk.len() / 2).div_ceil(256) as u32, 1, 1), 256, 0,
                &[crate::gpu::Arg::P(src), crate::gpu::Arg::P(temp.ptr), crate::gpu::Arg::I(first as i32), crate::gpu::Arg::I((chunk.len() / 2) as i32)])?;
            self.g.dtoh(chunk, temp.ptr)?;
            self.g.sync()?;
        }
        Ok(())
    }

    fn session_htod(&self, dst: u64, offset: usize, src: &[u8]) -> Result<()> {
        if dst & 3 == 0 { return self.g.htod(dst + offset as u64, src); }
        ensure!(offset % 64 == 0 && src.len() % 64 == 0, "unaligned packed cache import");
        let block = if dst & 3 == 1 { 16 } else { 32 };
        let capacity = src.len().min(32 << 20);
        let temp = self.g.alloc(capacity + 4)?;
        let error = temp.ptr + capacity as u64;
        self.g.memset(error, 4)?;
        for (i, chunk) in src.chunks(capacity).enumerate() {
            let first = (offset + i * capacity) / 2;
            self.g.htod_sync(temp.ptr, chunk)?;
            let n = chunk.len() / 2;
            self.g.launch(self.k["cache_pack_bf16"], ((n / block).div_ceil(4) as u32, 1, 1), 128, 0,
                &[crate::gpu::Arg::P(temp.ptr), crate::gpu::Arg::P(dst & !3), crate::gpu::Arg::I(n as i32),
                  crate::gpu::Arg::I(block as i32), crate::gpu::Arg::I((block == 16) as i32), crate::gpu::Arg::I(first as i32), crate::gpu::Arg::P(error)])?;
            self.g.sync()?;
        }
        ensure!(self.g.download::<u32>(error, 1)?[0] == 0, "archive contains cache values outside its FP4 grid");
        Ok(())
    }

    fn session_data_bytes(&self, snap: Option<&Snapshot>) -> usize {
        let c = &self.m.cfg;
        let mut bytes = 0;
        for l in &self.layers {
            bytes += l.window.bytes;
            if let Some((k, s)) = &l.comp_state {
                bytes += k.bytes + s.bytes;
            }
            if l.comp_cache.is_some() {
                bytes += l.n_comp * c.head_dim * 2;
            }
            if l.idx_cache.is_some() {
                bytes += l.n_idx * c.idx_dim * 2;
            }
        }
        if let Some(s) = snap {
            bytes += s.rings.bytes + s.states.bytes;
        }
        bytes
    }

    /// Heap footprint (the SessionCache adds the inline struct and token key). Includes page rounding, full
    /// Engram history, all row counters and the previous prompt's rollback buffers and metadata.
    pub fn session_snapshot_bytes(&self, snap: Option<&Snapshot>) -> usize {
        SessionBuffer::allocation_bytes(self.session_data_bytes(snap)).max(PAGE)
            + self.layers.len() * size_of::<(usize, usize)>()
            + self.engram_hist.len() * size_of::<i64>()
            + self.topk_rows.len() * size_of::<i32>()
            + self.candidates.len() * size_of::<bool>()
            + snap.map_or(0, |s| s.counters.len() * size_of::<(usize, usize)>())
    }

    /// The bulk state as (device pointer, bytes) pieces, in the order host snapshots and files keep it. `counters`
    /// are the occupied compressed / index rows per layer (the live ones to save, the saved ones to load); `snap`
    /// adds the prompt rewind point.
    fn segments(&self, counters: &[(usize, usize)], snap: Option<&Snapshot>) -> Result<Vec<(u64, usize)>> {
        let c = &self.m.cfg;
        ensure!(counters.len() == self.layers.len(), "invalid session shape");
        let mut v = Vec::with_capacity(self.layers.len() * 5 + 2);
        for (l, &(nc, ni)) in self.layers.iter().zip(counters) {
            v.push((l.window.ptr, l.window.bytes));
            if let Some((k, s)) = &l.comp_state {
                v.push((k.ptr, k.bytes));
                v.push((s.ptr, s.bytes));
            }
            if let Some(b) = &l.comp_cache {
                ensure!(Self::cache_bytes(self.packed_kv, nc * c.head_dim, 16) <= b.bytes, "session KV overflow");
                v.push((self.cache_pointer(b.ptr, 16), nc * c.head_dim * 2));
            }
            if let Some(b) = &l.idx_cache {
                ensure!(Self::cache_bytes(self.packed_kv, ni * c.idx_dim, 32) <= b.bytes, "session index overflow");
                v.push((self.cache_pointer(b.ptr, 32), ni * c.idx_dim * 2));
            }
        }
        if let Some(s) = snap {
            v.push((s.rings.ptr, s.rings.bytes));
            v.push((s.states.ptr, s.states.bytes));
        }
        Ok(v)
    }

    /// What a session file of the live sequence records besides the bulk state.
    pub fn live_meta(&self, snap: Option<&Snapshot>) -> SessionMeta {
        SessionMeta {
            pos: self.pos,
            counters: self.layers.iter().map(|l| (l.n_comp, l.n_idx)).collect(),
            engram_hist: self.engram_hist.clone(),
            topk_rows: self.topk_rows.clone(),
            candidates: self.candidates.clone(),
            topk_k: self.topk_k,
            comp_src: self.comp_src,
            idxk_src: self.idxk_src,
            rollback: snap.map(|s| Rollback { pos: s.pos, counters: s.counters.clone(), engram_len: s.engram_len }),
            data_len: self.session_data_bytes(snap),
        }
    }

    /// A fingerprint of the bulk state's layout for this model and context: session files from another layout
    /// (another model, another engine format) are not loaded.
    pub fn session_layout_id(&self) -> u64 {
        let c = &self.m.cfg;
        let mut h = FNV0;
        for x in [SESSION_FORMAT, c.head_dim as u64, c.idx_dim as u64, c.window as u64, self.layers.len() as u64, self.ring_bytes() as u64, self.state_bytes() as u64] {
            h = fnv64(h, &x.to_le_bytes());
        }
        for l in &self.layers {
            let st = l.comp_state.as_ref().map_or((0, 0), |(k, s)| (k.bytes, s.bytes));
            for x in [l.window.bytes, st.0, st.1, l.comp_cache.is_some() as usize, l.idx_cache.is_some() as usize] {
                h = fnv64(h, &(x as u64).to_le_bytes());
            }
        }
        h
    }

    fn check_switchable(&self) -> Result<()> {
        ensure!(
            self.verify_t == 0 && self.vpass.is_none(),
            "cannot switch during speculative verification"
        );
        Ok(())
    }

    /// The rewind point's device buffers, when the session to load has one (reuses the existing Snapshot so
    /// inactive sessions cost no VRAM).
    fn rollback_buffers(&self, needed: bool, snap: &mut Option<Snapshot>) -> Result<()> {
        if needed && snap.is_none() {
            let (rb, sb, n) = (self.ring_bytes(), self.state_bytes(), self.layers.len());
            *snap = Some(Snapshot {
                pos: 0,
                rings: self.g.alloc(rb * n)?,
                states: self.g.alloc(sb * n)?,
                counters: vec![],
                engram_len: 0,
            });
        }
        Ok(())
    }

    /// Call only at a completed token/prompt boundary. Compressed KV and index keys are copied through their
    /// occupied rows, rather than merely saving counters as the same-sequence Snapshot does.
    pub fn save_session(&self, snap: Option<&Snapshot>) -> Result<SessionSnapshot> {
        self.check_switchable()?;
        self.g.sync()?;
        let meta = self.live_meta(snap);
        let mut data = SessionBuffer::new(meta.data_len)?;
        let mut off = 0;
        let mut r = Ok(());
        for (src, bytes) in self.segments(&meta.counters, snap)? {
            if bytes > 0 {
                if let Err(e) = self.session_dtoh(data.slice_mut(off, bytes), src, 0) {
                    r = Err(e);
                    break;
                }
            }
            off += bytes;
        }
        // Even on an enqueue error, keep host storage alive until earlier async copies finish.
        let sync = self.g.sync();
        r?;
        sync?;
        Ok(SessionSnapshot { meta, data })
    }

    /// Restore the live sequence AND its prompt rewind point. Reuse the one existing GPU Snapshot so inactive
    /// sessions cost no VRAM. Transport epochs (seq_base/Xch flags), expert residency and workers stay global.
    pub fn load_session(&mut self, s: &SessionSnapshot, snap: &mut Option<Snapshot>) -> Result<()> {
        self.check_switchable()?;
        let m = &s.meta;
        ensure!(m.counters.len() == self.layers.len() && m.pos <= self.max_ctx, "invalid session shape");
        self.g.sync()?;
        self.ensure_context_capacity(m.pos, false)?;
        self.rollback_buffers(m.rollback.is_some(), snap)?;
        let segs = self.segments(&m.counters, if m.rollback.is_some() { snap.as_ref() } else { None })?;
        ensure!(segs.iter().map(|s| s.1).sum::<usize>() == m.data_len, "session layout mismatch");
        let mut off = 0;
        let mut r = Ok(());
        for (dst, bytes) in segs {
            if bytes > 0 {
                if let Err(e) = self.session_htod(dst, 0, s.data.slice(off, bytes)) {
                    r = Err(e);
                    break;
                }
            }
            off += bytes;
        }
        let sync = self.g.sync();
        r?;
        sync?;
        self.apply_session_meta(m, snap);
        #[cfg(test)]
        if self.packed_kv {
            let restored = self.save_session(snap.as_ref())?;
            if let Some(bad) = restored.data().iter().zip(s.data()).position(|(a,b)| a != b) {
                let mut off = 0;
                let segment = self.segments(&m.counters, snap.as_ref())?.iter().position(|&(_, n)| {
                    let yes = bad >= off && bad < off + n; off += n; yes
                });
                anyhow::bail!("packed RAM roundtrip byte {bad}, segment {segment:?}: expected {:?}, actual {:?}",
                    &s.data()[bad..(bad+32).min(m.data_len)], &restored.data()[bad..(bad+32).min(m.data_len)]);
            }
        }
        Ok(())
    }

    /// Write the live sequence's bulk state into `w` through `st` (`w` was created for `live_meta(snap).data_len`).
    pub fn write_session_stream(&self, snap: Option<&Snapshot>, w: &mut Writer, st: &mut Staging) -> Result<()> {
        self.check_switchable()?;
        self.g.sync()?;
        let counters: Vec<_> = self.layers.iter().map(|l| (l.n_comp, l.n_idx)).collect();
        let cap = st.0.bytes;
        let mut fill = 0;
        let r = (|| -> Result<()> {
            for (src, n) in self.segments(&counters, snap)? {
                let mut done = 0;
                while done < n {
                    let k = (n - done).min(cap - fill);
                    self.session_dtoh(st.0.slice_mut(fill, k), src, done)?;
                    fill += k;
                    done += k;
                    if fill == cap {
                        self.g.sync()?;
                        w.write(st.0.slice(0, fill))?;
                        fill = 0;
                    }
                }
            }
            if fill > 0 {
                self.g.sync()?;
                w.write(st.0.slice(0, fill))?;
            }
            Ok(())
        })();
        // the staging buffer must not be refilled while a copy into it may still run
        let sync = self.g.sync();
        r?;
        sync
    }

    /// Load a session file's bulk state into the live sequence through `st`, check its CRC, then take its
    /// metadata. On error the live sequence is undefined: the caller resets it.
    pub fn read_session_stream(&mut self, meta: &SessionMeta, mut r: Reader, st: &mut Staging, snap: &mut Option<Snapshot>) -> Result<()> {
        self.check_switchable()?;
        ensure!(meta.counters.len() == self.layers.len() && meta.pos <= self.max_ctx, "invalid session shape");
        ensure!(meta.data_len == r.info.data_len && meta.pos == r.info.pos, "session metadata disagrees with its header");
        self.g.sync()?;
        self.ensure_context_capacity(meta.pos, false)?;
        self.rollback_buffers(meta.rollback.is_some(), snap)?;
        let segs = self.segments(&meta.counters, if meta.rollback.is_some() { snap.as_ref() } else { None })?;
        ensure!(segs.iter().map(|s| s.1).sum::<usize>() == meta.data_len, "session layout mismatch");
        let cap = st.0.bytes;
        let (mut avail, mut used) = (0usize, 0usize);
        let res = (|| -> Result<()> {
            for (dst, n) in segs {
                let mut done = 0;
                while done < n {
                    if used == avail {
                        // the previous copies out of the staging buffer finish before it is refilled
                        self.g.sync()?;
                        avail = r.read(st.0.slice_mut(0, cap))?;
                        used = 0;
                        ensure!(avail > 0, "session file ended early");
                    }
                    let k = (n - done).min(avail - used);
                    self.session_htod(dst, done, st.0.slice(used, k))?;
                    used += k;
                    done += k;
                }
            }
            ensure!(used == avail, "session file longer than its layout");
            Ok(())
        })();
        let sync = self.g.sync();
        res?;
        sync?;
        r.finish()?;
        self.apply_session_meta(meta, snap);
        Ok(())
    }

    fn apply_session_meta(&mut self, m: &SessionMeta, snap: &mut Option<Snapshot>) {
        for (l, &(nc, ni)) in self.layers.iter_mut().zip(&m.counters) {
            l.n_comp = nc;
            l.n_idx = ni;
        }
        self.engram_hist.clear();
        self.engram_hist.extend_from_slice(&m.engram_hist);
        self.topk_rows.clear();
        self.topk_rows.extend_from_slice(&m.topk_rows);
        self.candidates.clear();
        self.candidates.extend_from_slice(&m.candidates);
        self.topk_k = m.topk_k;
        self.comp_src = m.comp_src;
        self.idxk_src = m.idxk_src;
        self.pos = m.pos;
        if let Some(r) = &m.rollback {
            let b = snap.as_mut().unwrap();
            b.pos = r.pos;
            b.counters = r.counters.clone();
            b.engram_len = r.engram_len;
        } else {
            *snap = None;
        }
        // No speculative pass or image encoding is live at the API boundary. Their scratch is overwritten by the
        // next pass. Invalidate stale verify metadata; retaining vsnap/vrows storage is safe and avoids allocations.
        self.vpass = None;
        self.images.clear();
        self.img_rows.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_round_trips_and_rejects_damage() {
        let m = SessionMeta {
            pos: 12345,
            counters: vec![(1, 2), (3, 4), (0, 0)],
            engram_hist: vec![-1, 5, i64::MAX],
            topk_rows: vec![7, -9],
            candidates: vec![true, false, true],
            topk_k: 512,
            comp_src: usize::MAX,
            idxk_src: 3,
            rollback: Some(Rollback { pos: 12000, counters: vec![(1, 1), (2, 2), (0, 0)], engram_len: 2 }),
            data_len: 999_999,
        };
        let b = m.encode();
        assert_eq!(SessionMeta::decode(&b).unwrap(), m);
        let none = SessionMeta { rollback: None, ..m.clone() };
        assert_eq!(SessionMeta::decode(&none.encode()).unwrap(), none);
        assert!(SessionMeta::decode(&b[..b.len() - 1]).is_err(), "truncated");
        let mut long = b.clone();
        long.push(0);
        assert!(SessionMeta::decode(&long).is_err(), "trailing");
        let mut huge = b.clone();
        huge[48..56].copy_from_slice(&u64::MAX.to_le_bytes()); // counters length
        assert!(SessionMeta::decode(&huge).is_err(), "absurd length");
    }
}
