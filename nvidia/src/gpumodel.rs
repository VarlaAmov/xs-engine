//! Decode on GPU0: the dense half of every layer on the GPU, routed experts on the CPU.
//! Step for step the same math as model.rs's reference (`ref_step`), so the two can be compared per token.

use crate::gguf::GType;
use crate::gpu::{Arg, Buf, Gpu, Kernel};
use crate::model::{Mat, Model, dot, matvec};
use crate::numerics::{round_bf16, silu};
use anyhow::{Result, bail};

mod prefill;
mod vision;
mod session;
pub use session::{SessionMeta, SessionSnapshot, Staging};
pub use vision::{VisionWeights, encode_image};
pub use prefill::PrefillBufs;
use std::collections::HashMap;

struct DMat {
    buf: Buf,
    scale: Option<Buf>,
    ty: GType,
    rows: usize,
    cols: usize,
    act_fp8: bool,
    /// Q8_0 repacked at upload: buf = int8 [rows][cols], scale = f16 [rows][cols/32] (coalesced 16-byte loads)
    q8r: bool,
    /// DSX_DENSE_Q6: requantized to 6 bits (q6.rs planes): buf = lo nibbles, q6 = (hi codes, sub-scales, super-scales)
    q6: Option<(Buf, Buf, Buf)>,
}

impl DMat {
    /// plane weights (q6 or Q5_K) at row r0: [lo, hi, sc, d] pointers and the (GEMV, f16 dequant) kernels
    fn planes(&self, r0: usize) -> ([u64; 4], &'static str, &'static str) {
        let (hi, sc, d) = self.q6.as_ref().unwrap();
        let k = self.cols;
        if self.ty == GType::Q5K {
            return ([self.buf.ptr + (r0 * k / 2) as u64, hi.ptr + (r0 * k / 8) as u64, sc.ptr + (r0 * k / 16) as u64, d.ptr + (r0 * k / 128 * 2) as u64], "gemv_q5p", "dequant_q5p_f16");
        }
        ([self.buf.ptr + (r0 * k / 2) as u64, hi.ptr + (r0 * k / 4) as u64, sc.ptr + (r0 * k / 16) as u64, d.ptr + (r0 * k / 256 * 2) as u64], "gemv_q6p", "dequant_q6p_f16")
    }

    /// (weight pointer, scale pointer, dequant type id) for rows starting at r0
    fn ptrs(&self, r0: usize) -> (u64, u64, i32) {
        if self.q8r {
            return (self.buf.ptr + (r0 * self.cols) as u64, self.scale.as_ref().unwrap().ptr + (r0 * (self.cols / 32) * 2) as u64, 1008);
        }
        let rb = self.ty.row_bytes(self.cols);
        let sp = self.scale.as_ref().map_or(0, |s| s.ptr + (r0 / 32 * (self.cols / 32)) as u64);
        let id = if self.ty == GType::Fp8B32 { 1000 } else { self.ty as u32 as i32 };
        (self.buf.ptr + (r0 * rb) as u64, sp, id)
    }
}

/// Q8_0 blocks -> (int8 [rows][k], f16 bits [rows][k/32])
fn repack_q8(raw: &[u8], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let nb = k / 32;
    let mut qs = vec![0u8; rows * k];
    let mut d = vec![0u8; rows * nb * 2];
    for b in 0..rows * nb {
        let blk = &raw[b * 34..b * 34 + 34];
        d[2 * b..2 * b + 2].copy_from_slice(&blk[..2]);
        qs[b * 32..b * 32 + 32].copy_from_slice(&blk[2..]);
    }
    (qs, d)
}

struct DLayer {
    q_a: DMat,
    q_a_norm: Buf,
    q_b: DMat,
    kv: DMat,
    kv_norm: Buf,
    o_a: DMat,
    o_b: DMat,
    sinks: Buf,
    attn_norm: Buf,
    ffn_norm: Buf,
    comp_kv: Option<DMat>,
    comp_gate: Option<DMat>,
    comp_norm: Option<Buf>,
    idx_q_b: Option<DMat>,
    idx_proj: Option<DMat>,
    idx_k: Option<DMat>,
    idx_k_norm: Option<Buf>,
    hc_attn: (DMat, Buf, Buf), // fn (f16), scale, base
    hc_ffn: (DMat, Buf, Buf),
    gate_w: DMat,
    gate_bias: Buf,
    gate_bias_vl: Option<Buf>,
    sh_gate: DMat,
    sh_up: DMat,
    sh_down: DMat,
    engram_kv: Option<DMat>,
    engram_qk: Option<Buf>,
    // caches
    window: Buf,       // bf16 [W][hd]
    comp_cache: Option<Buf>, // bf16 [max/ratio][hd] (kv source layers)
    idx_cache: Option<Buf>,  // bf16 [max/ratio][idx_dim] (index-key owners)
    comp_state: Option<(Buf, Buf)>, // kv_state, score_state f32 [ratio][hd]
    n_comp: usize,
    n_idx: usize,
}

/// A background thread that borrows the model; joined on drop so it never outlives it.
pub(crate) struct EngramPrefetch(pub(crate) Option<std::thread::JoinHandle<()>>);

impl Drop for EngramPrefetch {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            let _ = h.join();
        }
    }
}

pub struct GpuModel<'m> {
    pub m: &'m Model,
    pub g: Gpu,
    k: HashMap<&'static str, Kernel>,
    layers: Vec<DLayer>,
    head: DMat,
    out_norm: Buf,
    freqs: (Buf, Buf),
    // work buffers (T = 1)
    h: Buf,
    x: Buf,
    a: Buf,
    qr: Buf,
    q: Buf,
    kvv: Buf,
    o: Buf,
    r: Buf,
    mix_a: Buf,
    mix_f: Buf,
    /// decode hc mixes on GPU0's par stream: partial sums, and the fork / join events
    hc_part: Buf,
    hc_fork: crate::gpu::Event,
    hc_join: crate::gpu::Event,
    /// DSX_PAR_ATTN fork / join (see attention)
    attn_fork: crate::gpu::Event,
    attn_join: crate::gpu::Event,
    par_attn: bool,
    tmp: Buf,
    tmp2: Buf,
    iq: Buf,
    iw: Buf,
    score: Buf,
    /// split-K attention partials [8 tokens][ATTN_SPLITS tiles][heads][head_dim + 2] f32
    attn_part: Buf,
    nvis: Buf,
    win_idx: Buf,
    comp_idx: Buf,
    logits: Buf,
    lat: Buf,
    actq: Buf,
    pub pos: usize,
    max_ctx: usize,
    ctx_capacity: usize,
    dynamic_ctx: bool,
    dynamic_failed: bool,
    packed_kv: bool,
    prefill_cpu_estimate: Vec<f64>,
    /// XS_ORACLE_LIST ablation: per GPU ordinal and (gate/up, down) format, VRAM copies of one real expert
    oracle: Vec<(i32, (i32, i32), [Buf; 3])>,
    dynamic_kv_base: usize,
    dynamic_exp_base: usize,
    comp_src: usize,
    idxk_src: usize,
    topk_rows: Vec<i32>,
    candidates: Vec<bool>,
    engram_hist: Vec<i64>,
    /// routed experts on the CPU pool (GSQ block types); None = reference matvec (FP4 checkpoint)
    cpu_moe: Option<crate::cpu::moe::CpuMoe>,
    /// seconds spent in CPU experts / everything else, for profiling
    pub t_experts: f64,
    /// experts resident on the main GPU / on the aux GPUs (each in its own context)
    exp0: Option<crate::experts::GpuExperts>,
    aux: Vec<Aux>,
    /// Decode sums delivered directly to GPU0. Non-peer aux GPUs keep their mapped-host slots.
    aux_results: Option<Buf>,
    aux_peer_mask: u32,
    out0: Buf,
    /// per-token constants (window slots, visible compressed rows per ratio), uploaded once per token
    tokp: Buf,
    tokp_host: crate::gpu::HostBuf,
    topk: crate::topk::TopK,
    topk_scratch: Buf,
    keep: Buf,
    topk_k: usize,
    /// sync-free expert handoff (GGUF models): mapped exchange buffer, CPU worker thread, sequence base
    xch: Option<&'static crate::handoff::Xch>,
    worker: Option<crate::handoff::CpuWorker>,
    arena: Option<&'static crate::experts::Arena>,
    /// adaptive residency (the main GPU's store and fast-linked aux stores): replacements started per token per store
    /// (DSX_ADAPT, default 8; 0 = static placement)
    adapt: usize,
    pub adapt_swaps: usize,
    pub adapt_s: f64,
    /// on_cpu flag changes from adapt_step, applied after the token's sync (the CPU worker may still be running the token)
    adapt_flags: Vec<(usize, bool)>,
    /// one-token Q8_0 GEMVs on gemv_q8r1 (DSX_Q8R1=0: gemv_q8r, for A/Bs)
    q8r1: bool,
    /// one-token split-k GEMVs on the *_sk1 kernels (DSX_SK1=0: the generic ones, for A/Bs)
    sk1: bool,
    /// xs-engine: one-token Q5_K GEMVs on the xs_q5_* kernels (q_b, o_a in one grouped launch, o_b, shared expert
    /// gate+up+SwiGLU fused, down, head); XS_DENSE=0 keeps gemv_q5p / gemv_q5p4 (A/Bs)
    xs_dense: bool,
    /// xs-engine: indexer scores on xs_idx_score* (a thread per key, rows share key loads, candidate-masked keys
    /// skipped); XS_IDX=0 keeps indexer_score
    pub(crate) xs_idx: bool,
    /// xs-engine: aux GPUs read decode picks / weights / x from the main GPU over peer access (XS_PEER_READ=0: copies)
    xs_peer_read: bool,
    /// one-token deferred decode: injection + post-weights copy inside the combine launch (DSX_COMBINE_FUSED=0: separate)
    combine_fused: bool,
    /// one token: wo_a's rounding inside its GEMV, shared gate + up + swiglu in one launch (DSX_FUSE2=0: separate)
    fuse2: bool,
    /// one token: attention merge + inverse RoPE + fake FP8 in one launch (DSX_FUSE3=0: separate)
    fuse3: bool,
    qb_rope: bool,
    /// one token: the sublayers' closing hc_post inside o_b's GEMV and combine_post (DSX_FUSE4=1; off by default)
    fuse4: bool,
    post_attn_done: bool,
    post_ffn_done: bool,
    warned_nan: bool,
    /// DSX_ENGRAM_PERSIST=1: persistent threads gather the rows into eng_host (declared first: dropped before it)
    eng_gather: Option<crate::engram_gather::EngramGather>,
    /// prefill: the next chunk's tokens (set by `prefill`), whose Engram rows are prefetched while this one computes
    pub(crate) engram_next: Vec<u32>,
    /// prefill: the background Engram page prefetch (joined before the next one, at the end of `prefill`, on drop)
    pub(crate) engram_prefetch: EngramPrefetch,
    /// pinned staging for this token's Engram rows ([engram layer][rows][dim] f32)
    eng_host: crate::gpu::HostBuf,
    /// DSX_ROUTE_BIAS=x: per-layer selection bonus x for GPU-resident experts ([layers][E] f32), off by default
    bonus: Option<Buf>,
    pub pb: Option<PrefillBufs>,
    /// > 0 while a verify pass runs: prefill's experts go through the decode handoff
    verify_t: usize,
    /// speculative verify passes: the state before the pass (rewound for rejected tokens) and each ratio-2 compressor
    /// layer's per-token kv / score rows from the pass ([layer][2][MAXT][hd] f32), for the carried half pair
    vsnap: Option<Snapshot>,
    vrows: Option<Buf>,
    vpass: Option<(usize, usize, Vec<(usize, usize)>)>, // (first position, tokens, counters after the pass)
    /// DSX_PREGATE=1 (decode statistics): layer L+1's router applied to layer L's expert input, top-8 per layer, next to
    /// the actual top-6 ([layer][16] i32: 8 predicted, 6 actual); counted per token into pregate_stats
    pregate: Option<Buf>,
    /// [picks, predicted in top-6, predicted in top-8, CPU picks, CPU picks predicted in top-8]
    pub pregate_stats: [u64; 5],
    /// missed picks, and how many of them the first 1 / 2 / 3 / 4 predicted experts that are themselves misses would have covered
    pub pregate_cov: [u64; 5],
    /// DSX_DEFER: each layer's ffn output mix (post, 4 f32) by layer parity, so its deferred experts can enter the
    /// residual with their own layer's weights one layer later
    dpost: Buf,
    /// image spans of the current prompt: (first position, span rows [len][dim] f32)
    pub images: Vec<(usize, Vec<f32>)>,
    /// image rows of the chunk being prefilled, chunk-local [lo, hi)
    img_rows: Vec<(usize, usize)>,
    skip_experts: bool,
    prof_host: bool,
    /// the exchange's address in the main GPU's context
    xch_dev: u64,
    /// the main GPU's host -> device GB/s (weighs its share of prefill's streamed experts against streaming aux GPUs)
    main_gbs: f64,
    seq_base: u32,
    pub n_routed: [usize; 3], // routed expert evaluations served by the main GPU, the aux GPUs, the CPU
    /// DSX_EVPROF=1: timed events per decode layer on the main GPU ([layer][EV_N]) and each aux GPU; ms sums per stage
    ev0: Vec<crate::gpu::Event>,
    pub evprof: Option<std::collections::BTreeMap<&'static str, f64>>,
    spin_wait: bool,
    /// DSX_PROF=1: seconds per section (synchronizing after each), summed over steps
    pub prof: Option<std::collections::BTreeMap<&'static str, f64>>,
    /// xs-engine tri-split decode (XS_TRI=0: off, the CPU computes missed experts whole)
    tri: Option<Tri>,
}

/// A GPU besides the main one, holding routed experts only: decode experts through the exchange, its share of
/// prefill on shipped activations.
struct Aux {
    g: Gpu,
    exp: crate::experts::GpuExperts,
    pre: Option<prefill::AuxPrefill>,
    /// the exchange's address in this GPU's context
    xdev: u64,
    /// replacements at run time like the main GPU's store (its host link is at least half as fast as the main one's;
    /// slower links keep a static hot set)
    adaptive: bool,
    /// DSX_EVPROF: [layer][4] timed events
    ev: Vec<crate::gpu::Event>,
    peer: bool,
    /// One event per layer: all records are enqueued before GPU0 waits, including decode graph capture.
    peer_done: Vec<crate::gpu::Event>,
}

/// xs-engine tri-split, one GPU's part: the kernels in its context, its slice [c0, c1) of every missed expert's
/// intermediate dimension, the miss map and arena addresses it reads, its scratch and the fork / join events.
struct TriGpu {
    k_qx: Kernel,
    /// per expert format (xs_fmt): IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS
    k_gu: [Kernel; 4],
    k_down: [Kernel; 4],
    k_add: Kernel,
    /// CTAs of each pull kernel (one per SM at most): XS_PULL_SMS_MAIN / XS_PULL_SMS_AUX, default every SM
    ctas: std::cell::Cell<u32>,
    c0: usize,
    c1: usize,
    /// u8 [layers][experts]: 1 = on no GPU (the CPU's on_cpu), the pulls take its slices
    miss: Buf,
    /// u64 [layers][experts][gate, up, down]: device addresses of the arena copies in this GPU's context
    ptrs: Buf,
    xq: Buf,
    xd: Buf,
    h: Buf,
    out: Buf,
    fork: crate::gpu::Event,
    join: crate::gpu::Event,
    /// XS_PREFETCH=R: pull prefetch (xs_prefetch.cuh); None = off
    pf: Option<Pf>,
}

/// Pull prefetch state of one GPU: staging areas for the slices of the R experts predicted to miss in the next layer,
/// the pre-gate's router weights in this GPU's context, and the unpatched copy of the pointer table.
struct Pf {
    r: usize,
    ctas: u32,
    k_gemv: Kernel,
    k_topk: Kernel,
    k_pick: Kernel,
    k_gu: Kernel,
    k_dn: Kernel,
    /// [layers][experts][3] original arena addresses (the pull kernels read `TriGpu::ptrs`, which is patched)
    host_ptrs: Buf,
    pred: Buf,
    logits: Buf,
    psel: Buf,
    tag: Buf,
    /// this layer's input x, copied by the first pull kernel: the main stream reuses x once the real pulls are joined
    xc: Buf,
    gu: Buf,
    dn: Buf,
    /// u64 [4]: prefetched slices, of them picked, missed picks, rounds (xs_pf_pick)
    stats: Buf,
    gu_stride: usize,
    dn_stride: usize,
    /// per layer (router weights, router bias) in this GPU's context
    gate: Vec<(u64, u64)>,
    _own: Vec<Buf>,
}

/// xs-engine tri-split decode: a missed expert is split by 256-row blocks of its intermediate dimension between the
/// CPU [0, cpu_end), the main GPU and the aux GPUs, each GPU reading its slice from the pinned arena over PCIe.
struct Tri {
    cpu_end: usize,
    /// None: the main GPU takes no slice (XS_SPLIT=c,0,a): it is busy with the dense half and its resident experts
    main: Option<TriGpu>,
    /// XS_PULL_AFTER=1: the main GPU's pulls start after its resident and shared experts (no SM contention with them)
    main_after: bool,
    aux: Vec<TriGpu>,
    /// per layer: (gate/up, down) ggml type ids; (0, 0) = this layer's missed experts stay whole on the CPU
    ty: Vec<(i32, i32)>,
    /// the miss map as last uploaded, and its pinned staging
    last: Vec<u8>,
    stage: crate::gpu::HostBuf,
}

impl TriGpu {
    #[allow(clippy::too_many_arguments)]
    fn new(g: &Gpu, c: &crate::model::Config, arena: &crate::experts::Arena, c0: usize, c1: usize, main: bool, gate: Option<(Vec<(u64, u64)>, Vec<Buf>)>) -> Result<TriGpu> {
        g.bind();
        let (nl, ne) = (c.n_layers, c.n_experts);
        let pieces = arena.buf.piece_dev_ptrs(g)?;
        // experts without an arena copy (static GPU stores' sets) are never missed: address 0, never read
        let mut ptrs = vec![0u64; nl * ne * 3];
        for l in 0..nl {
            for e in 0..ne {
                let Some(s) = arena.slots.get(&(l, e)) else { continue };
                for m in 0..3 {
                    ptrs[(l * ne + e) * 3 + m] = crate::gpu::HostBuf::dev_at(&pieces, s[m].0).ok_or_else(|| anyhow::anyhow!("tri-split: no device address"))?;
                }
            }
        }
        let k = |base: &str| -> Result<[Kernel; 4]> { Ok([g.kernel(&format!("{base}16"))?, g.kernel(&format!("{base}17"))?, g.kernel(&format!("{base}22"))?, g.kernel(&format!("{base}18"))?]) };
        let pf = match gate {
            Some((gates, own)) => {
                let r: usize = std::env::var("XS_PREFETCH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                if r == 0 {
                    None
                } else {
                    anyhow::ensure!(r <= 4, "XS_PREFETCH: 1..=4 slots");
                    let rows = c1 - c0;
                    let gu_stride = (rows * 20 * 98 + 64).next_multiple_of(256);
                    let dn_stride = (c.dim * (c.inter / 256) * 98 + 64).next_multiple_of(256);
                    Some(Pf {
                        r,
                        ctas: std::env::var("XS_PF_CTAS").ok().and_then(|v| v.parse().ok()).unwrap_or(12u32).clamp(1, g.sms as u32),
                        k_gemv: g.kernel("gemv_bf16_sk1")?,
                        k_topk: g.kernel("router_topk")?,
                        k_pick: g.kernel("xs_pf_pick")?,
                        k_gu: g.kernel("xs_pf_gu")?,
                        k_dn: g.kernel("xs_pf_down")?,
                        host_ptrs: g.upload(&ptrs)?,
                        pred: g.alloc_zeroed(64)?,
                        logits: g.alloc_zeroed(c.n_experts * 4)?,
                        psel: g.alloc_zeroed(16)?,
                        tag: g.upload(&vec![-1i32; 4])?,
                        xc: g.alloc_zeroed(c.dim * 4)?,
                        gu: g.alloc_zeroed(2 * r * gu_stride)?,
                        dn: g.alloc_zeroed(r * dn_stride)?,
                        stats: g.alloc_zeroed(32)?,
                        gu_stride,
                        dn_stride,
                        gate: gates,
                        _own: own,
                    })
                }
            }
            None => None,
        };
        Ok(TriGpu {
            k_qx: g.kernel("xs_pull_qx")?,
            k_gu: k("xs_pull_gu")?,
            k_down: k("xs_pull_down")?,
            k_add: g.kernel("xs_add")?,
            ctas: std::cell::Cell::new(std::env::var(if main { "XS_PULL_SMS_MAIN" } else { "XS_PULL_SMS_AUX" }).ok().and_then(|v| v.parse().ok()).unwrap_or(g.sms as u32).clamp(1, g.sms as u32)),
            c0,
            c1,
            miss: g.alloc_zeroed(nl * ne)?,
            ptrs: g.upload(&ptrs)?,
            xq: g.alloc_zeroed(c.dim)?,
            xd: g.alloc_zeroed(c.dim / 256 * 4)?,
            h: g.alloc_zeroed(c.top_k * c.inter * 4)?,
            out: g.alloc_zeroed(c.dim * 4)?,
            fork: g.event()?,
            join: g.event()?,
            pf,
        })
    }

    /// Enqueue this GPU's slices of layer `li`'s missed picks on its pull stream (forked from its main stream after
    /// the picks and x are in place): out = their partial sum. `next` = (layer li + 1, its formats) when the pull
    /// prefetch should prepare that layer's predicted misses afterwards.
    #[allow(clippy::too_many_arguments)]
    fn enqueue(&self, g: &Gpu, c: &crate::model::Config, li: usize, ty: (i32, i32), sel: u64, wts: u64, x: u64, next: Option<(usize, (i32, i32))>) -> Result<()> {
        let (dim, inter, e, k) = (c.dim as i32, c.inter as i32, c.n_experts as i32, c.top_k as i32);
        let fmt = |t: i32| match t { 16 => 0, 17 => 1, 22 => 2, _ => 3 };
        g.pull_after_main(self.fork)?;
        g.launch_pull(self.k_qx, ((dim / 256) as u32, 1, 1), 256, 0, &[Arg::P(x), Arg::P(self.xq.ptr), Arg::P(self.xd.ptr), Arg::P(self.pf.as_ref().map_or(0, |p| p.xc.ptr))])?;
        g.launch_pull(
            self.k_gu[fmt(ty.0)],
            (self.ctas.get(), 1, 1),
            256,
            0,
            &[Arg::P(sel), Arg::P(wts), Arg::P(self.miss.ptr), Arg::P(self.ptrs.ptr), Arg::I(li as i32), Arg::I(e), Arg::I(k), Arg::P(self.xq.ptr), Arg::P(self.xd.ptr), Arg::P(self.h.ptr), Arg::I(self.c0 as i32), Arg::I(self.c1 as i32), Arg::I(inter), Arg::F(c.swiglu_limit)],
        )?;
        g.launch_pull(
            self.k_down[fmt(ty.1)],
            (self.ctas.get(), 1, 1),
            256,
            0,
            &[Arg::P(sel), Arg::P(self.miss.ptr), Arg::P(self.ptrs.ptr), Arg::I(li as i32), Arg::I(e), Arg::I(k), Arg::P(self.h.ptr), Arg::P(self.out.ptr), Arg::I(self.c0 as i32), Arg::I(self.c1 as i32), Arg::I(inter), Arg::I(dim)],
        )?;
        // the main stream may go on as soon as the real pulls are done; the prefetch for the next layer runs behind them
        if self.pf.is_some() {
            g.record_pull(self.join)?;
        }
        let (Some(pf), Some((nl, nty))) = (self.pf.as_ref(), next) else { return Ok(()) };
        let bs = |t: i32| match t { 16 => 66, 17 => 74, 22 => 82, _ => 98 } as i32;
        let (rb, rbd) = (20 * bs(nty.0), c.inter as i32 / 256 * bs(nty.1));
        let (off, sb) = (self.c0 as i32 / 256 * bs(nty.1), (self.c1 - self.c0) as i32 / 256 * bs(nty.1));
        let (gw, gb) = pf.gate[nl];
        // layer nl's router on this layer's input: the top 8 predicted experts
        g.launch_pull(pf.k_gemv, (c.n_experts as u32, 1, 1), 128, 0, &[Arg::P(gw), Arg::P(pf.xc.ptr), Arg::P(pf.logits.ptr), Arg::I(e), Arg::I(dim), Arg::I(1), Arg::I(dim), Arg::I(e), Arg::P(0)])?;
        g.launch_pull(pf.k_topk, (1, 1, 1), 32, 0, &[Arg::P(pf.logits.ptr), Arg::P(gb), Arg::P(pf.pred.ptr), Arg::P(pf.pred.ptr + 32), Arg::I(e), Arg::I(8), Arg::F(c.route_scale), Arg::P(0)])?;
        let (gu0, dn0) = (pf.gu.ptr, pf.dn.ptr);
        g.launch_pull(
            pf.k_pick,
            (1, 1, 1),
            32,
            0,
            &[Arg::P(pf.pred.ptr), Arg::P(self.miss.ptr), Arg::P(pf.host_ptrs.ptr), Arg::P(self.ptrs.ptr), Arg::P(pf.tag.ptr), Arg::P(pf.psel.ptr), Arg::I(nl as i32), Arg::I(e), Arg::I(pf.r as i32), Arg::I(self.c0 as i32), Arg::I(rb), Arg::I(off),
              Arg::P(gu0), Arg::P(pf.gu_stride as u64), Arg::P(dn0), Arg::P(pf.dn_stride as u64), Arg::P(sel), Arg::I(k), Arg::P(pf.stats.ptr)],
        )?;
        g.launch_pull(pf.k_gu, (pf.ctas, 2 * pf.r as u32, 1), 256, 0, &[Arg::P(pf.psel.ptr), Arg::P(pf.host_ptrs.ptr), Arg::I(nl as i32), Arg::I(e), Arg::I(self.c0 as i32), Arg::I(self.c1 as i32), Arg::I(rb), Arg::P(gu0), Arg::P(pf.gu_stride as u64)])?;
        g.launch_pull(pf.k_dn, (pf.ctas, pf.r as u32, 1), 256, 0, &[Arg::P(pf.psel.ptr), Arg::P(pf.host_ptrs.ptr), Arg::I(nl as i32), Arg::I(e), Arg::I(dim), Arg::I(rbd), Arg::I(off), Arg::I(sb), Arg::P(dn0), Arg::P(pf.dn_stride as u64)])
    }

    /// The main stream waits for the pulls and adds their partial sum into `y` [dim].
    fn join_add(&self, g: &Gpu, dim: usize, y: u64) -> Result<()> {
        if self.pf.is_some() {
            // the join event was recorded right behind the real pulls (before the prefetch kernels)
            g.main_wait(self.join)?;
        } else {
            g.main_after_pull(self.join)?;
        }
        g.launch(self.k_add, ((dim as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(y), Arg::P(self.out.ptr), Arg::I(dim as i32)])
    }
}

/// A visible GPU as placement sees it.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub ordinal: i32,
    pub name: String,
    pub total: usize,
    /// host -> device GB/s from pinned memory (the link: x16 vs x4)
    pub h2d_gbs: f64,
}

/// Every visible GPU (DSX_GPUS="0,2" limits them): memory and measured host -> device bandwidth.
pub fn probe_gpus() -> Result<Vec<GpuInfo>> {
    let only: Option<Vec<i32>> = std::env::var("DSX_GPUS").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let gs: Vec<Gpu> = (0..Gpu::count()?).filter(|i| only.as_ref().is_none_or(|o| o.contains(i))).map(Gpu::new).collect::<Result<_>>()?;
    let Some(g0) = gs.first() else { bail!("no CUDA devices") };
    g0.bind();
    let stage = crate::gpu::HostBuf::new(64 << 20)?;
    gs.iter().map(|g| Ok(GpuInfo { ordinal: g.ordinal, name: g.name.clone(), total: g.mem_info()?.1, h2d_gbs: g.h2d_gbs(&stage)? })).collect()
}

/// The main GPU (dense weights, KV, prefill, the expert cache that swaps): the best host link, within 15%, then
/// the most memory.
pub fn pick_main(gpus: &[GpuInfo]) -> i32 {
    let best = gpus.iter().map(|g| g.h2d_gbs).fold(0.0, f64::max);
    gpus.iter().filter(|g| g.h2d_gbs >= 0.85 * best).max_by_key(|g| g.total).map_or(0, |g| g.ordinal)
}

/// Where routed experts live besides the CPU.
#[derive(Clone, Debug)]
pub struct Placement {
    /// < 0: everything the main GPU has free after the dense half, caches and prefill buffers, minus `-gpu0_gb` GB
    /// of headroom
    pub gpu0_gb: f64,
    /// per aux GPU (every visible GPU but the main one); < 0: its free memory minus that headroom; 0 = no aux GPUs
    pub aux_gb: f64,
    /// CPU experts copied into a hugepage RAM arena (the rest stay in the mmapped file)
    pub arena_gb: f64,
    /// prefill chunk (tokens per pass); 0 = no prefill buffers
    pub prefill_chunk: usize,
    /// expert hotness ranking (rank.json: [[layer, expert, tokens], ...] hottest first); GPUs take the hottest,
    /// then the arena, the coldest stay in the file. None = layer-major round robin.
    pub rank: Option<std::path::PathBuf>,
}

pub const KERNELS: &[&str] = &[
    "gemv_q8_0", "gemv_f16", "gemv_f32", "rmsnorm", "round_bf16", "rope", "fake_fp8", "fake_fp4", "hc_mixes", "hc_pre",
    "hc_post", "sparse_attn", "sparse_attn_tile", "sparse_attn_merge", "store_bf16", "indexer_score", "compress_pool", "router_topk", "swiglu", "add_inplace",
    "engram_gate", "scale_bf16", "gemv_fp8b32", "gemv_fp8b32_o", "gemv_bf16", "fake_fp8_copy", "gemv_q2_0", "gemv_ptq1_0", "gemv_q3_k", "publish_route", "combine_experts", "spin_wait", "rope_pos", "compress_pairs", "gather_f16", "swiglu_rows_f16", "scatter_add", "engram_gate_t", "dequant_f16", "dequant_f16_iq8", "dequant_t16_f16", "t16_pre", "t16_post", "to_f16", "ring_scatter", "pack_f16", "gemv_q8r", "gemv_q8r4", "hc_dots", "hc_dots_t", "hc_finish", "hc_finish_w", "gemv_q8g", "hc_pre_mb", "hc_post_mb", "to_bf16", "gemv_q8r_sk", "gemv_f16_sk", "gemv_bf16_sk", "gemv_f32_sk", "hc_part", "hc_reduce", "gemv_q6p", "dequant_q6p_f16", "gemv_q5p", "dequant_q5p_f16", "gemv_q6g4", "gemv_q6g8", "gemv_q6s8", "gemv_q5p4",
    "cache_pack_f32", "cache_unpack", "cache_pack_bf16", "fwht_rows_f16", "hc_reduce_finish", "kv_norm_rope_store", "add_noise_rows", "hc_add_post", "hc_pre_norm", "router_publish", "gemv_q8r1", "gemv_bf16_sk1", "gemv_f16_sk1", "gemv_q8r_sk1", "combine_post", "gemv_q8g_bf", "gemv_q8r_gu", "sparse_attn_merge_rf", "gemv_q8r1_rope", "gemv_q8r1_post", "vit_rmsnorm", "vit_bias_add", "vit_rope_split", "vit_softmax", "vit_heads_merge", "vit_swiglu", "vit_unfold3", "vit_gelu_bias", "vit_to_bf16",
    "xs_q5_32x4", "xs_q5_16x4", "xs_q5_8x4", "xs_q5_8x2", "xs_q5_gu", "xs_idx_score1", "xs_idx_score2",
];

fn f32buf(g: &Gpu, v: &[f32]) -> Result<Buf> {
    g.upload(v)
}

impl<'m> GpuModel<'m> {
    pub fn new(m: &'m Model, ordinal: i32, max_ctx: usize) -> Result<GpuModel<'m>> {
        Self::with_placement(m, ordinal, max_ctx, Placement { gpu0_gb: 0.0, aux_gb: 0.0, arena_gb: 0.0, prefill_chunk: 0, rank: None })
    }

    /// `ordinal` < 0: pick the main GPU by measurement (pick_main); every other GPU becomes an aux expert store
    /// unless `pl.aux_gb` is 0.
    pub fn with_placement(m: &'m Model, ordinal: i32, max_ctx: usize, pl: Placement) -> Result<GpuModel<'m>> {
        let probe = if ordinal < 0 || pl.aux_gb != 0.0 { probe_gpus()? } else { vec![] };
        let ordinal = if ordinal >= 0 { ordinal } else { pick_main(&probe) };
        if probe.len() > 1 {
            for p in &probe {
                eprintln!("GPU{} {}: {:.1} GB, host link {:.1} GB/s{}", p.ordinal, p.name, p.total as f64 / 1e9, p.h2d_gbs, if p.ordinal == ordinal { " (main)" } else { "" });
            }
        }
        let g = Gpu::new(ordinal)?;
        let mut k = HashMap::new();
        for &n in KERNELS {
            k.insert(n, g.kernel(n)?);
        }
        let c = &m.cfg;
        let dense_q6 = crate::q6::DenseQ6::parse(std::env::var("DSX_DENSE_Q6").ok().as_deref())?;
        let up = |w: &Mat| -> Result<DMat> {
            if w.ty == GType::Q8_0 && dense_q6.includes(&w.name) && w.cols % 256 == 0 {
                // decode the Q8_0 rows, requantize to 6 bits (ggml's Q6_K algorithm), upload as planes
                let (rows, k) = (w.rows, w.cols);
                let (mut lo, mut hi, mut sc, mut d) = (vec![0u8; rows * k / 2], vec![0u8; rows * k / 4], vec![0i8; rows * k / 16], vec![0u16; rows * k / 256]);
                use rayon::prelude::*;
                lo.par_chunks_mut(k / 2).zip(hi.par_chunks_mut(k / 4)).zip(sc.par_chunks_mut(k / 16)).zip(d.par_chunks_mut(k / 256)).enumerate().for_each(|(r, (((lo, hi), sc), d))| {
                    let mut x = vec![0f32; k];
                    w.dequant(r, &mut x);
                    crate::q6::quantize_row(&x, lo, hi, sc, d);
                });
                return Ok(DMat { buf: g.upload(&lo)?, scale: None, ty: w.ty, rows, cols: k, act_fp8: w.act_fp8, q8r: false, q6: Some((g.upload(&hi)?, g.upload(&sc)?, g.upload(&d)?)) });
            }
            if w.ty == GType::Q6K {
                // llama.cpp Q6_K (e.g. Unsloth's attention / head): the same codes, scales and d in the q6 planes, exact
                let (rows, k) = (w.rows, w.cols);
                let (mut lo, mut hi, mut sc, mut d) = (vec![0u8; rows * k / 2], vec![0u8; rows * k / 4], vec![0i8; rows * k / 16], vec![0u16; rows * k / 256]);
                use rayon::prelude::*;
                lo.par_chunks_mut(k / 2).zip(hi.par_chunks_mut(k / 4)).zip(sc.par_chunks_mut(k / 16)).zip(d.par_chunks_mut(k / 256)).enumerate().for_each(|(r, (((lo, hi), sc), d))| {
                    crate::q6::from_q6k_row(w.row(r), lo, hi, sc, d);
                });
                return Ok(DMat { buf: g.upload(&lo)?, scale: None, ty: w.ty, rows, cols: k, act_fp8: w.act_fp8, q8r: false, q6: Some((g.upload(&hi)?, g.upload(&sc)?, g.upload(&d)?)) });
            }
            if w.ty == GType::Q5K {
                // llama.cpp Q5_K (shared experts in some layers): exact 5-bit planes (q6.rs from_q5k_row)
                let (rows, k) = (w.rows, w.cols);
                let (mut lo, mut hb, mut sm, mut dd) = (vec![0u8; rows * k / 2], vec![0u8; rows * k / 8], vec![0u8; rows * k / 16], vec![0u16; rows * k / 128]);
                use rayon::prelude::*;
                lo.par_chunks_mut(k / 2).zip(hb.par_chunks_mut(k / 8)).zip(sm.par_chunks_mut(k / 16)).zip(dd.par_chunks_mut(k / 128)).enumerate().for_each(|(r, (((lo, hb), sm), dd))| {
                    crate::q6::from_q5k_row(w.row(r), lo, hb, sm, dd);
                });
                return Ok(DMat { buf: g.upload(&lo)?, scale: None, ty: w.ty, rows, cols: k, act_fp8: false, q8r: false, q6: Some((g.upload(&hb)?, g.upload(&sm)?, g.upload(&dd)?)) });
            }
            if w.ty == GType::Q8_0 {
                let (qs, d) = repack_q8(w.bytes(), w.rows, w.cols);
                return Ok(DMat { buf: g.upload(&qs)?, scale: Some(g.upload(&d)?), ty: w.ty, rows: w.rows, cols: w.cols, act_fp8: w.act_fp8, q8r: true, q6: None });
            }
            let scale = if w.scale_bytes().is_empty() { None } else { Some(g.upload(w.scale_bytes())?) };
            Ok(DMat { buf: g.upload(w.bytes())?, scale, ty: w.ty, rows: w.rows, cols: w.cols, act_fp8: w.act_fp8, q8r: false, q6: None })
        };
        let upo = |w: &Option<Mat>| -> Result<Option<DMat>> { w.as_ref().map(up).transpose() };
        // DSX_ENGRAM_FP8=1: an F16 engram wkv goes up as FP8 + 32x32 E8M0 scales (the checkpoint's layout), half the
        // bytes for its bandwidth-bound decode GEMV; only when the conversion is exact (the F16 came from that FP8)
        let engram_fp8 = std::env::var("DSX_ENGRAM_FP8").is_ok_and(|v| v == "1");
        let up_engram = |w: &Option<Mat>| -> Result<Option<DMat>> {
            let Some(w) = w else { return Ok(None) };
            if engram_fp8 && w.ty == GType::F16 {
                let h: &[half::f16] = unsafe { std::slice::from_raw_parts(w.bytes().as_ptr().cast(), w.rows * w.cols) };
                if let Some((q, sc)) = crate::numerics::f16_to_fp8b32(h, w.rows, w.cols) {
                    eprintln!("engram: {} as FP8 (exact), {:.0} MB", w.name, (q.len() + sc.len()) as f64 / 1e6);
                    return Ok(Some(DMat { buf: g.upload(&q)?, scale: Some(g.upload(&sc)?), ty: GType::Fp8B32, rows: w.rows, cols: w.cols, act_fp8: w.act_fp8, q8r: false, q6: None }));
                }
                eprintln!("engram: {} is not exact FP8, kept F16", w.name);
            }
            up(w).map(Some)
        };
        let upv = |v: &Option<Vec<f32>>| -> Result<Option<Buf>> { v.as_ref().map(|v| f32buf(&g, v)).transpose() };
        let f32mat = |v: &[f32], rows: usize| -> Result<DMat> {
            Ok(DMat { buf: g.upload(v)?, scale: None, ty: GType::F32, rows, cols: v.len() / rows, act_fp8: false, q8r: false, q6: None })
        };
        let bf16mat = |v: &[f32], rows: usize| -> Result<DMat> {
            let h: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
            debug_assert!(v.iter().zip(&h).all(|(a, b)| b.to_f32() == *a), "router weights are not exact bf16");
            Ok(DMat { buf: g.upload(&h)?, scale: None, ty: GType::BF16, rows, cols: v.len() / rows, act_fp8: false, q8r: false, q6: None })
        };
        let f16mat = f32mat; // hc fn matrices: kept f32 (exact for both F16 GGUF and F32 checkpoint values)
        let bf16zeros = |n: usize| g.alloc_zeroed(n * 2);
        let dynamic_ctx = std::env::var("XS_DYNAMIC_CTX").is_ok_and(|v| v == "1");
        let packed_kv = std::env::var("XS_PACKED_KV").is_ok_and(|v| v == "1");
        let ctx_capacity = if dynamic_ctx { max_ctx.min(65536) } else { max_ctx };
        let cache = |dim: usize, ratio: usize, block: usize| {
            let bytes = |ctx| Self::cache_bytes(packed_kv, ctx / ratio * dim, block);
            if dynamic_ctx { g.alloc_virtual(bytes(max_ctx), bytes(ctx_capacity)) }
            else { g.alloc_zeroed(bytes(max_ctx)) }
        };
        let mut layers = vec![];
        for l in &m.layers {
            let ratio = l.ratio.max(1);
            let is_src = c.kv_sources.contains(&l.id);
            layers.push(DLayer {
                q_a: up(&l.q_a)?,
                q_a_norm: f32buf(&g, &l.q_a_norm)?,
                q_b: up(&l.q_b)?,
                kv: up(&l.kv)?,
                kv_norm: f32buf(&g, &l.kv_norm)?,
                o_a: up(&l.o_a)?,
                o_b: up(&l.o_b)?,
                sinks: f32buf(&g, &l.sinks)?,
                attn_norm: f32buf(&g, &l.attn_norm)?,
                ffn_norm: f32buf(&g, &l.ffn_norm)?,
                comp_kv: upo(&l.comp_kv)?,
                comp_gate: upo(&l.comp_gate)?,
                comp_norm: upv(&l.comp_norm)?,
                idx_q_b: upo(&l.idx_q_b)?,
                idx_proj: upo(&l.idx_proj)?,
                idx_k: upo(&l.idx_k)?,
                idx_k_norm: upv(&l.idx_k_norm)?,
                // the hc fn matrices are F16 in the GGUF; the reference keeps them as f32 copies of those f16 values
                hc_attn: (f16mat(&l.hc_attn_fn, 24)?, f32buf(&g, &l.hc_attn_scale)?, f32buf(&g, &l.hc_attn_base)?),
                hc_ffn: (f16mat(&l.hc_ffn_fn, 24)?, f32buf(&g, &l.hc_ffn_scale)?, f32buf(&g, &l.hc_ffn_base)?),
                gate_w: bf16mat(&l.gate_w, c.n_experts)?,
                gate_bias: f32buf(&g, &l.gate_bias)?,
                gate_bias_vl: l.gate_bias_vl.as_ref().map(|b| f32buf(&g, b)).transpose()?,
                sh_gate: up(&l.sh_gate)?,
                sh_up: up(&l.sh_up)?,
                sh_down: up(&l.sh_down)?,
                engram_kv: up_engram(&l.engram_kv)?,
                engram_qk: upv(&l.engram_qk)?,
                window: bf16zeros(c.window * c.head_dim)?,
                comp_cache: if is_src { Some(cache(c.head_dim, ratio, 16)?) } else { None },
                idx_cache: if is_src { Some(cache(c.idx_dim, ratio, 32)?) } else { None },
                comp_state: if is_src && ratio > 1 {
                    Some((g.alloc_zeroed(ratio * c.head_dim * 4)?, g.alloc_zeroed(ratio * c.head_dim * 4)?))
                } else {
                    None
                },
                n_comp: 0,
                n_idx: 0,
            });
        }
        let d = c.dim;
        let fz = |n: usize| g.alloc_zeroed(n * 4);
        let gm = GpuModel {
            head: up(&m.head)?,
            out_norm: f32buf(&g, &m.out_norm)?,
            freqs: (f32buf(&g, &m.ropes.0.freqs)?, f32buf(&g, &m.ropes.1.freqs)?),
            h: fz(c.hc * d)?,
            x: fz(d)?,
            a: fz(d)?,
            qr: fz(c.q_lora)?,
            q: fz(c.n_heads * c.head_dim)?,
            kvv: fz(c.head_dim)?,
            o: fz(c.n_heads * c.head_dim)?,
            r: fz(c.o_groups * c.o_lora)?,
            mix_a: fz(24)?,
            mix_f: fz(24)?,
            hc_part: fz(25 * 8)?,
            hc_fork: g.event()?,
            hc_join: g.event()?,
            attn_fork: g.event()?,
            attn_join: g.event()?,
            // window 34: +0.04 ms/token (36.01 -> 36.05 tok/s teacher-forced), under the 0.1 ms bar: opt-in
            par_attn: std::env::var("DSX_PAR_ATTN").is_ok_and(|v| v == "1"),
            tmp: fz(1 << 18)?,
            tmp2: fz(32768)?,
            iq: fz(c.idx_heads * c.idx_dim)?,
            iw: fz(c.idx_heads)?,
            score: fz(max_ctx)?,
            attn_part: fz(8 * ATTN_SPLITS * c.n_heads * (c.head_dim + 2))?,
            nvis: g.alloc_zeroed(4 * 8)?,
            win_idx: g.alloc_zeroed(4 * c.window)?,
            comp_idx: g.alloc_zeroed(4 * c.idx_topk)?,
            logits: fz(c.vocab)?,
            lat: fz(c.head_dim)?,
            actq: fz(8 * 32768)?,
            out0: fz(crate::handoff::MAXT * c.dim)?, // [t][dim]: verify passes run up to MAXT tokens
            tokp: g.alloc_zeroed(4 * 256)?,
            tokp_host: crate::gpu::HostBuf::new(1 << 20)?,
            topk: crate::topk::TopK::new(&g)?,
            // shared by decode (1 row) and the prefill indexer (up to prefill::index_sub() rows per pass)
            topk_scratch: g.alloc_zeroed(crate::topk::TopK::topk_scratch(max_ctx, c.idx_topk, prefill::index_sub()).max(crate::topk::TopK::blocks_scratch(max_ctx, c.cand_block, c.cand_blocks, prefill::index_sub())).max(64))?,
            keep: g.alloc_zeroed(max_ctx / c.cand_block + 64)?,
            topk_k: 0,
            xch: None,
            worker: None,
            arena: None,
            adapt: std::env::var("DSX_ADAPT").ok().and_then(|v| v.parse().ok()).unwrap_or(8),
            adapt_swaps: 0,
            adapt_s: 0.0,
            adapt_flags: vec![],
            q8r1: std::env::var("DSX_Q8R1").map_or(true, |v| v != "0"),
            sk1: std::env::var("DSX_SK1").map_or(true, |v| v != "0"),
            xs_dense: std::env::var("XS_DENSE").map_or(true, |v| v != "0"),
            xs_idx: std::env::var("XS_IDX").map_or(true, |v| v != "0"),
            xs_peer_read: std::env::var("XS_PEER_READ").map_or(true, |v| v != "0"),
            combine_fused: std::env::var("DSX_COMBINE_FUSED").map_or(true, |v| v != "0"),
            fuse2: std::env::var("DSX_FUSE2").map_or(true, |v| v != "0"),
            fuse3: std::env::var("DSX_FUSE3").map_or(true, |v| v != "0"),
            qb_rope: std::env::var("DSX_QB_ROPE").is_ok_and(|v| v == "1"),
            // window 34: 36.05 -> 36.03 tok/s teacher-forced (no gain): opt-in
            fuse4: std::env::var("DSX_FUSE4").is_ok_and(|v| v == "1"),
            post_attn_done: false,
            post_ffn_done: false,
            warned_nan: false,
            eng_gather: None,
            engram_next: vec![],
            engram_prefetch: EngramPrefetch(None),
            eng_host: crate::gpu::HostBuf::new(4 << 20)?,
            bonus: None,
            pb: None,
            verify_t: 0,
            vsnap: None,
            vrows: None,
            vpass: None,
            pregate: if std::env::var_os("DSX_PREGATE").is_some() { Some(g.alloc_zeroed(c.n_layers * 16 * 4 + 4096)?) } else { None },
            pregate_stats: [0; 5],
            pregate_cov: [0; 5],
            dpost: g.alloc_zeroed(64)?,
            images: Vec::new(),
            img_rows: Vec::new(),
            skip_experts: std::env::var_os("DSX_SKIP_EXPERTS").is_some(),
            prof_host: std::env::var("DSX_PROF").is_ok_and(|v| v == "host"),
            xch_dev: 0,
            main_gbs: probe.iter().find(|p| p.ordinal == ordinal).map_or(1.0, |p| p.h2d_gbs),
            seq_base: 0,
            layers,
            k,
            g,
            m,
            pos: 0,
            max_ctx,
            ctx_capacity, dynamic_ctx, dynamic_failed: false, packed_kv, prefill_cpu_estimate: vec![0.0; c.n_layers], oracle: vec![], dynamic_kv_base: 0, dynamic_exp_base: 0,
            comp_src: usize::MAX,
            idxk_src: usize::MAX,
            topk_rows: vec![],
            candidates: vec![],
            engram_hist: vec![],
            cpu_moe: if matches!(m.src, crate::model::Source::Gguf(_)) {
                crate::cpu::pool::pin_to(crate::cpu::pool::layout().main); // the caller spins too; keep it off the workers' cores
                Some(crate::cpu::moe::CpuMoe::new(crate::cpu::pool::Pool::new(
                    &crate::cpu::pool::P_CORES,
                    std::time::Duration::from_millis(200),
                )))
            } else {
                None
            },
            t_experts: 0.0,
            exp0: None,
            aux: vec![],
            aux_results: None,
            aux_peer_mask: 0,
            n_routed: [0; 3],
            ev0: vec![],
            evprof: None,
            spin_wait: std::env::var_os("DSX_SPIN_WAIT").is_some(),
            prof: std::env::var_os("DSX_PROF").map(|_| Default::default()),
            tri: None,
        };
        let mut gm = gm;
        if m.engram.is_some() && std::env::var("DSX_ENGRAM_PERSIST").is_ok_and(|v| v == "1") {
            let out = gm.eng_host.as_mut_slice::<f32>();
            gm.eng_gather = Some(crate::engram_gather::EngramGather::new(m, out, gm.prof.is_some())?);
            eprintln!("engram: rows gathered by persistent threads (layer 1's first)");
        }
        // prefill buffers before the experts take the rest of VRAM
        gm.pb = Some(PrefillBufs::new(&gm.g, m, pl.prefill_chunk, max_ctx)?);
        let gpu0_bytes = if pl.gpu0_gb < 0.0 {
            let (free, _) = gm.g.mem_info()?;
            free.saturating_sub((-pl.gpu0_gb * 1e9) as usize)
        } else {
            (pl.gpu0_gb * 1e9) as usize
        };
        // aux GPUs: every other probed GPU, each with its free memory minus headroom and its prefill buffers
        let mut aux_g: Vec<(Gpu, usize, bool, bool, f64)> = vec![]; // GPU, expert bytes, adaptive, streams prefill, link GB/s
        if pl.aux_gb != 0.0 {
            let main_bw = probe.iter().find(|p| p.ordinal == ordinal).map_or(0.0, |p| p.h2d_gbs);
            let others = probe.iter().filter(|p| p.ordinal != ordinal).count();
            if others > crate::handoff::MAX_AUX {
                eprintln!("{others} other GPUs: using the first {} (DSX_GPUS picks which)", crate::handoff::MAX_AUX);
            }
            for p in probe.iter().filter(|p| p.ordinal != ordinal).take(crate::handoff::MAX_AUX) {
                let ga = Gpu::new(p.ordinal)?;
                // a link at least half as fast as the main GPU's: the store swaps experts like the main one and takes a
                // share of prefill's streamed experts (DSX_AUX_ADAPTIVE=1/0 and DSX_AUX_STREAM=1/0 override each)
                let fast = p.h2d_gbs >= 0.5 * main_bw;
                let adaptive = std::env::var("DSX_AUX_ADAPTIVE").map_or(fast, |v| v == "1");
                let stream = pl.prefill_chunk > 0 && std::env::var("DSX_AUX_STREAM").map_or(fast, |v| v == "1");
                let pre1 = if pl.prefill_chunk > 0 { prefill::aux_prefill_bytes(c, pl.prefill_chunk, stream) } else { 0 };
                let (free, _) = ga.mem_info()?;
                let bytes = if pl.aux_gb < 0.0 { free.saturating_sub((-pl.aux_gb * 1e9) as usize + pre1) } else { ((pl.aux_gb * 1e9) as usize).min(free.saturating_sub(pre1)) };
                aux_g.push((ga, bytes, gm.adapt > 0 && adaptive, stream, p.h2d_gbs));
            }
            gm.g.bind();
            // swapping aux stores need RAM copies of their experts: when the arena cannot hold them, they would push
            // more than 5% of the expert bytes out to the file (prefill then streams those from the page cache, the CPU
            // computes them from there), so the stores keep fixed sets unless DSX_AUX_ADAPTIVE=1 asks otherwise
            let swap_bytes: usize = aux_g.iter().filter(|a| a.2).map(|a| a.1).sum();
            if swap_bytes > 0 && pl.arena_gb > 0.0 && std::env::var_os("DSX_AUX_ADAPTIVE").is_none() {
                let total: usize = m.layers.iter().flat_map(|l| l.experts.iter()).map(|x| x.gate.bytes().len() + x.up.bytes().len() + x.down.bytes().len()).sum();
                let cap = (pl.arena_gb * 1e9) as usize;
                let extra = total.saturating_sub(cap) - total.saturating_sub(cap + swap_bytes);
                if extra > total / 20 {
                    eprintln!("aux GPUs keep fixed expert sets: their RAM copies need ~{:.0} GB more arena than --arena-gb {} (DSX_AUX_ADAPTIVE=1 swaps anyway)", extra as f64 / 1e9, pl.arena_gb);
                    for a in aux_g.iter_mut() {
                        a.2 = false;
                    }
                }
            }
        }
        let mut prior: Vec<(usize, usize, u64)> = vec![];
        let rank: Option<Vec<(usize, usize)>> = match &pl.rank {
            None => None,
            Some(p) => {
                let v: Vec<(usize, usize, u64)> = serde_json::from_slice(&std::fs::read(p)?)?;
                prior = v.clone();
                let mut seen = std::collections::HashSet::new();
                let mut r: Vec<(usize, usize)> = v.into_iter().map(|(l, e, _)| (l, e)).filter(|x| seen.insert(*x)).collect();
                // experts the ranking does not cover go last, layer-major
                for e in 0..c.n_experts {
                    for l in 0..c.n_layers {
                        if seen.insert((l, e)) {
                            r.push((l, e));
                        }
                    }
                }
                Some(r)
            }
        };
        // static stores take the hottest experts first; the adaptive ones (the main GPU, fast-linked aux GPUs) the next
        let caps: Vec<usize> = std::iter::once(gpu0_bytes).chain(aux_g.iter().map(|a| a.1)).collect();
        let group: Vec<u8> = std::iter::once(1).chain(aux_g.iter().map(|a| a.2 as u8)).collect();
        if caps.iter().any(|&b| b > 0) {
            let sets = crate::experts::place(m, rank.as_deref(), &caps, &group);
            // staging for in-flight replacements: the largest expert's raw bytes
            let max_raw = (0..c.n_layers)
                .flat_map(|l| (0..c.n_experts).map(move |e| (l, e)))
                .map(|(l, e)| { let x = &m.layers[l].experts[e]; x.gate.bytes().len() + x.up.bytes().len() + x.down.bytes().len() })
                .max()
                .unwrap_or(0);
            if !sets[0].is_empty() {
                let mut s0 = crate::experts::GpuExperts::new(&gm.g, m, &sets[0])?;
                // Replacement uploads start between tokens on both paths; early planning needs no extra stages.
                s0.enable_swaps(&gm.g, gm.adapt, max_raw)?;
                gm.exp0 = Some(s0);
            }
            let mut msg = format!("experts resident: GPU{ordinal} {} ({:.1} GB)", sets[0].len(), gm.exp0.as_ref().map_or(0, |x| x.bytes) as f64 / 1e9);
            for ((ga, _, adaptive, stream, bw), set) in aux_g.into_iter().zip(sets.iter().skip(1)) {
                if set.is_empty() {
                    continue;
                }
                let mut exp = crate::experts::GpuExperts::new(&ga, m, set)?;
                let adaptive = adaptive && exp.is_fast();
                if adaptive {
                    // DSX_ADAPT_AUX: in-flight replacements on aux GPUs (default DSX_ADAPT)
                    let n = std::env::var("DSX_ADAPT_AUX").ok().and_then(|v| v.parse().ok()).unwrap_or(gm.adapt);
                    exp.enable_swaps(&ga, n, max_raw)?;
                }
                let pre = if pl.prefill_chunk > 0 { Some(prefill::AuxPrefill::new(&ga, &gm.g, m, pl.prefill_chunk, stream, bw)?) } else { None };
                msg += &format!(", GPU{} {} ({:.1} GB, {}{}, {bw:.1} GB/s link)", ga.ordinal, set.len(), exp.bytes as f64 / 1e9, if adaptive { "adaptive" } else { "static" }, if stream { ", streams prefill experts" } else { "" });
                let mode = std::env::var("DSX_P2P").unwrap_or_else(|_| "auto".into());
                anyhow::ensure!(matches!(mode.as_str(), "0" | "1" | "auto"), "DSX_P2P: expected 0, 1 or auto");
                let peer = if mode == "0" { false } else {
                    let forward = gm.g.enable_peer(&ga)?;
                    let reverse = ga.enable_peer(&gm.g)?;
                    forward && reverse
                };
                anyhow::ensure!(mode != "1" || peer, "DSX_P2P=1: GPU{} <-> GPU{} peer access unavailable", gm.g.ordinal, ga.ordinal);
                eprintln!("GPU{} <-> GPU{} transport: {}", gm.g.ordinal, ga.ordinal, if peer { "peer (direct device copies)" } else { "pinned host staging" });
                if peer { gm.aux_peer_mask |= 1 << gm.aux.len(); }
                let peer_done = if peer { (0..c.n_layers).map(|_| ga.event()).collect::<Result<_>>()? } else { vec![] };
                gm.aux.push(Aux { g: ga, exp, pre, xdev: 0, adaptive, ev: vec![], peer, peer_done });
            }
            gm.g.bind();
            eprintln!("{msg}");
        }
        if gm.aux_peer_mask != 0 {
            gm.aux_results = Some(gm.g.alloc(gm.aux.len() * crate::handoff::MAXT * c.dim * 4)?);
        }
        if let Some(x) = std::env::var("DSX_ROUTE_BIAS").ok().and_then(|v| v.parse::<f32>().ok()) {
            let mut b = vec![0f32; c.n_layers * c.n_experts];
            for l in 0..c.n_layers {
                for e in 0..c.n_experts {
                    if gm.resident(l, e) {
                        b[l * c.n_experts + e] = x;
                    }
                }
            }
            gm.bonus = Some(gm.g.upload(&b)?);
            eprintln!("route bias {x} toward GPU-resident experts (changes the model's routing)");
        }
        if gm.cpu_moe.is_some() && std::env::var_os("DSX_SYNC_MOE").is_none() {
            // the CPU worker owns the pool from here on
            gm.cpu_moe = None;
            let xch: &'static crate::handoff::Xch = Box::leak(Box::new(crate::handoff::Xch::new(c.dim)?));
            let on_cpu: Vec<Vec<bool>> = (0..c.n_layers)
                .map(|l| {
                    (0..c.n_experts)
                        .map(|e| !gm.resident(l, e))
                        .collect()
                })
                .collect();
            let ms: &'static Model = unsafe { std::mem::transmute::<&Model, &'static Model>(m) };
            let arena: Option<&'static crate::experts::Arena> = if pl.arena_gb > 0.0 {
                let order: Vec<(usize, usize)> = rank.clone().unwrap_or_else(|| (0..c.n_experts).flat_map(|e| (0..c.n_layers).map(move |l| (l, e))).collect());
                // the CPU's experts, and those an adaptive store may evict back to it
                let adaptive0 = gm.adapt > 0 && gm.exp0.as_ref().is_some_and(|s| s.is_fast());
                let cpu: Vec<(usize, usize)> = order
                    .into_iter()
                    .filter(|&(l, e)| {
                        on_cpu[l][e] || (adaptive0 && gm.exp0.as_ref().unwrap().has(l, e)) || gm.aux.iter().any(|a| a.adaptive && a.exp.has(l, e))
                    })
                    .collect();
                if let crate::model::Source::Gguf(g) = &m.src {
                    crate::gpu::drop_page_cache(&g.paths);
                }
                let t = std::time::Instant::now();
                let (a, rest) = crate::experts::Arena::new(m, &cpu, (pl.arena_gb * 1e9) as usize)?;
                eprintln!("arena: {} experts, {:.1} GB in {:.0} s; {} left in the file", a.slots.len(), a.bytes as f64 / 1e9, t.elapsed().as_secs_f64(), rest.len());
                Some(Box::leak(Box::new(a)))
            } else {
                None
            };
            gm.arena = arena;
            gm.worker = Some(crate::handoff::CpuWorker::spawn(ms, xch, on_cpu, c.top_k, arena));
            // adaptive residency starts from the calibration prior: each expert's expected decayed score (its share of
            // the calibration tokens per layer x 1 / (1 - decay)), so placed experts keep their slots until the live
            // routing disagrees, instead of every slot looking idle at the first token
            if let Some(w) = gm.worker.as_ref() {
                let per_layer: u64 = prior.iter().filter(|x| x.0 == 0).map(|x| x.2).sum::<u64>().max(1) / c.top_k as u64;
                for &(l, e, n) in &prior {
                    let v = n as f32 / per_layer as f32 / (1.0 - crate::handoff::decay());
                    w.score[l * c.n_experts + e].store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
            }
            gm.xch_dev = xch.buf.dev_ptr(&gm.g)?;
            for a in gm.aux.iter_mut() {
                a.xdev = xch.buf.dev_ptr(&a.g)?;
            }
            gm.g.bind();
            gm.xch = Some(xch);
        }
        if gm.worker.is_some() && std::env::var("XS_TRI").map_or(true, |v| v != "0") {
            match gm.tri_init() {
                Ok(()) => {}
                Err(e) => eprintln!("tri-split off: {e}"),
            }
        }
        if std::env::var_os("DSX_EVPROF").is_some() {
            for _ in 0..c.n_layers * EV_N {
                gm.ev0.push(gm.g.timed_event()?);
            }
            for a in gm.aux.iter_mut() {
                for _ in 0..c.n_layers * 4 {
                    a.ev.push(a.g.timed_event()?);
                }
            }
            gm.g.bind();
            gm.evprof = Some(Default::default());
        }
        if gm.dynamic_ctx {
            gm.dynamic_kv_base = gm.kv_mapped_bytes();
            gm.dynamic_exp_base = gm.exp0.as_ref().map_or(0, |s| s.dynamic_capacity());
            anyhow::ensure!(gm.worker.is_some() && gm.arena.is_some() && gm.adapt > 0,
                "XS_DYNAMIC_CTX needs the adaptive expert stores and RAM arena");
            tracing::info!(capacity = gm.ctx_capacity, limit = gm.max_ctx, kv_bytes = gm.dynamic_kv_base,
                "dynamic context initialized");
        }
        Ok(gm)
    }

    /// xs-engine: set up tri-split decode. XS_SPLIT=c,m[,a...] gives the CPU's, the main GPU's and each aux GPU's share
    /// in 256-row blocks of the intermediate dimension (default: equal thirds, 3,3,3 for XS). Needs every expert in the
    /// arena, XS's expert formats and no DSX_DEFER.
    fn tri_init(&mut self) -> Result<()> {
        let c = &self.m.cfg;
        anyhow::ensure!(crate::handoff::defer().is_none(), "DSX_DEFER is set");
        let arena = self.arena.ok_or_else(|| anyhow::anyhow!("no arena"))?;
        let nb = c.inter / 256;
        let parts: Vec<usize> = match std::env::var("XS_SPLIT") {
            Ok(v) => v.split(',').map(|x| x.trim().parse::<usize>()).collect::<Result<_, _>>()?,
            Err(_) => {
                let n = 2 + self.aux.len();
                (0..n).map(|i| nb / n + (i < nb % n) as usize).collect()
            }
        };
        anyhow::ensure!(parts.len() == 2 + self.aux.len() && parts.iter().sum::<usize>() == nb && parts[0] > 0 && parts.iter().all(|&p| p <= 4 || (p <= 6 && p == parts[parts.len() - 1])), "XS_SPLIT {parts:?}: one share per party (CPU, main, aux...), CPU > 0, at most 4 blocks each (6 for the last aux), {nb} in total");
        let ty: Vec<(i32, i32)> = (0..c.n_layers)
            .map(|l| {
                let x = &self.m.layers[l].experts[0];
                let ok = |t: GType| matches!(t, GType::Iq2Xxs | GType::Iq2Xs | GType::Iq2S | GType::Iq3Xxs);
                let same = self.m.layers[l].experts.iter().all(|y| y.gate.ty == x.gate.ty && y.up.ty == x.gate.ty && y.down.ty == x.down.ty);
                if ok(x.gate.ty) && ok(x.down.ty) && same && !c.expert_rot[l] { (x.gate.ty as u32 as i32, x.down.ty as u32 as i32) } else { (0, 0) }
            })
            .collect();
        anyhow::ensure!(ty.iter().all(|t| t.0 != 0), "a layer's experts are not in XS's formats (every layer must split: the CPU worker slices by token count)");
        // every expert that is or may become a miss (the CPU's now, and adaptive stores' sets they can evict) must be
        // in the arena: the pulls read only pinned memory
        let w = self.worker.as_ref().ok_or_else(|| anyhow::anyhow!("no CPU worker"))?;
        let ne = c.n_experts;
        for l in 0..c.n_layers {
            for e in 0..ne {
                let may_miss = w.on_cpu[l * ne + e].load(std::sync::atomic::Ordering::Relaxed)
                    || (self.adapt > 0 && self.exp0.as_ref().is_some_and(|s| s.is_fast() && s.has(l, e)))
                    || self.aux.iter().any(|a| a.adaptive && a.exp.has(l, e));
                anyhow::ensure!(!may_miss || arena.slots.contains_key(&(l, e)), "expert {l}/{e} may be missed but is not in the arena");
            }
        }
        let mut c0 = parts[0] * 256;
        // XS_PREFETCH: each GPU needs every layer's router in its own context (the main GPU's are the model's own)
        let prefetch = std::env::var("XS_PREFETCH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) > 0;
        let main_gates = prefetch.then(|| (self.layers.iter().map(|l| (l.gate_w.buf.ptr, l.gate_bias.ptr)).collect::<Vec<_>>(), vec![]));
        let main = if parts[1] > 0 { Some(TriGpu::new(&self.g, c, arena, c0, c0 + parts[1] * 256, true, main_gates)?) } else { None };
        c0 += parts[1] * 256;
        let mut aux = vec![];
        for (i, a) in self.aux.iter().enumerate() {
            let gates = if prefetch {
                let mut own = vec![];
                let mut ptrs = vec![];
                for l in &self.layers {
                    let w: Vec<u8> = self.g.download(l.gate_w.buf.ptr, l.gate_w.rows * l.gate_w.cols * 2)?;
                    let b: Vec<u8> = self.g.download(l.gate_bias.ptr, c.n_experts * 4)?;
                    a.g.bind();
                    let (wb, bb) = (a.g.upload(&w)?, a.g.upload(&b)?);
                    ptrs.push((wb.ptr, bb.ptr));
                    own.push(wb);
                    own.push(bb);
                    self.g.bind();
                }
                Some((ptrs, own))
            } else {
                None
            };
            aux.push(TriGpu::new(&a.g, c, arena, c0, c0 + parts[2 + i] * 256, false, gates)?);
            c0 += parts[2 + i] * 256;
        }
        self.g.bind();
        let stage = crate::gpu::HostBuf::new(c.n_layers * c.n_experts)?;
        let main_after = std::env::var("XS_PULL_AFTER").is_ok_and(|v| v == "1");
        self.tri = Some(Tri { cpu_end: parts[0] * 256, main, main_after, aux, ty, last: vec![2; c.n_layers * c.n_experts], stage });
        self.tri_sync()?;
        crate::handoff::TRI_CPU_END.store((parts[0] * 256) as u32, std::sync::atomic::Ordering::Relaxed);
        eprintln!("tri-split: missed experts' intermediate rows CPU 0..{}, GPU{} {}..{}{}", parts[0] * 256, self.g.ordinal, parts[0] * 256, (parts[0] + parts[1]) * 256,
            self.aux.iter().zip(&self.tri.as_ref().unwrap().aux).map(|(a, t)| format!(", GPU{} {}..{}", a.g.ordinal, t.c0, t.c1)).collect::<String>());
        Ok(())
    }

    /// xs-engine: upload the miss map (the CPU worker's on_cpu flags) to every GPU if it changed; stream-ordered before
    /// the next token's work. Called at the start of each decode token (the flags change only between tokens).
    fn tri_sync(&mut self) -> Result<()> {
        let (Some(tr), Some(w)) = (self.tri.as_mut(), self.worker.as_ref()) else { return Ok(()) };
        let map: Vec<u8> = w.on_cpu.iter().map(|f| f.load(std::sync::atomic::Ordering::Relaxed) as u8).collect();
        if map == tr.last {
            return Ok(());
        }
        tr.stage.as_mut_slice::<u8>()[..map.len()].copy_from_slice(&map);
        let src = &tr.stage.as_slice::<u8>()[..map.len()];
        if let Some(m) = &tr.main {
            self.g.htod(m.miss.ptr, src)?;
        }
        for (a, t) in self.aux.iter().zip(&tr.aux) {
            a.g.htod(t.miss.ptr, src)?;
        }
        self.g.bind();
        tr.last = map;
        Ok(())
    }

    /// true when a GPU store (main or aux) holds expert e of layer l
    fn resident(&self, l: usize, e: usize) -> bool {
        self.exp0.as_ref().is_some_and(|s| s.has(l, e)) || self.aux.iter().any(|a| a.exp.has(l, e))
    }

    /// The aux GPUs' part of one pass over all layers (T tokens): per layer each waits for the main GPU's picks, runs
    /// its resident ones and hands its sum back in its own slot.
    fn enqueue_aux(&self, t: usize, x: u64) -> Result<()> {
        self.enqueue_aux_layers(t, x, 0..self.m.cfg.n_layers)
    }

    /// `enqueue_aux` for some layers only; each aux stream still gets its layers in order. DSX_AUX_SPLIT=1 (decode)
    /// enqueues layer li after GPU0's layer-li attention, so the ~600 host calls of all 40 layers no longer sit between
    /// GPU0's layer-0 attention and its layer-0 MoE.
    fn enqueue_aux_layers(&self, t: usize, x: u64, layers: std::ops::Range<usize>) -> Result<()> {
        let Some(xch) = self.xch else { return Ok(()) };
        let c = &self.m.cfg;
        for (i, a) in self.aux.iter().enumerate() {
            let (g1, s1) = (&a.g, &a.exp);
            for li in layers.clone() {
                let seq = self.seq_base + li as u32 + 1;
                g1.wait_value(a.xdev + crate::handoff::ROUTE as u64, seq)?;
                let ev = |k: usize| if a.ev.is_empty() { Ok(()) } else { g1.record(a.ev[li * 4 + k], false) };
                ev(0)?;
                // xs-engine (XS_PEER_READ, default on with peer access): the aux kernels read the picks, weights and x
                // from the main GPU's memory over NVLink instead of three peer copies per layer (~40 us of the aux path)
                let peer_read = a.peer && self.xs_peer_read;
                let (sel, wts, xin) = if peer_read { (self.tmp2.ptr, self.tmp2.ptr + 256, x) } else { (s1.sel.ptr, s1.wts.ptr, s1.x.ptr) };
                if peer_read {
                    // ROUTE is published with a system fence after tmp2 and x are ready; GPU0 does not overwrite them
                    // before this aux GPU has delivered its sum (the combine waits for it)
                } else if a.peer {
                    // ROUTE is published with a system fence after tmp2 and x are ready. GPU0 cannot
                    // reuse these buffers until this aux GPU has delivered its sum and raised its flag.
                    g1.peer_from(s1.sel.ptr, &self.g, self.tmp2.ptr, t * c.top_k * 4)?;
                    g1.peer_from(s1.wts.ptr, &self.g, self.tmp2.ptr + 256, t * c.top_k * 4)?;
                    g1.peer_from(s1.x.ptr, &self.g, x, t * c.dim * 4)?;
                } else {
                    g1.htod(s1.sel.ptr, xch.sel(t * c.top_k))?;
                    g1.htod(s1.wts.ptr, xch.wts(t * c.top_k))?;
                    g1.htod(s1.x.ptr, xch.x(t))?;
                }
                ev(1)?;
                let tri = if t == 1 { self.tri.as_ref().map(|tr| (&tr.aux[i], tr.ty[li])) } else { None };
                if let Some((tg, ty)) = tri {
                    let nxt = (li + 1 < c.n_layers).then(|| (li + 1, self.tri.as_ref().unwrap().ty[li + 1]));
                    tg.enqueue(g1, c, li, ty, sel, wts, xin, nxt)?;
                }
                if t == 1 {
                    s1.run_device(g1, li, sel, wts, c.top_k, xin, s1.out.ptr)?;
                } else {
                    s1.run_device_fast(g1, li, sel, wts, c.top_k, t, xin, s1.out.ptr)?;
                }
                if let Some((tg, _)) = tri {
                    tg.join_add(g1, c.dim, s1.out.ptr)?;
                }
                ev(2)?;
                if a.peer {
                    let dst = self.aux_results.as_ref().unwrap().ptr + (i * crate::handoff::MAXT * c.dim * 4) as u64;
                    g1.peer_to(&self.g, dst, s1.out.ptr, t * c.dim * 4)?;
                    g1.record(a.peer_done[li], false)?;
                } else {
                    g1.dtoh(xch.y_aux(i, t), s1.out.ptr)?;
                }
                ev(3)?;
                g1.write_value(a.xdev + crate::handoff::aux_flag(i) as u64, seq)?;
            }
        }
        Ok(())
    }

    /// Router, shared expert and the main GPU's resident experts here; CPU and aux GPUs through the exchange; no host
    /// sync.
    /// `x` [t][dim] in, `a` [t][dim] out (t <= 8).
    fn moe_fast_t(&mut self, li: usize, x: u64, a: u64, t: usize) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let d = c.dim;
        let seq = self.seq_base + li as u32 + 1;
        let l = &self.layers[li];
        self.ev(li, 1)?;
        self.gemv_t(&l.gate_w, x, self.tmp.ptr, t)?;
        let idx_buf = self.tmp2.ptr; // [t][k] i32 then [t][k] f32 at +256
        // one token: top-k and publish in one launch (router_publish, below)
        let fused = t == 1 && !self.skip_experts && self.pregate.is_none();
        if !fused {
            self.g.launch(self.k["router_topk"], (t as u32, 1, 1), 32, 0, &[Arg::P(self.tmp.ptr), Arg::P(l.gate_bias.ptr), Arg::P(idx_buf), Arg::P(idx_buf + 256), Arg::I(c.n_experts as i32), Arg::I(c.top_k as i32), Arg::F(c.route_scale), Arg::P(self.bonus_ptr(li))])?;
        }
        if let (Some(pg), 1) = (self.pregate.as_ref(), t) {
            // actual picks of this layer next to the prediction made one layer earlier
            self.g.dtod(pg.ptr + (li * 64 + 32) as u64, idx_buf, c.top_k * 4)?;
            if li + 1 < c.n_layers && std::env::var("DSX_PREGATE").is_ok_and(|v| v == "1") {
                let ln = &self.layers[li + 1];
                let scratch = pg.ptr + (c.n_layers * 64) as u64; // [8] i32 then [8] f32 weights
                self.gemv_t(&ln.gate_w, x, self.tmp.ptr, 1)?;
                self.g.launch(self.k["router_topk"], (1, 1, 1), 32, 0, &[Arg::P(self.tmp.ptr), Arg::P(ln.gate_bias.ptr), Arg::P(pg.ptr + ((li + 1) * 64) as u64), Arg::P(scratch + 32), Arg::I(c.n_experts as i32), Arg::I(8), Arg::F(c.route_scale), Arg::P(0)])?;
            }
        }
        self.ev(li, 2)?;
        let d0 = self.xch_dev;
        if self.skip_experts {
            // profiling only (DSX_SKIP_EXPERTS=1): the dense half without routed experts (wrong output)
            let l = &self.layers[li];
            let inter = c.inter;
            let (g_ptr, u_ptr, hm) = (self.tmp.ptr, self.tmp.ptr + (t * inter * 4) as u64, self.tmp.ptr + (2 * t * inter * 4) as u64);
            self.gemv_t(&l.sh_gate, x, g_ptr, t)?;
            self.gemv_t(&l.sh_up, x, u_ptr, t)?;
            self.g.launch(self.k["swiglu"], ((inter as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(g_ptr), Arg::P(u_ptr), Arg::I(inter as i32), Arg::P(0), Arg::P(hm), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
            return self.gemv_t(&l.sh_down, hm, a, t);
        }
        // DSX_DEFER (decode): the CPU may still be reading the previous layer's picks and input; publishing overwrites them
        let deferred = crate::handoff::defer().is_some() && t == 1;
        if deferred && li > 0 {
            self.flag_wait(d0 + crate::handoff::XACK as u64, seq - 1)?;
        }
        if fused {
            self.g.launch(self.k["router_publish"], (1, 1, 1), 1024, 0, &[Arg::P(self.tmp.ptr), Arg::P(self.layers[li].gate_bias.ptr), Arg::P(idx_buf), Arg::P(idx_buf + 256), Arg::I(c.n_experts as i32), Arg::I(c.top_k as i32), Arg::F(c.route_scale), Arg::P(self.bonus_ptr(li)), Arg::P(x), Arg::I(d as i32), Arg::P(d0), Arg::I(seq as i32)])?;
        } else {
            self.g.launch(self.k["publish_route"], (1, 1, 1), 1024, 0, &[Arg::P(idx_buf), Arg::P(idx_buf + 256), Arg::I(c.top_k as i32), Arg::P(x), Arg::I(d as i32), Arg::P(d0), Arg::I(seq as i32), Arg::I(t as i32)])?;
        }
        let tri_main = t == 1 && !self.skip_experts && self.tri.as_ref().is_some_and(|tr| tr.main.is_some());
        let tri_after = tri_main && self.tri.as_ref().unwrap().main_after;
        if tri_main && !tri_after {
            let tr = self.tri.as_ref().unwrap();
            let nxt = (li + 1 < c.n_layers).then(|| (li + 1, tr.ty[li + 1]));
            tr.main.as_ref().unwrap().enqueue(&self.g, c, li, tr.ty[li], idx_buf, idx_buf + 256, x, nxt)?;
        }
        match self.exp0.as_ref() {
            Some(s0) if t == 1 => s0.run_device(&self.g, li, idx_buf, idx_buf + 256, c.top_k, x, self.out0.ptr)?,
            Some(s0) => s0.run_device_fast(&self.g, li, idx_buf, idx_buf + 256, c.top_k, t, x, self.out0.ptr)?,
            None => self.g.memset(self.out0.ptr, t * d * 4)?,
        }
        self.ev(li, 3)?;
        let inter = c.inter;
        let (g_ptr, u_ptr, hm) = (self.tmp.ptr, self.tmp.ptr + (t * inter * 4) as u64, self.tmp.ptr + (2 * t * inter * 4) as u64);
        let l = &self.layers[li];
        // one token, Q8_0 gate and up on the one-row-per-warp path: both dots and swiglu in one launch (gemv_q8r_gu)
        let gu = |w: &DMat| w.ty == GType::Q8_0 && w.q8r && !w.act_fp8 && w.rows >= 2048;
        let xs_gu = |w: &DMat| w.ty == GType::Q5K && w.q6.is_some() && w.cols % 1024 == 0;
        if t == 1 && self.xs_dense && xs_gu(&l.sh_gate) && xs_gu(&l.sh_up) && l.sh_gate.rows == l.sh_up.rows {
            // xs-engine: Q5_K gate and up + SwiGLU in one launch (xs_q5_gu)
            let (([gl, gh, gs, gd], _, _), ([ul, uh, us, ud], _, _)) = (l.sh_gate.planes(0), l.sh_up.planes(0));
            self.g.launch(
                self.k["xs_q5_gu"],
                ((inter as u32).div_ceil(16), 1, 1),
                256,
                0,
                &[Arg::P(gl), Arg::P(gh), Arg::P(gs), Arg::P(gd), Arg::P(ul), Arg::P(uh), Arg::P(us), Arg::P(ud), Arg::P(x), Arg::P(hm), Arg::I(inter as i32), Arg::I(l.sh_gate.cols as i32), Arg::F(c.swiglu_limit)],
            )?;
        } else if t == 1 && self.fuse2 && self.q8r1 && gu(&l.sh_gate) && gu(&l.sh_up) {
            let ((gw, gs, _), (uw, us, _)) = (l.sh_gate.ptrs(0), l.sh_up.ptrs(0));
            self.g.launch(self.k["gemv_q8r_gu"], ((inter as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(gw), Arg::P(gs), Arg::P(uw), Arg::P(us), Arg::P(x), Arg::P(hm), Arg::I(inter as i32), Arg::I(l.sh_gate.cols as i32), Arg::F(c.swiglu_limit)])?;
        } else {
            self.gemv_t(&l.sh_gate, x, g_ptr, t)?;
            self.gemv_t(&l.sh_up, x, u_ptr, t)?;
            self.g.launch(self.k["swiglu"], ((inter as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(g_ptr), Arg::P(u_ptr), Arg::I(inter as i32), Arg::P(0), Arg::P(hm), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
        }
        self.gemv_t(&l.sh_down, hm, a, t)?;
        if tri_after {
            let tr = self.tri.as_ref().unwrap();
            let nxt = (li + 1 < c.n_layers).then(|| (li + 1, tr.ty[li + 1]));
            tr.main.as_ref().unwrap().enqueue(&self.g, c, li, tr.ty[li], idx_buf, idx_buf + 256, x, nxt)?;
        }
        self.ev(li, 4)?;
        // the CPU's experts: this layer's (or, deferred, the previous layer's; the last layer waits for its own too)
        let last = li + 1 == c.n_layers;
        self.flag_wait(d0 + crate::handoff::CPU as u64, seq)?;
        let mut cpu_mask = 1;
        // one token, deferred: the injection and the 16-byte post-weights copy go into the combine launch (combine_post)
        let fuse = deferred && self.combine_fused;
        let mut inj_off = -1i32;
        if deferred {
            // deferred slots: even layers bit 1, odd layers bit 2
            let slot = |l: usize| if l % 2 == 0 { 2 } else { 4 };
            // deferred experts enter the residual with their own layer's output weights (adding them to the next layer's
            // expert output instead mixes them with the wrong weights: KL 0.395 vs 0.383 at DSX_DEFER=3)
            let inject = true;
            if inject {
                // this layer's output weights (mix_f, from the par stream), for its own deferred experts next layer
                self.hc_join_main()?;
                if !fuse {
                    self.g.dtod(self.dpost.ptr + ((li % 2) * 16) as u64, self.mix_f.ptr + 16, 16)?;
                }
            }
            if li > 0 {
                self.flag_wait(d0 + crate::handoff::CPUD as u64, seq - 1)?;
                if inject {
                    let off = if (li - 1) % 2 == 0 { crate::handoff::y_cpu2(d) } else { crate::handoff::y_cpu3(d) };
                    if fuse {
                        inj_off = off as i32;
                    } else {
                        self.hc_join_main()?;
                        self.g.launch(self.k["hc_add_post"], ((d as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(self.h.ptr), Arg::P(d0 + off as u64), Arg::P(self.dpost.ptr + (((li - 1) % 2) * 16) as u64), Arg::I(d as i32)])?;
                    }
                } else {
                    cpu_mask |= slot(li - 1);
                }
            }
            if last {
                self.flag_wait(d0 + crate::handoff::CPUD as u64, seq)?;
                cpu_mask |= slot(li);
            }
        }
        self.ev(li, 5)?;
        if tri_main {
            self.tri.as_ref().unwrap().main.as_ref().unwrap().join_add(&self.g, d, self.out0.ptr)?;
        }
        for (i, aux) in self.aux.iter().enumerate() {
            if aux.peer {
                self.g.wait(aux.peer_done[li], false)?;
            } else {
                self.flag_wait(d0 + crate::handoff::aux_flag(i) as u64, seq)?;
            }
        }
        self.ev(li, 6)?;
        if fuse {
            let prev = self.dpost.ptr + (((li + 1) % 2) * 16) as u64; // (li - 1) % 2, li > 0 when used
            self.g.launch(
                self.k["combine_post"],
                ((d as u32).div_ceil(256), 1, 1),
                256,
                0,
                &[Arg::P(a), Arg::P(self.out0.ptr), Arg::P(d0), Arg::I(d as i32), Arg::I(cpu_mask), Arg::I(self.aux.len() as i32), Arg::I(crate::handoff::y_cpu2(d) as i32), Arg::I(crate::handoff::y_cpu3(d) as i32), Arg::P(self.h.ptr), Arg::I(inj_off), Arg::P(prev), Arg::P(self.dpost.ptr + ((li % 2) * 16) as u64), Arg::P(self.mix_f.ptr), Arg::I(self.fuse4 as i32), Arg::P(self.aux_results.as_ref().map_or(0, |b| b.ptr)), Arg::I(self.aux_peer_mask as i32)],
            )?;
            // the layer's closing hc_post ran inside combine_post
            self.post_ffn_done = self.fuse4;
        } else {
            self.g.launch(self.k["combine_experts"], (((t * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(a), Arg::P(self.out0.ptr), Arg::P(d0), Arg::I(d as i32), Arg::I(cpu_mask), Arg::I(self.aux.len() as i32), Arg::I(t as i32), Arg::I(crate::handoff::y_cpu2(d) as i32), Arg::I(crate::handoff::y_cpu3(d) as i32), Arg::P(self.aux_results.as_ref().map_or(0, |b| b.ptr)), Arg::I(self.aux_peer_mask as i32)])?;
        }
        self.ev(li, 7)
    }

    /// attention for t <= 8 queries (sparse_attn's contract): split-K partials, then the merge with the sink.
    /// DSX_OLD_ATTN=1 runs the serial single-kernel version.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sparse_attn_small(&self, t: usize, q: u64, win: u64, win_idx: u64, w: usize, comp: u64, comp_idx: u64, k: usize, sinks: u64, o: u64) -> Result<()> {
        let c = &self.m.cfg;
        let (nh, hd) = (c.n_heads, c.head_dim);
        let scale = 1.0 / (hd as f32).sqrt();
        if std::env::var_os("DSX_OLD_ATTN").is_some() {
            return self.g.launch(self.k["sparse_attn"], (t as u32, (nh / 8) as u32, 1), 256, 0, &[Arg::P(q), Arg::P(win), Arg::P(win_idx), Arg::I(w as i32), Arg::P(comp), Arg::P(comp_idx), Arg::I(k as i32), Arg::P(sinks), Arg::P(o), Arg::I(nh as i32), Arg::I(hd as i32), Arg::F(scale)]);
        }
        let s = (w + k).div_ceil(32);
        anyhow::ensure!(s <= ATTN_SPLITS, "attention over {} rows: more than {ATTN_SPLITS} tiles", w + k);
        self.g.launch(self.k["sparse_attn_tile"], (t as u32, (nh / 16) as u32, s as u32), 256, 0, &[Arg::P(q), Arg::P(win), Arg::P(win_idx), Arg::I(w as i32), Arg::P(comp), Arg::P(comp_idx), Arg::I(k as i32), Arg::P(self.attn_part.ptr), Arg::I(nh as i32), Arg::I(hd as i32), Arg::F(scale)])?;
        self.g.launch(self.k["sparse_attn_merge"], (t as u32, nh as u32, 1), 256, 0, &[Arg::P(self.attn_part.ptr), Arg::I(s as i32), Arg::P(sinks), Arg::P(o), Arg::I(nh as i32), Arg::I(hd as i32)])
    }

    /// Rank non-resident experts and choose the coldest same-typed victims across stores with room. This only reads
    /// placement and scores: it may overlap a token once the worker has published that pass's completed score epoch.
    fn adapt_plan(&self) -> Vec<crate::adapt::Swap> {
        let ne = self.m.cfg.n_experts;
        let (Some(w), Some(arena)) = (self.worker.as_ref(), self.arena) else { return vec![] };
        let stores: Vec<&crate::experts::GpuExperts> =
            self.exp0.iter().chain(self.aux.iter().filter(|a| a.adaptive).map(|a| &a.exp)).collect();
        let rooms: Vec<usize> = stores.iter().map(|s| s.swap_room()).collect();
        let room: usize = rooms.iter().sum::<usize>().min(self.adapt);
        if room == 0 {
            return vec![];
        }
        let sc = |l: usize, e: usize| f32::from_bits(w.score[l * ne + e].load(std::sync::atomic::Ordering::Relaxed));
        let pending: std::collections::HashSet<(usize, usize)> = stores.iter().flat_map(|s| s.pending()).collect();
        let mut cand: Vec<(f32, usize, usize)> = (0..w.score.len())
            .filter_map(|i| {
                let (l, e) = (i / ne, i % ne);
                let v = sc(l, e);
                (v >= 1.0 && w.on_cpu[i].load(std::sync::atomic::Ordering::Relaxed) && !pending.contains(&(l, e)) && arena.get(l, e).is_some()).then_some((v, l, e))
            })
            .collect();
        let full = std::env::var("DSX_CACHE_SELECT").is_ok_and(|v| v == "full");
        if full {
            // The original production algorithm, including its unspecified order for tied scores.
            cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            cand.truncate(room * 4);
        } else {
            crate::cache_select::keep(&mut cand, room * 4, crate::cache_select::hottest, false);
        }
        // per store: residents by quant types, coldest last (popped first)
        let victims: Vec<crate::adapt::Victims> = stores
            .iter()
            .map(|s| {
                let mut v: crate::adapt::Victims = Default::default();
                for (vl, ve) in s.residents() {
                    v.entry(s.types(vl, ve).unwrap()).or_default().push((sc(vl, ve), vl, ve));
                }
                for x in v.values_mut() {
                    if full {
                        x.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    } else {
                        crate::cache_select::keep(x, room, crate::cache_select::coldest, false);
                        x.reverse(); // the coldest remaining victim is popped first
                    }
                }
                v
            })
            .collect();
        crate::adapt::choose(&cand, rooms, victims, room, |l, e| {
            let x = &self.m.layers[l].experts[e];
            (x.gate.ty, x.down.ty)
        })
    }

    /// Between tokens only: start the planned replacements. Evictions and uploads remain after the logits download;
    /// the existing side-stream event still protects each slot's contents.
    fn adapt_apply(&mut self, plan: Vec<crate::adapt::Swap>) -> Result<()> {
        let Some(arena) = self.arena else { return Ok(()) };
        let mut stores: Vec<(&Gpu, &mut crate::experts::GpuExperts)> =
            self.exp0.as_mut().map(|s| (&self.g, s)).into_iter().chain(self.aux.iter_mut().filter(|a| a.adaptive).map(|a| (&a.g, &mut a.exp))).collect();
        for p in plan {
            let raw = arena.get(p.new.0, p.new.1).unwrap();
            let (g, s) = &mut stores[p.store];
            if !s.begin_swap(g, p.victim, p.new, raw)? {
                break;
            }
            self.adapt_flags.push((p.victim.0 * self.m.cfg.n_experts + p.victim.1, true));
            self.adapt_swaps += 1;
        }
        self.refill_dynamic()?;
        Ok(())
    }

    /// Default path: keep the decision and replacement work together, after download and commit.
    fn adapt_step(&mut self) -> Result<()> {
        let plan = self.adapt_plan();
        self.adapt_apply(plan)
    }

    /// Plan against the same post-commit inputs as the default path. If any upload is unfinished, leave the whole
    /// operation for after download: a partial commit could change victim order, capacity and pending exclusions.
    fn adapt_early_plan(&mut self) -> Result<Option<Vec<crate::adapt::Swap>>> {
        // DSX_SKIP_EXPERTS publishes no routes, hence no completed worker epoch to acquire.
        if self.skip_experts || self.arena.is_none() {
            return Ok(None);
        }
        let Some(w) = &self.worker else { return Ok(None) };
        let tw = std::time::Instant::now();
        w.scores_ready.wait(self.seq_base);
        self.hmark(tw, "host: adapt scores wait");
        let tc = std::time::Instant::now();
        let stores = self.exp0.iter().map(|s| (&self.g, s))
            .chain(self.aux.iter().filter(|a| a.adaptive).map(|a| (&a.g, &a.exp)));
        let mut ready = true;
        for (g, s) in stores {
            if !s.swaps_ready(g)? {
                ready = false;
                break;
            }
        }
        if ready {
            // Every pending swap will also be ready after download. Commit in the same order as the default path.
            // The score epoch fences the worker's on_cpu reads; table writes follow all queued GPU uses.
            self.adapt_commit()?;
        }
        self.g.bind();
        self.hmark(tc, "host: adapt early commit");
        self.adapt_s += tc.elapsed().as_secs_f64();
        if !ready {
            return Ok(None);
        }
        let tp = std::time::Instant::now();
        let plan = self.adapt_plan();
        self.hmark(tp, "host: adapt plan");
        self.adapt_s += tp.elapsed().as_secs_f64();
        Ok(Some(plan))
    }

    /// After a token's sync, or its completed worker score epoch: make finished replacements resident. Table copies
    /// follow the token's queued kernels on each GPU's main stream; no worker reads on_cpu again until the next pass.
    fn adapt_commit(&mut self) -> Result<()> {
        let ne = self.m.cfg.n_experts;
        let Some(w) = self.worker.as_ref() else { return Ok(()) };
        let stores = self.exp0.as_mut().map(|s| (&self.g, s)).into_iter().chain(self.aux.iter_mut().filter(|a| a.adaptive).map(|a| (&a.g, &mut a.exp)));
        for (g, s) in stores {
            for (l, e) in s.commit_ready(g, false)? {
                w.on_cpu[l * ne + e].store(false, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.g.bind();
        Ok(())
    }

    /// Before a multi-token pass (prefill, verify, eval): finish and commit every in-flight replacement.
    pub(crate) fn adapt_flush(&mut self) -> Result<()> {
        let ne = self.m.cfg.n_experts;
        let Some(w) = self.worker.as_ref() else { return Ok(()) };
        let stores = self.exp0.as_mut().map(|s| (&self.g, s)).into_iter().chain(self.aux.iter_mut().filter(|a| a.adaptive).map(|a| (&a.g, &mut a.exp)));
        for (g, s) in stores {
            for (l, e) in s.commit_ready(g, true)? {
                w.on_cpu[l * ne + e].store(false, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.g.bind();
        Ok(())
    }

    /// DSX_PROF=1: synchronize GPU0 and add the time since `t0` under `name` (prefill profiling)
    pub(crate) fn prof_mark(&mut self, name: &'static str, t0: std::time::Instant) -> Result<()> {
        if self.prof.is_some() && !self.prof_host {
            self.g.sync()?;
            *self.prof.as_mut().unwrap().entry(name).or_default() += t0.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// DSX_PROF=host: add host time since `t0` under `name`
    fn hmark(&mut self, t0: std::time::Instant, name: &'static str) {
        if self.prof_host {
            *self.prof.as_mut().unwrap().entry(name).or_default() += t0.elapsed().as_secs_f64();
        }
    }

    /// GPU0's stream waits for a handoff flag: a spinning kernel (DSX_SPIN_WAIT) or cuStreamWaitValue32
    fn flag_wait(&self, addr: u64, seq: u32) -> Result<()> {
        if self.spin_wait {
            // the CPU flag's spin records GPU-side handoff time per layer
            let d0 = self.xch_dev;
            let diag = if addr == d0 + crate::handoff::CPU as u64 { d0 + (crate::handoff::DIAG + 16 * ((seq - 1) as usize % self.m.cfg.n_layers)) as u64 } else { 0 };
            self.g.launch(self.k["spin_wait"], (1, 1, 1), 1, 0, &[Arg::P(addr), Arg::I(seq as i32), Arg::P(diag)])
        } else {
            self.g.wait_value(addr, seq)
        }
    }

    /// DSX_EVPROF: record stage `k` of layer `li` on GPU0's stream
    fn ev(&self, li: usize, k: usize) -> Result<()> {
        if self.ev0.is_empty() { Ok(()) } else { self.g.record(self.ev0[li * EV_N + k], false) }
    }

    /// DSX_EVPROF: add this step's per-stage GPU times (events are complete once the logits are downloaded)
    fn ev_collect(&mut self) -> Result<()> {
        if self.ev0.is_empty() {
            return Ok(());
        }
        const N0: [&str; 7] = ["0 layer start -> moe (attn, engram, hc)", "1 router", "2 publish + GPU0 experts", "3 shared expert", "4 wait CPU experts", "5 wait aux GPUs", "6 combine"];
        // (from, to, name) inside stage 0; attention parts split encoder (0-19) / decoder
        const SUB: [(usize, usize, &str); 8] = [(0, 8, "0a engram"), (8, 9, "0b hc+norm"), (9, 11, "0c q"), (11, 12, "0d kv"), (12, 13, "0e compress+indexer"), (13, 14, "0f sparse_attn"), (14, 10, "0g o proj"), (10, 1, "0h hc post/pre+norm")];
        let mut acc = std::collections::BTreeMap::new();
        for li in 0..self.m.cfg.n_layers {
            for &(a, b, name) in &SUB {
                let v = self.g.elapsed_ms(self.ev0[li * EV_N + a], self.ev0[li * EV_N + b])? as f64;
                *acc.entry(name).or_insert(0f64) += v;
                if (12..=14).contains(&a) {
                    let tag: &'static str = match (a, li < 20) { (12, true) => "   enc compress+indexer", (12, false) => "   dec compress+indexer", (13, true) => "   enc sparse_attn", (13, false) => "   dec sparse_attn", (_, true) => "   enc o proj", (_, false) => "   dec o proj" };
                    *acc.entry(tag).or_insert(0f64) += v;
                }
            }
            for k in 0..7 {
                *acc.entry(N0[k]).or_insert(0f64) += self.g.elapsed_ms(self.ev0[li * EV_N + k], self.ev0[li * EV_N + k + 1])? as f64;
            }
            const LW: [&str; 40] = ["w00", "w01", "w02", "w03", "w04", "w05", "w06", "w07", "w08", "w09", "w10", "w11", "w12", "w13", "w14", "w15", "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23", "w24", "w25", "w26", "w27", "w28", "w29", "w30", "w31", "w32", "w33", "w34", "w35", "w36", "w37", "w38", "w39"];
            if li < 40 {
                *acc.entry(LW[li]).or_insert(0f64) += self.g.elapsed_ms(self.ev0[li * EV_N + 4], self.ev0[li * EV_N + 5])? as f64;
            }
            if li + 1 < self.m.cfg.n_layers {
                *acc.entry("7 combine -> next layer").or_insert(0f64) += self.g.elapsed_ms(self.ev0[li * EV_N + 7], self.ev0[(li + 1) * EV_N])? as f64;
            }
        }
        // aux GPUs: summed over all of them
        for a in &self.aux {
            const N1: [&str; 3] = ["aux: htod", "aux: experts", "aux: dtoh"];
            for li in 0..self.m.cfg.n_layers {
                for k in 0..3 {
                    *acc.entry(N1[k]).or_insert(0f64) += a.g.elapsed_ms(a.ev[li * 4 + k], a.ev[li * 4 + k + 1])? as f64;
                }
                if li + 1 < self.m.cfg.n_layers {
                    *acc.entry("aux: dtoh -> next route").or_insert(0f64) += a.g.elapsed_ms(a.ev[li * 4 + 3], a.ev[(li + 1) * 4])? as f64;
                }
            }
        }
        self.g.bind();
        let p = self.evprof.as_mut().unwrap();
        for (k, v) in acc {
            *p.entry(k).or_default() += v;
        }
        *p.entry("steps").or_default() += 1.0;
        Ok(())
    }

    fn moe_fast(&mut self, li: usize) -> Result<()> {
        let (x, a) = (self.x.ptr, self.a.ptr);
        self.moe_fast_t(li, x, a, 1)
    }

    /// y[t][w.rows] = W x[t] for t <= 8 rows of x (row stride w.cols)
    fn gemv_t(&self, w: &DMat, x: u64, y: u64, t: usize) -> Result<()> {
        self.gemv_rows_t(w, 0, w.rows, x, y, t)
    }

    /// seconds the CPU expert worker has spent computing (0 without a worker)
    pub fn cpu_busy_s(&self) -> f64 {
        self.worker.as_ref().map_or(0.0, |w| w.busy_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9)
    }

    /// per layer (GPU publish -> CPU sees picks, CPU flag store -> GPU sees it) in microseconds, averaged; the two
    /// clocks (GPU globaltimer, CPU wall clock) may be offset, which shifts the legs in opposite directions
    pub fn handoff_legs_us(&self) -> Vec<(f64, f64)> {
        let (Some(w), Some(x)) = (self.worker.as_ref(), self.xch) else { return vec![] };
        (0..self.m.cfg.n_layers)
            .map(|l| {
                let a = &w.leg1_ns[l];
                let (s1, n1) = (a[0].load(std::sync::atomic::Ordering::Relaxed) as i64, a[1].load(std::sync::atomic::Ordering::Relaxed).max(1) as f64);
                let d = unsafe { std::slice::from_raw_parts(x.at::<u64>(crate::handoff::DIAG + 16 * l), 2) };
                (s1 as f64 / n1 / 1e3, d[0] as i64 as f64 / (d[1].max(1) as f64) / 1e3)
            })
            .collect()
    }

    /// routed picks so far and how many the CPU computed
    pub fn cpu_picks(&self) -> (u64, u64) {
        self.worker.as_ref().map_or((0, 0), |w| (w.picks[0].load(std::sync::atomic::Ordering::Relaxed), w.picks[1].load(std::sync::atomic::Ordering::Relaxed)))
    }

    /// CPU worker [idle, prep, forward] seconds
    pub fn cpu_split_s(&self) -> [f64; 3] {
        self.worker.as_ref().map_or([0.0; 3], |w| std::array::from_fn(|i| w.split_ns[i].load(std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9))
    }

    fn bonus_ptr(&self, li: usize) -> u64 {
        self.bonus.as_ref().map_or(0, |b| b.ptr + (li * self.m.cfg.n_experts * 4) as u64)
    }

    fn gemv(&self, w: &DMat, x: u64, y: u64) -> Result<()> {
        self.gemv_rows(w, 0, w.rows, x, y)
    }

    /// rows [r0, r0 + n) of w
    fn gemv_rows(&self, w: &DMat, r0: usize, n: usize, x: u64, y: u64) -> Result<()> {
        self.gemv_rows_t(w, r0, n, x, y, 1)
    }

    fn gemv_rows_t(&self, w: &DMat, r0: usize, n: usize, x: u64, y: u64, t: usize) -> Result<()> {
        if w.q6.is_some() {
            return self.gemv_rows_q6(w, r0, n, x, y, t);
        }
        self.gemv_rows_ld(w, r0, n, x, w.cols, y, n, t)
    }

    /// Q6: 8 lanes per row (gemv_q6s8, 64 rows per block) for many rows of short k: more chunks per lane, and
    /// k / 32 need not split over 32 lanes (3090: q_b 32768 x 1280 85 -> 71 us, o_a 8 x 1024 x 4096 58 -> 48 us; o_b,
    /// the head and small matrices stay on 32 lanes)
    fn q6_s8(n: usize, k: usize) -> bool {
        n >= 4096 && k <= 4096 && (k / 32) % 8 == 0
    }

    /// rows from which Q5_K planes use the 4-row kernel (gemv_q5p4); Q6: from 512 rows (gemv_q6g4)
    const PLANES_R4_MIN: usize = 2048;

    /// xs-engine: the one-token Q5_K kernel for k (LPR lanes x R rows, rows per 256-thread block)
    fn xs_q5_kernel(k: usize) -> Option<(&'static str, usize)> {
        match k / 32 {
            n if n % 32 == 0 => Some(("xs_q5_32x4", 32)),
            n if n % 8 == 0 && k <= 1536 => Some(("xs_q5_8x4", 128)),
            n if n % 8 == 0 => Some(("xs_q5_8x2", 64)),
            _ => None,
        }
    }

    /// xs-engine: rows [r0, r0 + n) of a Q5_K matrix for one token; rows read x + (row / rpg) * xgs; `round`: bf16 out
    #[allow(clippy::too_many_arguments)]
    fn xs_gemv_q5(&self, w: &DMat, r0: usize, n: usize, x: u64, y: u64, rpg: usize, xgs: usize, round: bool, kn: &'static str, rpb: usize) -> Result<()> {
        let ([lo, hb, sm, dd], _, _) = w.planes(r0);
        self.g.launch(
            self.k[kn],
            ((n as u32).div_ceil(rpb as u32), 1, 1),
            256,
            0,
            &[Arg::P(lo), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(w.cols as i32), Arg::P(hb), Arg::P(sm), Arg::P(dd), Arg::I(rpg as i32), Arg::I(xgs as i32), Arg::I(round as i32)],
        )
    }

    fn gemv_rows_q6(&self, w: &DMat, r0: usize, n: usize, x: u64, y: u64, t: usize) -> Result<()> {
        let ([lo, hi, sc, d], kn, _) = w.planes(r0);
        let k = w.cols;
        if t == 1 && self.xs_dense && w.ty == GType::Q5K {
            if let Some((xk, rpb)) = Self::xs_q5_kernel(k) {
                return self.xs_gemv_q5(w, r0, n, x, y, n, 0, false, xk, rpb);
            }
        }
        // 4 rows per warp once there are enough rows to fill the GPU (fewer blocks hurt small matrices)
        let r4 = if kn == "gemv_q6p" { n >= 512 } else { n >= Self::PLANES_R4_MIN };
        let (kn, rpb) = if kn == "gemv_q6p" && Self::q6_s8(n, k) {
            ("gemv_q6s8", 64)
        } else if r4 {
            (if kn == "gemv_q6p" { "gemv_q6g4" } else { "gemv_q5p4" }, 16)
        } else {
            (kn, 4)
        };
        self.g.launch(
            self.k[kn],
            ((n as u32).div_ceil(rpb), 1, 1),
            128,
            0,
            // gemv_q6g's trailing (rows per group, x group stride): one group (the others ignore extra arguments)
            &[Arg::P(lo), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(k as i32), Arg::I(t as i32), Arg::I(k as i32), Arg::I(n as i32), Arg::P(hi), Arg::P(sc), Arg::P(d), Arg::I(n as i32), Arg::I(0)],
        )
    }

    /// y[t][ldy] (rows r0..r0+n of W) = W x[t][ldx], t <= 8
    #[allow(clippy::too_many_arguments)]
    fn gemv_rows_ld(&self, w: &DMat, r0: usize, n: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize) -> Result<()> {
        let kname = match w.ty {
            GType::Q8_0 if w.q8r => "gemv_q8r",
            GType::Q8_0 => "gemv_q8_0",
            GType::F16 => "gemv_f16",
            GType::BF16 => "gemv_bf16",
            GType::F32 => "gemv_f32",
            // converted from an F16 copy (DSX_ENGRAM_FP8; checkpoint FP8 weights quantize activations): gemv_f16's exact
            // arithmetic, bit-identical to the F16 path
            GType::Fp8B32 if !w.act_fp8 => "gemv_fp8b32_o",
            GType::Fp8B32 => "gemv_fp8b32",
            GType::Q2_0 => "gemv_q2_0",
            GType::Ptq1_0 => "gemv_ptq1_0",
            GType::Q3K => "gemv_q3_k",
            t => bail!("no GPU gemv for {t:?}"),
        };
        let (wp, sp, _) = w.ptrs(r0);
        let x = if w.act_fp8 {
            anyhow::ensure!(ldx == w.cols, "strided GEMV input with FP8 activations");
            self.g.launch(self.k["fake_fp8_copy"], ((((t * w.cols) / 32) as u32).div_ceil(8), 1, 1), 256, 0, &[Arg::P(x), Arg::P(self.actq.ptr), Arg::I((t * w.cols) as i32)])?;
            self.actq.ptr
        } else {
            x
        };
        let cols = w.cols as i32;
        // few rows: one block per row with the warps splitting k (measured faster below ~2K rows)
        let sk = n < 2048 && matches!(kname, "gemv_q8r" | "gemv_f16" | "gemv_bf16" | "gemv_f32");
        let kname = if sk { match kname { "gemv_q8r" => "gemv_q8r_sk", "gemv_f16" => "gemv_f16_sk", "gemv_bf16" => "gemv_bf16_sk", _ => "gemv_f32_sk" } } else { kname };
        // one token: gemv_q8r without the MAXT accumulators (bit-identical, more warps per SM)
        let kname = if kname == "gemv_q8r" && t == 1 && self.q8r1 { "gemv_q8r1" } else { kname };
        // one token, few rows: the split-k kernels without MAXT accumulators (bit-identical)
        let kname = match kname {
            "gemv_bf16_sk" if t == 1 && self.sk1 => "gemv_bf16_sk1",
            "gemv_f16_sk" if t == 1 && self.sk1 => "gemv_f16_sk1",
            "gemv_q8r_sk" if t == 1 && self.sk1 => "gemv_q8r_sk1",
            k => k,
        };
        let per_block = if sk { 1 } else if matches!(kname, "gemv_q8r4" | "gemv_fp8b32") { 16 } else { 4 };
        self.g.launch(
            self.k[kname],
            ((n as u32).div_ceil(per_block), 1, 1),
            128,
            0,
            &[Arg::P(wp), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(cols), Arg::I(t as i32), Arg::I(ldx as i32), Arg::I(ldy as i32), Arg::P(sp)],
        )
    }

    fn rmsnorm(&self, x: u64, w: &Buf, out: u64, n: usize, round: bool) -> Result<()> {
        let eps = self.m.cfg.eps;
        self.g.launch(
            self.k["rmsnorm"],
            (1, 1, 1),
            1024,
            0,
            &[Arg::P(x), Arg::P(w.ptr), Arg::P(out), Arg::I(n as i32), Arg::I(n as i32), Arg::I(n as i32), Arg::F(eps), Arg::I(round as i32)],
        )
    }

    /// One token: hc_pre with `mixes` (pre weights) into x, then rmsnorm(x, w), in one launch
    fn hc_pre_norm(&self, mixes: &Buf, w: &Buf, round: bool) -> Result<()> {
        let d = self.m.cfg.dim;
        self.g.launch(
            self.k["hc_pre_norm"],
            ((d as u32).div_ceil(1024), 1, 1),
            1024,
            0,
            &[Arg::P(self.h.ptr), Arg::P(mixes.ptr), Arg::P(self.x.ptr), Arg::P(w.ptr), Arg::I(d as i32), Arg::F(self.m.cfg.eps), Arg::I(round as i32)],
        )
    }

    fn rope(&self, x: u64, heads: usize, hd: usize, compressed: bool, pos: usize, inverse: bool) -> Result<()> {
        let c = &self.m.cfg;
        let f = if compressed { &self.freqs.1 } else { &self.freqs.0 };
        self.g.launch(
            self.k["rope"],
            (1, ((heads * c.rope_dim / 2) as u32).div_ceil(256), 1),
            256,
            0,
            &[Arg::P(x), Arg::P(f.ptr), Arg::I(heads as i32), Arg::I(hd as i32), Arg::I(c.rope_dim as i32), Arg::I((heads * hd) as i32), Arg::I(pos as i32), Arg::I(inverse as i32)],
        )
    }

    fn fake_fp8(&self, x: u64, n: usize) -> Result<()> {
        self.g.launch(self.k["fake_fp8"], ((n as u32 / 32).div_ceil(8), 1, 1), 256, 0, &[Arg::P(x), Arg::I(n as i32)])
    }

    fn fake_fp4(&self, x: u64, n: usize, block: usize, e4m3s: bool) -> Result<()> {
        self.g.launch(
            self.k["fake_fp4"],
            ((n as u32 / 32).div_ceil(8), 1, 1),
            256,
            0,
            &[Arg::P(x), Arg::I(n as i32), Arg::I(block as i32), Arg::I(e4m3s as i32)],
        )
    }

    fn store_bf16(&self, x: u64, dst: u64, n: usize) -> Result<()> {
        self.g.launch(self.k["store_bf16"], ((n as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(x), Arg::P(dst), Arg::I(n as i32)])
    }

    /// Gather this token's Engram rows for every engram layer into `out` ([layer][rows][dim] f32), all rows in
    /// parallel (each is a random 264-byte read from the mmapped table). Runs on a helper thread while layer 0 runs.
    fn prefetch_into(m: &Model, hist: &[i64], out: &mut [f32]) -> Result<()> {
        let c = &m.cfg;
        let Some(tabs) = m.engram.as_ref() else { return Ok(()) };
        let hd = c.engram_dim;
        let per = (c.engram_ngram - 1) * c.engram_heads;
        for eidx in 0..c.engram_layers.len() {
            let tab = tabs.layer(eidx, hd)?;
            let tab = &tab;
            let ids = m.engram_hashes_for(hist, eidx);
            let off = eidx * per * hd;
            // 8 plain threads (not rayon: its pool shares cores with the spinning CPU-expert workers); uncached rows
            // are NVMe reads, so the parallelism is what hides them behind layer 0
            let rows: Vec<(usize, &mut [f32])> = ids.iter().map(|&r| r as usize).zip(out[off..off + per * hd].chunks_mut(hd)).collect();
            let mut groups: Vec<Vec<(usize, &mut [f32])>> = (0..8).map(|_| vec![]).collect();
            for (i, r) in rows.into_iter().enumerate() {
                groups[i % 8].push(r);
            }
            std::thread::scope(|sc| {
                for grp in groups {
                    sc.spawn(move || {
                        crate::cpu::pool::pin_to_set(&crate::cpu::pool::layout().aux);
                        for (row, o) in grp {
                            tab.row(row, o);
                        }
                    });
                }
            });
        }
        Ok(())
    }

    fn engram(&mut self, li: usize, eidx: usize) -> Result<()> {
        // rows were staged by prefetch_into (joined before the first engram layer); copy in stream order
        let c = &self.m.cfg;
        let n = (c.engram_ngram - 1) * c.engram_heads * c.engram_dim;
        let hb = self.eng_host.as_mut_slice::<f32>();
        let off = eidx * n;
        self.g.htod(self.tmp2.ptr, &hb[off..off + n])?;
        let l = &self.layers[li];
        self.gemv(l.engram_kv.as_ref().unwrap(), self.tmp2.ptr, self.tmp.ptr)?;
        self.g.launch(
            self.k["engram_gate"],
            (c.hc as u32, 1, 1),
            1024,
            0,
            &[Arg::P(self.h.ptr), Arg::P(self.tmp.ptr), Arg::P(l.engram_qk.as_ref().unwrap().ptr), Arg::I(c.dim as i32), Arg::F(c.eps)],
        )
    }

    /// The sublayer's hc mixes into `out` on the par stream, beside the main stream's attention or MoE: they read h as
    /// it is now and are first needed by the hc_post after that work. `hc_join_main` must run before the main stream
    /// next writes h or reads `out`.
    fn hc_mixes(&self, fnw: &(DMat, Buf, Buf), out: &Buf) -> Result<()> {
        let c = &self.m.cfg;
        // dots split 8 ways for more blocks, then the per-token reduction
        let part = self.hc_part.ptr;
        self.g.par_after_main(self.hc_fork)?;
        self.g.launch_par(self.k["hc_part"], (25, 1, 8), 256, 0, &[Arg::P(fnw.0.buf.ptr), Arg::P(self.h.ptr), Arg::P(part), Arg::I((c.hc * c.dim) as i32), Arg::I(8)])?;
        self.g.launch_par(self.k["hc_reduce_finish"], (1, 1, 1), 32, 0, &[Arg::P(part), Arg::P(out.ptr), Arg::I((c.hc * c.dim) as i32), Arg::I(8), Arg::F(c.eps), Arg::P(fnw.1.ptr), Arg::P(fnw.2.ptr), Arg::F(c.hc_eps), Arg::I(c.sinkhorn_iters as i32)])?;
        self.g.record_par(self.hc_join)
    }

    /// the main stream waits for the last `hc_mixes`
    fn hc_join_main(&self) -> Result<()> {
        self.g.wait(self.hc_join, false)
    }

    fn attention(&mut self, li: usize) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let t = self.pos;
        let (hd, nh) = (c.head_dim, c.n_heads);
        let ratio = m.layers[li].ratio;
        let compressed = ratio > 0;
        let l = &self.layers[li];
        self.gemv(&l.q_a, self.x.ptr, self.qr.ptr)?;
        self.rmsnorm(self.qr.ptr, &l.q_a_norm, self.qr.ptr, c.q_lora, true)?;
        // DSX_PAR_ATTN: the kv / compressor / indexer branch (inputs x and qr, needed first by sparse attention) runs on the
        // par stream while the main stream runs q_b; joined before sparse attention
        let par_attn = self.par_attn;
        if par_attn {
            self.g.par_after_main(self.attn_fork)?;
        }
        // q_b's RoPE in the GEMV's epilogue (gemv_q8r1_rope) when q_b is the one-row-per-warp Q8_0 one
        let qb = &l.q_b;
        // off by default (DSX_QB_ROPE=1): on sm_86 its rotation differs from rope by 1 ulp in ~700 of 4096 values
        // (bench/qb_rope.cu on the 3090; exact on the 5090), and window 35's adapt-0 KL moved 0.160876 -> 0.170395
        if self.qb_rope && self.fuse3 && self.q8r1 && qb.ty == GType::Q8_0 && qb.q8r && !qb.act_fp8 && qb.rows >= 2048 && qb.rows % 4 == 0 && hd % 4 == 0 {
            let (wp, sp, _) = qb.ptrs(0);
            let f = if compressed { &self.freqs.1 } else { &self.freqs.0 };
            self.g.launch(self.k["gemv_q8r1_rope"], ((qb.rows / 4) as u32, 1, 1), 128, 0, &[Arg::P(wp), Arg::P(self.qr.ptr), Arg::P(self.q.ptr), Arg::I(qb.rows as i32), Arg::I(qb.cols as i32), Arg::P(sp), Arg::P(f.ptr), Arg::I(hd as i32), Arg::I(c.rope_dim as i32), Arg::I(t as i32)])?;
        } else {
            self.gemv(&l.q_b, self.qr.ptr, self.q.ptr)?;
            self.rope(self.q.ptr, nh, hd, compressed, t, false)?;
        }
        self.ev(li, 11)?;
        if par_attn {
            // every launch below goes to the par stream until the swap back
            std::mem::swap(&mut self.g.stream, &mut self.g.par);
        }
        let l = &self.layers[li];
        self.gemv(&l.kv, self.x.ptr, self.kvv.ptr)?;
        let w = c.window;
        // rmsnorm + rope + fake FP8 + bf16 into the window slot, one launch
        let f = if compressed { &self.freqs.1 } else { &self.freqs.0 };
        self.g.launch(self.k["kv_norm_rope_store"], (1, 1, 1), 1024, 0, &[Arg::P(self.kvv.ptr), Arg::P(l.kv_norm.ptr), Arg::I(hd as i32), Arg::F(c.eps), Arg::P(f.ptr), Arg::I(c.rope_dim as i32), Arg::I(t as i32), Arg::P(l.window.ptr + ((t % w) * hd * 2) as u64)])?;
        self.ev(li, 12)?;
        let mut n_comp_rows = 0usize;
        if compressed {
            let compress_len = (t + 1) / ratio;
            let is_src = c.kv_sources.contains(&li);
            let mut have_latent = false;
            if is_src {
                let l = &self.layers[li];
                if ratio == 1 {
                    self.gemv(l.comp_kv.as_ref().unwrap(), self.x.ptr, self.lat.ptr)?;
                    self.rmsnorm(self.lat.ptr, l.comp_norm.as_ref().unwrap(), self.lat.ptr, hd, false)?;
                    have_latent = true;
                } else {
                    let (ks, ss) = l.comp_state.as_ref().unwrap();
                    let slot = t % ratio;
                    self.gemv(l.comp_kv.as_ref().unwrap(), self.x.ptr, ks.ptr + (slot * hd * 4) as u64)?;
                    self.gemv(l.comp_gate.as_ref().unwrap(), self.x.ptr, ss.ptr + (slot * hd * 4) as u64)?;
                    if (t + 1) % ratio == 0 {
                        self.g.launch(self.k["compress_pool"], (1, 1, 1), 512, 0, &[Arg::P(ks.ptr), Arg::P(ss.ptr), Arg::P(self.lat.ptr), Arg::I(ratio as i32), Arg::I(hd as i32)])?;
                        self.rmsnorm(self.lat.ptr, l.comp_norm.as_ref().unwrap(), self.lat.ptr, hd, false)?;
                        have_latent = true;
                    }
                }
                self.comp_src = li;
            }
            if c.index_sources.contains(&li) {
                if compress_len == 0 {
                    self.topk_k = 0;
                } else {
                    self.indexer(li, have_latent)?;
                }
            }
            if have_latent {
                let l = &self.layers[li];
                self.rope(self.lat.ptr, 1, hd, true, t + 1 - ratio, false)?;
                let n = l.n_comp;
                self.store_cache(self.lat.ptr, l.comp_cache.as_ref().unwrap().ptr, n * hd, hd, 16)?;
                self.layers[li].n_comp += 1;
            }
            n_comp_rows = self.topk_k;
        }
        if par_attn {
            std::mem::swap(&mut self.g.stream, &mut self.g.par);
            self.g.record_par(self.attn_join)?;
            self.g.wait(self.attn_join, false)?;
        }
        self.ev(li, 13)?;
        let l = &self.layers[li];
        let comp_ptr = if n_comp_rows > 0 { self.cache_pointer(self.layers[self.comp_src].comp_cache.as_ref().unwrap().ptr, 16) } else { 0 };
        // the merge, the inverse RoPE and fake FP8 in one launch (sparse_attn_merge_rf) when the shapes fit it
        let fuse3 = self.fuse3 && hd == 512 && std::env::var_os("DSX_OLD_ATTN").is_none();
        if fuse3 {
            let s = (w + n_comp_rows).div_ceil(32);
            anyhow::ensure!(s <= ATTN_SPLITS, "attention over {} rows: more than {ATTN_SPLITS} tiles", w + n_comp_rows);
            let scale = 1.0 / (hd as f32).sqrt();
            self.g.launch(self.k["sparse_attn_tile"], (1, (nh / 16) as u32, s as u32), 256, 0, &[Arg::P(self.q.ptr), Arg::P(l.window.ptr), Arg::P(self.tokp.ptr), Arg::I(w as i32), Arg::P(comp_ptr), Arg::P(self.comp_idx.ptr), Arg::I(n_comp_rows as i32), Arg::P(self.attn_part.ptr), Arg::I(nh as i32), Arg::I(hd as i32), Arg::F(scale)])?;
            self.ev(li, 14)?;
            let f = if compressed { &self.freqs.1 } else { &self.freqs.0 };
            self.g.launch(self.k["sparse_attn_merge_rf"], (1, nh as u32, 1), (hd / 2) as u32, 0, &[Arg::P(self.attn_part.ptr), Arg::I(s as i32), Arg::P(l.sinks.ptr), Arg::P(self.o.ptr), Arg::I(nh as i32), Arg::I(hd as i32), Arg::P(f.ptr), Arg::I(c.rope_dim as i32), Arg::I(t as i32)])?;
        } else {
            self.sparse_attn_small(1, self.q.ptr, l.window.ptr, self.tokp.ptr, w, comp_ptr, self.comp_idx.ptr, n_comp_rows, l.sinks.ptr, self.o.ptr)?;
            self.ev(li, 14)?;
            self.rope(self.o.ptr, nh, hd, compressed, t, true)?;
            self.fake_fp8(self.o.ptr, nh * hd)?;
        }
        let gsz = nh * hd / c.o_groups;
        // wo_a's bf16 rounding inside the GEMV (gemv_q8g_bf) when it is the Q8_0 one, or xs_q5_16x4's epilogue
        let xs_oa = self.xs_dense && l.o_a.ty == GType::Q5K && l.o_a.q6.is_some() && (gsz / 32) % 16 == 0;
        let rounded = (l.o_a.q8r && self.fuse2) || xs_oa;
        if xs_oa {
            // xs-engine: every group in one launch (group gi's rows read o's slice gi), bf16 rounding in the epilogue
            self.xs_gemv_q5(&l.o_a, 0, c.o_groups * c.o_lora, self.o.ptr, self.r.ptr, c.o_lora, gsz, true, "xs_q5_16x4", 64)?;
        } else if l.o_a.q8r {
            let (wp, sp, _) = l.o_a.ptrs(0);
            self.g.launch(self.k[if rounded { "gemv_q8g_bf" } else { "gemv_q8g" }], (((c.o_groups * c.o_lora) as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(wp), Arg::P(self.o.ptr), Arg::P(self.r.ptr), Arg::I((c.o_groups * c.o_lora) as i32), Arg::I(gsz as i32), Arg::I(c.o_lora as i32), Arg::I(gsz as i32), Arg::P(sp)])?;
        } else if l.o_a.q6.is_some() && l.o_a.ty != GType::Q5K && c.o_lora % 16 == 0 {
            // 6-bit planes: every group in one launch (group gi's rows read o's slice gi)
            let ([lo, hi, sc, d], _, _) = l.o_a.planes(0);
            let rows = c.o_groups * c.o_lora;
            let (kn, rpb) = if Self::q6_s8(rows, gsz) { ("gemv_q6s8", 64) } else { ("gemv_q6g4", 16) };
            self.g.launch(
                self.k[kn],
                ((rows as u32).div_ceil(rpb), 1, 1),
                128,
                0,
                &[Arg::P(lo), Arg::P(self.o.ptr), Arg::P(self.r.ptr), Arg::I(rows as i32), Arg::I(gsz as i32), Arg::I(1), Arg::I(gsz as i32), Arg::I(rows as i32), Arg::P(hi), Arg::P(sc), Arg::P(d), Arg::I(c.o_lora as i32), Arg::I(gsz as i32)],
            )?;
        } else {
            for gi in 0..c.o_groups {
                self.gemv_rows(&l.o_a, gi * c.o_lora, c.o_lora, self.o.ptr + (gi * gsz * 4) as u64, self.r.ptr + (gi * c.o_lora * 4) as u64)?;
            }
        }
        if !rounded {
            self.g.launch(self.k["round_bf16"], (((c.o_groups * c.o_lora) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(self.r.ptr), Arg::I((c.o_groups * c.o_lora) as i32)])?;
        }
        // o_b with the attention half's hc_post in its epilogue (gemv_q8r1_post)
        let ob = &l.o_b;
        if self.fuse4 && self.q8r1 && ob.ty == GType::Q8_0 && ob.q8r && !ob.act_fp8 && ob.rows >= 2048 && ob.rows == c.dim {
            let (wp, sp, _) = ob.ptrs(0);
            let (rows, cols) = (ob.rows, ob.cols);
            self.hc_join_main()?;
            self.g.launch(self.k["gemv_q8r1_post"], ((rows as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(wp), Arg::P(self.r.ptr), Arg::P(self.a.ptr), Arg::I(rows as i32), Arg::I(cols as i32), Arg::P(sp), Arg::P(self.h.ptr), Arg::P(self.mix_a.ptr)])?;
            self.post_attn_done = true;
            return Ok(());
        }
        self.gemv(&l.o_b, self.r.ptr, self.a.ptr)
    }

    fn indexer(&mut self, li: usize, have_latent: bool) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let t = self.pos;
        let ratio = m.layers[li].ratio;
        let (id, nh) = (c.idx_dim, c.idx_heads);
        let owns_k = c.kv_sources.contains(&li);
        if owns_k {
            if have_latent {
                let l = &self.layers[li];
                self.gemv(l.idx_k.as_ref().unwrap(), self.lat.ptr, self.tmp.ptr)?;
                self.rmsnorm(self.tmp.ptr, l.idx_k_norm.as_ref().unwrap(), self.tmp.ptr, id, false)?;
                self.rope(self.tmp.ptr, 1, id, true, t + 1 - ratio, false)?;
                let n = l.n_idx;
                self.store_cache(self.tmp.ptr, l.idx_cache.as_ref().unwrap().ptr, n * id, id, 32)?;
                self.layers[li].n_idx += 1;
            }
            self.idxk_src = li;
        }
        let l = &self.layers[li];
        self.gemv(l.idx_q_b.as_ref().unwrap(), self.qr.ptr, self.iq.ptr)?;
        self.rope(self.iq.ptr, nh, id, true, t, false)?;
        self.fake_fp4(self.iq.ptr, nh * id, 32, false)?;
        self.gemv(l.idx_proj.as_ref().unwrap(), self.x.ptr, self.iw.ptr)?;
        let wscale = (id as f32).powf(-0.5) * (nh as f32).powf(-0.5);
        self.g.launch(self.k["scale_bf16"], (1, 1, 1), 32, 0, &[Arg::P(self.iw.ptr), Arg::I(nh as i32), Arg::F(wscale)])?;
        let end = (t + 1) / ratio;
        let nvis = self.tokp.ptr + 4 * (c.window + if ratio == 1 { 0 } else { 1 }) as u64;
        let kc = self.cache_pointer(self.layers[self.idxk_src].idx_cache.as_ref().unwrap().ptr, 32);
        if self.xs_idx && nh == 32 && id == 128 {
            // decoder index layers only need the keys of layer 20's kept candidate blocks: the others score -inf here
            let mask = if li > c.cand_layer { self.keep.ptr } else { 0 };
            self.g.launch(
                self.k["xs_idx_score1"],
                ((end as u32).div_ceil(512), 1, 1),
                128,
                0,
                &[Arg::P(self.iq.ptr), Arg::P(kc), Arg::P(self.iw.ptr), Arg::P(self.score.ptr), Arg::I(end as i32), Arg::P(nvis), Arg::I(end as i32), Arg::I(1), Arg::P(mask), Arg::I(0), Arg::I(c.cand_block as i32)],
            )?;
        } else {
            self.g.launch(
                self.k["indexer_score"],
                ((end as u32).div_ceil(8), 1, 1),
                256,
                0,
                &[Arg::P(self.iq.ptr), Arg::P(kc), Arg::P(self.iw.ptr), Arg::P(self.score.ptr), Arg::P(nvis), Arg::I(end as i32), Arg::I(nh as i32), Arg::I(id as i32)],
            )?;
        }
        let rows = crate::topk::Rows { scores: self.score.ptr, lds: end, rows: 1, ns: 0, n_max: end };
        if li == c.cand_layer {
            self.topk.blocks(&self.g, &rows, c.cand_block, c.cand_blocks, self.keep.ptr, end.div_ceil(c.cand_block), self.topk_scratch.ptr)?;
        }
        let mask = if li > c.cand_layer { self.keep.ptr } else { 0 };
        self.topk.topk(&self.g, &rows, c.idx_topk, mask, end.div_ceil(c.cand_block), c.cand_block, self.comp_idx.ptr, c.idx_topk, self.topk_scratch.ptr)?;
        self.topk_k = c.idx_topk.min(end);
        Ok(())
    }

    #[allow(dead_code)]
    fn indexer_cpu_topk(&mut self, li: usize, end: usize) -> Result<()> {
        let c = &self.m.cfg;
        let mut score: Vec<f32> = self.g.download(self.score.ptr, end)?;
        // top-k on the CPU for now (the same selection as the reference)
        if li == c.cand_layer {
            let bs = c.cand_block;
            let nb = end.div_ceil(bs);
            let mut bscore: Vec<f32> = (0..nb).map(|b| score[b * bs..((b + 1) * bs).min(end)].iter().cloned().fold(f32::NEG_INFINITY, f32::max)).collect();
            bscore[(end - 1) / bs] = f32::INFINITY;
            let mut order: Vec<usize> = (0..nb).collect();
            order.sort_by(|&a, &b| bscore[b].partial_cmp(&bscore[a]).unwrap());
            let mut keep = vec![false; end];
            for &b in order.iter().take(c.cand_blocks.min(nb)) {
                if bscore[b] > f32::NEG_INFINITY {
                    for i in b * bs..((b + 1) * bs).min(end) {
                        keep[i] = true;
                    }
                }
            }
            self.candidates = keep;
        } else if li > c.cand_layer {
            for (s, v) in score.iter_mut().enumerate() {
                if !self.candidates.get(s).copied().unwrap_or(false) {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        let k = c.idx_topk.min(end);
        let mut order: Vec<usize> = (0..end).collect();
        order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap().then(a.cmp(&b)));
        let mut idx: Vec<usize> = order[..k].to_vec();
        idx.sort();
        let rows: Vec<i32> = idx.iter().map(|&i| if score[i] > f32::NEG_INFINITY { i as i32 } else { -1 }).collect();
        self.g.htod_sync(self.comp_idx.ptr, &rows)?;
        self.topk_k = rows.len();
        Ok(())
    }

    fn moe(&mut self, li: usize) -> Result<()> {
        anyhow::ensure!(!self.m.cfg.expert_rot[li], "layer {li}: rotated experts run on the handoff path only (the serial reference path doesn't rotate)");
        // router + shared expert on the GPU; routed experts on the CPU (reference matvec for now)
        let m = self.m;
        let c = &m.cfg;
        let d = c.dim;
        let l = &self.layers[li];
        self.gemv(&l.gate_w, self.x.ptr, self.tmp.ptr)?;
        let idx_buf = self.tmp2.ptr; // [k] i32 then [k] f32
        self.g.launch(self.k["router_topk"], (1, 1, 1), 32, 0, &[Arg::P(self.tmp.ptr), Arg::P(l.gate_bias.ptr), Arg::P(idx_buf), Arg::P(idx_buf + 64), Arg::I(c.n_experts as i32), Arg::I(c.top_k as i32), Arg::F(c.route_scale), Arg::P(self.bonus_ptr(li))])?;
        // shared expert
        let inter = c.inter;
        let (g_ptr, u_ptr, hm) = (self.tmp.ptr, self.tmp.ptr + (inter * 4) as u64, self.tmp.ptr + (2 * inter * 4) as u64);
        self.gemv(&l.sh_gate, self.x.ptr, g_ptr)?;
        self.gemv(&l.sh_up, self.x.ptr, u_ptr)?;
        self.g.launch(self.k["swiglu"], ((inter as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(g_ptr), Arg::P(u_ptr), Arg::I(inter as i32), Arg::P(0), Arg::P(hm), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
        self.gemv(&l.sh_down, hm, self.a.ptr)?;
        // routed experts on the CPU
        let sel: Vec<i32> = self.g.download(idx_buf, c.top_k)?;
        if std::env::var_os("DSX_ROUTES").is_some() && self.pos < 4 {
            eprintln!("gpu p{} l{li} {:?}", self.pos, sel);
        }
        if let Ok(path) = std::env::var("DSX_LOG_ROUTES") {
            // one line per (position, layer): the picks, for offline expert-overlap / hotness statistics
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
            writeln!(f, "{} {} {:?}", self.pos, li, sel)?;
        }
        let wts: Vec<f32> = self.g.download(idx_buf + 64, c.top_k)?;
        let x: Vec<f32> = self.g.download(self.x.ptr, d)?;
        let ml = &m.layers[li];
        // split the picks: aux stores, main GPU store, CPU
        let (mut l0, mut l1, mut lc) = (vec![], vec![vec![]; self.aux.len()], vec![]);
        for (j, &e) in sel.iter().enumerate() {
            let e = e as usize;
            if let Some(i) = self.aux.iter().position(|a| a.exp.has(li, e)) {
                l1[i].push((e, wts[j]));
            } else if self.exp0.as_ref().is_some_and(|s| s.has(li, e)) {
                l0.push((e, wts[j]));
            } else {
                lc.push((e, wts[j]));
            }
        }
        self.n_routed[0] += l0.len();
        self.n_routed[1] += l1.iter().map(|v| v.len()).sum::<usize>();
        self.n_routed[2] += lc.len();
        let mut out1 = vec![vec![0f32; d]; self.aux.len()];
        for ((a, li1), o1) in self.aux.iter().zip(&l1).zip(out1.iter_mut()) {
            if !li1.is_empty() {
                a.g.htod(a.exp.x.ptr, &x)?;
                a.exp.run(&a.g, li, li1, a.exp.x.ptr, a.exp.out.ptr)?;
                a.g.dtoh(o1, a.exp.out.ptr)?;
            }
        }
        if !l0.is_empty() {
            self.exp0.as_ref().unwrap().run(&self.g, li, &l0, self.x.ptr, self.out0.ptr)?;
        }
        let mut y = vec![0f32; d];
        let te = std::time::Instant::now();
        if let Some(cm) = self.cpu_moe.as_mut() {
            let ex: Vec<crate::cpu::moe::ExpertW> = lc
                .iter()
                .map(|&(e, _)| {
                    let x = &ml.experts[e];
                    crate::cpu::moe::ExpertW { gu: x.gate.ty, gate: x.gate.bytes(), up: x.up.bytes(), dn: x.down.ty, down: x.down.bytes() }
                })
                .collect();
            let w: Vec<f32> = lc.iter().map(|&(_, w)| w).collect();
            cm.forward(&ex, d, &x, 1, &w, &mut y);
        } else {
            for &(e, wt) in &lc {
                let ex = &ml.experts[e];
                let mut gv = vec![0f32; inter];
                let mut uv = vec![0f32; inter];
                matvec(&ex.gate, &x, &mut gv);
                matvec(&ex.up, &x, &mut uv);
                let lim = c.swiglu_limit;
                let hmid: Vec<f32> = gv.iter().zip(&uv).map(|(&g, &u)| round_bf16(silu(g.min(lim)) * u.clamp(-lim, lim) * wt)).collect();
                let mut o = vec![0f32; d];
                matvec(&ex.down, &hmid, &mut o);
                for i in 0..d {
                    y[i] += o[i];
                }
            }
        }
        self.t_experts += te.elapsed().as_secs_f64();
        for (a, li1) in self.aux.iter().zip(&l1) {
            if !li1.is_empty() {
                a.g.sync()?;
            }
        }
        self.g.bind();
        // shared + GPU0 experts are on the device; add the host parts, then bf16 like the reference
        let sh: Vec<f32> = self.g.download(self.a.ptr, d)?;
        let o0: Vec<f32> = if l0.is_empty() { vec![0f32; d] } else { self.g.download(self.out0.ptr, d)? };
        let out: Vec<f32> = (0..d).map(|i| round_bf16(y[i] + sh[i] + o0[i] + out1.iter().map(|o| o[i]).sum::<f32>())).collect();
        self.g.htod_sync(self.a.ptr, &out)?;
        let _ = dot;
        Ok(())
    }

    fn mark(&mut self, t: &mut std::time::Instant, name: &'static str) -> Result<()> {
        if self.prof.is_some() && !self.prof_host {
            self.g.sync()?;
            let e = t.elapsed().as_secs_f64();
            *self.prof.as_mut().unwrap().entry(name).or_default() += e;
            *t = std::time::Instant::now();
        }
        Ok(())
    }

    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        let t_step = std::time::Instant::now();
        // DSX_ADAPT_EARLY=1: overlap only the decision work. The old early path read live scores (including partial
        // decay), planned before commit and doubled staging capacity, changing placement and thus CPU/GPU arithmetic.
        let early = adapt_early();
        let r = self.step_inner(token).and_then(|()| {
            let plan = if self.adapt > 0 && early { self.adapt_early_plan()? } else { None };
            let tl = std::time::Instant::now();
            let l = self.g.download::<f32>(self.logits.ptr, self.m.cfg.vocab)?;
            self.hmark(tl, "host: logits wait + copy");
            let t = std::time::Instant::now();
            if self.adapt > 0 {
                self.adapt_commit()?;
                self.hmark(t, "host: adapt commit");
                if let Some(plan) = plan {
                    // All pending swaps were committed before planning, and no uploads have started since: none
                    // can commit here and invalidate the plan. Evictions and new uploads start only after download.
                    let ta = std::time::Instant::now();
                    self.adapt_apply(plan)?;
                    self.hmark(ta, "host: adapt apply");
                } else {
                    let ts = std::time::Instant::now();
                    self.adapt_step()?;
                    self.hmark(ts, if early { "host: adapt late fallback" } else { "host: adapt step" });
                }
            }
            if let Some(w) = self.worker.as_ref() {
                for (i, v) in self.adapt_flags.drain(..) {
                    w.on_cpu[i].store(v, std::sync::atomic::Ordering::Relaxed);
                }
            }
            self.adapt_s += t.elapsed().as_secs_f64();
            Ok(l)
        });
        if let Ok(l) = &r {
            if l.iter().any(|v| !v.is_finite()) && !self.warned_nan {
                self.warned_nan = true;
                eprintln!("warning: non-finite logits at position {} (token {token}); the routers fall back to valid experts", self.pos);
            }
        }
        if r.is_ok() {
            self.ev_collect()?;
            if let Some(pg) = self.pregate.as_ref() {
                let (nl, ne, k) = (self.m.cfg.n_layers, self.m.cfg.n_experts, self.m.cfg.top_k);
                let v: Vec<i32> = self.g.download(pg.ptr, nl * 16)?;
                for l in 1..nl {
                    let (pred, act) = (&v[l * 16..l * 16 + 8], &v[l * 16 + 8..l * 16 + 8 + k]);
                    {
                        let missed = |e: i32| e >= 0 && self.worker.as_ref().is_some_and(|w| w.on_cpu[l * ne + e as usize].load(std::sync::atomic::Ordering::Relaxed));
                        let cand: Vec<i32> = pred.iter().copied().filter(|&p| missed(p)).collect();
                        let total = act.iter().filter(|&&e| missed(e)).count() as u64;
                        self.pregate_cov[0] += total;
                        for r in 1..=4usize {
                            let top = &cand[..r.min(cand.len())];
                            self.pregate_cov[r] += act.iter().filter(|&&e| missed(e) && top.contains(&e)).count() as u64;
                        }
                    }
                    for &e in act {
                        let cpu = self.worker.as_ref().is_some_and(|w| w.on_cpu[l * ne + e as usize].load(std::sync::atomic::Ordering::Relaxed));
                        let (in6, in8) = (pred[..k].contains(&e), pred.contains(&e));
                        self.pregate_stats[0] += 1;
                        self.pregate_stats[1] += in6 as u64;
                        self.pregate_stats[2] += in8 as u64;
                        self.pregate_stats[3] += cpu as u64;
                        self.pregate_stats[4] += (cpu && in8) as u64;
                    }
                }
            }
        }
        if let Some(p) = self.prof.as_mut() {
            *p.entry("step total").or_default() += t_step.elapsed().as_secs_f64();
        }
        r
    }

    /// Enqueue one token through the head; `step` downloads the logits.
    fn step_inner(&mut self, token: u32) -> Result<()> {
        let t_enq = std::time::Instant::now();
        self.tri_sync()?;
        let m = self.m;
        let c = &m.cfg;
        let d = c.dim;
        self.g.bind();
        if self.pos + 1 > self.max_ctx {
            bail!("context full");
        }
        self.ensure_context_capacity(self.pos + 1, false)?;
        self.engram_hist.push(c.engram_token_map[token as usize] as i64);
        {
            let t = self.pos;
            let w = c.window;
            let hp = self.tokp_host.as_mut_slice::<i32>();
            let oldest = t % w + 1;
            for (j, i) in (oldest..w).chain(0..oldest).enumerate() {
                hp[j] = if i > t { -1 } else { i as i32 };
            }
            hp[w] = ((t + 1) / 1) as i32;
            hp[w + 1] = ((t + 1) / 2) as i32;
            self.g.htod(self.tokp.ptr, &hp[..w + 2])?;
        }
        let mut e = vec![0f32; d];
        m.embed.dequant(token as usize, &mut e);
        let h: Vec<f32> = (0..c.hc).flat_map(|_| e.iter().cloned()).collect();
        self.g.htod_sync(self.h.ptr, &h)?;
        // pre_mix for layer 0 = one-hot; kept in mix_f[0..4] as the "previous ffn pre"
        let mut pre = vec![0f32; 24];
        pre[0] = 1.0;
        self.g.htod_sync(self.mix_f.ptr, &pre)?;
        // GPU1's work for the token is enqueued after GPU0's layer-0 attention (below): it cannot start before layer 0's
        // router publishes, and enqueueing it first (~0.15 ms of host calls) kept GPU0 idle at the token's start
        let mut aux_pending = self.worker.is_some() && !self.skip_experts;
        // DSX_AUX_LATE=0: enqueue GPU1's token first, as before (window 31: late is +0.35% teacher-forced)
        if aux_pending && std::env::var("DSX_AUX_LATE").is_ok_and(|v| v == "0") {
            aux_pending = false;
            self.enqueue_aux(1, self.x.ptr)?;
            self.g.bind();
        }
        let aux_split = std::env::var("DSX_AUX_SPLIT").is_ok_and(|v| v == "1");
        self.hmark(t_enq, "host: step prologue");
        // Engram rows on persistent threads (DSX_ENGRAM_PERSIST=1) or a helper thread while layer 0 (no engram) is
        // enqueued and runs
        let mut prefetch = if let Some(eg) = &self.eng_gather {
            let th = std::time::Instant::now();
            eg.publish(&self.engram_hist);
            self.hmark(th, "host: engram publish");
            None
        } else {
            let ms: &'static Model = unsafe { std::mem::transmute::<&Model, &'static Model>(m) };
            let hist = self.engram_hist.clone();
            let (ptr, len) = (self.eng_host.host_ptr() as usize, self.eng_host.bytes / 4);
            Some(std::thread::spawn(move || {
                crate::cpu::pool::pin_to_set(&crate::cpu::pool::layout().aux);
                let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, len) };
                Self::prefetch_into(ms, &hist, out)
            }))
        };
        let mut tm = std::time::Instant::now();
        for li in 0..c.n_layers {
            self.ev(li, 0)?;
            if let Some(eidx) = c.engram_layers.iter().position(|&x| x == li) {
                if let Some(j) = prefetch.take() {
                    let tpe = std::time::Instant::now();
                    j.join().map_err(|_| anyhow::anyhow!("engram prefetch thread panicked"))??;
                    if let Some(p) = self.prof.as_mut() {
                        *p.entry("host: engram join wait").or_default() += tpe.elapsed().as_secs_f64();
                    }
                }
                if let Some(eg) = &self.eng_gather {
                    let tpe = std::time::Instant::now();
                    eg.wait(eidx)?;
                    if let Some(p) = self.prof.as_mut() {
                        *p.entry("host: engram join wait").or_default() += tpe.elapsed().as_secs_f64();
                        let (cold, t) = eg.stats(eidx);
                        *p.entry(if eidx == 0 { "host: engram gather, 1st layer" } else { "host: engram gather, later" }).or_default() += t;
                        if eidx + 1 == c.engram_layers.len() {
                            // a count, printed in the ms/token column as rows per token
                            *p.entry("engram rows not cached (n)").or_default() += cold as f64 * 1e-3;
                        }
                    }
                }
                self.engram(li, eidx)?;
                self.mark(&mut tm, "engram")?;
            }
            self.ev(li, 8)?;
            // attention half: mixes for this sublayer; input collapsed with the previous FFN's pre
            self.hc_pre_norm(&self.mix_f, &self.layers[li].attn_norm, true)?;
            self.hc_mixes(&self.layers[li].hc_attn, &self.mix_a)?;
            self.mark(&mut tm, "hc+norm")?;
            self.ev(li, 9)?;
            let th = std::time::Instant::now();
            if let (Some(pg), true) = (self.pregate.as_ref(), std::env::var("DSX_PREGATE").is_ok_and(|v| v == "2")) {
                // this layer's own router on its attention input (before attention): what a prefetch could know
                let l = &self.layers[li];
                let scratch = pg.ptr + (c.n_layers * 64) as u64;
                self.gemv_t(&l.gate_w, self.x.ptr, self.tmp.ptr, 1)?;
                self.g.launch(self.k["router_topk"], (1, 1, 1), 32, 0, &[Arg::P(self.tmp.ptr), Arg::P(l.gate_bias.ptr), Arg::P(pg.ptr + (li * 64) as u64), Arg::P(scratch + 32), Arg::I(c.n_experts as i32), Arg::I(8), Arg::F(c.route_scale), Arg::P(0)])?;
            }
            self.attention(li)?;
            if aux_pending {
                let ta = std::time::Instant::now();
                if aux_split {
                    self.enqueue_aux_layers(1, self.x.ptr, li..li + 1)?;
                    aux_pending = li + 1 < c.n_layers;
                } else {
                    aux_pending = false;
                    self.enqueue_aux(1, self.x.ptr)?;
                }
                self.g.bind();
                self.hmark(ta, "host: enqueue aux");
            }
            self.ev(li, 10)?;
            self.hmark(th, if li < 20 { "host: attention enc" } else { "host: attention dec" });
            self.mark(&mut tm, "attention")?;
            if !std::mem::take(&mut self.post_attn_done) {
                self.hc_join_main()?;
                self.g.launch(self.k["hc_post_mb"], ((d as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(self.h.ptr), Arg::P(self.a.ptr), Arg::P(self.mix_a.ptr), Arg::I(d as i32)])?;
            }
            // ffn half
            self.hc_pre_norm(&self.mix_a, &self.layers[li].ffn_norm, true)?;
            self.hc_mixes(&self.layers[li].hc_ffn, &self.mix_f)?;
            self.mark(&mut tm, "hc+norm")?;
            let th = std::time::Instant::now();
            if self.worker.is_some() { self.moe_fast(li)?; } else { self.moe(li)?; }
            self.hmark(th, if li < 20 { "host: moe enc" } else { "host: moe dec" });
            self.mark(&mut tm, "moe")?;
            if !std::mem::take(&mut self.post_ffn_done) {
                self.hc_join_main()?;
                self.g.launch(self.k["hc_post_mb"], ((d as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(self.h.ptr), Arg::P(self.a.ptr), Arg::P(self.mix_f.ptr), Arg::I(d as i32)])?;
            }
            if std::env::var_os("DSX_DUMP").is_some() && self.pos < 4 {
                let h: Vec<f32> = self.g.download(self.h.ptr, c.hc * d)?;
                crate::model::dump("gpu", self.pos, li, &h);
            }
        }
        self.hc_pre_norm(&self.mix_f, &self.out_norm, false)?;
        self.gemv(&self.head, self.x.ptr, self.logits.ptr)?;
        self.pos += 1;
        self.seq_base += c.n_layers as u32;
        if let Some(p) = self.prof.as_mut() {
            *p.entry("host: enqueue (no-prof only)").or_default() += t_enq.elapsed().as_secs_f64();
        }
        Ok(())
    }
}

fn adapt_early() -> bool {
    std::env::var("DSX_ADAPT_EARLY").is_ok_and(|v| v == "1")
}

/// most 32-row tiles of the small-T attention (window 128 + top-512 = 20)
const ATTN_SPLITS: usize = 32;

/// timed events per decode layer on GPU0 (DSX_EVPROF)
const EV_N: usize = 16;

/// Decode state at a prompt boundary: enough to rewind there. Compressed KV and index-key caches are append-only,
/// so their row counters suffice; the window rings and the compressor's half groups are copied.
pub struct Snapshot {
    pub pos: usize,
    rings: Buf,
    states: Buf,
    counters: Vec<(usize, usize)>,
    engram_len: usize,
}

impl GpuModel<'_> {
    fn ring_bytes(&self) -> usize {
        self.m.cfg.window * self.m.cfg.head_dim * 2
    }

    fn state_bytes(&self) -> usize {
        2 * self.m.cfg.head_dim * 4 * 2 // kv + score, ratio 2
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn cache_bytes(packed: bool, elements: usize, block: usize) -> usize {
        if packed { elements / block * (block / 2 + 1) } else { elements * 2 }
    }
    fn cache_pointer(&self, p: u64, block: usize) -> u64 {
        p | if self.packed_kv { if block == 16 { 1 } else { 3 } } else { 0 }
    }
    fn store_cache(&self, x: u64, dst: u64, first: usize, n: usize, block: usize) -> Result<()> {
        if self.packed_kv {
            self.g.launch(self.k["cache_pack_f32"], ((n / block).div_ceil(4) as u32, 1, 1), 128, 0,
                &[Arg::P(x), Arg::P(dst), Arg::I(n as i32), Arg::I(block as i32), Arg::I((block == 16) as i32), Arg::I(first as i32)])
        } else {
            self.fake_fp4(x, n, block, block == 16)?;
            self.store_bf16(x, dst + (first * 2) as u64, n)
        }
    }

    /// XS_PREFETCH coverage so far, one line per GPU: prefetched slices, how many were picked, the layers' missed picks
    pub fn prefetch_report(&self) -> Vec<String> {
        let mut out = vec![];
        let Some(tr) = self.tri.as_ref() else { return out };
        let line = |g: &Gpu, t: &TriGpu, name: &str| -> Option<String> {
            let pf = t.pf.as_ref()?;
            g.bind();
            g.sync().ok()?;
            let v: Vec<u64> = g.download(pf.stats.ptr, 4).ok()?;
            let rounds = v[3].max(1) as f64;
            Some(format!("prefetch {name} (r = {}): {:.2} slices/layer, {:.1}% of them picked, {:.1}% of the missed picks covered (missed picks/layer {:.2})",
                pf.r, v[0] as f64 / rounds, 100.0 * v[1] as f64 / v[0].max(1) as f64, 100.0 * v[1] as f64 / v[2].max(1) as f64, v[2] as f64 / rounds))
        };
        if let Some(m) = &tr.main {
            out.extend(line(&self.g, m, "main"));
        }
        for (a, t) in self.aux.iter().zip(&tr.aux) {
            out.extend(line(&a.g, t, "aux"));
        }
        self.g.bind();
        out
    }

    /// Pull-kernel CTAs per GPU, changed between tokens (ablation / tuning runs): main, then the aux GPUs.
    pub fn set_pull_ctas(&self, main: u32, aux: u32) {
        if let Some(tr) = self.tri.as_ref() {
            if let Some(m) = &tr.main {
                m.ctas.set(main.clamp(1, self.g.sms as u32));
            }
            for (a, t) in self.aux.iter().zip(&tr.aux) {
                t.ctas.set(aux.clamp(1, a.g.sms as u32));
            }
        }
    }

    /// Wait for every stream of every GPU (profiler capture boundaries).
    pub fn sync_for_profiler(&self) -> Result<()> {
        for a in &self.aux {
            a.g.quiesce()?;
        }
        self.g.quiesce()?;
        self.g.bind();
        Ok(())
    }

    /// ABLATION ONLY, the output is wrong: serve `pct` percent of the missed experts' pull reads (a fixed pseudo-random
    /// subset) from small VRAM copies of one real expert per format instead of the pinned arena over PCIe. Bounds what
    /// any prefetch or staging of the pulls can win. 0 restores the arena addresses.
    pub fn set_pull_oracle(&mut self, pct: u32, which: &str) -> Result<()> {
        let (nl, ne) = (self.m.cfg.n_layers, self.m.cfg.n_experts);
        let (Some(arena), Some(tr)) = (self.arena, self.tri.as_ref()) else { return Ok(()) };
        let tys = tr.ty.clone();
        let host = arena.buf.host_ptr();
        // `which` = "main" or "aux" limits the ablation to one GPU's pulls
        let mut targets: Vec<(&Gpu, u64)> = vec![(&self.g, if which == "aux" { 0 } else { tr.main.as_ref().map_or(0, |m| m.ptrs.ptr) })];
        for (a, t) in self.aux.iter().zip(&tr.aux) {
            targets.push((&a.g, if which == "main" { 0 } else { t.ptrs.ptr }));
        }
        for (g, dev) in targets {
            if dev == 0 {
                continue;
            }
            g.bind();
            g.sync()?;
            let pieces = arena.buf.piece_dev_ptrs(g)?;
            let mut table = vec![0u64; nl * ne * 3];
            for l in 0..nl {
                let ty = tys[l];
                let at = pct > 0 && !self.oracle.iter().any(|o| o.0 == g.ordinal && o.1 == ty);
                if at {
                    let l0 = tys.iter().position(|&t| t == ty).unwrap();
                    let slots = arena.slots.get(&(l0, 0)).ok_or_else(|| anyhow::anyhow!("oracle: no arena copy of {l0}/0"))?;
                    let mut bufs = vec![];
                    for &(off, len) in slots {
                        let b = g.alloc(len)?;
                        g.htod_sync(b.ptr, unsafe { std::slice::from_raw_parts(host.add(off), len) })?;
                        bufs.push(b);
                    }
                    let [a, b, c]: [Buf; 3] = bufs.try_into().map_err(|_| anyhow::anyhow!("oracle: three matrices"))?;
                    self.oracle.push((g.ordinal, ty, [a, b, c]));
                }
                let dummy = self.oracle.iter().find(|o| o.0 == g.ordinal && o.1 == ty).map(|o| [o.2[0].ptr, o.2[1].ptr, o.2[2].ptr]);
                for e in 0..ne {
                    let Some(s) = arena.slots.get(&(l, e)) else { continue };
                    let chosen = pct > 0 && (((l * ne + e) as u64).wrapping_mul(2654435761) >> 7) % 100 < pct as u64;
                    for m in 0..3 {
                        table[(l * ne + e) * 3 + m] = match (chosen, dummy) {
                            (true, Some(d)) => d[m],
                            _ => crate::gpu::HostBuf::dev_at(&pieces, s[m].0).ok_or_else(|| anyhow::anyhow!("oracle: no device address"))?,
                        };
                    }
                }
            }
            g.htod_sync(dev, &table)?;
        }
        self.g.bind();
        Ok(())
    }

    pub fn decode_heat(&self) -> Vec<u32> {
        self.worker.as_ref().map_or_else(Vec::new, |w| w.score.iter().map(|s| s.load(std::sync::atomic::Ordering::Relaxed)).collect())
    }
    pub fn restore_decode_heat(&self, heat: &[u32]) {
        if let Some(w) = &self.worker {
            if heat.len() == w.score.len() {
                for (s, &v) in w.score.iter().zip(heat) { s.store(v, std::sync::atomic::Ordering::Relaxed); }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn context_memory(&self) -> (usize, usize, usize) {
        (self.ctx_capacity, self.kv_mapped_bytes(), self.exp0.as_ref().map_or(0, |s| s.bytes))
    }

    fn kv_mapped_bytes(&self) -> usize {
        self.layers.iter().flat_map(|l| l.comp_cache.iter().chain(l.idx_cache.iter()))
            .map(|b| b.mapped_bytes()).sum()
    }

    /// Resize only at a fenced boundary; KV pointers stay stable. Resident expert tail pages pay for growth.
    pub fn ensure_context_capacity(&mut self, needed: usize, shrink: bool) -> Result<()> {
        anyhow::ensure!(needed <= self.max_ctx, "context full");
        if !self.dynamic_ctx { return Ok(()); }
        anyhow::ensure!(!self.dynamic_failed, "context remapping previously failed; reload the model");
        let result = self.resize_context(needed, shrink);
        if result.is_err() { self.dynamic_failed = true; }
        result
    }

    fn resize_context(&mut self, needed: usize, shrink: bool) -> Result<()> {
        let need = needed.max(self.pos).max(1);
        let mut cap = 65536.min(self.max_ctx);
        while cap < need { cap = (cap * 2).min(self.max_ctx); }
        if cap == self.ctx_capacity || (!shrink && cap < self.ctx_capacity) { return Ok(()); }
        self.adapt_flush()?;
        for a in &self.aux { a.g.quiesce()?; }
        self.g.quiesce()?;
        let target_bytes: usize = self.layers.iter().enumerate().map(|(i, l)| {
            let ratio = self.m.layers[i].ratio.max(1);
            l.comp_cache.iter().map(|_| Self::cache_bytes(self.packed_kv, cap / ratio * self.m.cfg.head_dim, 16).max(1).next_multiple_of(2 << 20)).sum::<usize>()
                + l.idx_cache.iter().map(|_| Self::cache_bytes(self.packed_kv, cap / ratio * self.m.cfg.idx_dim, 32).max(1).next_multiple_of(2 << 20)).sum::<usize>()
        }).sum();
        let expert_bytes = self.dynamic_exp_base.saturating_sub(target_bytes.saturating_sub(self.dynamic_kv_base));
        if cap > self.ctx_capacity {
            anyhow::ensure!(expert_bytes > (256 << 20), "dynamic context leaves too little expert memory");
            if let Some(s) = &mut self.exp0 {
                let gone = s.resize_dynamic(&self.g, expert_bytes)?;
                if let Some(w) = &self.worker {
                    for (l, e) in gone { w.on_cpu[l * self.m.cfg.n_experts + e].store(true, std::sync::atomic::Ordering::Relaxed); }
                }
            }
        }
        for (i, l) in self.layers.iter_mut().enumerate() {
            let ratio = self.m.layers[i].ratio.max(1);
            if let Some(b) = &mut l.comp_cache { b.resize_mapped(Self::cache_bytes(self.packed_kv, cap / ratio * self.m.cfg.head_dim, 16))?; }
            if let Some(b) = &mut l.idx_cache { b.resize_mapped(Self::cache_bytes(self.packed_kv, cap / ratio * self.m.cfg.idx_dim, 32))?; }
        }
        if cap < self.ctx_capacity {
            if let Some(s) = &mut self.exp0 { s.resize_dynamic(&self.g, expert_bytes)?; }
        }
        self.tri_sync()?;
        let previous = self.ctx_capacity;
        self.ctx_capacity = cap;
        tracing::info!(previous, capacity = cap, limit = self.max_ctx, kv_bytes = self.kv_mapped_bytes(),
            expert_bytes, "dynamic context resized");
        Ok(())
    }

    fn refill_dynamic(&mut self) -> Result<()> {
        if !self.dynamic_ctx || self.exp0.as_ref().is_none_or(|s| s.dynamic_vacancies() == 0) { return Ok(()); }
        let (Some(w), Some(arena)) = (&self.worker, self.arena) else { return Ok(()) };
        let pending: std::collections::HashSet<_> = self.exp0.iter().flat_map(|s| s.pending())
            .chain(self.aux.iter().flat_map(|a| a.exp.pending())).collect();
        let ne = self.m.cfg.n_experts;
        let mut ranked: Vec<_> = (0..w.score.len()).filter_map(|i| {
            let key = (i / ne, i % ne);
            (w.on_cpu[i].load(std::sync::atomic::Ordering::Relaxed) && !pending.contains(&key))
                .then_some((f32::from_bits(w.score[i].load(std::sync::atomic::Ordering::Relaxed)), key.0, key.1))
        }).collect();
        crate::cache_select::keep(&mut ranked, 1024, crate::cache_select::hottest, false);
        let store = self.exp0.as_mut().unwrap();
        for (_, l, e) in ranked {
            if store.swap_room() == 0 { break; }
            if let Some(raw) = arena.get(l, e) {
                let x = &self.m.layers[l].experts[e];
                store.fill_dynamic(&self.g, (l, e), raw, (x.gate.ty, x.down.ty))?;
            }
        }
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Snapshot> {
        let rb = self.ring_bytes();
        let sb = self.state_bytes();
        let rings = self.g.alloc(rb * self.layers.len())?;
        let states = self.g.alloc(sb * self.layers.len())?;
        for (i, l) in self.layers.iter().enumerate() {
            self.g.dtod(rings.ptr + (i * rb) as u64, l.window.ptr, rb)?;
            if let Some((k, s)) = &l.comp_state {
                self.g.dtod(states.ptr + (i * sb) as u64, k.ptr, sb / 2)?;
                self.g.dtod(states.ptr + (i * sb + sb / 2) as u64, s.ptr, sb / 2)?;
            }
        }
        self.g.sync()?;
        Ok(Snapshot { pos: self.pos, rings, states, counters: self.layers.iter().map(|l| (l.n_comp, l.n_idx)).collect(), engram_len: self.engram_hist.len() })
    }

    pub fn restore(&mut self, s: &Snapshot) -> Result<()> {
        let rb = self.ring_bytes();
        let sb = self.state_bytes();
        for (i, l) in self.layers.iter_mut().enumerate() {
            self.g.dtod(l.window.ptr, s.rings.ptr + (i * rb) as u64, rb)?;
            if let Some((k, st)) = &l.comp_state {
                self.g.dtod(k.ptr, s.states.ptr + (i * sb) as u64, sb / 2)?;
                self.g.dtod(st.ptr, s.states.ptr + (i * sb + sb / 2) as u64, sb / 2)?;
            }
            (l.n_comp, l.n_idx) = s.counters[i];
        }
        self.engram_hist.truncate(s.engram_len);
        self.pos = s.pos;
        self.g.sync()
    }

    /// Copy the decode state into `s` (buffers reused): verify_spec's rewind point.
    fn snapshot_into(&self, s: &mut Snapshot) -> Result<()> {
        let (rb, sb) = (self.ring_bytes(), self.state_bytes());
        for (i, l) in self.layers.iter().enumerate() {
            self.g.dtod(s.rings.ptr + (i * rb) as u64, l.window.ptr, rb)?;
            if let Some((k, st)) = &l.comp_state {
                self.g.dtod(s.states.ptr + (i * sb) as u64, k.ptr, sb / 2)?;
                self.g.dtod(s.states.ptr + (i * sb + sb / 2) as u64, st.ptr, sb / 2)?;
            }
        }
        s.counters = self.layers.iter().map(|l| (l.n_comp, l.n_idx)).collect();
        s.engram_len = self.engram_hist.len();
        s.pos = self.pos;
        Ok(())
    }

    /// Speculative verify: like `verify` (logits after each of up to 8 tokens), but rewindable: `keep(k)` afterwards
    /// keeps the first k tokens' effects and drops the rest without another forward pass.
    pub fn verify_spec(&mut self, pb: &PrefillBufs, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let c = &self.m.cfg;
        if self.vsnap.is_none() {
            let (rb, sb) = (self.ring_bytes(), self.state_bytes());
            let n = self.layers.len();
            self.vsnap = Some(Snapshot { pos: 0, rings: self.g.alloc(rb * n)?, states: self.g.alloc(sb * n)?, counters: vec![], engram_len: 0 });
            self.vrows = Some(self.g.alloc(n * 2 * crate::handoff::MAXT * c.head_dim * 4)?);
        }
        let mut snap = self.vsnap.take().unwrap();
        self.snapshot_into(&mut snap)?;
        self.vsnap = Some(snap);
        let p0 = self.pos;
        let r = self.verify(pb, tokens)?;
        self.vpass = Some((p0, tokens.len(), self.layers.iter().map(|l| (l.n_comp, l.n_idx)).collect()));
        Ok(r)
    }

    /// After verify_spec: keep the first `k` (1..=T) verified tokens. Rejected positions get their window slots back;
    /// compressed / index rows and the ratio-2 carry are recounted for the kept prefix.
    pub fn keep(&mut self, k: usize) -> Result<()> {
        let (p0, t, after) = self.vpass.take().expect("keep without verify_spec");
        anyhow::ensure!(k >= 1 && k <= t, "keep {k} of {t}");
        if k == t {
            return Ok(());
        }
        let c = &self.m.cfg;
        let (w, hd) = (c.window, c.head_dim);
        let snap = self.vsnap.as_ref().unwrap();
        let rows = self.vrows.as_ref().unwrap().ptr;
        let (rb, slot) = (self.ring_bytes(), hd * 2);
        // positions in [p0, p0 + k) that end a group of `ratio` (the compressed rows they add)
        let ends = |ratio: usize, n: usize| (p0..p0 + n).filter(|p| (p + 1) % ratio == 0).count();
        for (i, l) in self.layers.iter_mut().enumerate() {
            for p in p0 + k..p0 + t {
                let o = ((p % w) * slot) as u64;
                self.g.dtod(l.window.ptr + o, snap.rings.ptr + (i * rb) as u64 + o, slot)?;
            }
            let ratio = self.m.layers[i].ratio.max(1);
            let (b, a) = (snap.counters[i], after[i]);
            if a.0 > b.0 {
                l.n_comp = b.0 + ends(ratio, k);
            }
            if a.1 > b.1 {
                l.n_idx = b.1 + ends(ratio, k);
            }
            // ratio 2: the kept prefix ending on an even position leaves that position waiting in the carry
            if let Some((ks, ss)) = &l.comp_state {
                if ratio == 2 && a.0 > b.0 && (p0 + k - 1) % 2 == 0 {
                    let base = rows + (i * 2 * crate::handoff::MAXT * hd * 4) as u64;
                    let r = ((k - 1) * hd * 4) as u64;
                    self.g.dtod(ks.ptr, base + r, hd * 4)?;
                    self.g.dtod(ss.ptr, base + (crate::handoff::MAXT * hd * 4) as u64 + r, hd * 4)?;
                }
            }
        }
        self.engram_hist.truncate(snap.engram_len + k);
        self.pos = p0 + k;
        Ok(())
    }

    /// Forget the sequence (caches are overwritten as positions are refilled; stale window slots are masked).
    pub fn reset(&mut self) {
        for l in self.layers.iter_mut() {
            l.n_comp = 0;
            l.n_idx = 0;
        }
        self.engram_hist.clear();
        self.pos = 0;
        self.topk_k = 0;
    }

    /// A/B transport with identical placement and weights, without loading a second 135 GB arena.
    #[cfg(test)]
    pub(crate) fn test_peer_transport(&mut self, enabled: bool) -> Result<u32> {
        self.g.sync()?;
        self.aux_peer_mask = 0;
        for (i, a) in self.aux.iter_mut().enumerate() {
            a.g.sync()?;
            a.peer = enabled && !a.peer_done.is_empty();
            if a.peer { self.aux_peer_mask |= 1 << i; }
        }
        self.g.bind();
        Ok(self.aux_peer_mask)
    }

    /// Logits after every token of `tokens` (one pass, decoder on every row): quality evaluation.
    pub fn eval_logits(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let pb = self.pb.take().expect("prefill buffers");
        anyhow::ensure!(tokens.len() <= pb.tmax, "eval sequence longer than the prefill chunk");
        let r = self.prefill_pass(&pb, tokens, true, true);
        self.pb = Some(pb);
        Ok(r?.unwrap())
    }

    /// Prefill `tokens` (continuing at self.pos) in chunks of at most `pb.tmax`; returns the last token's logits.
    /// Earlier chunks skip the decoder half, so the final chunk must cover the decoder cone (19 windows) unless the
    /// whole call fits in one chunk.
    pub fn prefill(&mut self, pb: &PrefillBufs, tokens: &[u32]) -> Result<Vec<f32>> {
        self.ensure_context_capacity(self.pos + tokens.len(), false)?;
        let c = &self.m.cfg;
        let cone = (c.n_layers - 1 - c.cand_layer) * c.window;
        let total = tokens.len();
        let end = |s0: usize| {
            let e = (s0 + pb.tmax).min(total);
            if e < total && total - e < cone { total.saturating_sub(cone).max(s0 + 1) } else { e }
        };
        let mut s0 = 0;
        let mut logits = None;
        let r = (|| {
            while s0 < total {
                let e = end(s0);
                self.engram_next = tokens[e..end(e)].to_vec();
                logits = self.prefill_chunk(pb, &tokens[s0..e], e == total)?;
                s0 = e;
            }
            anyhow::Ok(())
        })();
        self.engram_next.clear();
        self.join_engram_prefetch();
        r?;
        logits.ok_or_else(|| anyhow::anyhow!("empty prompt"))
    }
}
