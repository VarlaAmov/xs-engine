//! T16 routed experts on the CPU (core/src/t16.rs, docs/t16.md "v6"). A matrix is `trellis | suh | svh` in the stored
//! layout: tiles by column (tile (kb, nb) at nb * k/16 + kb), each tile's bytes in bit-stream order.
//!
//! y = had128(had128(x * suh) @ W_inner) * svh: `prep` makes the int8 activations (x * suh, Hadamard, per 256
//! inputs a scale and the sum of codes), `gemv` gives the raw dot products of tile columns nb0..nb1 (each column's
//! 64-bit view from byte-aligned loads and vpshufb, 16 windows by shifts, vpmullw by 0x3A2F, vpdpbusd of the product
//! bytes [lo hi] with [x x 0 0]: x * byte sum b, then 0.009567260742 b - 2.4375 per group), `post` the output
//! Hadamard and svh. AVX-VNNI / AVX-512 VNNI; other CPUs take the scalar decode.

use super::qdot::{Isa, isa};
use crate::numerics::fwht_blocks;
use crate::t16;
use half::f16;

pub const C_B: f32 = 0.009567260742;
pub const C_0: f32 = 2.4375;

/// EXL3's stock "mul1" codebook on the same bit streams, for speed comparisons only (DSX_T16_CODEBOOK=mul1, read
/// once): x = state * 0x83DCD12D (u32), b4 = sum of x's 4 bytes, value fp16(fma(fp16(1024 + b4), 0x1eee, 0xc931)) =
/// 0.006767272949 b4 - 3.453125 (taken as exactly affine here, like T16's)
pub const M1_MUL: u32 = 0x83DC_D12D;
pub const M1_CB: f32 = 0.006767272949;
pub const M1_C0: f32 = 3.453125;

/// DSX_T16_MULTI=0 (read once): gemv_multi runs gemv per token, the per-token baseline for benchmarks
pub fn multi() -> bool {
    static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("DSX_T16_MULTI").map_or(true, |v| v != "0"))
}

pub fn mul1() -> bool {
    static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("DSX_T16_CODEBOOK").is_ok_and(|v| v == "mul1"))
}

/// int8 activations of one matrix input as pair words: x4[i / 2] = bytes [q_i, q_i, q_i+1, q_i+1] for even i, so one
/// vpdpbusd of two windows' product bytes [lo_i hi_i lo_i+1 hi_i+1] gives q_i b_i + q_i+1 b_i+1; d / s per 256 inputs
#[derive(Default, Clone)]
pub struct Act {
    pub x4: Vec<i32>,
    /// mul1 only: x1[i] = bytes [q_i q_i q_i q_i]
    pub x1: Vec<i32>,
    pub d: Vec<f32>,
    pub s: Vec<i32>,
}

/// suh / svh of a stored matrix as f32
pub fn scales(m: &[u8], k2: usize, k: usize, n: usize) -> (&[u8], &[u8]) {
    let tb = t16::trellis_bytes(k2, k, n);
    (&m[tb..tb + 2 * k], &m[tb + 2 * k..tb + 2 * k + 2 * n])
}

#[inline]
fn h(b: &[u8], i: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
}

/// one 256-input group of `prep`: x * suh, Hadamard per 128, int8 (round half to even, d = amax / 127)
pub fn prep_group(x: &[f32], suh: &[u8], gr: usize, a: &mut Act) {
    let mut v = [0f32; 256];
    for (i, vi) in v.iter_mut().enumerate() {
        *vi = x[256 * gr + i] * h(suh, 256 * gr + i);
    }
    fwht_blocks(&mut v);
    let m = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    let id = if m > 0.0 { 127.0 / m } else { 0.0 };
    let mut sum = 0i32;
    for i in (0..256).step_by(2) {
        let (q0, q1) = ((v[i] * id).round_ties_even() as i32, (v[i + 1] * id).round_ties_even() as i32);
        sum += q0 + q1;
        let (b0, b1) = ((q0 & 255) as u32, (q1 & 255) as u32);
        a.x4[128 * gr + i / 2] = (b0 | b0 << 8 | b1 << 16 | b1 << 24) as i32;
        if mul1() {
            // sized by the caller (groups of one Act are prepped by several threads at once)
            a.x1[256 * gr + i] = (b0 * 0x0101_0101) as i32;
            a.x1[256 * gr + i + 1] = (b1 * 0x0101_0101) as i32;
        }
    }
    a.d[gr] = m / 127.0;
    a.s[gr] = sum;
}

pub fn prep(x: &[f32], suh: &[u8], k: usize, a: &mut Act) {
    a.x4.resize(k / 2, 0);
    if mul1() {
        a.x1.resize(k, 0);
    }
    a.d.resize(k / 256, 0.0);
    a.s.resize(k / 256, 0);
    for gr in 0..k / 256 {
        prep_group(x, suh, gr, a);
    }
}

/// y (n raw outputs, the slice of `nb0 * 16 ..`) -> had128(y) * svh, in place, for one 128-block at `o`
pub fn post_block(y: &mut [f32], svh: &[u8], o: usize) {
    fwht_blocks(&mut y[..128]);
    for (i, v) in y[..128].iter_mut().enumerate() {
        *v *= h(svh, o + i);
    }
}

fn hend(k2: usize, t: usize) -> usize {
    if k2 % 2 == 1 { (t + 1) * (k2 / 2) + (t + 1) / 2 } else { (t + 1) * (k2 / 2) }
}

/// raw outputs of tile columns nb0..nb1 into out[(nb - nb0) * 16 ..]: per 256-group d (C_B acc - C_0 s)
pub fn gemv(m: &[u8], k2: usize, k: usize, n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32]) {
    gemv_groups(m, k2, k, n, a, nb0, nb1, out, 0, k / 256)
}

/// gemv over the input groups g0..g1 only, adding to out's partial sums when g0 > 0: gemv_groups(0..g) then
/// gemv_groups(g..k/256) equals gemv bit for bit (the f32 sums go through memory unchanged); not with mul1
#[allow(clippy::too_many_arguments)]
pub fn gemv_groups(m: &[u8], k2: usize, k: usize, n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32], g0: usize, g1: usize) {
    assert!(g0 < g1 && g1 <= k / 256);
    if mul1() {
        assert!(g0 == 0 && g1 == k / 256, "mul1: whole matrices only");
        return match isa() {
            #[cfg(target_arch = "x86_64")]
            Isa::Vnni => unsafe { vnni::gemv_m1(m, k2, k, n, a, nb0, nb1, out) },
            #[cfg(target_arch = "x86_64")]
            Isa::Vnni512 => unsafe { vnni512::gemv_m1(m, k2, k, n, a, nb0, nb1, out) },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => unsafe { avx2::gemv_m1(m, k2, k, n, a, nb0, nb1, out) },
            _ => gemv_scalar_m1(m, k2, k, n, a, nb0, nb1, out),
        };
    }
    match isa() {
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni => unsafe { vnni::gemv(m, k2, k, n, a, nb0, nb1, out, g0, g1) },
        #[cfg(target_arch = "x86_64")]
        Isa::Vnni512 => unsafe { vnni512::gemv(m, k2, k, n, a, nb0, nb1, out, g0, g1) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { avx2::gemv(m, k2, k, n, a, nb0, nb1, out, g0, g1) },
        _ => gemv_scalar_g(m, k2, k, n, a, nb0, nb1, out, g0, g1),
    }
}

/// gemv for several tokens on the same columns: outs[i] = gemv(acts[i]) bit for bit, each tile decoded once per up
/// to 8 tokens (SIMD; the scalar and mul1 paths, and DSX_T16_MULTI=0, run gemv per token)
#[allow(clippy::too_many_arguments)]
pub fn gemv_multi(m: &[u8], k2: usize, k: usize, n: usize, acts: &[&Act], nb0: usize, nb1: usize, outs: &mut [&mut [f32]]) {
    gemv_multi_groups(m, k2, k, n, acts, nb0, nb1, outs, 0, k / 256)
}

/// gemv_multi over the input groups g0..g1 (see gemv_groups)
#[allow(clippy::too_many_arguments)]
pub fn gemv_multi_groups(m: &[u8], k2: usize, k: usize, n: usize, acts: &[&Act], nb0: usize, nb1: usize, outs: &mut [&mut [f32]], g0: usize, g1: usize) {
    assert!(g0 < g1 && g1 <= k / 256);
    if !mul1() && multi() {
        match isa() {
            #[cfg(target_arch = "x86_64")]
            Isa::Vnni => return unsafe { vnni::gemv_multi(m, k2, k, n, acts, nb0, nb1, outs, g0, g1) },
            #[cfg(target_arch = "x86_64")]
            Isa::Vnni512 => return unsafe { vnni512::gemv_multi(m, k2, k, n, acts, nb0, nb1, outs, g0, g1) },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => return unsafe { avx2::gemv_multi(m, k2, k, n, acts, nb0, nb1, outs, g0, g1) },
            _ => {}
        }
    }
    for (a, o) in acts.iter().zip(outs.iter_mut()) {
        gemv_groups(m, k2, k, n, a, nb0, nb1, o, g0, g1);
    }
}

/// the same with the scalar decode (reference and fallback)
pub fn gemv_scalar(m: &[u8], k2: usize, k: usize, n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32]) {
    gemv_scalar_g(m, k2, k, n, a, nb0, nb1, out, 0, k / 256)
}

#[allow(clippy::too_many_arguments)]
fn gemv_scalar_g(m: &[u8], k2: usize, k: usize, _n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32], g0: usize, g1: usize) {
    let (kt, tb) = (k / 16, t16::tile_bytes(k2));
    for nb in nb0..nb1 {
        for c in 0..16 {
            let mut f = if g0 == 0 { 0f32 } else { out[(nb - nb0) * 16 + c] };
            for gr in g0..g1 {
                let mut acc = 0i64;
                for kb in 16 * gr..16 * gr + 16 {
                    let tile = &m[(nb * kt + kb) * tb..][..tb];
                    for r in 0..16 {
                        let p = t16::state(tile, k2, 16 * c + r).wrapping_mul(t16::MUL) & 0xFFFF;
                        let b = (p & 0xFF) + (p >> 8);
                        let q = (a.x4[8 * kb + r / 2] >> (16 * (r % 2))) as i8;
                        acc += b as i64 * q as i64;
                    }
                }
                f += a.d[gr] * (C_B * acc as f32 - C_0 * a.s[gr] as f32);
            }
            out[(nb - nb0) * 16 + c] = f;
        }
    }
    let _ = hend;
}

/// gemv_scalar with the mul1 codebook (reference for gemv_m1)
pub fn gemv_scalar_m1(m: &[u8], k2: usize, k: usize, _n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32]) {
    let (kt, tb) = (k / 16, t16::tile_bytes(k2));
    for nb in nb0..nb1 {
        for c in 0..16 {
            let mut f = 0f32;
            for gr in 0..k / 256 {
                let mut acc = 0i64;
                for kb in 16 * gr..16 * gr + 16 {
                    let tile = &m[(nb * kt + kb) * tb..][..tb];
                    for r in 0..16 {
                        let x = (t16::state(tile, k2, 16 * c + r) as u32).wrapping_mul(M1_MUL);
                        let b4 = (x & 0xFF) + (x >> 8 & 0xFF) + (x >> 16 & 0xFF) + (x >> 24);
                        let q = (a.x4[8 * kb + r / 2] >> (16 * (r % 2))) as i8;
                        acc += b4 as i64 * q as i64;
                    }
                }
                // fused like the SIMD epilogue, so the two agree bit for bit
                f = a.d[gr].mul_add(M1_CB.mul_add(acc as f32, -(M1_C0 * a.s[gr] as f32)), f);
            }
            out[(nb - nb0) * 16 + c] = f;
        }
    }
}

/// per K2: for each of the 4 views (lo cols 0-7, lo 8-15, hi 0-7, hi 8-15) and 128-bit half (4 columns): one or two
/// 16-byte loads (stream byte offsets, may be negative = wrap to the tile end) and their vpshufb masks
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Plan {
    src: [[[i32; 2]; 2]; 4],
    mask: [[[[i8; 16]; 2]; 2]; 4],
    nsrc: [[u8; 2]; 4],
    tb: i32,
}

#[cfg(target_arch = "x86_64")]
fn plan(k2: usize) -> Plan {
    let mut p = Plan { src: [[[0; 2]; 2]; 4], mask: [[[[-1; 16]; 2]; 2]; 4], nsrc: [[0; 2]; 4], tb: (16 * k2) as i32 };
    for v in 0..4 {
        for hh in 0..2 {
            let mut off = [0i32; 4];
            for (i, o) in off.iter_mut().enumerate() {
                let c = 8 * (v & 1) + 4 * hh + i;
                let e = (hend(k2, 16 * c + 15) / 8) as i32;
                *o = if v < 2 { e - 4 } else { e - 8 };
            }
            let one = off[3] + 4 - off[0] <= 16;
            p.nsrc[v][hh] = if one { 1 } else { 2 };
            for part in 0..p.nsrc[v][hh] as usize {
                let base = if one { off[0] } else { off[2 * part] };
                p.src[v][hh][part] = base;
                for i in 0..4 {
                    let mine = one || i / 2 == part;
                    for b in 0..4 {
                        p.mask[v][hh][part][4 * i + b] = if mine { (off[i] - base + 3 - b as i32) as i8 } else { -1 };
                    }
                }
            }
        }
    }
    p
}

#[cfg(target_arch = "x86_64")]
fn plans() -> &'static [Plan; 3] {
    static P: std::sync::OnceLock<[Plan; 3]> = std::sync::OnceLock::new();
    P.get_or_init(|| [plan(3), plan(4), plan(5)])
}

// window macros of the SIMD kernels (outside t16_kernels so their $-variables are their own); `dp` and `view` resolve
// at the expansion site
// Two windows per 32-bit lane: window j (the view H:L shifted right by o_j) in the low half, window j + 1 in the high
// half (the view shifted by o_j+1 - 16; a negative shift is L shifted left), joined by vpblendw; one vpmullw and one
// vpdpbusd for both. `lo o`: (L >> o) | (H << 32 - o); `hi o`: H >> o; `zero`: L; `left o`: L << o.
macro_rules! sh {
    ($l:ident, $h:ident, lo $o:literal) => {
        _mm256_or_si256(_mm256_srli_epi32::<$o>($l), _mm256_slli_epi32::<{ 32 - $o }>($h))
    };
    ($l:ident, $h:ident, hi $o:literal) => {
        _mm256_srli_epi32::<$o>($h)
    };
    ($l:ident, $h:ident, zero) => {
        $l
    };
    ($l:ident, $h:ident, left $o:literal) => {
        _mm256_slli_epi32::<$o>($l)
    };
}

macro_rules! tile {
    ($t:ident, $p:ident, $msk:ident, $x:ident, $kb:ident, $a0:ident, $a1:ident, $mul:ident; $($q:literal: ($ka:ident $($oa:literal)?, $kb2:ident $($ob:literal)?)),*) => {{
        let l0 = view($t, $p, &$msk, 0);
        let l1 = view($t, $p, &$msk, 1);
        let h0 = view($t, $p, &$msk, 2);
        let h1 = view($t, $p, &$msk, 3);
        $(
            let xq = _mm256_set1_epi32(*$x.get_unchecked(8 * $kb + $q));
            let w0 = _mm256_blend_epi16::<0b1010_1010>(sh!(l0, h0, $ka $($oa)?), sh!(l0, h0, $kb2 $($ob)?));
            let w1 = _mm256_blend_epi16::<0b1010_1010>(sh!(l1, h1, $ka $($oa)?), sh!(l1, h1, $kb2 $($ob)?));
            $a0 = dp($a0, _mm256_mullo_epi16(w0, $mul), xq);
            $a1 = dp($a1, _mm256_mullo_epi16(w1, $mul), xq);
        )*
    }};
}

// tile! for N tokens (gemv_n), one half of the tile (columns 0-7: views 0 and 2, 8-15: 1 and 3): each pair's product
// is decoded once and dotted with every token's pair word; for N > 4 gemv_n runs a group's 16 tiles per half, so the
// N accumulators, two views and the shift temporaries fit the 16 registers up to N = 8
macro_rules! tile_n {
    ($t:ident, $p:ident, $msk:ident, $x:ident, $kb:ident, $acc:ident, $vl:literal, $vh:literal, $mul:ident; $($q:literal: ($ka:ident $($oa:literal)?, $kb2:ident $($ob:literal)?)),*) => {{
        let l = view($t, $p, &$msk, $vl);
        let h = view($t, $p, &$msk, $vh);
        $(
            let pr = _mm256_mullo_epi16(_mm256_blend_epi16::<0b1010_1010>(sh!(l, h, $ka $($oa)?), sh!(l, h, $kb2 $($ob)?)), $mul);
            for i in 0..N {
                $acc[i] = dp($acc[i], pr, _mm256_set1_epi32(*$x[i].add(8 * $kb + $q)));
            }
        )*
    }};
}

// tile_n for N <= 4: both halves per tile, 2N accumulator chains (vpdpbusd's latency, 5 clocks, would bound fewer)
macro_rules! tile_n2 {
    ($t:ident, $p:ident, $msk:ident, $x:ident, $kb:ident, $a0:ident, $a1:ident, $mul:ident; $($q:literal: ($ka:ident $($oa:literal)?, $kb2:ident $($ob:literal)?)),*) => {{
        let l0 = view($t, $p, &$msk, 0);
        let l1 = view($t, $p, &$msk, 1);
        let h0 = view($t, $p, &$msk, 2);
        let h1 = view($t, $p, &$msk, 3);
        $(
            let p0 = _mm256_mullo_epi16(_mm256_blend_epi16::<0b1010_1010>(sh!(l0, h0, $ka $($oa)?), sh!(l0, h0, $kb2 $($ob)?)), $mul);
            let p1 = _mm256_mullo_epi16(_mm256_blend_epi16::<0b1010_1010>(sh!(l1, h1, $ka $($oa)?), sh!(l1, h1, $kb2 $($ob)?)), $mul);
            for i in 0..N {
                let xq = _mm256_set1_epi32(*$x[i].add(8 * $kb + $q));
                $a0[i] = dp($a0[i], p0, xq);
                $a1[i] = dp($a1[i], p1, xq);
            }
        )*
    }};
}

// mul1: one window per 32-bit lane (window j = the view shifted right by o_j = end(15) - end(j), masked to 16 bits),
// vpmulld by 0x83DCD12D, vpdpbusd of the product's 4 bytes with [q_j q_j q_j q_j]: 8 weights per multiply and dot
macro_rules! tile_m1 {
    ($t:ident, $p:ident, $msk:ident, $x1:ident, $kb:ident, $a0:ident, $a1:ident, $mul:ident, $m16:ident; $($j:literal: ($k:ident $($o:literal)?)),*) => {{
        let l0 = view($t, $p, &$msk, 0);
        let l1 = view($t, $p, &$msk, 1);
        let h0 = view($t, $p, &$msk, 2);
        let h1 = view($t, $p, &$msk, 3);
        $(
            let xq = _mm256_set1_epi32(*$x1.get_unchecked(16 * $kb + $j));
            let w0 = _mm256_and_si256(sh!(l0, h0, $k $($o)?), $m16);
            let w1 = _mm256_and_si256(sh!(l1, h1, $k $($o)?), $m16);
            $a0 = dp($a0, _mm256_mullo_epi32(w0, $mul), xq);
            $a1 = dp($a1, _mm256_mullo_epi32(w1, $mul), xq);
        )*
    }};
}

macro_rules! t16_kernels {
    ($m:ident, $feat:literal, |$s:ident, $a:ident, $b:ident| $dp:expr) => {
        #[cfg(target_arch = "x86_64")]
        mod $m {
            use super::*;
            use std::arch::x86_64::*;

            #[inline]
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn dp($s: __m256i, $a: __m256i, $b: __m256i) -> __m256i {
                $dp
            }

            /// 16 bytes of the tile's stream from byte s (negative: wraps to the tile end)
            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn load16w(t: *const u8, s: i32, tb: i32) -> __m128i {
                unsafe {
                    if s >= 0 {
                        return _mm_loadu_si128(t.add(s as usize) as *const __m128i);
                    }
                    let head = _mm_loadu_si128(t as *const __m128i);
                    let tail = _mm_loadu_si128(t.add((tb - 16) as usize) as *const __m128i);
                    match -s {
                        1 => _mm_alignr_epi8::<15>(head, tail),
                        2 => _mm_alignr_epi8::<14>(head, tail),
                        3 => _mm_alignr_epi8::<13>(head, tail),
                        4 => _mm_alignr_epi8::<12>(head, tail),
                        5 => _mm_alignr_epi8::<11>(head, tail),
                        6 => _mm_alignr_epi8::<10>(head, tail),
                        7 => _mm_alignr_epi8::<9>(head, tail),
                        8 => _mm_alignr_epi8::<8>(head, tail),
                        9 => _mm_alignr_epi8::<7>(head, tail),
                        10 => _mm_alignr_epi8::<6>(head, tail),
                        11 => _mm_alignr_epi8::<5>(head, tail),
                        12 => _mm_alignr_epi8::<4>(head, tail),
                        _ => head,
                    }
                }
            }

            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn view(t: *const u8, p: &Plan, msk: &[[[__m128i; 2]; 2]; 4], v: usize) -> __m256i {
                unsafe {
                    let mut r = [_mm_setzero_si128(); 2];
                    for hh in 0..2 {
                        r[hh] = _mm_shuffle_epi8(load16w(t, p.src[v][hh][0], p.tb), msk[v][hh][0]);
                        if p.nsrc[v][hh] == 2 {
                            r[hh] = _mm_or_si128(r[hh], _mm_shuffle_epi8(load16w(t, p.src[v][hh][1], p.tb), msk[v][hh][1]));
                        }
                    }
                    _mm256_set_m128i(r[1], r[0])
                }
            }

            #[allow(clippy::too_many_arguments)]
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn gemv(m: &[u8], k2: usize, k: usize, n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32], g0: usize, g1: usize) {
                unsafe {
                    assert!(m.len() >= t16::trellis_bytes(k2, k, n) + 16 && nb1 <= n / 16 && a.x4.len() >= k / 2 && out.len() >= 16 * (nb1 - nb0));
                    let p = &plans()[k2 - 3];
                    let mut msk = [[[_mm_setzero_si128(); 2]; 2]; 4];
                    for v in 0..4 {
                        for hh in 0..2 {
                            for q in 0..2 {
                                msk[v][hh][q] = _mm_loadu_si128(p.mask[v][hh][q].as_ptr() as *const __m128i);
                            }
                        }
                    }
                    let (kt, tb) = (k / 16, 16 * k2);
                    let mul = _mm256_set1_epi16(t16::MUL as i16);
                    let x = &a.x4[..];
                    for nb in nb0..nb1 {
                        let o = out.as_mut_ptr().add((nb - nb0) * 16);
                        let (mut f0, mut f1) = if g0 == 0 { (_mm256_setzero_ps(), _mm256_setzero_ps()) } else { (_mm256_loadu_ps(o), _mm256_loadu_ps(o.add(8))) };
                        let col = m.as_ptr().add(nb * kt * tb);
                        for gr in g0..g1 {
                            let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                            for kb in 16 * gr..16 * gr + 16 {
                                let t = col.add(kb * tb);
                                match k2 {
                                    // o_j = end(15) - end(j); pair q = windows 2q (shift o) and 2q + 1 (shift o' - 16)
                                    3 => tile!(t, p, msk, x, kb, a0, a1, mul; 0: (lo 23, lo 5), 1: (lo 20, lo 2), 2: (lo 17, left 1), 3: (lo 14, left 4), 4: (lo 11, left 7), 5: (lo 8, left 10), 6: (lo 5, left 13), 7: (lo 2, left 16)),
                                    4 => tile!(t, p, msk, x, kb, a0, a1, mul; 0: (lo 30, lo 12), 1: (lo 26, lo 8), 2: (lo 22, lo 4), 3: (lo 18, zero), 4: (lo 14, left 4), 5: (lo 10, left 8), 6: (lo 6, left 12), 7: (lo 2, left 16)),
                                    _ => tile!(t, p, msk, x, kb, a0, a1, mul; 0: (hi 6, lo 19), 1: (hi 1, lo 14), 2: (lo 28, lo 9), 3: (lo 23, lo 4), 4: (lo 18, left 1), 5: (lo 13, left 6), 6: (lo 8, left 11), 7: (lo 3, left 16)),
                                }
                            }
                            let d = _mm256_set1_ps(a.d[gr]);
                            let c0 = _mm256_set1_ps(C_0 * a.s[gr] as f32);
                            let cb = _mm256_set1_ps(C_B);
                            f0 = _mm256_fmadd_ps(d, _mm256_fmsub_ps(cb, _mm256_cvtepi32_ps(a0), c0), f0);
                            f1 = _mm256_fmadd_ps(d, _mm256_fmsub_ps(cb, _mm256_cvtepi32_ps(a1), c0), f1);
                        }
                        _mm256_storeu_ps(o, f0);
                        _mm256_storeu_ps(o.add(8), f1);
                    }
                }
            }

            /// gemv for N tokens at once: the same integer accumulators and epilogue per token, the decode shared
            #[inline]
            #[target_feature(enable = $feat)]
            #[allow(clippy::too_many_arguments)]
            unsafe fn gemv_n<const N: usize, const BOTH: bool>(m: &[u8], k2: usize, k: usize, acts: &[&Act], nb0: usize, nb1: usize, outs: &mut [&mut [f32]], g0: usize, g1: usize) {
                unsafe {
                    let p = &plans()[k2 - 3];
                    let mut msk = [[[_mm_setzero_si128(); 2]; 2]; 4];
                    for v in 0..4 {
                        for hh in 0..2 {
                            for q in 0..2 {
                                msk[v][hh][q] = _mm_loadu_si128(p.mask[v][hh][q].as_ptr() as *const __m128i);
                            }
                        }
                    }
                    let (kt, tb) = (k / 16, 16 * k2);
                    let mul = _mm256_set1_epi16(t16::MUL as i16);
                    let x: [*const i32; N] = std::array::from_fn(|i| acts[i].x4.as_ptr());
                    for nb in nb0..nb1 {
                        let (mut f0, mut f1) = ([_mm256_setzero_ps(); N], [_mm256_setzero_ps(); N]);
                        if g0 > 0 {
                            for i in 0..N {
                                let o = outs[i].as_ptr().add((nb - nb0) * 16);
                                (f0[i], f1[i]) = (_mm256_loadu_ps(o), _mm256_loadu_ps(o.add(8)));
                            }
                        }
                        let col = m.as_ptr().add(nb * kt * tb);
                        for gr in g0..g1 {
                            let (mut a0, mut a1) = ([_mm256_setzero_si256(); N], [_mm256_setzero_si256(); N]);
                            if BOTH {
                                for kb in 16 * gr..16 * gr + 16 {
                                    let t = col.add(kb * tb);
                                    match k2 {
                                        3 => tile_n2!(t, p, msk, x, kb, a0, a1, mul; 0: (lo 23, lo 5), 1: (lo 20, lo 2), 2: (lo 17, left 1), 3: (lo 14, left 4), 4: (lo 11, left 7), 5: (lo 8, left 10), 6: (lo 5, left 13), 7: (lo 2, left 16)),
                                        4 => tile_n2!(t, p, msk, x, kb, a0, a1, mul; 0: (lo 30, lo 12), 1: (lo 26, lo 8), 2: (lo 22, lo 4), 3: (lo 18, zero), 4: (lo 14, left 4), 5: (lo 10, left 8), 6: (lo 6, left 12), 7: (lo 2, left 16)),
                                        _ => tile_n2!(t, p, msk, x, kb, a0, a1, mul; 0: (hi 6, lo 19), 1: (hi 1, lo 14), 2: (lo 28, lo 9), 3: (lo 23, lo 4), 4: (lo 18, left 1), 5: (lo 13, left 6), 6: (lo 8, left 11), 7: (lo 3, left 16)),
                                    }
                                }
                            } else {
                                // columns 0-7 of the group's 16 tiles, then 8-15: N accumulators live, not 2N
                                for kb in 16 * gr..16 * gr + 16 {
                                    let t = col.add(kb * tb);
                                    match k2 {
                                        3 => tile_n!(t, p, msk, x, kb, a0, 0, 2, mul; 0: (lo 23, lo 5), 1: (lo 20, lo 2), 2: (lo 17, left 1), 3: (lo 14, left 4), 4: (lo 11, left 7), 5: (lo 8, left 10), 6: (lo 5, left 13), 7: (lo 2, left 16)),
                                        4 => tile_n!(t, p, msk, x, kb, a0, 0, 2, mul; 0: (lo 30, lo 12), 1: (lo 26, lo 8), 2: (lo 22, lo 4), 3: (lo 18, zero), 4: (lo 14, left 4), 5: (lo 10, left 8), 6: (lo 6, left 12), 7: (lo 2, left 16)),
                                        _ => tile_n!(t, p, msk, x, kb, a0, 0, 2, mul; 0: (hi 6, lo 19), 1: (hi 1, lo 14), 2: (lo 28, lo 9), 3: (lo 23, lo 4), 4: (lo 18, left 1), 5: (lo 13, left 6), 6: (lo 8, left 11), 7: (lo 3, left 16)),
                                    }
                                }
                                for kb in 16 * gr..16 * gr + 16 {
                                    let t = col.add(kb * tb);
                                    match k2 {
                                        3 => tile_n!(t, p, msk, x, kb, a1, 1, 3, mul; 0: (lo 23, lo 5), 1: (lo 20, lo 2), 2: (lo 17, left 1), 3: (lo 14, left 4), 4: (lo 11, left 7), 5: (lo 8, left 10), 6: (lo 5, left 13), 7: (lo 2, left 16)),
                                        4 => tile_n!(t, p, msk, x, kb, a1, 1, 3, mul; 0: (lo 30, lo 12), 1: (lo 26, lo 8), 2: (lo 22, lo 4), 3: (lo 18, zero), 4: (lo 14, left 4), 5: (lo 10, left 8), 6: (lo 6, left 12), 7: (lo 2, left 16)),
                                        _ => tile_n!(t, p, msk, x, kb, a1, 1, 3, mul; 0: (hi 6, lo 19), 1: (hi 1, lo 14), 2: (lo 28, lo 9), 3: (lo 23, lo 4), 4: (lo 18, left 1), 5: (lo 13, left 6), 6: (lo 8, left 11), 7: (lo 3, left 16)),
                                    }
                                }
                            }
                            let cb = _mm256_set1_ps(C_B);
                            for i in 0..N {
                                let d = _mm256_set1_ps(*acts[i].d.get_unchecked(gr));
                                let c0 = _mm256_set1_ps(C_0 * *acts[i].s.get_unchecked(gr) as f32);
                                f0[i] = _mm256_fmadd_ps(d, _mm256_fmsub_ps(cb, _mm256_cvtepi32_ps(a0[i]), c0), f0[i]);
                                f1[i] = _mm256_fmadd_ps(d, _mm256_fmsub_ps(cb, _mm256_cvtepi32_ps(a1[i]), c0), f1[i]);
                            }
                        }
                        for i in 0..N {
                            let o = outs[i].as_mut_ptr().add((nb - nb0) * 16);
                            _mm256_storeu_ps(o, f0[i]);
                            _mm256_storeu_ps(o.add(8), f1[i]);
                        }
                    }
                }
            }

            /// gemv of several tokens' activations against the same columns: chunks of up to 8 tokens per decode. Chunks
            /// of up to 4 tokens at K 1.5 and 2, 2 at K 2.5 (longer shifts, more temporaries) take both halves of a tile
            /// at once (2N accumulator chains), larger ones a group's tiles per half (N live accumulators, no spills):
            /// the faster of the two per (K, N) on Raptor Cove
            #[target_feature(enable = $feat)]
            #[allow(clippy::too_many_arguments)]
            pub(super) unsafe fn gemv_multi(m: &[u8], k2: usize, k: usize, n: usize, acts: &[&Act], nb0: usize, nb1: usize, outs: &mut [&mut [f32]], g0: usize, g1: usize) {
                unsafe {
                    assert!(m.len() >= t16::trellis_bytes(k2, k, n) + 16 && nb1 <= n / 16 && acts.len() == outs.len());
                    for (a, o) in acts.iter().zip(outs.iter()) {
                        assert!(a.x4.len() >= k / 2 && a.d.len() >= k / 256 && a.s.len() >= k / 256 && o.len() >= 16 * (nb1 - nb0));
                    }
                    let mut i = 0;
                    while i < acts.len() {
                        let (a, o) = (&acts[i..], &mut outs[i..]);
                        let both = |c: usize| c <= [4, 4, 2][k2 - 3];
                        i += match a.len() {
                            1 => { gemv(m, k2, k, n, a[0], nb0, nb1, o[0], g0, g1); 1 }
                            2 => { gemv_n::<2, true>(m, k2, k, a, nb0, nb1, o, g0, g1); 2 }
                            3 if both(3) => { gemv_n::<3, true>(m, k2, k, a, nb0, nb1, o, g0, g1); 3 }
                            3 => { gemv_n::<3, false>(m, k2, k, a, nb0, nb1, o, g0, g1); 3 }
                            4 if both(4) => { gemv_n::<4, true>(m, k2, k, a, nb0, nb1, o, g0, g1); 4 }
                            4 => { gemv_n::<4, false>(m, k2, k, a, nb0, nb1, o, g0, g1); 4 }
                            5 => { gemv_n::<5, false>(m, k2, k, a, nb0, nb1, o, g0, g1); 5 }
                            6 => { gemv_n::<6, false>(m, k2, k, a, nb0, nb1, o, g0, g1); 6 }
                            7 => { gemv_n::<7, false>(m, k2, k, a, nb0, nb1, o, g0, g1); 7 }
                            _ => { gemv_n::<8, false>(m, k2, k, a, nb0, nb1, o, g0, g1); 8 }
                        };
                    }
                }
            }

            #[target_feature(enable = $feat)]
            pub(super) unsafe fn gemv_m1(m: &[u8], k2: usize, k: usize, n: usize, a: &Act, nb0: usize, nb1: usize, out: &mut [f32]) {
                unsafe {
                    assert!(m.len() >= t16::trellis_bytes(k2, k, n) + 16 && nb1 <= n / 16 && a.x1.len() >= k);
                    let p = &plans()[k2 - 3];
                    let mut msk = [[[_mm_setzero_si128(); 2]; 2]; 4];
                    for v in 0..4 {
                        for hh in 0..2 {
                            for q in 0..2 {
                                msk[v][hh][q] = _mm_loadu_si128(p.mask[v][hh][q].as_ptr() as *const __m128i);
                            }
                        }
                    }
                    let (kt, tb) = (k / 16, 16 * k2);
                    let mul = _mm256_set1_epi32(M1_MUL as i32);
                    let m16 = _mm256_set1_epi32(0xFFFF);
                    let x1 = &a.x1[..];
                    for nb in nb0..nb1 {
                        let (mut f0, mut f1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                        let col = m.as_ptr().add(nb * kt * tb);
                        for gr in 0..k / 256 {
                            let (mut a0, mut a1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                            for kb in 16 * gr..16 * gr + 16 {
                                let t = col.add(kb * tb);
                                match k2 {
                                    3 => tile_m1!(t, p, msk, x1, kb, a0, a1, mul, m16; 0: (lo 23), 1: (lo 21), 2: (lo 20), 3: (lo 18), 4: (lo 17), 5: (lo 15), 6: (lo 14), 7: (lo 12), 8: (lo 11), 9: (lo 9), 10: (lo 8), 11: (lo 6), 12: (lo 5), 13: (lo 3), 14: (lo 2), 15: (zero)),
                                    4 => tile_m1!(t, p, msk, x1, kb, a0, a1, mul, m16; 0: (lo 30), 1: (lo 28), 2: (lo 26), 3: (lo 24), 4: (lo 22), 5: (lo 20), 6: (lo 18), 7: (lo 16), 8: (lo 14), 9: (lo 12), 10: (lo 10), 11: (lo 8), 12: (lo 6), 13: (lo 4), 14: (lo 2), 15: (zero)),
                                    _ => tile_m1!(t, p, msk, x1, kb, a0, a1, mul, m16; 0: (hi 6), 1: (hi 3), 2: (hi 1), 3: (lo 30), 4: (lo 28), 5: (lo 25), 6: (lo 23), 7: (lo 20), 8: (lo 18), 9: (lo 15), 10: (lo 13), 11: (lo 10), 12: (lo 8), 13: (lo 5), 14: (lo 3), 15: (zero)),
                                }
                            }
                            let d = _mm256_set1_ps(a.d[gr]);
                            let c0 = _mm256_set1_ps(M1_C0 * a.s[gr] as f32);
                            let cb = _mm256_set1_ps(M1_CB);
                            f0 = _mm256_fmadd_ps(d, _mm256_fmsub_ps(cb, _mm256_cvtepi32_ps(a0), c0), f0);
                            f1 = _mm256_fmadd_ps(d, _mm256_fmsub_ps(cb, _mm256_cvtepi32_ps(a1), c0), f1);
                        }
                        let o = out.as_mut_ptr().add((nb - nb0) * 16);
                        _mm256_storeu_ps(o, f0);
                        _mm256_storeu_ps(o.add(8), f1);
                    }
                }
            }
        }
    };
}

// AVX2 without VNNI (Zen 2 EPYC): the kernels' dot products always pair a lane's bytes as [lo hi lo' hi'] (two
// 16-bit products, or mul1's 4 bytes) with activation bytes [q q q' q']: q (lo + hi) + q' (lo' + hi'). vpmaddubsw by
// ones gives each 16-bit lane's byte sum (at most 510, no saturation), the activation word shifted right by 8
// (arithmetic) gives [q q'] as i16, vpmaddwd sums the pair: exactly vpdpbusd's result for these operands. 3 ops (the
// shift is shared by a pair's two views) instead of the split-maddubs emulation's 8. Reference CPU (6-expert
// decode call): 2.23 -> 1.55 ms at 14 threads, 1.22 at 20; 8-token prefill calls 1.6 -> 0.71 ms (cpubench t16).
// Summing the bytes with and/shift/add instead of the multiplier was slower (single thread 24-26 vs 19-20 ms).
t16_kernels!(avx2, "avx2,fma,f16c,ssse3", |s, a, b| {
    let bs = _mm256_maddubs_epi16(a, _mm256_set1_epi8(1));
    _mm256_add_epi32(s, _mm256_madd_epi16(bs, _mm256_srai_epi16::<8>(b)))
});

t16_kernels!(vnni, "avx2,avxvnni,fma,f16c,ssse3", |s, a, b| _mm256_dpbusd_avx_epi32(s, a, b));
t16_kernels!(vnni512, "avx2,avx512vnni,avx512vl,fma,f16c,ssse3", |s, a, b| _mm256_dpbusd_epi32(s, a, b));

#[cfg(test)]
mod tests {
    use super::*;

    /// gemv_m1 (SIMD, mul1 codebook) against gemv_scalar_m1 on random streams; called directly, not via DSX_T16_CODEBOOK
    #[test]
    fn mul1_simd_matches_scalar() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for (k2, k, n) in [(3usize, 512usize, 256usize), (4, 768, 128), (5, 256, 384)] {
            let ty = t16::ty_for_k2(k2).unwrap();
            let m: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| rng(&mut seed) as u8).collect();
            let mut a = Act { x4: vec![0; k / 2], x1: vec![0; k], d: vec![0.0; k / 256], s: vec![0; k / 256] };
            for i in 0..k {
                let q = (rng(&mut seed) % 255) as i32 - 127;
                let b = (q & 255) as u32;
                a.x1[i] = (b * 0x0101_0101) as i32;
                a.x4[i / 2] |= ((b | b << 8) << (16 * (i % 2))) as i32;
                a.s[i / 256] += q;
            }
            for d in a.d.iter_mut() {
                *d = 0.01;
            }
            let (mut ys, mut yv) = (vec![0f32; n], vec![0f32; n]);
            gemv_scalar_m1(&m, k2, k, n, &a, 0, n / 16, &mut ys);
            match isa() {
                #[cfg(target_arch = "x86_64")]
                Isa::Vnni => unsafe { vnni::gemv_m1(&m, k2, k, n, &a, 0, n / 16, &mut yv) },
                #[cfg(target_arch = "x86_64")]
                Isa::Vnni512 => unsafe { vnni512::gemv_m1(&m, k2, k, n, &a, 0, n / 16, &mut yv) },
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2 => unsafe { avx2::gemv_m1(&m, k2, k, n, &a, 0, n / 16, &mut yv) },
                _ => return,
            }
            assert_eq!(ys, yv, "k2 {k2}: mul1 SIMD vs scalar ({:?})", isa());
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_dot_extremes_match_integer_reference() {
        if !std::is_x86_feature_detected!("avx2") { return; }
        use std::arch::x86_64::*;
        let mut seed = 0x42d2_7ab1_87d4_12a7;
        for round in 0..4096 {
            let weights: [u8; 32] = std::array::from_fn(|_| if round < 4 { 255 } else { rng(&mut seed) as u8 });
            // the kernels' activation words: [q q q' q'] (pair words) or [q q q q] (mul1)
            let q: [i8; 16] = std::array::from_fn(|_| match round { 0 => -128, 1 => 127, 2 => -127, 3 => 0, _ => rng(&mut seed) as i8 });
            let acts: [i8; 32] = std::array::from_fn(|i| if round % 2 == 1 { q[i / 4] } else { q[i / 2] });
            let initial: [i32; 8] = std::array::from_fn(|_| rng(&mut seed) as i32);
            let mut got = [0i32; 8];
            unsafe {
                let r = avx2::dp(_mm256_loadu_si256(initial.as_ptr().cast()),
                    _mm256_loadu_si256(weights.as_ptr().cast()), _mm256_loadu_si256(acts.as_ptr().cast()));
                _mm256_storeu_si256(got.as_mut_ptr().cast(), r);
            }
            for lane in 0..8 {
                let dot: i32 = (0..4).map(|j| weights[4*lane+j] as i32 * acts[4*lane+j] as i32).sum();
                assert_eq!(got[lane], initial[lane].wrapping_add(dot), "round {round}, lane {lane}");
            }
        }
    }

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// gemv_multi for 1..=17 tokens (chunks of 8 and every remainder) equals gemv per token bit for bit, on every K, also
    /// in two group ranges
    #[test]
    fn multi_matches_per_token() {
        let mut seed = 0x51d7_348f_00c4_2b77u64;
        for (k2, k, n) in [(3usize, 512usize, 256usize), (4, 768, 128), (5, 256, 384)] {
            let ty = t16::ty_for_k2(k2).unwrap();
            let m: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| rng(&mut seed) as u8).collect();
            let (suh, _) = scales(&m, k2, k, n);
            let acts: Vec<Act> = (0..17)
                .map(|_| {
                    let x: Vec<f32> = (0..k).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
                    let mut a = Act::default();
                    prep(&x, suh, k, &mut a);
                    a
                })
                .collect();
            for nt in 1..=17 {
                for (nb0, nb1) in [(0, n / 16), (1, 4)] {
                    let want: Vec<Vec<f32>> = (0..nt)
                        .map(|i| {
                            let mut y = vec![0f32; 16 * (nb1 - nb0)];
                            gemv(&m, k2, k, n, &acts[i], nb0, nb1, &mut y);
                            y
                        })
                        .collect();
                    let mut got = vec![vec![0f32; 16 * (nb1 - nb0)]; nt];
                    let ar: Vec<&Act> = acts[..nt].iter().collect();
                    let mut or: Vec<&mut [f32]> = got.iter_mut().map(|v| &mut v[..]).collect();
                    gemv_multi(&m, k2, k, n, &ar, nb0, nb1, &mut or);
                    for i in 0..nt {
                        assert!(want[i].iter().zip(&got[i]).all(|(a, b)| a.to_bits() == b.to_bits()), "k2 {k2} nt {nt} token {i} cols {nb0}..{nb1}");
                    }
                    // the same in two group ranges (the pipelined down phase), every split point
                    for g in 1..k / 256 {
                        let mut got = vec![vec![0f32; 16 * (nb1 - nb0)]; nt];
                        let mut or: Vec<&mut [f32]> = got.iter_mut().map(|v| &mut v[..]).collect();
                        gemv_multi_groups(&m, k2, k, n, &ar, nb0, nb1, &mut or, 0, g);
                        gemv_multi_groups(&m, k2, k, n, &ar, nb0, nb1, &mut or, g, k / 256);
                        for i in 0..nt {
                            assert!(want[i].iter().zip(&got[i]).all(|(a, b)| a.to_bits() == b.to_bits()), "k2 {k2} nt {nt} token {i} split {g}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simd_matches_scalar_and_reference() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for (k2, k, n) in [(3usize, 512usize, 256usize), (4, 768, 128), (5, 256, 384)] {
            let ty = t16::ty_for_k2(k2).unwrap();
            let mut m: Vec<u8> = (0..t16::matrix_bytes(ty, k, n)).map(|_| rng(&mut seed) as u8).collect();
            let tb = t16::trellis_bytes(k2, k, n);
            for c in m[tb..].chunks_exact_mut(2) {
                let v = 0.02 * (0.5 + (rng(&mut seed) % 1000) as f32 / 1000.0) * if rng(&mut seed) % 2 == 0 { 1.0 } else { -1.0 };
                c.copy_from_slice(&f16::from_f32(v).to_le_bytes());
            }
            let x: Vec<f32> = (0..k).map(|_| (rng(&mut seed) % 2001) as f32 / 1000.0 - 1.0).collect();
            let (suh, svh) = scales(&m, k2, k, n);
            let mut a = Act::default();
            prep(&x, suh, k, &mut a);
            let (mut ys, mut yv) = (vec![0f32; n], vec![0f32; n]);
            gemv_scalar(&m, k2, k, n, &a, 0, n / 16, &mut ys);
            gemv(&m, k2, k, n, &a, 0, n / 16, &mut yv);
            let rms = (ys.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
            let e = ys.iter().zip(&yv).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / rms;
            assert!(e < 1e-5, "k2 {k2}: SIMD vs scalar {e} ({:?})", isa());
            for o in (0..n).step_by(128) {
                post_block(&mut yv[o..], svh, o);
            }
            let want = t16::forward(&m, ty, k, n, &x);
            let wr = (want.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
            let er = (want.iter().zip(&yv).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / n as f32).sqrt() / wr;
            println!("k2 {k2}: SIMD = scalar ({e:.1e}); vs f32 reference rms {er:.2e} (int8 activations), {:?}", isa());
            assert!(er < 0.02, "k2 {k2}: vs reference {er}");
        }
    }
}
