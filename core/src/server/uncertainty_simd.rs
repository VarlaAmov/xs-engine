//! AVX2/FMA implementation of the raw-distribution reduction. Called only after feature detection.

use super::uncertainty::Distribution;
use std::arch::x86_64::*;

/// exp(x) on [-30, 0]: range reduction and a degree-7 Taylor polynomial on [-ln(2)/2, ln(2)/2].
/// The truncation error is below 8e-9 relative; f32 rounding dominates (tested below 4e-7).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp(x: __m256) -> __m256 {
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.4286068e-6), _mm256_fnmadd_ps(n, _mm256_set1_ps(0.69314575), x));
    let r2 = _mm256_mul_ps(r, r);
    let p23 = _mm256_fmadd_ps(r, _mm256_set1_ps(1.0 / 6.0), _mm256_set1_ps(0.5));
    let p45 = _mm256_fmadd_ps(r, _mm256_set1_ps(1.0 / 120.0), _mm256_set1_ps(1.0 / 24.0));
    let p67 = _mm256_fmadd_ps(r, _mm256_set1_ps(1.0 / 5040.0), _mm256_set1_ps(1.0 / 720.0));
    let p = _mm256_fmadd_ps(r2, _mm256_fmadd_ps(r2, _mm256_fmadd_ps(r2, p67, p45), p23), _mm256_add_ps(_mm256_set1_ps(1.0), r));
    let power = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127))));
    _mm256_mul_ps(p, power)
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn distribution(logits: &[f32], k: u8) -> Distribution {
    unsafe {
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut mx = neg_inf;
        let mut chunks = logits.chunks_exact(8);
        for chunk in &mut chunks {
            let v = _mm256_loadu_ps(chunk.as_ptr());
            mx = _mm256_max_ps(v, mx);
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), mx);
        let max = lanes.iter().chain(chunks.remainder()).copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return Distribution::scalar(logits, k);
        }
        let k = usize::from(k.min(20));
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(if k == 0 { 0 } else { k + 1 });
        let (mut sum_lo, mut sum_hi, mut moment_lo, mut moment_hi) = (_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd());
        let (mut first, mut second) = (neg_inf, neg_inf);
        let (mut partial_sum, mut partial_moment) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        for (block, chunk) in logits.chunks_exact(8).enumerate() {
            let v = _mm256_loadu_ps(chunk.as_ptr());
            let v = _mm256_max_ps(v, neg_inf);
            second = _mm256_max_ps(second, _mm256_min_ps(first, v));
            first = _mm256_max_ps(first, v);
            let d = _mm256_sub_ps(v, _mm256_set1_ps(max));
            let active = _mm256_cmp_ps::<_CMP_GE_OQ>(d, _mm256_set1_ps(-30.0));
            if _mm256_movemask_ps(active) != 0 {
                let d = _mm256_max_ps(d, _mm256_set1_ps(-30.0));
                let w = _mm256_and_ps(exp(d), active);
                partial_sum = _mm256_add_ps(partial_sum, w);
                partial_moment = _mm256_fmadd_ps(w, d, partial_moment);
            }
            // Bound f32 accumulation to 16 terms per lane, then accumulate both moments in f64.
            if block & 15 == 15 {
                sum_lo = _mm256_add_pd(sum_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(partial_sum)));
                sum_hi = _mm256_add_pd(sum_hi, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(partial_sum)));
                moment_lo = _mm256_add_pd(moment_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(partial_moment)));
                moment_hi = _mm256_add_pd(moment_hi, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(partial_moment)));
                partial_sum = _mm256_setzero_ps();
                partial_moment = _mm256_setzero_ps();
            }
            if k > 0 {
                let threshold = if top.len() < k { f32::NEG_INFINITY } else { top.last().unwrap().1 };
                let mut candidates = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(v, _mm256_set1_ps(threshold))) as u32;
                while candidates != 0 {
                    let lane = candidates.trailing_zeros() as usize;
                    candidates &= candidates - 1;
                    insert(&mut top, k, (block * 8 + lane) as u32, chunk[lane]);
                }
            }
        }
        sum_lo = _mm256_add_pd(sum_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(partial_sum)));
        sum_hi = _mm256_add_pd(sum_hi, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(partial_sum)));
        moment_lo = _mm256_add_pd(moment_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(partial_moment)));
        moment_hi = _mm256_add_pd(moment_hi, _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(partial_moment)));
        let mut sums = [0.0f64; 4];
        let mut moments = [0.0f64; 4];
        _mm256_storeu_pd(sums.as_mut_ptr(), _mm256_add_pd(sum_lo, sum_hi));
        _mm256_storeu_pd(moments.as_mut_ptr(), _mm256_add_pd(moment_lo, moment_hi));
        let mut sum: f64 = sums.iter().sum();
        let mut moment: f64 = moments.iter().sum();
        let mut maxima = [0.0f32; 16];
        _mm256_storeu_ps(maxima.as_mut_ptr(), first);
        _mm256_storeu_ps(maxima.as_mut_ptr().add(8), second);
        let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &l in &maxima {
            if l > first {
                second = first;
                first = l;
            } else if l > second {
                second = l;
            }
        }
        let offset = logits.len() / 8 * 8;
        for (i, &l) in logits[offset..].iter().enumerate() {
            if !l.is_finite() {
                continue;
            }
            if l > first {
                second = first;
                first = l;
            } else if l > second {
                second = l;
            }
            if k > 0 {
                insert(&mut top, k, (offset + i) as u32, l);
            }
            let d = l as f64 - max as f64;
            if d >= -30.0 {
                let w = d.exp();
                sum += w;
                moment += w * d;
            }
        }
        let log_sum = sum.ln();
        let mut dist = Distribution { max: max as f64, log_sum, entropy: (log_sum - moment / sum).max(0.0) as f32, margin: first - second, top: Vec::new() };
        dist.top = top.into_iter().map(|(id, l)| (id, dist.logprob(l))).collect();
        dist
    }
}

fn insert(top: &mut Vec<(u32, f32)>, k: usize, id: u32, l: f32) {
    if top.len() < k || l > top.last().unwrap().1 {
        let pos = top.partition_point(|&(_, v)| v >= l);
        top.insert(pos, (id, l));
        if top.len() > k {
            top.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_accuracy() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        for i in 0..30_000 {
            let x = -(i as f32) / 1000.0;
            let mut values = [0.0f32; 8];
            unsafe {
                _mm256_storeu_ps(values.as_mut_ptr(), exp(_mm256_set1_ps(x)));
            }
            let exact = (x as f64).exp();
            assert!((values[0] as f64 / exact - 1.0).abs() < 4e-7, "x={x}");
        }
    }
}
