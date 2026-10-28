//! CPU token sampling on a logits vector: greedy at temperature 0, else temperature + min-p + top-p with a seeded RNG.
//! While the model reasons, `sample_with_penalty` also applies the reasoning control: the hesitation penalty and the
//! thinking budget (`engine::Hesitation`).

use super::engine::{GenParams, Hesitation};
use super::uncertainty::{Distribution, TokenStats};

/// splitmix64: tiny, seedable, good enough for sampling
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    /// keep tokens whose weight is at least this fraction of the top token's (0 = off); applied before top-p
    pub min_p: f32,
    /// reasoning tokens sampled so far (counted while the reasoning control is active)
    think_n: usize,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            t.as_nanos() as u64 ^ (std::process::id() as u64) << 32
        });
        Self { temperature, top_p, min_p: 0.0, think_n: 0, rng: Rng::new(seed) }
    }

    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = if min_p.is_finite() { min_p.clamp(0.0, 1.0) } else { 0.0 };
        self
    }

    pub fn from_params(p: &GenParams) -> Self {
        Self::new(p.temperature, p.top_p, p.seed).with_min_p(p.min_p)
    }

    /// The sampling distribution: (token, weight) pairs and their total weight, or None for greedy. Only tokens within
    /// 30 temperature-scaled logits of the max are candidates (each other token weighs < e^-30 of the top one: under
    /// 1.2e-8 of the mass over the whole vocabulary). With min_p > 0 only tokens within that fraction of the top token's
    /// weight stay. With top_p < 1 the support is the nucleus, heaviest first.
    fn dist(&self, logits: &[f32]) -> Option<(Vec<(u32, f64)>, f64)> {
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return None;
        }
        let mx = logits.iter().copied().filter(|v| !v.is_nan()).fold(f32::NEG_INFINITY, f32::max);
        if !mx.is_finite() {
            return None;
        }
        let th = mx - 30.0 * self.temperature;
        let inv_t = 1.0 / self.temperature as f64;
        let mut c: Vec<(u32, f64)> = logits.iter().enumerate().filter(|&(_, &l)| l >= th).map(|(i, &l)| (i as u32, ((l - mx) as f64 * inv_t).exp())).collect();
        let mut sum: f64 = c.iter().map(|x| x.1).sum();
        if self.min_p > 0.0 {
            // weights are relative to the top token (weight 1): this keeps p_i / p_max >= min_p
            let min_p = self.min_p as f64;
            c.retain(|x| x.1 >= min_p);
            sum = c.iter().map(|x| x.1).sum();
        }
        let top_p = self.top_p as f64;
        if !(top_p < 1.0) {
            return Some((c, sum));
        }
        // nucleus: grow a sorted top-k until it covers top_p of the mass (usually a few dozen tokens)
        let target = top_p.max(0.0) * sum;
        let mut k = 64.min(c.len());
        loop {
            if k < c.len() {
                c.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            }
            c[..k].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut acc = 0.0;
            for j in 0..k {
                acc += c[j].1;
                if acc >= target {
                    c.truncate(j + 1);
                    return Some((c, acc));
                }
            }
            if k == c.len() {
                return Some((c, acc));
            }
            k = (k * 4).min(c.len());
        }
    }

    /// Capture the raw distribution, apply the reasoning control (hesitation penalty, thinking budget) and sample.
    /// The statistics describe the model: they use the raw logits of every id the control changed.
    pub fn sample_with_penalty(&mut self, logits: &mut [f32], hesitation: Option<&Hesitation>, logprobs: Option<u8>) -> (u32, Option<TokenStats>) {
        let distribution = logprobs.map(|k| Distribution::new(logits, k));
        let raw_changed: Option<Vec<(u32, f32)>> = distribution.as_ref().and(hesitation).map(|h| {
            let budget_ids = h.budget.iter().flat_map(|b| b.touched(h.close));
            h.ids.iter().copied().chain(budget_ids).filter_map(|id| logits.get(id as usize).map(|&l| (id, l))).collect()
        });
        if let Some(h) = hesitation {
            self.think_n += 1;
            if h.lambda > 0.0 {
                for &i in h.ids.iter() {
                    if let Some(l) = logits.get_mut(i as usize) { *l -= h.lambda; }
                }
            }
            if let Some(b) = &h.budget {
                b.apply(self.think_n, h.close, logits);
            }
        }
        let tok = self.sample(logits);
        let stats = distribution.map(|d| {
            let raw = raw_changed.as_ref().and_then(|v| v.iter().find(|&&(id, _)| id == tok))
                .map_or_else(|| logits.get(tok as usize).copied().unwrap_or(f32::NEG_INFINITY), |&(_, l)| l);
            d.chosen(raw)
        });
        (tok, stats)
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let Some((c, mass)) = self.dist(logits) else { return argmax(logits) };
        let mut r = self.rng.next_f64() * mass;
        for &(i, w) in &c {
            r -= w;
            if r < 0.0 {
                return i;
            }
        }
        c.last().map(|x| x.0).unwrap_or_else(|| argmax(logits))
    }
}

impl Sampler {
    /// Speculative sampling against a deterministic draft token `d` (prompt lookup or an argmax draft): accept `d`
    /// with its probability under this sampler's distribution (temperature, top-p); otherwise draw from that
    /// distribution with `d` removed and renormalized. The result is distributed exactly as `sample(logits)`.
    /// Returns (accepted, token).
    pub fn accept_or_resample(&mut self, logits: &[f32], d: u32) -> (bool, u32) {
        let Some((c, mass)) = self.dist(logits) else {
            let a = argmax(logits);
            return (a == d, a);
        };
        let wd = c.iter().find(|x| x.0 == d).map(|x| x.1).unwrap_or(0.0);
        if wd > 0.0 && self.rng.next_f64() < wd / mass {
            return (true, d);
        }
        // residual: the distribution without d
        let rest = mass - wd;
        if rest <= 0.0 {
            return (true, d);
        }
        let mut r = self.rng.next_f64() * rest;
        let mut last = d;
        for &(i, w) in &c {
            if i == d {
                continue;
            }
            last = i;
            r -= w;
            if r < 0.0 {
                return (false, i);
            }
        }
        (false, last)
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_matches_sampling() {
        // accept_or_resample's token distribution equals sample's, whatever the draft
        let l = logits();
        for (t, p) in [(1.0f32, 1.0f32), (0.7, 0.9)] {
            let mut a = Sampler::new(t, p, Some(7));
            let mut b = Sampler::new(t, p, Some(9));
            let n = 200_000;
            let ha = hist(&mut a, &l, n);
            let mut hb = std::collections::HashMap::new();
            for k in 0..n {
                let d = [17u32, 400, 3, 5][k % 4];
                *hb.entry(b.accept_or_resample(&l, d).1).or_insert(0usize) += 1;
            }
            for tok in [17u32, 400, 3, 999] {
                let (x, y) = (*ha.get(&tok).unwrap_or(&0) as f64 / n as f64, *hb.get(&tok).unwrap_or(&0) as f64 / n as f64);
                assert!((x - y).abs() < 0.006, "t {t} p {p} token {tok}: {x} vs {y}");
            }
        }
    }

    fn logits() -> Vec<f32> {
        // probabilities at T=1 ~ softmax([3, 2, 1, 0, -1, ...])
        let mut l = vec![-10.0f32; 1000];
        l[17] = 3.0;
        l[400] = 2.0;
        l[3] = 1.0;
        l[999] = 0.0;
        l
    }

    fn hist(s: &mut Sampler, l: &[f32], n: usize) -> std::collections::HashMap<u32, usize> {
        let mut h = std::collections::HashMap::new();
        for _ in 0..n {
            *h.entry(s.sample(l)).or_insert(0) += 1;
        }
        h
    }

    #[test]
    fn greedy_is_argmax() {
        let l = logits();
        let mut s = Sampler::new(0.0, 0.9, Some(1));
        for _ in 0..10 {
            assert_eq!(s.sample(&l), 17);
        }
        assert_eq!(argmax(&[f32::NAN, 1.0, 2.0, -1.0]), 2);
    }

    #[test]
    fn seeded_is_deterministic() {
        let l = logits();
        let a: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(42)); (0..200).map(|_| s.sample(&l)).collect() };
        let b: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(42)); (0..200).map(|_| s.sample(&l)).collect() };
        let c: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(43)); (0..200).map(|_| s.sample(&l)).collect() };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn temperature_matches_softmax() {
        let l = logits();
        let n = 200_000;
        let h = hist(&mut Sampler::new(1.0, 1.0, Some(7)), &l, n);
        let z: f64 = l.iter().map(|&x| (x as f64).exp()).sum();
        for i in [17u32, 400, 3, 999] {
            let p = (l[i as usize] as f64).exp() / z;
            let got = *h.get(&i).unwrap_or(&0) as f64 / n as f64;
            assert!((got - p).abs() < 0.01, "token {i}: {got} vs {p}");
        }
        // T=0.5 sharpens: the top token gets more mass than at T=1
        let h2 = hist(&mut Sampler::new(0.5, 1.0, Some(7)), &l, 20_000);
        assert!(h2[&17] as f64 / 20_000.0 > 0.85);
    }

    #[test]
    fn top_p_truncates() {
        let l = logits();
        // p(17) ~ 0.64, p(400) ~ 0.24: top_p 0.5 keeps only 17, 0.8 keeps {17, 400}
        let h = hist(&mut Sampler::new(1.0, 0.5, Some(3)), &l, 5_000);
        assert_eq!(h.len(), 1);
        assert!(h.contains_key(&17));
        let h = hist(&mut Sampler::new(1.0, 0.8, Some(3)), &l, 20_000);
        assert_eq!(h.keys().copied().collect::<std::collections::BTreeSet<_>>(), [17, 400].into());
        let r = h[&17] as f64 / h[&400] as f64; // renormalized ratio stays e^1
        assert!((r - std::f64::consts::E).abs() < 0.2, "{r}");
    }

    #[test]
    fn min_p_keeps_tokens_near_the_top() {
        // weights relative to the top at T=1: 400 -> e^-1 = 0.37, 3 -> e^-2 = 0.14, 999 -> e^-3 = 0.05, the rest ~2e-6
        let l = logits();
        let keys = |min_p: f32| hist(&mut Sampler::new(1.0, 1.0, Some(5)).with_min_p(min_p), &l, 30_000).keys().copied().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(keys(0.3), [17, 400].into());
        assert_eq!(keys(0.1), [17, 400, 3].into());
        assert_eq!(keys(0.04), [17, 400, 3, 999].into());
        assert!(keys(0.0).len() > 4, "min_p 0 keeps the whole tail");
        // the top token always stays, whatever min_p is
        assert_eq!(keys(1.0), [17].into());
        assert_eq!(Sampler::new(1.0, 1.0, Some(1)).with_min_p(f32::NAN).min_p, 0.0);
        assert_eq!(Sampler::new(1.0, 1.0, Some(1)).with_min_p(7.0).min_p, 1.0);
        // renormalized: the ratio of the survivors is unchanged (e^1)
        let h = hist(&mut Sampler::new(1.0, 1.0, Some(3)).with_min_p(0.3), &l, 40_000);
        let r = h[&17] as f64 / h[&400] as f64;
        assert!((r - std::f64::consts::E).abs() < 0.15, "{r}");
    }

    #[test]
    fn min_p_zero_changes_nothing_and_composes_with_top_p() {
        let l = logits();
        let plain: Vec<u32> = { let mut s = Sampler::new(1.0, 0.9, Some(42)); (0..300).map(|_| s.sample(&l)).collect() };
        let off: Vec<u32> = { let mut s = Sampler::new(1.0, 0.9, Some(42)).with_min_p(0.0); (0..300).map(|_| s.sample(&l)).collect() };
        assert_eq!(plain, off);
        // min_p 0.3 then top_p 0.5: the nucleus of {17, 400} at p(17) ~ 0.73 is {17}
        let h = hist(&mut Sampler::new(1.0, 0.5, Some(2)).with_min_p(0.3), &l, 5_000);
        assert_eq!(h.keys().copied().collect::<Vec<_>>(), vec![17]);
        // speculative sampling sees the same support
        let mut s = Sampler::new(1.0, 1.0, Some(8)).with_min_p(0.3);
        for _ in 0..500 {
            let (_, t) = s.accept_or_resample(&l, 999);
            assert!(t == 17 || t == 400, "{t}");
        }
    }

    #[test]
    fn top_p_grows_past_initial_k() {
        // flat distribution over 129k tokens: the nucleus is far larger than the first top-k window
        let l = vec![0.0f32; 129_280];
        let mut s = Sampler::new(1.0, 0.95, Some(9));
        let mut seen = std::collections::HashSet::new();
        for _ in 0..300 {
            let t = s.sample(&l);
            assert!((t as usize) < l.len());
            seen.insert(t);
        }
        assert!(seen.len() > 290);
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use crate::server::ThinkBudget;
    use std::sync::Arc;

    const CLOSE: u32 = 5;
    const BREAK: u32 = 6;

    fn control(tokens: usize, boost: f32, lambda: f32) -> Hesitation {
        Hesitation { lambda, ids: Arc::new(vec![1]), close: CLOSE, budget: Some(ThinkBudget::new(tokens, boost, Arc::new(vec![BREAK]))) }
    }

    /// the model wants token 0 (logit 4), then 1 (3); `</think>` and the paragraph break are unlikely (-3, -2)
    fn raw() -> [f32; 8] {
        [4.0, 3.0, 0.0, 0.0, 0.0, -3.0, -2.0, 0.0]
    }

    #[test]
    fn window_is_a_quarter_of_the_budget_between_8_and_256() {
        assert_eq!(ThinkBudget::new(16, 1.0, Arc::new(vec![])).window, 8);
        assert_eq!(ThinkBudget::new(400, 1.0, Arc::new(vec![])).window, 100);
        assert_eq!(ThinkBudget::new(100_000, 1.0, Arc::new(vec![])).window, 256);
    }

    #[test]
    fn nothing_changes_before_the_window_then_the_bonus_grows_and_past_the_budget_close_is_forced() {
        let b = ThinkBudget::new(400, 8.0, Arc::new(vec![BREAK])); // window 100: bonus starts after token 300
        let mut l = raw();
        b.apply(300, CLOSE, &mut l);
        assert_eq!(l, raw(), "at n = tokens - window the window has not begun");
        let mut prev = f32::MIN;
        for n in [301, 340, 380, 400] {
            let mut l = raw();
            b.apply(n, CLOSE, &mut l);
            assert!(l[CLOSE as usize] > prev, "bonus grows with n");
            let (bc, bb) = (l[CLOSE as usize] - raw()[CLOSE as usize], l[BREAK as usize] - raw()[BREAK as usize]);
            assert!((bc - bb).abs() < 1e-4, "close and the break get the same bonus: {bc} vs {bb}");
            assert_eq!(l[0], raw()[0], "other logits are untouched");
            prev = l[CLOSE as usize];
        }
        let mut l = raw();
        b.apply(400, CLOSE, &mut l);
        assert!((l[CLOSE as usize] - (raw()[CLOSE as usize] + 8.0)).abs() < 1e-5, "full bonus on the last allowed token");
        let mut l = raw();
        b.apply(401, CLOSE, &mut l);
        assert_eq!(l[CLOSE as usize], 0.0);
        assert!(l.iter().enumerate().all(|(i, &v)| i == CLOSE as usize || v == f32::NEG_INFINITY), "everything else is out");
    }

    #[test]
    fn the_sampler_counts_reasoning_tokens_and_closes_after_the_budget_at_any_temperature() {
        for t in [0.0f32, 1.0] {
            let h = control(40, 6.0, 0.0);
            let mut s = Sampler::new(t, 1.0, Some(11));
            let mut out = vec![];
            for _ in 0..41 {
                out.push(s.sample_with_penalty(&mut raw(), Some(&h), None).0);
            }
            assert_eq!(*out.last().unwrap(), CLOSE, "token 41 is </think> (t {t})");
            assert!(out[..40].iter().all(|&x| x != CLOSE || t > 0.0), "greedy never closes early on its own");
        }
    }

    #[test]
    fn soft_landing_makes_an_early_close_likely_before_the_limit() {
        // near the end of the window the bonus (6) outweighs the model's wish: close (-3 + 6 = 3) ties with the favourite 4 - and the break (-2 + 6 = 4) with it
        let h = control(40, 6.0, 0.0); // window 10
        let mut s = Sampler::new(1.0, 1.0, Some(3));
        for _ in 0..39 {
            s.sample_with_penalty(&mut raw(), Some(&h), None);
        }
        let mut hits = 0;
        for k in 0..2_000 {
            let mut s2 = s.clone();
            s2.rng = Rng::new(k);
            let t = s2.sample_with_penalty(&mut raw(), Some(&h), None).0;
            hits += (t == CLOSE || t == BREAK) as usize;
        }
        assert!(hits > 800, "close or break at the last allowed token: {hits} of 2000");
    }

    #[test]
    fn budget_and_penalty_work_together_and_without_a_budget_nothing_counts_as_forced() {
        // penalty only: no forced close however long the reasoning
        let h = Hesitation { lambda: 1.5, ids: Arc::new(vec![0]), close: CLOSE, budget: None };
        let mut s = Sampler::new(0.0, 1.0, Some(1));
        for _ in 0..1000 {
            assert_eq!(s.sample_with_penalty(&mut raw(), Some(&h), None).0, 1, "token 0 is penalized to 2.5, token 1 (3.0) wins");
        }
        // no control at all: the sampler is the plain one
        let mut s = Sampler::new(0.0, 1.0, Some(1));
        assert_eq!(s.sample_with_penalty(&mut raw(), None, None).0, 0);
        // both: the penalty acts before the limit, the close after it
        let h = control(3, 6.0, 1.5);
        let mut s = Sampler::new(0.0, 1.0, Some(1));
        let got: Vec<u32> = (0..4).map(|_| s.sample_with_penalty(&mut raw(), Some(&h), None).0).collect();
        assert_eq!(got[3], CLOSE);
        assert!(got[..3].iter().all(|&x| x != CLOSE));
    }

    #[test]
    fn statistics_keep_the_raw_logits_of_close_and_the_breaks() {
        let h = control(2, 6.0, 0.0);
        let mut s = Sampler::new(0.0, 1.0, Some(1));
        s.sample_with_penalty(&mut raw(), Some(&h), Some(4));
        s.sample_with_penalty(&mut raw(), Some(&h), Some(4));
        let (tok, stats) = s.sample_with_penalty(&mut raw(), Some(&h), Some(4));
        assert_eq!(tok, CLOSE);
        let reference = Distribution::new(&raw(), 4).chosen(raw()[CLOSE as usize]);
        let stats = stats.unwrap();
        assert_eq!(stats.logprob, reference.logprob, "the log-probability is the model's own, not the forced one");
        assert_eq!(stats.entropy, reference.entropy);
    }
}

#[cfg(test)]
mod uncertainty_tests {
    use super::*;

    #[test]
    fn raw_statistics_survive_penalty_temperature_and_nucleus() {
        let raw = [3.0, 2.9, 2.0, -4.0];
        let h = Hesitation { lambda: 1.5, ids: std::sync::Arc::new(vec![0, 1]), close: 7, budget: None };
        for (t, p) in [(0.0, 1.0), (0.8, 0.9)] {
            let mut plain = Sampler::new(t, p, Some(15));
            let mut measured = plain.clone();
            let mut disabled = plain.clone();
            for _ in 0..100 {
                let mut penalized = raw;
                for &id in h.ids.iter() { penalized[id as usize] -= h.lambda; }
                let expected = plain.sample(&penalized);
                let (tok, stats) = measured.sample_with_penalty(&mut raw.clone(), Some(&h), Some(4));
                assert_eq!(tok, expected);
                let stats = stats.unwrap();
                let reference = Distribution::new(&raw, 4).chosen(raw[tok as usize]);
                assert_eq!(stats.logprob, reference.logprob);
                assert_eq!(stats.entropy, reference.entropy);
                assert_eq!(stats.top, reference.top);
                let (off, stats) = disabled.sample_with_penalty(&mut raw.clone(), Some(&h), None);
                assert_eq!(off, expected);
                assert!(stats.is_none());
            }
        }
    }
}
