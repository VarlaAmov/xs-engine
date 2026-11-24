//! Raw T=1 model statistics, independent of sampling and logit penalties.

use serde::Serialize;

#[derive(Debug, Clone)]
pub struct TokenStats {
    pub logprob: f32,
    pub entropy: f32,
    pub margin: f32,
    pub top: Vec<(u32, f32)>,
}

pub struct Distribution {
    pub(super) max: f64,
    pub(super) log_sum: f64,
    pub(super) entropy: f32,
    pub(super) margin: f32,
    pub(super) top: Vec<(u32, f32)>,
}

impl Distribution {
    /// One max reduction followed by one pass for both moments and the bounded top-k.
    /// NaNs have zero mass. An entirely invalid vector has zero entropy and no alternatives.
    pub fn new(logits: &[f32], k: u8) -> Self {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { super::uncertainty_simd::distribution(logits, k) };
        }
        Self::scalar(logits, k)
    }

    pub(super) fn scalar(logits: &[f32], k: u8) -> Self {
        let max = logits.iter().copied().filter(|v| !v.is_nan()).fold(f32::NEG_INFINITY, f32::max);
        let k = usize::from(k.min(20));
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(if k == 0 { 0 } else { k + 1 });
        let (mut sum, mut moment) = (0.0f64, 0.0f64);
        let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        let infinite = max == f32::INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if !l.is_finite() && !(infinite && l == max) {
                continue;
            }
            if l > first {
                second = first;
                first = l;
            } else if l > second {
                second = l;
            }
            if k > 0 && (top.len() < k || l > top.last().unwrap().1) {
                let pos = top.partition_point(|&(_, v)| v >= l);
                top.insert(pos, (i as u32, l));
                if top.len() > k {
                    top.pop();
                }
            }
            let d = if infinite { if l == max { 0.0 } else { continue } } else { l as f64 - max as f64 };
            if d >= -30.0 {
                let w = d.exp();
                sum += w;
                moment += w * d;
            }
        }
        let log_sum = if sum > 0.0 { sum.ln() } else { 0.0 };
        let entropy = if sum > 0.0 { (log_sum - moment / sum).max(0.0) as f32 } else { 0.0 };
        let margin = if first == second { 0.0 } else { first - second };
        let mut dist = Self { max: max as f64, log_sum, entropy, margin, top: Vec::new() };
        dist.top = top.into_iter().map(|(i, l)| (i, dist.logprob(l))).collect();
        dist
    }

    pub(super) fn logprob(&self, logit: f32) -> f32 {
        if self.max == f64::INFINITY && logit == f32::INFINITY {
            return -self.log_sum as f32;
        }
        if !logit.is_finite() || !self.max.is_finite() {
            return f32::NEG_INFINITY;
        }
        (logit as f64 - self.max - self.log_sum) as f32
    }

    pub fn chosen(self, logit: f32) -> TokenStats {
        TokenStats { logprob: self.logprob(logit), entropy: self.entropy, margin: self.margin, top: self.top }
    }
}

#[derive(Default, Serialize)]
pub struct Summary {
    pub tokens: usize,
    mean_logprob: f64,
    pub min_logprob: Option<f32>,
    pub mean_entropy: f64,
    max_entropy: Option<f32>,
    low_prob_tokens: usize,
    low_margin_tokens: usize,
}

impl Summary {
    pub fn add(&mut self, s: &TokenStats) {
        self.tokens += 1;
        if s.logprob == f32::NEG_INFINITY {
            self.mean_logprob = f64::NEG_INFINITY;
        } else if self.mean_logprob.is_finite() {
            self.mean_logprob += (s.logprob as f64 - self.mean_logprob) / self.tokens as f64;
        }
        self.mean_entropy += (s.entropy as f64 - self.mean_entropy) / self.tokens as f64;
        self.min_logprob = Some(self.min_logprob.map_or(s.logprob, |v| v.min(s.logprob)));
        self.max_entropy = Some(self.max_entropy.map_or(s.entropy, |v| v.max(s.entropy)));
        self.low_prob_tokens += usize::from(s.logprob < 0.3f32.ln());
        self.low_margin_tokens += usize::from(s.margin < 0.5);
    }
}

pub struct Channels {
    think: Option<(u32, u32)>,
    reasoning: bool,
    pub reasoning_stats: Summary,
    pub content_stats: Summary,
}

impl Channels {
    pub fn new(think: Option<(u32, u32)>, prompt: &[u32]) -> Self {
        Self {
            think,
            reasoning: think.is_some_and(|(open, _)| prompt.last() == Some(&open)),
            reasoning_stats: Summary::default(),
            content_stats: Summary::default(),
        }
    }
    /// Delimiters belong to the channel they close/open; every generated id is counted once.
    pub fn token(&mut self, tok: u32, stats: &TokenStats) -> bool {
        if self.think.is_some_and(|(open, _)| tok == open) {
            self.reasoning = true;
        }
        let reasoning = self.reasoning;
        if reasoning {
            self.reasoning_stats.add(stats);
        } else {
            self.content_stats.add(stats);
        }
        if self.think.is_some_and(|(_, close)| tok == close) {
            self.reasoning = false;
        }
        reasoning
    }
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"reasoning": self.reasoning_stats, "content": self.content_stats})
    }
}

#[derive(Default, serde::Deserialize, Clone, Copy)]
#[serde(default)]
pub(super) struct Options {
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
    x_uncertainty: Option<bool>,
    x_guard: Option<super::guard::Request>,
}

impl Options {
    pub fn parse(body: &[u8]) -> Result<Self, deepseek_recipe::request::ConversionError> {
        let options: Self = super::parse(body)?;
        if options.top_logprobs.is_some_and(|k| k > 20) {
            return Err(deepseek_recipe::request::ConversionError::bad_request("top_logprobs must be an integer in 0..=20"));
        }
        Ok(options)
    }
    pub fn enabled(self) -> bool {
        self.logprobs == Some(true) || self.x_uncertainty == Some(true) || self.guard().is_some()
    }
    pub fn guard(self) -> Option<super::guard::Options> {
        self.x_guard.and_then(super::guard::Request::options)
    }
    pub fn k(self) -> Option<u8> {
        self.enabled().then(|| if self.logprobs == Some(true) { self.top_logprobs.unwrap_or(0) } else { 0 })
    }
}

pub(super) struct Capture {
    channels: Channels,
    entries: bool,
    reasoning: Vec<serde_json::Value>,
    content: Vec<serde_json::Value>,
    tok: std::sync::Arc<tokenizers::Tokenizer>,
    guard: Option<super::guard::Guard>,
    /// a chunk or the response carried a tool call
    tool_call: bool,
}

impl Capture {
    pub fn new(options: Options, think: Option<(u32, u32)>, prompt: &[u32], tok: std::sync::Arc<tokenizers::Tokenizer>, guard: Option<super::guard::Guard>) -> Self {
        Self { channels: Channels::new(think, prompt), entries: options.logprobs == Some(true), reasoning: Vec::new(), content: Vec::new(), tok, guard, tool_call: false }
    }
    pub fn token(&mut self, tok: u32, stats: &TokenStats) {
        let reasoning = self.channels.token(tok, stats);
        if self.entries {
            let mut entry = token_json(&self.tok, tok, stats.logprob);
            entry["top_logprobs"] = serde_json::Value::Array(stats.top.iter().map(|&(id, lp)| token_json(&self.tok, id, lp)).collect());
            if reasoning {
                self.reasoning.push(entry);
            } else {
                self.content.push(entry);
            }
        }
    }
    pub fn attach(&mut self, json: &mut serde_json::Value, final_chunk: bool) {
        if self.entries {
            if let Some(choice) = json["choices"].get_mut(0) {
                choice["logprobs"] = serde_json::json!({
                    "content": std::mem::take(&mut self.content),
                    "reasoning_content": std::mem::take(&mut self.reasoning),
                });
            }
        }
        // a response that calls a tool is not a short factual answer: the guard reports not_applicable
        let choice = &json["choices"][0];
        let calls = |v: &serde_json::Value| v["tool_calls"].as_array().is_some_and(|c| !c.is_empty()) || v["function_call"].is_object();
        self.tool_call |= calls(&choice["delta"]) || calls(&choice["message"]);
        if final_chunk {
            json["x_uncertainty"] = self.channels.json();
            if let Some(guard) = &self.guard {
                guard.attach(json, &self.channels, self.tool_call);
            }
        }
    }
}

fn token_json(tok: &tokenizers::Tokenizer, id: u32, logprob: f32) -> serde_json::Value {
    let bytes = token_bytes(tok, id);
    serde_json::json!({"token": String::from_utf8_lossy(&bytes), "logprob": logprob, "bytes": bytes})
}

/// Invert the model's ByteLevel alphabet before UTF-8 decoding, preserving partial characters.
fn token_bytes(tok: &tokenizers::Tokenizer, id: u32) -> Vec<u8> {
    static INVERSE: std::sync::OnceLock<[Option<u8>; 512]> = std::sync::OnceLock::new();
    let inverse = INVERSE.get_or_init(|| {
        let mut table = [None; 512];
        let mut extra = 256;
        for b in 0..=255u8 {
            let c = if matches!(b, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
                b as usize
            } else {
                let c = extra;
                extra += 1;
                c
            };
            table[c] = Some(b);
        }
        table
    });
    let token = tok.id_to_token(id).unwrap_or_default();
    token.chars().map(|c| inverse.get(c as usize).copied().flatten()).collect::<Option<Vec<_>>>().unwrap_or_else(|| token.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::sample::{Rng, Sampler};

    fn random_logits(n: usize, scale: f64) -> Vec<f32> {
        let mut rng = Rng::new(13);
        (0..n).map(|_| ((rng.next_f64() - 0.5) * scale) as f32).collect()
    }

    #[test]
    fn matches_naive_f64() {
        let mut peaked = vec![-100.0; 129_280];
        peaked[7] = 5.0;
        peaked[123] = 4.8;
        peaked[234] = -24.9;
        for logits in [random_logits(129_280, 20.0), random_logits(129_280, 100.0), peaked, vec![0.0; 129_280]] {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
            let log_z = max + sum.ln();
            let entropy: f64 = logits
                .iter()
                .map(|&l| {
                    let lp = l as f64 - log_z;
                    -lp.exp() * lp
                })
                .sum();
            for k in [0, 1, 20] {
                let stats = Distribution::new(&logits, k).chosen(logits[7]);
                assert!((stats.logprob as f64 - (logits[7] as f64 - log_z)).abs() < 1e-4);
                assert!((stats.entropy as f64 - entropy).abs() < 1e-4, "{} vs {entropy}", stats.entropy);
                assert_eq!(stats.top.len(), k as usize);
                for &(id, lp) in &stats.top {
                    assert!((lp as f64 - (logits[id as usize] as f64 - log_z)).abs() < 1e-4);
                }
                assert!(stats.top.windows(2).all(|w| w[0].1 >= w[1].1));
                let mut sorted = logits.clone();
                sorted.sort_unstable_by(|a, b| b.total_cmp(a));
                for (&(_, lp), &l) in stats.top.iter().zip(&sorted) {
                    assert!((lp as f64 - (l as f64 - log_z)).abs() < 1e-4);
                }
                assert!((stats.margin - (sorted[0] - sorted[1])).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn simd_and_scalar_agree_with_invalid_lanes_and_remainders() {
        for n in [0, 1, 7, 8, 9, 15, 16, 129_280] {
            let mut logits = random_logits(n, 70.0);
            for i in (0..n).step_by(11) {
                logits[i] = f32::NAN;
            }
            for i in (1..n).step_by(13) {
                logits[i] = f32::NEG_INFINITY;
            }
            for k in [0, 20] {
                let a = Distribution::new(&logits, k).chosen(1.0);
                let b = Distribution::scalar(&logits, k).chosen(1.0);
                if a.logprob.is_finite() {
                    assert!((a.logprob - b.logprob).abs() < 1e-4);
                } else {
                    assert_eq!(a.logprob, b.logprob);
                }
                assert!((a.entropy - b.entropy).abs() < 1e-4);
                assert_eq!(a.margin, b.margin);
                assert_eq!(a.top.len(), b.top.len());
                for ((i, lp), (j, expected)) in a.top.iter().zip(&b.top) {
                    assert_eq!(i, j);
                    assert!((lp - expected).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn invalid_logits_are_safe() {
        for logits in [vec![], vec![f32::NAN, f32::NEG_INFINITY], vec![f32::NAN, 0.0, f32::NEG_INFINITY], vec![f32::INFINITY, f32::INFINITY, 1.0]] {
            let s = Distribution::new(&logits, 20).chosen(f32::NEG_INFINITY);
            assert_eq!(s.logprob, f32::NEG_INFINITY);
            assert!(s.entropy.is_finite());
            assert!(!s.margin.is_nan());
            assert!(s.top.iter().all(|&(_, lp)| !lp.is_nan()));
        }
        let s = Distribution::new(&[f32::NAN, 3.0, f32::NEG_INFINITY], 20).chosen(3.0);
        assert_eq!(s.logprob, 0.0);
        assert_eq!(s.entropy, 0.0);
        assert_eq!(s.top, [(1, 0.0)]);
    }

    #[test]
    fn summary_thresholds_and_channel_boundaries() {
        let s = |lp, entropy, margin| TokenStats { logprob: lp, entropy, margin, top: Vec::new() };
        let mut channels = Channels::new(Some((10, 11)), &[10]);
        channels.token(1, &s(-2.0, 1.0, 0.49));
        channels.token(11, &s(-1.0, 3.0, 0.5));
        channels.token(2, &s(-0.5, 2.0, 1.0));
        let json = channels.json();
        assert_eq!(
            json["reasoning"],
            serde_json::json!({"tokens": 2, "mean_logprob": -1.5,
            "min_logprob": -2.0, "mean_entropy": 2.0, "max_entropy": 3.0, "low_prob_tokens": 1, "low_margin_tokens": 1})
        );
        assert_eq!(json["content"]["tokens"], 1);
        assert_eq!(json["content"]["mean_logprob"], -0.5);
        assert_eq!(json["content"]["low_prob_tokens"], 0);
    }

    #[test]
    fn statistics_do_not_change_seeded_sampling() {
        let logits = random_logits(1000, 15.0);
        for (t, p) in [(0.0, 1.0), (0.7, 0.9), (1.0, 1.0)] {
            let mut plain = Sampler::new(t, p, Some(42));
            let mut measured = plain.clone();
            for _ in 0..100 {
                let dist = Distribution::new(&logits, 20);
                let chosen = measured.sample(&logits);
                let _stats = dist.chosen(logits[chosen as usize]);
                assert_eq!(plain.sample(&logits), chosen);
            }
        }
    }

    #[test]
    #[ignore = "run with --release --ignored --nocapture to measure raw statistics"]
    fn uncertainty_cost() {
        use std::hint::black_box;
        let dense = random_logits(129_280, 20.0);
        let mut peaked = vec![-40.0; 129_280];
        for (i, l) in peaked[..64].iter_mut().enumerate() {
            *l = -(i as f32) * 0.25;
        }
        for (name, logits) in [("dense", &dense), ("peaked", &peaked)] {
            for k in [0, 20] {
                for _ in 0..20 {
                    black_box(Distribution::new(black_box(logits), k).chosen(logits[0]));
                }
                let start = std::time::Instant::now();
                let n = 1000;
                for _ in 0..n {
                    black_box(Distribution::new(black_box(logits), k).chosen(logits[0]));
                }
                eprintln!("uncertainty {name} k={k}: {:.3} us/token", start.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
        }
    }
}
