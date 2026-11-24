//! Opt-in short-answer guard calibrated on raw answer entropy and reasoning length.

use deepseek_recipe_core::messages::InputMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ServerConfig, uncertainty::Channels};

#[derive(Clone, Copy, Deserialize)]
#[serde(untagged)]
pub(super) enum Request {
    Enabled(bool),
    Options(Options),
}

impl Request {
    pub fn options(self) -> Option<Options> {
        match self {
            Self::Enabled(true) => Some(Options::default()),
            Self::Enabled(false) => None,
            Self::Options(options) => Some(options),
        }
    }
}

#[derive(Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Mode {
    #[default]
    Flag,
    Hedge,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Options {
    mode: Mode,
    max_answer_tokens: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { mode: Mode::Flag, max_answer_tokens: 64 }
    }
}

pub(super) struct Guard {
    options: Options,
    entropy: f64,
    think_ratio: f64,
    hedge: Option<&'static str>,
}

#[derive(Serialize)]
struct Verdict {
    verdict: &'static str,
    risk: f64,
    threshold: f64,
    signal: &'static str,
    answer_tokens: usize,
    reasoning_tokens: usize,
}

impl Guard {
    pub fn new(options: Options, cfg: &ServerConfig, messages: &[InputMessage], stream: bool) -> Self {
        let hedge = (options.mode == Mode::Hedge && !stream).then(|| {
            let russian = messages.iter().rev().find_map(|m| match m {
                InputMessage::User { content, .. } => Some(content.chars().any(|c| matches!(c,
                    '\u{0400}'..='\u{052f}' | '\u{1c80}'..='\u{1c8f}' | '\u{2de0}'..='\u{2dff}' | '\u{a640}'..='\u{a69f}'
                ))),
                _ => None,
            }).unwrap_or(false);
            if russian { "Не уверен в этом ответе — стоит проверить." }
            else { "I am not sure about this answer — worth checking." }
        });
        Self { options, entropy: cfg.guard_entropy, think_ratio: cfg.guard_think_ratio, hedge }
    }

    /// Use original generated tokens, before any hedge is added to the response text.
    pub fn attach(&self, json: &mut Value, channels: &Channels, tool_call: bool) {
        let answer_tokens = channels.content_stats.tokens;
        let reasoning_tokens = channels.reasoning_stats.tokens;
        let (risk, threshold, signal) = if reasoning_tokens == 0 {
            (channels.content_stats.mean_entropy, self.entropy, "answer_entropy")
        } else {
            (reasoning_tokens as f64 / answer_tokens.max(1) as f64, self.think_ratio, "reasoning_ratio")
        };
        let verdict = if tool_call || answer_tokens == 0 || answer_tokens > self.options.max_answer_tokens { "not_applicable" }
            else if risk > threshold { "uncertain" } else { "ok" };
        json["x_guard"] = serde_json::to_value(Verdict { verdict, risk, threshold, signal, answer_tokens, reasoning_tokens }).unwrap();
        if verdict == "uncertain" {
            if let Some(prefix) = self.hedge {
                let message = &mut json["choices"][0]["message"];
                let tool_call = message["tool_calls"].as_array().is_some_and(|calls| !calls.is_empty()) || message["function_call"].is_object();
                if !tool_call {
                    if let Some(content) = message["content"].as_str() {
                        message["content"] = Value::String(format!("{prefix}\n\n{content}"));
                    }
                }
            }
        }
    }
}
