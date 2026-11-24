//! dsx-server: OpenAI / Anthropic compatible HTTP API for the dsx engine.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use clap::Parser;
use dsx::server::mock::{MockEngine, MockMode};
use dsx::server::{AppState, EngineHandle, ServerConfig};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(about = "OpenAI / Anthropic compatible API server for DeepSeek-V4.1-Flash (dsx)")]
struct Cli {
    /// DeepSeek V4.1 tokenizer.json; default: the one stored in the model GGUF (`dsx.tokenizer_json`)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8090)]
    port: u16,
    /// model id reported by /v1/models
    #[arg(long, default_value = "deepseek-v4.1-flash")]
    model_name: String,
    /// context window (prompt + generated tokens)
    #[arg(long, default_value_t = 32768)]
    ctx: usize,
    /// generation cap when a request sets no max_tokens
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,
    /// default sampling when a request sets none
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// logit penalty on hesitation tokens ("Wait", "But", "Alternatively", ...) while reasoning; 0 = off.
    /// A request's `hesitation_penalty` field overrides it.
    #[arg(long, default_value_t = 0.0)]
    hesitation_penalty: f32,
    /// min-p default: drop tokens whose probability is below this fraction of the top token's (0 = off). A request's `min_p` overrides it.
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    /// reasoning budget in tokens with a soft landing on a paragraph break and `</think>` (0 = none).
    /// A request's `thinking_budget` overrides it; `hesitation_words` replaces the penalty's word list.
    #[arg(long, default_value_t = 0)]
    thinking_budget: usize,
    /// Short-answer guard: mean answer entropy threshold in nats (requests opt in with x_guard).
    #[arg(long, default_value_t = 0.389, value_parser = positive_threshold)]
    guard_entropy: f64,
    /// Short-answer guard: reasoning/content token ratio threshold.
    #[arg(long, default_value_t = 83.0, value_parser = positive_threshold)]
    guard_think_ratio: f64,
    /// faster decode, slightly different outputs: the CPU's experts the router ranked R or lower (0 = best of 6) are added one
    /// layer later instead of being waited for (3: ~+20% decode for KL +0.014 on 2x RTX 3090). Same as DSX_DEFER=R
    #[arg(long)]
    defer_rank: Option<usize>,
    /// run the deterministic mock engine instead of the model
    #[arg(long)]
    mock: bool,
    /// mock: repeat the prompt tokens instead of the scripted replies
    #[arg(long)]
    mock_echo: bool,
    /// mock: sleep per generated token
    #[arg(long, default_value_t = 0)]
    mock_delay_ms: u64,
    /// require this key on every API request (`Authorization: Bearer <key>` or `x-api-key: <key>`);
    /// `openssl rand -hex 32` makes one. Health checks and the web UI's files stay open.
    #[arg(long, env = "DSX_API_KEY")]
    api_key: Option<String>,
    /// model: GGUF export or HF checkpoint dir
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    engram: Option<PathBuf>,
    #[arg(long, default_value_t = 4096)]
    chunk: usize,
    /// Conversation slots including the live GPU sequence (1 = original single-conversation behavior)
    #[arg(long, default_value_t = 1)]
    sessions: usize,
    /// Hard host-memory budget for all saved sessions, including snapshots in transit (decimal GB)
    #[arg(long, default_value_t = 2.0)]
    session_gb: f64,
    /// Minimum sequence length to save and minimum prefix advantage required to switch
    #[arg(long, default_value_t = 2048)]
    session_min_tokens: usize,
    /// Shared prompt snapshots per client pool, kept only in host memory (0 disables them)
    #[arg(long, default_value_t = 4)]
    session_prefix_max: usize,
    /// A client's own pool of saved conversations, `CLIENT=SESSIONS:GB` (repeatable; the client is the request's
    /// `X-DSX-Client` header). Only that client's requests evict from it; inside it the cheapest to re-read goes first.
    /// Clients without a quota share `default`, which gets what the quotas leave of --sessions and --session-gb
    #[arg(long = "session-quota", value_name = "CLIENT=SESSIONS:GB")]
    session_quota: Vec<String>,
    /// with quotas: a conversation used in the last N minutes is evicted only when nothing else in its pool can be
    #[arg(long, default_value_t = 10)]
    session_recent_min: u64,
    /// with quotas: a conversation idle N minutes goes first, whatever its size
    #[arg(long, default_value_t = 180)]
    session_stale_min: u64,
    /// `X-DSX-Client` whose requests are served before other clients' queued ones (repeatable; never interrupts a
    /// running request)
    #[arg(long = "priority-client", value_name = "CLIENT")]
    priority_client: Vec<String>,
    /// Archive of saved conversations on disk (turns the disk tier on): conversations pushed off the host-memory
    /// shelf are written here instead of dropped, the shelf is copied here while the server is idle, the live
    /// conversation is checkpointed, and everything here survives a restart. Owner-only directory
    #[arg(long, value_name = "DIR")]
    session_dir: Option<PathBuf>,
    /// archive: disk budget for all conversations (decimal GB)
    #[arg(long, default_value_t = 40.0)]
    session_disk_gb: f64,
    /// archive: conversations kept at most
    #[arg(long, default_value_t = 256)]
    session_disk_sessions: usize,
    /// archive: a client's own pool, `CLIENT=SESSIONS:GB[:DAYS]` (repeatable); clients without one share
    /// `default`, which gets what the quotas leave
    #[arg(long = "session-disk-quota", value_name = "CLIENT=SESSIONS:GB[:DAYS]")]
    session_disk_quota: Vec<String>,
    /// archive: a conversation not used for this many days is deleted (per client: the DAYS of its quota)
    #[arg(long, default_value_t = 7.0)]
    session_disk_days: f64,
    /// archive: copy the shelf to disk and checkpoint the live conversation after this many seconds without requests
    #[arg(long, default_value_t = 20)]
    session_idle_s: u64,
    /// archive: always leave this much free on its filesystem (decimal GB)
    #[arg(long, default_value_t = 20.0)]
    session_disk_reserve_gb: f64,
    /// GB of routed experts on the main GPU (picked by host-link bandwidth); negative = its free memory minus that
    #[arg(long = "main-gb", alias = "gpu0-gb", value_name = "GB", default_value_t = -1.5, allow_negative_numbers = true)]
    gpu0_gb: f64,
    /// GB of routed experts on each other GPU; negative = its free memory minus that; 0 = main GPU only
    #[arg(long = "aux-gb", alias = "gpu1-gb", value_name = "GB", default_value_t = -1.0, allow_negative_numbers = true)]
    gpu1_gb: f64,
    #[arg(long, default_value_t = 0.0)]
    arena_gb: f64,
    /// expert hotness ranking (rank.json from calibration)
    #[arg(long)]
    rank: Option<std::path::PathBuf>,
    /// vision weights (`vision.safetensors`, or the official checkpoint directory): enables image input.
    /// Default: `vision.safetensors` next to the model, when present.
    #[arg(long)]
    vision: Option<std::path::PathBuf>,
    /// text only: don't load the vision weights even if `vision.safetensors` sits next to the model
    #[arg(long)]
    no_vision: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let mut cli = Cli::parse();
    session_config(&cli)?;
    for c in &cli.priority_client {
        anyhow::ensure!(dsx::server::sessions::valid_client(c), "--priority-client {c:?}: client names are 1-32 of [a-z0-9._-]");
    }
    if let Some(r) = cli.defer_rank {
        // read once by the engine (handoff::defer) before the model starts
        unsafe { std::env::set_var("DSX_DEFER", r.to_string()) };
    }
    if cli.no_vision {
        cli.vision = None;
    } else if cli.vision.is_none() {
        // vision.safetensors next to the model (the quant repo ships it beside the shards)
        let dir = cli.model.as_ref().and_then(|m| if m.is_dir() { Some(m.clone()) } else { m.parent().map(|p| p.to_path_buf()) });
        cli.vision = dir.map(|d| d.join("vision.safetensors")).filter(|p| p.is_file());
    }
    if let Some(v) = &cli.vision {
        tracing::info!("vision: {}", v.display());
    }
    let tok = Arc::new(match (&cli.tokenizer, &cli.model) {
        (Some(t), _) => Tokenizer::from_file(t).map_err(|e| anyhow!("{}: {e}", t.display()))?,
        (None, Some(m)) if m.is_file() => {
            let g = dsx::gguf::Gguf::open(m)?;
            let s = g.kv.get("dsx.tokenizer_json").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("{} has no dsx.tokenizer_json: pass --tokenizer", m.display()))?;
            s.parse::<Tokenizer>().map_err(|e| anyhow!("dsx.tokenizer_json in {}: {e}", m.display()))?
        }
        _ => return Err(anyhow!("--tokenizer is required unless the model GGUF stores dsx.tokenizer_json")),
    });
    let engine = if cli.mock {
        let (tok, mode, delay) = (tok.clone(), if cli.mock_echo { MockMode::Echo } else { MockMode::Scripted }, cli.mock_delay_ms);
        EngineHandle::spawn(move || {
            let mut m = MockEngine::new(tok, mode);
            m.delay = Duration::from_millis(delay);
            Ok(Box::new(m) as Box<dyn dsx::server::Engine>)
        })?
    } else {
        real(&cli, &tok)?
    };
    let cfg = ServerConfig {
        model_name: cli.model_name.clone(),
        ctx: cli.ctx,
        default_max_tokens: cli.max_tokens,
        default_temperature: cli.temperature,
        default_top_p: cli.top_p,
        hesitation_penalty: cli.hesitation_penalty,
        default_min_p: cli.min_p,
        thinking_budget: cli.thinking_budget,
        guard_entropy: cli.guard_entropy,
        guard_think_ratio: cli.guard_think_ratio,
        vision: cli.vision.is_some() && !cli.mock,
        api_key: cli.api_key.clone(),
        priority_clients: cli.priority_client.clone(),
        ..ServerConfig::default()
    };
    let monitor = engine.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if !monitor.is_alive() && !monitor.is_stopping() {
                tracing::error!("engine thread exited unexpectedly; exiting for systemd recovery");
                std::process::exit(1);
            }
        }
    });
    let st = AppState::new(engine, tok, cfg)?;
    let listener = tokio::net::TcpListener::bind((cli.host.as_str(), cli.port)).await?;
    tracing::info!("listening on http://{} ({})", listener.local_addr()?, if cli.mock { "mock engine" } else { "dsx engine" });
    dsx::server::serve(listener, st).await?;
    Ok(())
}

fn positive_threshold(value: &str) -> std::result::Result<f64, String> {
    let threshold: f64 = value.parse().map_err(|_| "expected a number".to_string())?;
    if threshold.is_finite() && threshold > 0.0 { Ok(threshold) }
    else { Err("threshold must be finite and > 0".into()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_threshold_flags() {
        let defaults = Cli::try_parse_from(["dsx-server"]).unwrap();
        assert_eq!(defaults.guard_entropy, 0.389);
        assert_eq!(defaults.guard_think_ratio, 83.0);
        let custom = Cli::try_parse_from(["dsx-server", "--guard-entropy=0.5", "--guard-think-ratio=65"]).unwrap();
        assert_eq!(custom.guard_entropy, 0.5);
        assert_eq!(custom.guard_think_ratio, 65.0);
        for flag in ["--guard-entropy", "--guard-think-ratio"] {
            for value in ["0", "-1", "NaN", "inf", "-inf"] {
                let error = Cli::try_parse_from(["dsx-server", &format!("{flag}={value}")]).err().unwrap();
                assert!(error.to_string().contains("must be finite and > 0"));
            }
        }
    }
}

fn session_config(cli: &Cli) -> Result<dsx::server::sessions::SessionConfig> {
    use dsx::server::sessions::{DiskConfig, DiskQuota, Quota, SessionConfig};
    let quotas = cli.session_quota.iter().map(|q| Quota::parse(q)).collect::<Result<Vec<_>>>()?;
    let min = |m: u64| Duration::from_secs(60 * m);
    let mut cfg = SessionConfig::new(cli.sessions, cli.session_gb, cli.session_min_tokens)?.with_quotas(quotas, min(cli.session_recent_min), min(cli.session_stale_min))?;
    cfg.prefix_max = cli.session_prefix_max;
    for q in &cfg.quotas {
        tracing::info!(client = q.client, sessions = q.sessions, gb = q.max_bytes as f64 / 1e9, "session quota");
    }
    if let Some(dir) = &cli.session_dir {
        let dq = cli.session_disk_quota.iter().map(|q| DiskQuota::parse(q)).collect::<Result<Vec<_>>>()?;
        let d = DiskConfig::new(dir.clone(), cli.session_disk_sessions, cli.session_disk_gb, cli.session_disk_days, Duration::from_secs(cli.session_idle_s), cli.session_disk_reserve_gb)?
            .with_quotas(dq)?;
        tracing::info!(dir = %d.dir.display(), sessions = d.sessions, gb = d.max_bytes as f64 / 1e9, days = d.ttl.as_secs_f64() / 86400.0, idle_s = cli.session_idle_s, "session archive");
        for q in &d.quotas {
            tracing::info!(client = q.client, sessions = q.sessions, gb = q.max_bytes as f64 / 1e9, days = q.ttl.unwrap_or(d.ttl).as_secs_f64() / 86400.0, "session archive quota");
        }
        cfg = cfg.with_disk(d)?;
    }
    Ok(cfg)
}

#[cfg(feature = "cuda")]
fn real(cli: &Cli, tok: &Tokenizer) -> Result<EngineHandle> {
    let args = dsx::server::real::RealEngineArgs {
        model: cli.model.clone().ok_or_else(|| anyhow!("--model is required (or --mock)"))?,
        engram: cli.engram.clone(),
        ctx: cli.ctx,
        chunk: cli.chunk,
        gpu0_gb: cli.gpu0_gb,
        gpu1_gb: cli.gpu1_gb,
        arena_gb: cli.arena_gb,
        rank: cli.rank.clone(),
        vision: cli.vision.clone(),
        sessions: session_config(cli)?,
        assistant_token: tok.token_to_id("<｜Assistant｜>"),
        user_token: tok.token_to_id("<｜User｜>").ok_or_else(|| anyhow!("tokenizer has no <｜User｜> token"))?,
    };
    EngineHandle::spawn(move || dsx::server::real::real_engine(&args))
}

#[cfg(not(feature = "cuda"))]
fn real(_: &Cli, _: &Tokenizer) -> Result<EngineHandle> {
    anyhow::bail!("built without the cuda feature: only --mock is available")
}
