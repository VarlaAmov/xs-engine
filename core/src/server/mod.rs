//! OpenAI / Anthropic compatible HTTP API over one engine thread, built on DeepSeek's `deepseek-recipe` crates:
//! request -> Conversation -> V4.1 prompt -> token ids -> engine -> token ids -> StreamProcessor -> protocol events.
//! Text only: image inputs are rejected.

pub mod auth;
pub mod engine;
mod guard;
pub mod mock;
pub mod sample;
pub mod session_store;
pub mod sessions;
pub mod uncertainty;
#[cfg(target_arch = "x86_64")]
mod uncertainty_simd;
pub mod webui;

use std::pin::{Pin, pin};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{ConversationRequest, ConversionError, ConversionOptions, ProtocolRequest};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::{ChunkGenerator, InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor};
use deepseek_recipe::util::append_delta::AppendDelta;
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_core::multimodal::ImageSource;
use crate::vision::ImageInput;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use tokenizers::Tokenizer;
use tokio_stream::{Stream, StreamExt};

pub use engine::{BatchEngine, Engine, EngineHandle, Event, Finish, GenParams, GenStats, Hesitation, ThinkBudget};

/// Tokens that end generation (never forwarded to the parser).
pub const STOP_TOKENS: [&str; 2] = ["<｜end▁of▁sentence｜>", "<｜User｜>"];

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// reported by /v1/models and used when a request names no model
    pub model_name: String,
    /// context window: prompt + generated tokens
    pub ctx: usize,
    /// generation cap when the request sets none
    pub default_max_tokens: usize,
    pub default_temperature: f32,
    pub default_top_p: f32,
    /// min-p filter (fraction of the top token's probability); 0 = off. Requests can set `min_p`.
    pub default_min_p: f32,
    /// hesitation-token logit penalty while reasoning; 0 = off. Requests can set `hesitation_penalty` (and `hesitation_words`).
    pub hesitation_penalty: f32,
    /// reasoning budget in tokens with a soft landing; 0 = none. Requests can set `thinking_budget`.
    pub thinking_budget: usize,
    /// Short-answer guard thresholds (raw entropy in nats; reasoning/content token ratio).
    pub guard_entropy: f64,
    pub guard_think_ratio: f64,
    /// the engine has the vision tower loaded: image inputs are accepted
    pub vision: bool,
    /// required from every API request, as `Authorization: Bearer <key>` or `x-api-key: <key>`; None = open
    pub api_key: Option<String>,
    /// SSE comment interval while nothing else is sent (long prefills)
    pub keepalive: Duration,
    /// `X-DSX-Client` values whose requests are served before the others' queued ones
    pub priority_clients: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_name: "deepseek-v4.1-flash".into(),
            ctx: 32768,
            default_max_tokens: 8192,
            default_temperature: 1.0,
            default_top_p: 1.0,
            default_min_p: 0.0,
            hesitation_penalty: 0.0,
            thinking_budget: 0,
            guard_entropy: 0.389,
            guard_think_ratio: 83.0,
            vision: false,
            api_key: None,
            keepalive: Duration::from_secs(10),
            priority_clients: vec![],
        }
    }
}

pub struct AppState {
    pub engine: EngineHandle,
    pub tok: Arc<Tokenizer>,
    pub cfg: ServerConfig,
    stop_ids: Vec<u32>,
    /// single-token spellings of HESITATION_WORDS, and the reasoning delimiters
    hes_ids: Arc<Vec<u32>>,
    /// ids of the paragraph-break token ("\n\n"): the thinking budget's soft landing favours them
    break_ids: Arc<Vec<u32>>,
    think: Option<(u32, u32)>,
}

/// Words a quantized model over-uses when it second-guesses itself (Lotfi et al., arXiv 2606.00206). Each is
/// penalized in the spellings the tokenizer has as one token: as is, lowercase, with a leading space.
pub const HESITATION_WORDS: [&str; 16] = [
    "Wait", "But", "Alternatively", "Hmm", "However", "Actually", "Hold", "Perhaps", "Maybe", "Oh", "Oops", "Instead",
    "Again", "Recheck", "Double", "Verify",
];

/// A string's id when the tokenizer has it as one token.
fn single_token(tok: &Tokenizer, v: &str) -> Option<u32> {
    tok.token_to_id(v).or_else(|| {
        let e = tok.encode(v, false).ok()?;
        (e.get_ids().len() == 1).then(|| e.get_ids()[0])
    })
}

/// The ids whose logits a hesitation penalty lowers: each word in the spellings the tokenizer has as one token
/// (as is, lowercase, with a leading space).
pub fn word_ids<S: AsRef<str>>(tok: &Tokenizer, words: &[S]) -> Vec<u32> {
    let mut ids: Vec<u32> = vec![];
    for w in words {
        let w = w.as_ref();
        let lw = w.to_lowercase();
        for v in [w.to_string(), lw.clone(), format!(" {w}"), format!(" {lw}")] {
            ids.extend(single_token(tok, &v));
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Largest thinking budget a request may set (tokens).
pub const MAX_BUDGET: usize = 262_144;
/// Smallest budget that is applied (a smaller positive one is raised to this).
pub const MIN_BUDGET: usize = 16;
/// Bonus (logit units) the soft landing reaches on `</think>` and the paragraph break at the end of the budget.
pub const BUDGET_BOOST: f32 = 8.0;

/// Per-request reasoning controls and sampling extras: dsx extension fields of the request body (the protocol parsers
/// ignore unknown fields); a bad value is ignored, as if the field were absent.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Overrides {
    /// `hesitation_penalty`: logit penalty on the hesitation words while the model reasons (0 = none)
    pub penalty: Option<f32>,
    /// `hesitation_words`: replaces the built-in word list (at most 64 words of at most 32 characters)
    pub words: Option<Vec<String>>,
    /// `thinking_budget`: reasoning tokens before the soft landing (0 = no budget)
    pub budget: Option<usize>,
    /// `min_p`: 0..1
    pub min_p: Option<f32>,
}

impl Overrides {
    pub fn parse(body: &[u8]) -> Self {
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else { return Self::default() };
        let num = |k: &str| v.get(k).and_then(|x| x.as_f64()).filter(|x| x.is_finite());
        Self {
            penalty: num("hesitation_penalty").map(|x| x as f32),
            words: v.get("hesitation_words").and_then(|x| x.as_array()).map(|a| {
                a.iter().filter_map(|w| w.as_str()).filter(|w| !w.is_empty() && w.chars().count() <= 32).take(64).map(String::from).collect()
            }),
            budget: num("thinking_budget").filter(|x| *x >= 0.0).map(|x| (x as usize).min(MAX_BUDGET)),
            min_p: num("min_p").map(|x| x.clamp(0.0, 1.0) as f32),
        }
    }
}

impl AppState {
    pub fn new(engine: EngineHandle, tok: Arc<Tokenizer>, mut cfg: ServerConfig) -> anyhow::Result<Arc<Self>> {
        anyhow::ensure!(cfg.guard_entropy.is_finite() && cfg.guard_entropy > 0.0, "--guard-entropy must be finite and > 0");
        anyhow::ensure!(cfg.guard_think_ratio.is_finite() && cfg.guard_think_ratio > 0.0, "--guard-think-ratio must be finite and > 0");
        // an empty --api-key would be a lock whose key is the empty string; treat it as what it is, no lock
        if cfg.api_key.as_deref().is_some_and(|k| k.trim().is_empty()) {
            tracing::warn!("api key: empty, the API is unauthenticated");
            cfg.api_key = None;
        }
        match &cfg.api_key {
            Some(k) => {
                tracing::info!("api key: required on the API routes ({} chars)", k.len());
                if k.trim() != k {
                    tracing::warn!("api key: leading or trailing whitespace, it is compared verbatim");
                }
                if k.len() < 16 {
                    tracing::warn!("api key: {} chars is short; 32 random hex chars is better (openssl rand -hex 32)", k.len());
                }
            }
            None => tracing::info!("api key: none, the API is open to anyone who can reach the port"),
        }
        let stop_ids = STOP_TOKENS
            .iter()
            .map(|t| tok.token_to_id(t).ok_or_else(|| anyhow::anyhow!("tokenizer has no {t} token: not a DeepSeek V4.1 tokenizer?")))
            .collect::<anyhow::Result<_>>()?;
        let hes = word_ids(&tok, &HESITATION_WORDS);
        let breaks: Vec<u32> = single_token(&tok, "\n\n").into_iter().collect();
        let think = tok.token_to_id("<think>").zip(tok.token_to_id("</think>"));
        tracing::info!(tokens = hes.len(), "hesitation penalty: {} tokens, default {}; thinking budget default {}, break tokens {:?}; min-p default {}",
            hes.len(), cfg.hesitation_penalty, cfg.thinking_budget, breaks, cfg.default_min_p);
        Ok(Arc::new(Self { engine, tok, cfg, stop_ids, hes_ids: Arc::new(hes), break_ids: Arc::new(breaks), think }))
    }
}

pub fn router(st: Arc<AppState>) -> Router {
    // everything that runs the model or reports on it: behind the key when one is set
    let api = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/responses", post(responses))
        .route("/v1/models", get(models))
        .route("/props", get(webui::props))
        .route("/slots", get(webui::slots))
        .route("/tools", get(webui::no_tools).post(webui::no_tools))
        .route("/v1/streams/lookup", post(webui::no_streams))
        .route_layer(middleware::from_fn_with_state(st.clone(), auth::require_key));
    // open: health checks, and the UI shell and its assets, which load before the UI can ask for a key
    // (a browser cannot put a header on a <script src>)
    let open = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .fallback(get(webui::asset));
    // request bodies up to 32 MB (base64 images; axum's default is 2 MB)
    api.merge(open).layer(axum::extract::DefaultBodyLimit::max(32 * 1024 * 1024)).with_state(st)
}

pub async fn serve(listener: tokio::net::TcpListener, st: Arc<AppState>) -> std::io::Result<()> {
    if st.cfg.api_key.is_none() {
        if let Some(a) = listener.local_addr().ok().filter(|a| !a.ip().is_loopback()) {
            tracing::warn!("{a} is not loopback and there is no api key: anyone who can reach this port can use the model");
        }
    }
    // ConnectInfo hands auth::require_key the peer address for its rejection log
    let engine = st.engine.clone();
    axum::serve(listener, router(st).into_make_service_with_connect_info::<std::net::SocketAddr>())
        .with_graceful_shutdown(async move {
            stop_signal().await;
            tracing::info!("shutdown: stopping the running request and saving conversations");
            let e = engine.clone();
            let _ = tokio::task::spawn_blocking(move || e.shutdown_and_wait()).await;
            tracing::info!("shutdown: engine stopped");
        })
        .await
}

static SIGNALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

extern "C" fn on_stop_signal(_: libc::c_int) {
    // a second signal while saving: leave now (only async-signal-safe calls here)
    if SIGNALLED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        unsafe { libc::_exit(130) };
    }
}

/// SIGTERM (systemd stop/restart) or SIGINT. Plain libc handlers: the tokio `signal` feature is not vendored here.
async fn stop_signal() {
    unsafe {
        libc::signal(libc::SIGTERM, on_stop_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_stop_signal as *const () as libc::sighandler_t);
    }
    while !SIGNALLED.load(std::sync::atomic::Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ---- errors

#[derive(Clone, Copy, PartialEq)]
enum Proto {
    OpenAi,
    Anthropic,
}

/// Which protocol's error envelope a path expects: Anthropic under /v1/messages, OpenAI everywhere else.
fn proto_for(path: &str) -> Proto {
    if path.starts_with("/v1/messages") { Proto::Anthropic } else { Proto::OpenAi }
}

struct ApiError(Proto, ConversionError);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.1.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = match self.0 {
            Proto::OpenAi => serde_json::to_value(self.1.into_response()).unwrap_or_default(),
            Proto::Anthropic => {
                let kind = if status.is_client_error() { "invalid_request_error" } else { "api_error" };
                json!({"type": "error", "error": {"type": kind, "message": self.1.to_string()}})
            }
        };
        (status, axum::Json(body)).into_response()
    }
}

fn parse<T: DeserializeOwned>(body: &[u8]) -> Result<T, ConversionError> {
    serde_json::from_slice(body).map_err(|e| ConversionError::bad_request(format!("invalid JSON request: {e}")))
}

/// parse, filling in this server's model name when the request has none (llama.cpp's web UI omits it in
/// single-model mode; there is only one model here anyway)
fn parse_with_model<T: DeserializeOwned>(st: &AppState, body: &[u8]) -> Result<T, ConversionError> {
    parse_with_default_model(body, &st.cfg.model_name)
}

fn parse_with_default_model<T: DeserializeOwned>(body: &[u8], model: &str) -> Result<T, ConversionError> {
    let mut v: serde_json::Value = parse(body)?;
    if let Some(o) = v.as_object_mut() {
        if !o.get("model").is_some_and(|m| m.as_str().is_some_and(|s| !s.is_empty())) {
            o.insert("model".into(), json!(model));
        }
    }
    serde_json::from_value(v).map_err(|e| ConversionError::bad_request(format!("invalid JSON request: {e}")))
}

/// llama.cpp's thinking switches on a chat request, mapped to the DeepSeek fields when those are absent:
/// chat_template_kwargs.enable_thinking (llama.cpp's web UI) or a boolean "thinking" -> thinking.type, and
/// chat_template_kwargs.reasoning_effort -> reasoning_effort
fn llama_thinking_compat(body: &[u8]) -> Result<Vec<u8>, ConversionError> {
    let mut v: serde_json::Value = parse(body)?;
    let Some(o) = v.as_object_mut() else { return Ok(body.to_vec()) };
    let kwargs = o.get("chat_template_kwargs").cloned().unwrap_or_default();
    let on = match o.get("thinking") {
        Some(t) => t.as_bool(),
        None => kwargs["enable_thinking"].as_bool(),
    };
    if let Some(on) = on {
        o.insert("thinking".into(), json!({"type": if on { "enabled" } else { "disabled" }}));
    }
    if !o.contains_key("reasoning_effort") {
        if let Some(effort) = kwargs.get("reasoning_effort") {
            o.insert("reasoning_effort".into(), effort.clone());
        }
    }
    // The crate's effort enum can express these numeric budgets exactly. An
    // explicitly disabled thinking mode never renders a budget. Leave other
    // active budgets to its validator rather than silently rounding them.
    let effort = match o.get("reasoning_effort").and_then(|e| e.as_u64()) {
        Some(50) => Some("low"),
        Some(75) => Some("high"),
        Some(100) => Some("max"),
        Some(1..=100) if o.get("thinking").is_some_and(|t| t["type"] == "disabled") => Some("high"),
        _ => None,
    };
    if let Some(effort) = effort {
        o.insert("reasoning_effort".into(), json!(effort));
    }
    Ok(serde_json::to_vec(&v).expect("a JSON value serializes"))
}

/// Chat request conversion shared by the HTTP handler and prompt golden tests.
/// Returns the conversation, seed, and streaming usage setting before tokenization.
pub fn convert_chat_request(body: &[u8], default_model: &str) -> Result<(ConversationRequest, Option<u64>, bool), ConversionError> {
    let body = llama_thinking_compat(body)?;
    let req: ChatCompletionRequest = parse_with_default_model(&body, default_model)?;
    let (seed, include_usage) = (req.seed, req.include_usage());
    Ok((req.convert(ConversionOptions::default())?, seed, include_usage))
}

// ---- routes

/// The request's `X-DSX-Client` label (lowercased), or "default" when it is absent or not a valid label. A fronting
/// proxy for other people must overwrite it: the server cannot tell who set it.
pub fn client_of(headers: &HeaderMap) -> Arc<str> {
    headers
        .get("x-dsx-client")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| sessions::valid_client(s))
        .map_or_else(|| sessions::DEFAULT_CLIENT.into(), Arc::from)
}

async fn chat_completions(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::OpenAi, err);
    let ov = Overrides::parse(&body);
    let uncertainty = uncertainty::Options::parse(&body).map_err(e)?;
    let (conv, seed, include_usage) = convert_chat_request(&body, &st.cfg.model_name).map_err(e)?;
    let streaming = conv.stream;
    run::<ChatCompletionRequest>(st, conv, client_of(&headers), "chatcmpl-", seed, ov, uncertainty, |g| if streaming { g.with_include_usage(include_usage) } else { g })
        .await
        .map_err(e)
}

async fn messages(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::Anthropic, err);
    let ov = Overrides::parse(&body);
    let req: MessagesRequest = parse_with_model(&st, &body).map_err(e)?;
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    run::<MessagesRequest>(st, conv, client_of(&headers), "msg_", None, ov, uncertainty::Options::default(), |g| g).await.map_err(e)
}

async fn responses(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::OpenAi, err);
    let ov = Overrides::parse(&body);
    let req: ResponsesRequest = parse_with_model(&st, &body).map_err(e)?;
    let custom = req.custom_tool_names();
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    run::<ResponsesRequest>(st, conv, client_of(&headers), "resp_", None, ov, uncertainty::Options::default(), move |g| g.with_custom_tool_names(custom)).await.map_err(e)
}

/// Anthropic token counting: the exact prompt length this server would prefill.
async fn count_tokens(State(st): State<Arc<AppState>>, body: Bytes) -> Result<Response, ApiError> {
    let e = |err| ApiError(Proto::Anthropic, err);
    let req: MessagesRequest = parse_with_model(&st, &body).map_err(e)?;
    let conv = req.convert(ConversionOptions::default()).map_err(e)?;
    let (ids, _) = encode(&st, &conv).map_err(e)?;
    Ok(axum::Json(json!({"input_tokens": ids.len()})).into_response())
}

async fn models(State(st): State<Arc<AppState>>) -> Response {
    let created = now();
    axum::Json(json!({
        "object": "list",
        "data": [{"id": st.cfg.model_name, "object": "model", "created": created, "owned_by": "dsx",
                  "display_name": st.cfg.model_name, "type": "model", "context_length": st.cfg.ctx}],
    }))
    .into_response()
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    let alive = st.engine.is_alive();
    (if alive { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE },
        axum::Json(json!({"status": if alive { "ok" } else { "engine_failed" }, "model": st.cfg.model_name, "pending": st.engine.pending()}))).into_response()
}

// ---- the shared request path

type Gen<T> = <<T as ProtocolRequest>::Response as ProtocolResponse>::ChunkGenerator;
type Chunk<T> = <Gen<T> as ChunkGenerator>::Chunk;


/// Render the exact pre-tokenization prompt, using the loaded tokenizer's image token.
pub fn render_prompt(conv: &ConversationRequest, image_token: &str) -> deepseek_recipe_encoding::RenderedPrompt {
    let mut rendered = DeepseekV41Encoding::new().render_conversation(&conv.conversation);
    if !rendered.image_sources.is_empty() {
        rendered.prompt = rendered.prompt.replace(deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN, image_token);
    }
    rendered
}

/// Render and tokenize the conversation (V4.1 template), then expand images.
pub fn encode(st: &AppState, conv: &ConversationRequest) -> Result<(Vec<u32>, Vec<ImageInput>), ConversionError> {
    // Tokenizer revisions spell the image token differently (1017: `<｜deepseek_image｜>`).
    let image_token = st.tok.id_to_token(crate::vision::IMAGE_TOKEN_ID).unwrap_or_else(|| deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN.into());
    let rendered = render_prompt(conv, &image_token);
    if !rendered.image_sources.is_empty() && !st.cfg.vision {
        return Err(ConversionError::bad_request(format!(
            "this server was started without the vision weights (--vision); the request has {} image(s)",
            rendered.image_sources.len()
        )));
    }
    tracing::debug!(prompt = rendered.prompt, "rendered prompt");
    let enc = st.tok.encode(rendered.prompt.as_str(), false).map_err(|e| ConversionError::internal(format!("tokenizer: {e}")))?;
    let bytes = rendered
        .image_sources
        .iter()
        .map(|s| match s {
            ImageSource::DataUrl { data_url, .. } => crate::vision::data_url_bytes(data_url).map_err(|e| ConversionError::bad_request(e.to_string())),
            ImageSource::Bytes { data, .. } => Ok(data.clone()),
            ImageSource::Url { .. } => Err(ConversionError::bad_request("remote image URLs are not fetched by this server; send the image as a data URL (base64)")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    crate::vision::expand(enc.get_ids(), bytes).map_err(|e| ConversionError::bad_request(e.to_string()))
}

async fn run<T>(
    st: Arc<AppState>,
    conv: ConversationRequest,
    client: Arc<str>,
    id_prefix: &str,
    seed: Option<u64>,
    ov: Overrides,
    uncertainty: uncertainty::Options,
    configure: impl FnOnce(Gen<T>) -> Gen<T>,
) -> Result<Response, ConversionError>
where
    T: ProtocolRequest,
    Chunk<T>: Serialize + Send,
{
    let (prompt, images) = encode(&st, &conv)?;
    let cfg = &st.cfg;
    let lambda = ov.penalty.unwrap_or(cfg.hesitation_penalty);
    let budget = ov.budget.unwrap_or(cfg.thinking_budget);
    let budget = if budget == 0 { None } else { Some(budget.max(MIN_BUDGET)) };
    let hesitation = match st.think {
        Some((open, close)) if (lambda > 0.0 || budget.is_some()) && prompt.last() == Some(&open) => {
            let ids = match &ov.words {
                Some(words) => Arc::new(word_ids(&st.tok, words)),
                None => st.hes_ids.clone(),
            };
            let budget = budget.map(|t| ThinkBudget::new(t, BUDGET_BOOST, st.break_ids.clone()));
            Some(Hesitation { lambda, ids, close, budget })
        }
        _ => None,
    };
    if prompt.len() >= cfg.ctx {
        return Err(ConversionError::bad_request(format!(
            "prompt is {} tokens; this server's context window is {} tokens",
            prompt.len(),
            cfg.ctx
        )));
    }
    let io = &conv.inference_options;
    let room = cfg.ctx - prompt.len();
    let params = GenParams {
        max_new: io.max_tokens.map(|m| m as usize).unwrap_or(cfg.default_max_tokens).min(room),
        temperature: io.temperature.unwrap_or(cfg.default_temperature),
        top_p: io.top_p.unwrap_or(cfg.default_top_p),
        min_p: ov.min_p.unwrap_or(cfg.default_min_p),
        seed,
        logprobs: uncertainty.k(),
        think: st.think,
        stop_token_ids: st.stop_ids.clone(),
        hesitation,
        images: Arc::new(images),
        priority: cfg.priority_clients.iter().any(|c| **c == *client),
        client,
    };
    let id = engine::request_id(id_prefix);
    let model = conv.model.clone().filter(|m| !m.is_empty()).unwrap_or_else(|| cfg.model_name.clone());
    let c = &conv.conversation;
    tracing::info!(
        %id, %model, client = &*params.client, priority = params.priority, stream = conv.stream, thinking = c.thinking_mode, effort = ?c.reasoning_effort,
        messages = c.messages.len(), tools = c.tools.len(), temperature = params.temperature, top_p = params.top_p, min_p = params.min_p, hesitation = params.hesitation.as_ref().map_or(0.0, |h| h.lambda),
        words = params.hesitation.as_ref().map_or(0, |h| h.ids.len()), budget = params.hesitation.as_ref().and_then(|h| h.budget.as_ref()).map_or(0, |b| b.tokens),
        "request"
    );
    let generator = configure(T::chunk_generator(&conv, id.clone(), model.clone()));
    let processor = StreamProcessor::new(generator, conv.parsing_options).with_tokenizer(st.tok.clone());
    let guard = uncertainty.guard().map(|options| guard::Guard::new(options, cfg, &conv.conversation.messages, conv.stream));
    let capture = uncertainty.enabled().then(|| Arc::new(Mutex::new(uncertainty::Capture::new(uncertainty, st.think, &prompt, st.tok.clone(), guard))));
    let n_prompt = prompt.len();
    let rx = st.engine.submit(id.clone(), prompt, params).map_err(|e| ConversionError::internal(e.to_string()))?;
    let failed = Arc::new(Mutex::new(None::<String>));
    let timing = Arc::new(Mutex::new(Timing::new()));
    let inference = inference_stream(rx, n_prompt, failed.clone(), timing.clone(), capture.clone());

    if !conv.stream {
        let mut resp = T::Response::new(id, model, now(), n_prompt, 0);
        let mut chunks = pin!(processor.process(inference));
        while let Some(chunk) = chunks.next().await {
            resp.append(chunk.map_err(|e| ConversionError::internal(e.to_string()))?);
        }
        if let Some(err) = failed.lock().unwrap().take() {
            return Err(ConversionError::internal(format!("engine error: {err}")));
        }
        if let Some(capture) = &capture {
            let mut json = serde_json::to_value(resp).map_err(|e| ConversionError::internal(e.to_string()))?;
            capture.lock().unwrap().attach(&mut json, true);
            return Ok(axum::Json(json).into_response());
        }
        return Ok(axum::Json(resp).into_response());
    }

    let events = processor.process(inference).map(move |chunk| {
        let chunk = chunk.map_err(|e| std::io::Error::other(e.to_string()))?;
        let data = if let Some(capture) = &capture {
            let mut json = serde_json::to_value(&chunk).map_err(std::io::Error::other)?;
            // Recipe always emits final usage, even when include_usage is false.
            let final_chunk = json["usage"].is_object();
            capture.lock().unwrap().attach(&mut json, final_chunk);
            serde_json::to_string(&json).map_err(std::io::Error::other)?
        } else { serde_json::to_string(&chunk).map_err(std::io::Error::other)? };
        Ok::<_, std::io::Error>(match <T::Response as ProtocolResponse>::chunk_event_type(&chunk) {
            Some(ev) => format!("event: {ev}\ndata: {data}\n\n"),
            None => format!("data: {data}\n\n"),
        })
    });
    // llama.cpp extension for chat completions: a last chunk with `timings` (its web UI shows tok/s from it)
    let llama_timings = id_prefix == "chatcmpl-";
    let (tid, tmodel) = (id.clone(), model.clone());
    let timings = tokio_stream::iter([()]).filter_map(move |_| {
        llama_timings.then(|| {
            let t = timing.lock().unwrap().json(n_prompt);
            Ok(format!("data: {}\n\n", json!({"id": tid, "object": "chat.completion.chunk", "model": tmodel, "choices": [], "timings": t})))
        })
    });
    let done = <T::Response as ProtocolResponse>::done_message().map(|m| Ok(format!("data: {m}\n\n")));
    let events = events.chain(timings).chain(tokio_stream::iter(done));
    // SSE comments keep idle connections (long prefill, slow decode) alive; parsers ignore them
    let events = events.timeout(cfg.keepalive).map(|r| r.unwrap_or_else(|_| Ok(": keep-alive\n\n".to_string())));
    let mut res = Response::new(Body::from_stream(events));
    let h = res.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream; charset=utf-8"));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    Ok(res)
}

/// Engine events -> recipe inference chunks. Dropping the stream drops the receiver, which cancels generation.
fn inference_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    n_prompt: usize,
    failed: Arc<Mutex<Option<String>>>,
    timing: Arc<Mutex<Timing>>,
    capture: Option<Arc<Mutex<uncertainty::Capture>>>,
) -> Pin<Box<dyn Stream<Item = InferenceChunk> + Send>> {
    let ready = InferenceChunk::Ready {
        system_fingerprint: Some("dsx".into()),
        prompt_usage: PromptUsage { prompt_tokens: n_prompt, prompt_cache_hit_tokens: 0 },
    };
    let rest = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map_while(move |ev| match ev {
        Event::TokenWithStats(t, stats) => {
            timing.lock().unwrap().token();
            if let Some(capture) = &capture { capture.lock().unwrap().token(t, &stats); }
            Some(InferenceChunk::Token { token_id: t })
        }
        Event::Token(t) => {
            timing.lock().unwrap().token();
            Some(InferenceChunk::Token { token_id: t })
        }
        Event::Done { finish: Finish::Stop } => Some(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop }),
        Event::Done { finish: Finish::Length } => Some(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length }),
        Event::Done { finish: Finish::Cancelled } => None,
        Event::Error(e) => {
            *failed.lock().unwrap() = Some(e);
            None
        }
    });
    Box::pin(tokio_stream::iter([ready]).chain(rest))
}

/// Request timing as seen by the HTTP layer: prompt = submit -> first token, predicted = first -> last token.
struct Timing {
    start: std::time::Instant,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
    n: usize,
}

impl Timing {
    fn new() -> Self {
        Self { start: std::time::Instant::now(), first: None, last: None, n: 0 }
    }
    fn token(&mut self) {
        let t = std::time::Instant::now();
        self.first.get_or_insert(t);
        self.last = Some(t);
        self.n += 1;
    }
    /// llama.cpp's `timings` object
    fn json(&self, n_prompt: usize) -> serde_json::Value {
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        let prompt_ms = self.first.map_or(0.0, |f| ms(self.start, f));
        // the first token comes with the prefill; decode speed is over the tokens after it
        let predicted_ms = match (self.first, self.last) { (Some(f), Some(l)) => ms(f, l), _ => 0.0 };
        let dn = self.n.saturating_sub(1);
        json!({
            "prompt_n": n_prompt, "prompt_ms": prompt_ms,
            "prompt_per_second": if prompt_ms > 0.0 { n_prompt as f64 * 1e3 / prompt_ms } else { 0.0 },
            "predicted_n": self.n, "predicted_ms": predicted_ms,
            "predicted_per_second": if predicted_ms > 0.0 { dn as f64 * 1e3 / predicted_ms } else { 0.0 },
        })
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod override_tests {
    use super::*;

    #[test]
    fn request_fields_are_parsed_and_clamped() {
        let o = Overrides::parse(br#"{"hesitation_penalty": 1.5, "hesitation_words": ["Maybe", "But", "", 7], "thinking_budget": 800, "min_p": 0.05, "model": "x"}"#);
        assert_eq!(o, Overrides { penalty: Some(1.5), words: Some(vec!["Maybe".into(), "But".into()]), budget: Some(800), min_p: Some(0.05) });
        // out-of-range values are clamped, absurd or wrong-typed ones are ignored
        let o = Overrides::parse(br#"{"thinking_budget": 99999999999, "min_p": 3, "hesitation_penalty": "lots", "hesitation_words": "Maybe"}"#);
        assert_eq!(o, Overrides { penalty: None, words: None, budget: Some(MAX_BUDGET), min_p: Some(1.0) });
        let o = Overrides::parse(br#"{"thinking_budget": -5, "min_p": -1}"#);
        assert_eq!(o, Overrides { penalty: None, words: None, budget: None, min_p: Some(0.0) });
        // not JSON, an array, or nothing: no overrides, and no panic
        for body in [&b"not json"[..], b"[1,2]", b"", b"{}"] {
            assert_eq!(Overrides::parse(body), Overrides::default());
        }
    }

    #[test]
    fn word_list_is_bounded() {
        let words: Vec<String> = (0..100).map(|i| format!("w{i}")).collect();
        let body = serde_json::json!({ "hesitation_words": words, "hesitation_penalty": 1.0 }).to_string();
        assert_eq!(Overrides::parse(body.as_bytes()).words.unwrap().len(), 64);
        let long = serde_json::json!({ "hesitation_words": ["x".repeat(33), "ok"] }).to_string();
        assert_eq!(Overrides::parse(long.as_bytes()).words.unwrap(), vec!["ok".to_string()]);
    }

    fn tokenizer() -> Tokenizer {
        // a tiny word-level vocabulary; a word it lacks is an encoding error (the unknown token is not in the vocabulary)
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,
            "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"Maybe":0," maybe":1,"But":2,"but":3,"\n\n":4,"</think>":5,"cat":6},"unk_token":"<missing>"}}"#;
        json.parse::<Tokenizer>().unwrap()
    }

    #[test]
    fn word_ids_take_the_single_token_spellings_of_each_word() {
        let tok = tokenizer();
        assert_eq!(word_ids(&tok, &["Maybe", "But"]), vec![0, 1, 2, 3]);
        assert_eq!(word_ids(&tok, &["Absent"]), Vec::<u32>::new(), "a word the vocabulary lacks adds nothing");
        assert_eq!(word_ids(&tok, &[String::from("But")]), vec![2, 3], "owned strings work too; duplicates collapse");
        assert_eq!(single_token(&tok, "\n\n"), Some(4));
        assert_eq!(single_token(&tok, "</think>"), Some(5));
        assert_eq!(single_token(&tok, "a b"), None);
    }
}
