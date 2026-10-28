//! HTTP API end to end against the mock engine: OpenAI chat, Anthropic messages, Responses, tool calls, limits.

use std::io::Read;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::Request;
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use dsx_core::server::mock::{MOCK_REASONING, MOCK_REPLY, MockEngine, MockMode};
use dsx_core::server::{AppState, Engine, EngineHandle, ServerConfig};
use serde_json::{Value, json};
use tokenizers::Tokenizer;
use tower::ServiceExt;

/// The bundled V4.1 tokenizer (MIT, DeepSeek; see tests/data/LICENSE-tokenizer).
fn tokenizer() -> Arc<Tokenizer> {
    static TOK: OnceLock<Arc<Tokenizer>> = OnceLock::new();
    TOK.get_or_init(|| {
        let gz = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tokenizer-v41.json.gz")).unwrap();
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut json).unwrap();
        Arc::new(Tokenizer::from_bytes(json).unwrap())
    })
    .clone()
}

async fn start(mode: MockMode, delay_ms: u64) -> Router {
    start_cfg(mode, delay_ms, ServerConfig::default()).await
}

async fn start_cfg(mode: MockMode, delay_ms: u64, cfg: ServerConfig) -> Router {
    let tok = tokenizer();
    let t2 = tok.clone();
    let engine = EngineHandle::spawn(move || {
        let mut m = MockEngine::new(t2, mode);
        m.delay = Duration::from_millis(delay_ms);
        Ok(Box::new(m) as Box<dyn Engine>)
    })
    .unwrap();
    let st = AppState::new(engine, tok, cfg).unwrap();
    dsx_core::server::router(st)
}

async fn request(base: &Router, method: &str, path: &str, body: Value, headers: &[(&str, &str)]) -> axum::response::Response {
    let mut req = Request::builder().method(method).uri(path).header("content-type", "application/json");
    for (k, v) in headers { req = req.header(*k, *v); }
    base.clone().oneshot(req.body(Body::from(body.to_string())).unwrap()).await.unwrap()
}

async fn post(base: &Router, path: &str, body: Value) -> (u16, String) {
    post_h(base, path, body, &[]).await
}

async fn post_h(base: &Router, path: &str, body: Value, headers: &[(&str, &str)]) -> (u16, String) {
    let r = request(base, "POST", path, body, headers).await;
    (r.status().as_u16(), String::from_utf8(to_bytes(r.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap())
}

async fn get(base: &Router, path: &str) -> u16 {
    request(base, "GET", path, Value::Null, &[]).await.status().as_u16()
}

async fn get_json(base: &Router, path: &str) -> Value {
    let r = request(base, "GET", path, Value::Null, &[]).await;
    serde_json::from_slice(&to_bytes(r.into_body(), usize::MAX).await.unwrap()).unwrap()
}

async fn post_json(base: &Router, path: &str, body: Value) -> Value {
    let (status, text) = post(base, path, body).await;
    assert_eq!(status, 200, "{text}");
    serde_json::from_str(&text).unwrap()
}

/// SSE body -> (event name, data JSON) pairs; `[DONE]` becomes a null data.
fn sse(text: &str) -> Vec<(String, Value)> {
    text.split("\n\n")
        .filter(|f| !f.trim().is_empty() && !f.starts_with(':'))
        .map(|f| {
            let mut ev = String::new();
            let mut data = Value::Null;
            for line in f.lines() {
                if let Some(e) = line.strip_prefix("event: ") {
                    ev = e.into();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = if d == "[DONE]" { Value::Null } else { serde_json::from_str(d).unwrap() };
                }
            }
            (ev, data)
        })
        .collect()
}

fn weather_tool_openai() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
}

fn weather_tool_anthropic() -> Value {
    json!([{"name": "get_weather", "description": "Weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}])
}

fn chat_prompt_tokens(body: &Value) -> usize {
    let conv = serde_json::from_value::<ChatCompletionRequest>(body.clone()).unwrap().convert(ConversionOptions::default()).unwrap();
    DeepseekV41Encoding::new().with_tokenizer(tokenizer()).encode(&conv.conversation).unwrap().len()
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_chat_nonstream() {
    let base = start(MockMode::Scripted, 0).await;
    let body = json!({"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Hello"}]});
    let r = post_json(&base, "/v1/chat/completions", body.clone()).await;
    let m = &r["choices"][0]["message"];
    assert_eq!(m["content"], MOCK_REPLY);
    assert_eq!(m["reasoning_content"], MOCK_REASONING);
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert_eq!(r["usage"]["prompt_tokens"], chat_prompt_tokens(&body));
    assert!(r["usage"]["completion_tokens"].as_u64().unwrap() > 5);
    // thinking disabled: no reasoning, the prompt ends in </think>
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["choices"][0]["message"]["content"], MOCK_REPLY);
    assert!(r["choices"][0]["message"]["reasoning_content"].is_null());
    // llama.cpp's switches (its web UI sends chat_template_kwargs.enable_thinking) do the same
    for off in [json!({"chat_template_kwargs": {"enable_thinking": false}}), json!({"thinking": false}),
                json!({"chat_template_kwargs": {"reasoning_effort": "none"}})] {
        let mut b = json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}]});
        b.as_object_mut().unwrap().extend(off.as_object().unwrap().clone());
        let r = post_json(&base, "/v1/chat/completions", b).await;
        assert_eq!(r["choices"][0]["message"]["content"], MOCK_REPLY, "{off}");
        assert!(r["choices"][0]["message"]["reasoning_content"].is_null(), "{off}");
    }
    // an explicit thinking object wins over chat_template_kwargs
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "thinking": {"type": "enabled"},
        "chat_template_kwargs": {"enable_thinking": false}, "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["choices"][0]["message"]["reasoning_content"], MOCK_REASONING);
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_chat_stream() {
    let base = start(MockMode::Scripted, 0).await;
    let (status, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "stream": true,
        "stream_options": {"include_usage": true}, "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(status, 200);
    let ev = sse(&text);
    assert!(ev.last().unwrap().1.is_null(), "ends with [DONE]");
    let (mut content, mut reasoning, mut finish, mut usage) = (String::new(), String::new(), None, None);
    for (_, d) in &ev {
        if let Some(c) = d["choices"].get(0) {
            content += c["delta"]["content"].as_str().unwrap_or("");
            reasoning += c["delta"]["reasoning_content"].as_str().unwrap_or("");
            if let Some(f) = c["finish_reason"].as_str() {
                finish = Some(f.to_string());
            }
        }
        if d["usage"].is_object() {
            usage = Some(d["usage"].clone());
        }
    }
    assert_eq!(content, MOCK_REPLY);
    assert_eq!(reasoning, MOCK_REASONING);
    assert_eq!(finish.as_deref(), Some("stop"));
    assert!(usage.unwrap()["completion_tokens"].as_u64().unwrap() > 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_messages_nonstream() {
    let base = start(MockMode::Scripted, 0).await;
    let r = post_json(&base, "/v1/messages", json!({"model": "m", "max_tokens": 256,
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["type"], "message");
    assert_eq!(r["content"][0]["type"], "thinking");
    assert_eq!(r["content"][0]["thinking"], MOCK_REASONING);
    assert_eq!(r["content"][1]["type"], "text");
    assert_eq!(r["content"][1]["text"], MOCK_REPLY);
    assert_eq!(r["stop_reason"], "end_turn");
    assert!(r["usage"]["input_tokens"].as_u64().unwrap() > 5);
    // errors use Anthropic's envelope
    let (status, text) = post(&base, "/v1/messages", json!({"model": "m"})).await;
    assert_eq!(status, 400);
    let e: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(e["type"], "error");
    assert_eq!(e["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_messages_stream() {
    let base = start(MockMode::Scripted, 0).await;
    let (status, text) = post(&base, "/v1/messages", json!({"model": "m", "max_tokens": 256, "stream": true,
        "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(status, 200);
    let ev = sse(&text);
    let names: Vec<&str> = ev.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));
    assert!(names.contains(&"content_block_start") && names.contains(&"message_delta"));
    let text: String = ev
        .iter()
        .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "text_delta")
        .map(|(_, d)| d["delta"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(text, MOCK_REPLY);
    let delta = ev.iter().find(|(e, _)| e == "message_delta").unwrap();
    assert_eq!(delta.1["delta"]["stop_reason"], "end_turn");
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_tool_call_round_trip() {
    let base = start(MockMode::Scripted, 0).await;
    let user = json!({"role": "user", "content": "Weather in Paris?"});
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "tools": weather_tool_openai(), "messages": [user]})).await;
    let c = &r["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls");
    let call = &c["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_weather");
    let args: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "mock"}));
    // send the result back with the previous turn (thinking included) as history
    let mut assistant = c["message"].clone();
    assistant.as_object_mut().unwrap().retain(|k, _| ["role", "content", "reasoning_content", "tool_calls"].contains(&k.as_str()));
    let tool = json!({"role": "tool", "tool_call_id": call["id"], "content": "Sunny, 21C"});
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "tools": weather_tool_openai(),
        "messages": [user, assistant, tool]})).await;
    assert_eq!(r["choices"][0]["message"]["content"], "The tool returned: Sunny, 21C");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");

    // streamed tool call: name and argument deltas reassemble
    let (_, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "stream": true, "tools": weather_tool_openai(),
        "messages": [{"role": "user", "content": "Weather?"}]})).await;
    let (mut name, mut args, mut finish) = (String::new(), String::new(), String::new());
    for (_, d) in sse(&text) {
        if let Some(c) = d["choices"].get(0) {
            if let Some(tc) = c["delta"]["tool_calls"].get(0) {
                name += tc["function"]["name"].as_str().unwrap_or("");
                args += tc["function"]["arguments"].as_str().unwrap_or("");
            }
            if let Some(f) = c["finish_reason"].as_str() {
                finish = f.into();
            }
        }
    }
    assert_eq!(name, "get_weather");
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"city": "mock"}));
    assert_eq!(finish, "tool_calls");
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_tool_use_round_trip() {
    let base = start(MockMode::Scripted, 0).await;
    let user = json!({"role": "user", "content": "Weather in Paris?"});
    let r = post_json(&base, "/v1/messages", json!({"model": "m", "max_tokens": 512, "tools": weather_tool_anthropic(),
        "messages": [user]})).await;
    assert_eq!(r["stop_reason"], "tool_use");
    let blocks = r["content"].as_array().unwrap();
    let tu = blocks.iter().find(|b| b["type"] == "tool_use").unwrap();
    assert_eq!(tu["name"], "get_weather");
    assert_eq!(tu["input"], json!({"city": "mock"}));
    let result = json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu["id"], "content": "Rain"}]});
    let r = post_json(&base, "/v1/messages", json!({"model": "m", "max_tokens": 512, "tools": weather_tool_anthropic(),
        "messages": [user, {"role": "assistant", "content": r["content"]}, result]})).await;
    let text = r["content"].as_array().unwrap().iter().find(|b| b["type"] == "text").unwrap();
    assert_eq!(text["text"], "The tool returned: Rain");
    assert_eq!(r["stop_reason"], "end_turn");
}

#[tokio::test(flavor = "multi_thread")]
async fn responses_api() {
    let base = start(MockMode::Scripted, 0).await;
    let r = post_json(&base, "/v1/responses", json!({"model": "m", "input": "Hello"})).await;
    assert_eq!(r["status"], "completed");
    let out = r["output"].as_array().unwrap();
    let msg = out.iter().find(|o| o["type"] == "message").unwrap();
    assert_eq!(msg["content"][0]["text"], MOCK_REPLY);
    let (status, text) = post(&base, "/v1/responses", json!({"model": "m", "input": "Hello", "stream": true})).await;
    assert_eq!(status, 200);
    let ev = sse(&text);
    assert!(ev.iter().any(|(e, _)| e == "response.output_text.delta"));
    assert_eq!(ev.last().unwrap().0, "response.completed");
}

#[tokio::test(flavor = "multi_thread")]
async fn long_prompt_token_count() {
    let base = start(MockMode::Scripted, 0).await;
    let long: String = (0..6000).map(|i| format!("word{} ", i % 97)).collect();
    let body = json!({"model": "m", "max_tokens": 64, "messages": [{"role": "system", "content": "Be terse."},
        {"role": "user", "content": long}]});
    let r = post_json(&base, "/v1/chat/completions", body.clone()).await;
    let n = r["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
    assert_eq!(n, chat_prompt_tokens(&body));
    assert!(n > 12_000 && n < 40_000, "{n}");
    // Anthropic count_tokens agrees with what /v1/messages prefills
    let mb = json!({"model": "m", "max_tokens": 64, "system": "Be terse.", "messages": [{"role": "user", "content": long}]});
    let c = post_json(&base, "/v1/messages/count_tokens", mb.clone()).await["input_tokens"].as_u64().unwrap();
    let conv = serde_json::from_value::<MessagesRequest>(mb.clone()).unwrap().convert(ConversionOptions::default()).unwrap();
    assert_eq!(c as usize, DeepseekV41Encoding::new().with_tokenizer(tokenizer()).encode(&conv.conversation).unwrap().len());
    let m = post_json(&base, "/v1/messages", mb).await;
    assert_eq!(m["usage"]["input_tokens"].as_u64().unwrap(), c);
    // over the context window: 400
    let huge: String = (0..40_000).map(|i| format!("w{i} ")).collect();
    let (status, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "messages": [{"role": "user", "content": huge}]})).await;
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("context window"));
}

#[tokio::test(flavor = "multi_thread")]
async fn limits_and_stops() {
    let base = start(MockMode::Scripted, 0).await;
    let off = json!({"type": "disabled"});
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "thinking": off, "max_tokens": 3,
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["choices"][0]["finish_reason"], "length");
    assert_eq!(r["usage"]["completion_tokens"], 3);
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "thinking": off, "stop": [" mock"],
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["choices"][0]["message"]["content"], "Hello from the dsx");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    let r = post_json(&base, "/v1/messages", json!({"model": "m", "max_tokens": 100, "thinking": off, "stop_sequences": [" mock"],
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(r["stop_reason"], "stop_sequence");
    assert_eq!(r["stop_sequence"], " mock");
    // images need the vision weights: rejected up front by a server without them
    let (status, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "messages": [{"role": "user",
        "content": [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]})).await;
    assert_eq!(status, 400);
    assert!(text.contains("vision weights"), "{text}");
}

#[tokio::test(flavor = "multi_thread")]
async fn disconnect_cancels_generation() {
    // echo a ~300-token prompt at 20 ms/token: ~6 s if nobody cancels
    let base = start(MockMode::Echo, 20).await;
    let long: String = (0..250).map(|i| format!("t{i} ")).collect();
    let mut r = request(&base, "POST", "/v1/chat/completions",
        json!({"model": "m", "stream": true, "messages": [{"role": "user", "content": long}]}), &[]).await.into_body().into_data_stream();
    use tokio_stream::StreamExt;
    assert!(r.next().await.unwrap().is_ok());
    let health = || get_json(&base, "/health");
    assert_eq!(health().await["pending"], 1);
    drop(r);
    let t = Instant::now();
    while health().await["pending"] != 0 {
        assert!(t.elapsed() < Duration::from_secs(2), "generation was not cancelled");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let models: Value = get_json(&base, "/v1/models").await;
    assert_eq!(models["data"][0]["id"], "deepseek-v4.1-flash");
}

// ---- api key

const KEY: &str = "sk-dsx-test-0123456789abcdef";

fn keyed() -> ServerConfig {
    ServerConfig { api_key: Some(KEY.into()), ..ServerConfig::default() }
}

async fn chat_status(base: &Router, headers: &[(&str, &str)]) -> (u16, String) {
    let body = json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}]});
    post_h(base, "/v1/chat/completions", body, headers).await
}

#[tokio::test(flavor = "multi_thread")]
async fn api_key_guards_the_api() {
    let base = start_cfg(MockMode::Scripted, 0, keyed()).await;
    let bearer = format!("Bearer {KEY}");

    // no credentials: 401 in the OpenAI envelope, and no generation started
    let (status, text) = chat_status(&base, &[]).await;
    assert_eq!(status, 401, "{text}");
    let e: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(e["error"]["type"], "invalid_request_error");
    assert_eq!(e["error"]["code"], "invalid_api_key");
    let health: Value = get_json(&base, "/health").await;
    assert_eq!(health["pending"], 0);

    // either header, either protocol
    assert_eq!(chat_status(&base, &[("authorization", &bearer)]).await.0, 200);
    assert_eq!(chat_status(&base, &[("x-api-key", KEY)]).await.0, 200);
    let msgs = json!({"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "Hello"}]});
    assert_eq!(post_h(&base, "/v1/messages", msgs.clone(), &[("x-api-key", KEY)]).await.0, 200);
    let lower = format!("bearer {KEY}");
    assert_eq!(chat_status(&base, &[("authorization", &lower)]).await.0, 200);

    // wrong key, a prefix of the key, and a bare key without the scheme are all rejected
    assert_eq!(chat_status(&base, &[("authorization", "Bearer nope")]).await.0, 401);
    assert_eq!(chat_status(&base, &[("x-api-key", &KEY[..8])]).await.0, 401);
    assert_eq!(chat_status(&base, &[("authorization", KEY)]).await.0, 401);
    assert_eq!(chat_status(&base, &[("x-api-key", "")]).await.0, 401);

    // Anthropic's error envelope on /v1/messages
    let (status, text) = post(&base, "/v1/messages", msgs).await;
    assert_eq!(status, 401);
    let e: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(e["type"], "error");
    assert_eq!(e["error"]["type"], "authentication_error");

    // the other API surfaces are guarded too
    for (path, body) in [
        ("/v1/responses", json!({"model": "m", "input": "hi"})),
        ("/v1/messages/count_tokens", json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})),
        ("/v1/streams/lookup", json!({})),
        ("/tools", json!({})),
    ] {
        let (status, text) = post(&base, path, body).await;
        assert_eq!(status, 401, "{path}: {text}");
    }
    for path in ["/v1/models", "/props", "/slots", "/tools"] {
        assert_eq!(get(&base, path).await, 401, "{path}");
    }

    // health stays open (tunnel health checks) and the UI shell loads so its key field is reachable
    assert_eq!(get(&base, "/health").await, 200);
    assert_eq!(get(&base, "/v1/health").await, 200);
    assert_eq!(get(&base, "/").await, 200);
    assert_eq!(get(&base, "/_app/immutable/entry/start.js").await, 404, "assets need no key, missing ones are just missing");
}

#[tokio::test(flavor = "multi_thread")]
async fn no_api_key_leaves_the_api_open() {
    // the default: no key configured, every request is served (as before the flag existed)
    let base = start(MockMode::Scripted, 0).await;
    assert_eq!(chat_status(&base, &[]).await.0, 200);
    assert_eq!(chat_status(&base, &[("authorization", "Bearer whatever")]).await.0, 200);
    // an empty --api-key is not a lock with no key, it is no lock at all
    let base = start_cfg(MockMode::Scripted, 0, ServerConfig { api_key: Some("  ".into()), ..ServerConfig::default() }).await;
    assert_eq!(chat_status(&base, &[]).await.0, 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn large_request_bodies_are_accepted() {
    // base64 images make multi-MB requests; axum's default 2 MB body limit used to reject them with 413
    let base = start(MockMode::Scripted, 0).await;
    let big = "a ".repeat(4 * 1024 * 1024); // 8 MB of text
    let body = json!({"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": big}]});
    let (status, text) = post(&base, "/v1/chat/completions", body).await;
    assert_ne!(status, 413, "{}", &text[..text.len().min(200)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_unchanged_response() {
    let base = start(MockMode::Scripted, 0).await;
    let (status, text) = post(&base, "/v1/chat/completions", json!({"model": "m",
        "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert_eq!(status, 200, "{text}");
    let mut response: Value = serde_json::from_str(&text).unwrap();
    // Preserve the wire representation; normalize only the two nondeterministic fields.
    let normalized = text.replacen(&response["id"].to_string(), "\"chatcmpl-baseline\"", 1)
        .replacen(&format!("\"created\":{}", response["created"]), "\"created\":0", 1);
    assert_eq!(normalized.as_bytes(), include_str!("data/chat-baseline.json").trim().as_bytes());
    response["id"] = json!("chatcmpl-baseline");
    response["created"] = json!(0);
    assert_eq!(response, serde_json::from_str::<Value>(include_str!("data/chat-baseline.json")).unwrap());
}

fn entries<'a>(value: &'a Value, channel: &str) -> &'a Vec<Value> {
    value["choices"][0]["logprobs"][channel].as_array().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_uncertainty_nonstream() {
    let base = start(MockMode::Scripted, 0).await;
    let request = json!({"model": "m", "logprobs": true, "top_logprobs": 20,
        "messages": [{"role": "user", "content": "Hello"}]});
    let r = post_json(&base, "/v1/chat/completions", request).await;
    let reasoning = entries(&r, "reasoning_content");
    let content = entries(&r, "content");
    let ids = |text: &str| tokenizer().encode(text, false).unwrap().get_ids().len();
    assert_eq!(reasoning.len(), ids(&format!("{MOCK_REASONING}</think>")));
    assert_eq!(content.len(), ids(MOCK_REPLY));
    assert_eq!(reasoning.len() + content.len(), r["usage"]["completion_tokens"].as_u64().unwrap() as usize);
    assert_eq!(reasoning.last().unwrap()["token"], "</think>");
    assert_eq!(content.iter().map(|e| e["token"].as_str().unwrap()).collect::<String>(), MOCK_REPLY);
    for entry in reasoning.iter().chain(content) {
        let bytes: Vec<u8> = serde_json::from_value(entry["bytes"].clone()).unwrap();
        assert_eq!(entry["token"], String::from_utf8_lossy(&bytes).as_ref());
        assert!(entry["logprob"].as_f64().unwrap() < 0.0);
        assert_eq!(entry["top_logprobs"].as_array().unwrap().len(), 20);
        assert!(entry["top_logprobs"][0]["bytes"].is_array());
        assert_ne!(entry["token"], "<｜end▁of▁sentence｜>");
    }
    for (name, n) in [("reasoning", reasoning.len()), ("content", content.len())] {
        let summary = &r["x_uncertainty"][name];
        assert_eq!(summary["tokens"], n);
        assert_eq!(summary["low_margin_tokens"], n);
        assert_eq!(summary["low_prob_tokens"], 0);
        let logprob = content[0]["logprob"].as_f64().unwrap();
        assert!((summary["mean_logprob"].as_f64().unwrap() - logprob).abs() < 1e-6);
        assert_eq!(summary["min_logprob"], content[0]["logprob"]);
        assert!(summary["mean_entropy"].as_f64().unwrap() > 1.0);
        assert_eq!(summary["mean_entropy"], summary["max_entropy"]);
    }
    // Summary only: identical text, no per-token payload and no alternatives requested.
    let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "x_uncertainty": true,
        "thinking": false, "messages": [{"role": "user", "content": "Hello"}]})).await;
    assert!(r["choices"][0]["logprobs"].is_null());
    assert_eq!(r["choices"][0]["message"]["content"], MOCK_REPLY);
    assert_eq!(r["x_uncertainty"]["reasoning"]["tokens"], 0);
    assert!(r["x_uncertainty"]["reasoning"]["min_logprob"].is_null());
    assert_eq!(r["x_uncertainty"]["content"]["tokens"], ids(MOCK_REPLY));
    for k in [0, 3] {
        let r = post_json(&base, "/v1/chat/completions", json!({"model": "m", "logprobs": true,
            "top_logprobs": k, "max_tokens": 3, "messages": [{"role": "user", "content": "Hello"}]})).await;
        assert_eq!(entries(&r, "reasoning_content").len(), 3);
        assert_eq!(entries(&r, "reasoning_content")[0]["top_logprobs"].as_array().unwrap().len(), k);
        assert_eq!(r["x_uncertainty"]["reasoning"]["tokens"], 3);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_uncertainty_stream() {
    let base = start(MockMode::Scripted, 0).await;
    for tool in [false, true] {
        let mut body = json!({"model": "m", "logprobs": true, "top_logprobs": 3,
            "messages": [{"role": "user", "content": "Hello"}]});
        if tool { body["tools"] = weather_tool_openai(); }
        let expected = post_json(&base, "/v1/chat/completions", body.clone()).await;
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
        let (status, text) = post(&base, "/v1/chat/completions", body).await;
        assert_eq!(status, 200, "{text}");
        let (mut reasoning, mut content, mut summary) = (Vec::new(), Vec::new(), None);
        for (_, chunk) in sse(&text) {
            if chunk["choices"][0]["logprobs"].is_object() {
                reasoning.extend(entries(&chunk, "reasoning_content").clone());
                content.extend(entries(&chunk, "content").clone());
            }
            if chunk.get("x_uncertainty").is_some() {
                assert!(summary.is_none(), "summary sent only once");
                assert!(chunk["usage"].is_object(), "summary accompanies final usage");
                summary = Some(chunk["x_uncertainty"].clone());
            }
        }
        assert_eq!(reasoning, *entries(&expected, "reasoning_content"));
        assert_eq!(content, *entries(&expected, "content"));
        assert_eq!(summary.unwrap(), expected["x_uncertainty"]);
        if tool {
            let text: String = content.iter().map(|e| e["token"].as_str().unwrap()).collect();
            assert!(text.contains("DSML"), "tool markup is content");
        }
    }
    for usage in [None, Some(false)] {
        let mut body = json!({"model": "m", "stream": true, "x_uncertainty": true, "thinking": false,
            "messages": [{"role": "user", "content": "Hello"}]});
        if let Some(usage) = usage { body["stream_options"] = json!({"include_usage": usage}); }
        let (_, text) = post(&base, "/v1/chat/completions", body).await;
        let chunks = sse(&text);
        let final_chunks: Vec<_> = chunks.iter().filter(|(_, c)| c["usage"].is_object()).collect();
        assert_eq!(final_chunks.len(), 1);
        assert!(final_chunks[0].1["x_uncertainty"].is_object());
        assert!(chunks.iter().all(|(_, c)| !c["choices"][0]["logprobs"].is_object() && c["x_guard"].is_null()));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_uncertainty_validation() {
    let base = start(MockMode::Scripted, 0).await;
    for k in [json!(-1), json!(21), json!(256), json!(1.5), json!("3")] {
        let (status, text) = post(&base, "/v1/chat/completions", json!({"model": "m", "logprobs": true,
            "top_logprobs": k, "messages": [{"role": "user", "content": "Hello"}]})).await;
        assert_eq!(status, 400, "{text}");
    }
    let mut response = post_json(&base, "/v1/chat/completions", json!({"model": "m", "logprobs": false,
        "x_uncertainty": false, "x_guard": false, "messages": [{"role": "user", "content": "Hello"}]})).await;
    response["id"] = json!("chatcmpl-baseline");
    response["created"] = json!(0);
    assert_eq!(response, serde_json::from_str::<Value>(include_str!("data/chat-baseline.json")).unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_uncertainty_buffered_utf8_and_think_markers() {
    use dsx_core::server::{Finish, GenParams};
    use dsx_core::server::uncertainty::TokenStats;

    struct Reply { inner: MockEngine, ids: Vec<u32> }
    impl Engine for Reply {
        fn generate(&mut self, _: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> anyhow::Result<Finish> {
            self.inner.generate(&self.ids, p, emit)
        }
        fn generate_with_stats(&mut self, _: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32, Option<TokenStats>) -> bool) -> anyhow::Result<Finish> {
            self.inner.generate_with_stats(&self.ids, p, emit)
        }
    }
    let tok = tokenizer();
    let text = "<think>🙂漢字</think>Ответ 🦊<｜end▁of▁sentence｜>";
    let ids = tok.encode(text, false).unwrap().get_ids().to_vec();
    let generated = ids.len() - 1;
    let t2 = tok.clone();
    let engine = EngineHandle::spawn(move || Ok(Box::new(Reply { inner: MockEngine::new(t2, MockMode::Echo), ids }))).unwrap();
    let base = dsx_core::server::router(AppState::new(engine, tok, ServerConfig::default()).unwrap());
    // Start outside thinking, then let the generated open/close ids change the channel.
    let mut body = json!({"model": "m", "thinking": false, "logprobs": true,
        "messages": [{"role": "user", "content": "Hello"}]});
    let expected = post_json(&base, "/v1/chat/completions", body.clone()).await;
    assert_eq!(entries(&expected, "reasoning_content").len() + entries(&expected, "content").len(), generated);
    let all: Vec<u8> = entries(&expected, "reasoning_content").iter().chain(entries(&expected, "content"))
        .flat_map(|e| serde_json::from_value::<Vec<u8>>(e["bytes"].clone()).unwrap()).collect();
    assert_eq!(String::from_utf8(all).unwrap(), text.trim_end_matches("<｜end▁of▁sentence｜>"));
    assert!(entries(&expected, "reasoning_content").iter().any(|e| {
        let bytes: Vec<u8> = serde_json::from_value(e["bytes"].clone()).unwrap();
        std::str::from_utf8(&bytes).is_err()
    }), "fixture includes an incomplete UTF-8 token");
    body["stream"] = json!(true);
    let (_, text) = post(&base, "/v1/chat/completions", body).await;
    for channel in ["reasoning_content", "content"] {
        let actual: Vec<Value> = sse(&text).iter().filter_map(|(_, c)| c["choices"][0]["logprobs"][channel].as_array())
            .flatten().cloned().collect();
        assert_eq!(actual, *entries(&expected, channel));
    }
}

/// A fixed token script with deterministic entropy, through the same stats path as A1.
async fn guard_script(ids: Vec<u32>, entropy: f32, cfg: ServerConfig) -> Router {
    use dsx_core::server::{Finish, GenParams};
    use dsx_core::server::uncertainty::TokenStats;

    struct Reply { inner: MockEngine, ids: Vec<u32>, entropy: f32 }
    impl Engine for Reply {
        fn generate(&mut self, _: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> anyhow::Result<Finish> {
            self.inner.generate(&self.ids, p, emit)
        }
        fn generate_with_stats(&mut self, _: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32, Option<TokenStats>) -> bool) -> anyhow::Result<Finish> {
            let entropy = self.entropy;
            self.inner.generate_with_stats(&self.ids, p, &mut |id, stats| {
                emit(id, stats.map(|mut s| { s.entropy = entropy; s }))
            })
        }
    }
    let tok = tokenizer();
    let t2 = tok.clone();
    let engine = EngineHandle::spawn(move || Ok(Box::new(Reply { inner: MockEngine::new(t2, MockMode::Echo), ids, entropy }))).unwrap();
    dsx_core::server::router(AppState::new(engine, tok, cfg).unwrap())
}

fn guard_ids(reasoning: usize, answer: usize) -> Vec<u32> {
    let tok = tokenizer();
    let single = |text: &str| {
        let ids = tok.encode(text, false).unwrap().get_ids().to_vec();
        assert_eq!(ids.len(), 1, "fixture must be one token: {text}");
        ids[0]
    };
    let mut ids = vec![single("Wait"); reasoning.saturating_sub(1)];
    if reasoning > 0 { ids.push(single("</think>")); }
    ids.extend(vec![single("Yes"); answer]);
    ids.push(single("<｜end▁of▁sentence｜>"));
    ids
}

fn guard_request(thinking: bool) -> Value {
    json!({"model": "m", "thinking": thinking, "x_guard": true,
        "messages": [{"role": "user", "content": "Hello"}]})
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_guard_entropy_thresholds_and_logprobs() {
    for (entropy, verdict) in [(0.388, "ok"), (0.390, "uncertain")] {
        let base = guard_script(guard_ids(0, 1), entropy, ServerConfig::default()).await;
        let mut body = guard_request(false);
        let r = post_json(&base, "/v1/chat/completions", body.clone()).await;
        assert_eq!(r["x_guard"], json!({"verdict": verdict, "risk": entropy as f64,
            "threshold": 0.389, "signal": "answer_entropy", "answer_tokens": 1, "reasoning_tokens": 0}));
        assert!(r["choices"][0]["logprobs"].is_null());
        assert_eq!(r["choices"][0]["message"]["content"], "Yes");
        assert_eq!(r["x_uncertainty"]["content"]["mean_entropy"], r["x_guard"]["risk"]);
        body["logprobs"] = json!(true);
        body["top_logprobs"] = json!(3);
        let with_logprobs = post_json(&base, "/v1/chat/completions", body).await;
        assert_eq!(with_logprobs["x_guard"], r["x_guard"]);
        assert_eq!(entries(&with_logprobs, "content").len(), 1);
        assert_eq!(entries(&with_logprobs, "content")[0]["top_logprobs"].as_array().unwrap().len(), 3);
    }
    // Exact equality is ok; use the same f32-derived value as the engine's entropy.
    let entropy = 0.389f32;
    let base = guard_script(guard_ids(0, 1), entropy, ServerConfig { guard_entropy: entropy as f64, ..ServerConfig::default() }).await;
    let r = post_json(&base, "/v1/chat/completions", guard_request(false)).await;
    assert_eq!(r["x_guard"]["verdict"], "ok");
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_guard_reasoning_ratio_thresholds() {
    for (reasoning, answer, verdict) in [(82, 1, "ok"), (83, 1, "ok"), (84, 1, "uncertain"), (84, 0, "not_applicable")] {
        let base = guard_script(guard_ids(reasoning, answer), 0.0, ServerConfig::default()).await;
        let r = post_json(&base, "/v1/chat/completions", guard_request(true)).await;
        assert_eq!(r["x_guard"], json!({"verdict": verdict, "risk": reasoning as f64,
            "threshold": 83.0, "signal": "reasoning_ratio", "answer_tokens": answer, "reasoning_tokens": reasoning}));
        assert_eq!(r["x_uncertainty"]["reasoning"]["tokens"], reasoning);
    }
    let base = guard_script(guard_ids(66, 1), 0.0, ServerConfig { guard_think_ratio: 65.0, ..ServerConfig::default() }).await;
    let r = post_json(&base, "/v1/chat/completions", guard_request(true)).await;
    assert_eq!(r["x_guard"]["verdict"], "uncertain");
    assert_eq!(r["x_guard"]["threshold"], 65.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_guard_answer_limit() {
    for reasoning in [0, 84] {
        for (answer, limit, verdict) in [(64, None, "uncertain"), (65, None, "not_applicable"),
            (2, Some(2), "uncertain"), (3, Some(2), "not_applicable")] {
            let cfg = ServerConfig { guard_think_ratio: 0.5, ..ServerConfig::default() };
            let base = guard_script(guard_ids(reasoning, answer), 1.0, cfg).await;
            let mut body = guard_request(reasoning > 0);
            body["x_guard"] = json!({"mode": "hedge"});
            if let Some(limit) = limit { body["x_guard"]["max_answer_tokens"] = json!(limit); }
            let r = post_json(&base, "/v1/chat/completions", body).await;
            assert_eq!(r["x_guard"]["verdict"], verdict);
            assert_eq!(r["x_guard"]["answer_tokens"], answer);
            if verdict == "not_applicable" {
                assert_eq!(r["choices"][0]["message"]["content"], "Yes".repeat(answer));
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_guard_hedge_language_and_tools() {
    let base = start(MockMode::Scripted, 0).await;
    for (last, prefix) in [(json!("Привет"), "Не уверен в этом ответе — стоит проверить."),
        (json!([{"type": "text", "text": "Вопрос"}]), "Не уверен в этом ответе — стоит проверить."),
        (json!("Hello"), "I am not sure about this answer — worth checking.")] {
        let mut body = guard_request(false);
        body["messages"] = json!([{"role": "user", "content": "Русский"},
            {"role": "assistant", "content": "Hello"}, {"role": "user", "content": last}]);
        body["x_guard"] = json!({"mode": "hedge"});
        let r = post_json(&base, "/v1/chat/completions", body.clone()).await;
        assert_eq!(r["x_guard"]["verdict"], "uncertain");
        assert_eq!(r["choices"][0]["message"]["content"], format!("{prefix}\n\n{MOCK_REPLY}"));
        // The prefix does not inflate usage or the captured token count.
        let n = tokenizer().encode(MOCK_REPLY, false).unwrap().get_ids().len();
        assert_eq!(r["x_guard"]["answer_tokens"], n);
        assert_eq!(r["usage"]["completion_tokens"], n);
        body["tools"] = weather_tool_openai();
        body["x_guard"]["max_answer_tokens"] = json!(1024);
        let mut hedged = post_json(&base, "/v1/chat/completions", body.clone()).await;
        // a tool call is not a short factual answer
        assert_eq!(hedged["x_guard"]["verdict"], "not_applicable");
        assert!(!hedged["choices"][0]["message"]["tool_calls"].as_array().unwrap().is_empty());
        body.as_object_mut().unwrap().remove("x_guard");
        let original = post_json(&base, "/v1/chat/completions", body).await;
        // Tool IDs incorporate the per-request response ID.
        hedged["choices"][0]["message"]["tool_calls"][0]["id"] = original["choices"][0]["message"]["tool_calls"][0]["id"].clone();
        assert_eq!(hedged["choices"], original["choices"]);
    }
    let base = guard_script(guard_ids(0, 1), 0.1, ServerConfig::default()).await;
    let mut body = guard_request(false);
    body["x_guard"] = json!({"mode": "hedge"});
    let r = post_json(&base, "/v1/chat/completions", body).await;
    assert_eq!(r["x_guard"]["verdict"], "ok");
    assert_eq!(r["choices"][0]["message"]["content"], "Yes");
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_guard_stream_flags_without_changing_content() {
    let base = start(MockMode::Scripted, 0).await;
    for usage in [None, Some(false), Some(true)] {
        for logprobs in [false, true] {
            let mut body = guard_request(false);
            body["stream"] = json!(true);
            body["logprobs"] = json!(logprobs);
            body["x_guard"] = json!({"mode": "hedge"});
            if let Some(usage) = usage { body["stream_options"] = json!({"include_usage": usage}); }
            let (status, text) = post(&base, "/v1/chat/completions", body).await;
            assert_eq!(status, 200, "{text}");
            let chunks = sse(&text);
            assert!(chunks.last().unwrap().1.is_null());
            let content: String = chunks.iter().filter_map(|(_, c)| c["choices"][0]["delta"]["content"].as_str()).collect();
            assert_eq!(content, MOCK_REPLY);
            let final_chunks: Vec<_> = chunks.iter().filter(|(_, c)| c["usage"].is_object()).collect();
            assert_eq!(final_chunks.len(), 1);
            let final_chunk = &final_chunks[0].1;
            assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
            assert_eq!(final_chunk["x_guard"]["verdict"], "uncertain");
            assert_eq!(final_chunk["x_guard"]["signal"], "answer_entropy");
            assert!(final_chunk["x_uncertainty"].is_object());
            assert_eq!(chunks.iter().filter(|(_, c)| c["x_guard"].is_object()).count(), 1);
            assert_eq!(chunks.iter().filter(|(_, c)| c["x_uncertainty"].is_object()).count(), 1);
            let n: usize = chunks.iter().filter_map(|(_, c)| c["choices"][0]["logprobs"]["content"].as_array()).map(Vec::len).sum();
            assert_eq!(n, if logprobs { final_chunk["x_guard"]["answer_tokens"].as_u64().unwrap() as usize } else { 0 });
        }
    }
    // With thinking, both transports use the ratio regardless of answer entropy.
    let base = guard_script(guard_ids(84, 1), 0.0, ServerConfig::default()).await;
    let mut body = guard_request(true);
    let expected = post_json(&base, "/v1/chat/completions", body.clone()).await;
    body["stream"] = json!(true);
    body["x_guard"] = json!({"mode": "hedge"});
    let (_, text) = post(&base, "/v1/chat/completions", body).await;
    let chunks = sse(&text);
    let final_chunk = &chunks.iter().find(|(_, c)| c["usage"].is_object()).unwrap().1;
    assert_eq!(final_chunk["x_guard"], expected["x_guard"]);
    let content: String = chunks.iter().filter_map(|(_, c)| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(content, "Yes");
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_guard_validation_and_defaults() {
    let base = start(MockMode::Scripted, 0).await;
    for guard in [json!("true"), json!(1), json!({"mode": "guess"}), json!({"max_answer_tokens": -1}),
        json!({"max_answer_tokens": 1.5}), json!({"max_answer_tokens": "64"})] {
        let mut body = guard_request(false);
        body["x_guard"] = guard;
        let (status, text) = post(&base, "/v1/chat/completions", body).await;
        assert_eq!(status, 400, "{text}");
    }
    let mut body = guard_request(false);
    let expected = post_json(&base, "/v1/chat/completions", body.clone()).await;
    body["x_guard"] = json!({"mode": "flag", "max_answer_tokens": 64});
    let explicit = post_json(&base, "/v1/chat/completions", body.clone()).await;
    assert_eq!(explicit["x_guard"], expected["x_guard"]);
    body["x_guard"] = json!({});
    let defaults = post_json(&base, "/v1/chat/completions", body).await;
    assert_eq!(defaults["x_guard"], expected["x_guard"]);
}

#[test]
fn guard_server_threshold_validation() {
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        for entropy in [true, false] {
            let tok = tokenizer();
            let t2 = tok.clone();
            let engine = EngineHandle::spawn(move || Ok(Box::new(MockEngine::new(t2, MockMode::Scripted)))).unwrap();
            let mut cfg = ServerConfig::default();
            if entropy { cfg.guard_entropy = invalid; } else { cfg.guard_think_ratio = invalid; }
            let error = AppState::new(engine, tok, cfg).err().unwrap().to_string();
            assert!(error.contains(if entropy { "--guard-entropy" } else { "--guard-think-ratio" }));
        }
    }
}
