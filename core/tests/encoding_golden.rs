//! Byte comparisons against files produced only by DeepSeek's Python reference.
//! Known upstream gaps assert the entire observed server prompt and first byte;
//! they are never skipped or accepted as reference-equivalent.
use dsx_core::server::{convert_chat_request, render_prompt};
use serde_json::{Value, json};
use std::{io::Read, path::Path, sync::LazyLock};

static ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/dsv41_encoding");
static CASES: LazyLock<Value> = LazyLock::new(|| read_json("cases.json"));
static EXPECTATIONS: LazyLock<Value> = LazyLock::new(|| read_json("expectations.json"));
static LEGACY_IMAGE_TOKEN: LazyLock<String> = LazyLock::new(|| {
    let gz = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tokenizer-v41.json.gz")).unwrap();
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut bytes).unwrap();
    tokenizers::Tokenizer::from_bytes(bytes).unwrap().id_to_token(dsx_core::vision::IMAGE_TOKEN_ID).unwrap()
});

fn read(name: &str) -> Vec<u8> {
    std::fs::read(Path::new(ROOT).join(name)).unwrap()
}

fn read_json(name: &str) -> Value {
    serde_json::from_slice(&read(name)).unwrap()
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

fn snippet(bytes: &[u8], offset: usize) -> String {
    // Escape bytes, so even a mismatch within a UTF-8 character is unambiguous.
    bytes[offset.saturating_sub(24)..bytes.len().min(offset + 100)].iter().flat_map(|b| std::ascii::escape_default(*b)).map(char::from).collect()
}

fn assert_bytes(name: &str, actual: &[u8], expected: &[u8]) {
    if let Some(offset) = first_difference(actual, expected) {
        panic!(
            "{name}: first differing byte {offset}; server ({} bytes): {}; reference ({} bytes): {}",
            actual.len(),
            snippet(actual, offset),
            expected.len(),
            snippet(expected, offset)
        );
    }
}

fn compare(name: &str) {
    let case = CASES["cases"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap();
    let expected = &EXPECTATIONS[name];
    assert!(expected.is_object(), "missing explicit expectation: {name}");
    let body = read(case["request"].as_str().unwrap());
    let conversion = convert_chat_request(&body, "test-model");
    if let Some(error) = expected["expected_error"].as_str() {
        let error_actual = conversion.unwrap_err();
        assert_eq!(error_actual.status_code(), 400, "{name}");
        assert_eq!(error_actual.to_string(), error, "{name}");
        // The encoder accepts this budget; a protocol error is an open gap.
        assert!(!read(case["reference"].as_str().unwrap()).is_empty());
        return;
    }
    let (conv, _, _) = conversion.unwrap();
    let marker = if case["image_token"] == "repository_fixture" { LEGACY_IMAGE_TOKEN.as_str() } else { "<｜deepseek_image｜>" };
    let rendered = render_prompt(&conv, marker);
    assert_eq!(rendered.image_sources.len() as u64, expected["images"].as_u64().unwrap(), "{name}");
    for source in &rendered.image_sources {
        let deepseek_recipe_core::multimodal::ImageSource::DataUrl { data_url, .. } = source else {
            panic!("{name}: fixture must be self-contained");
        };
        let bytes = dsx_core::vision::data_url_bytes(data_url).unwrap();
        image::load_from_memory(&bytes).expect("valid image bytes accepted by the production vision loader");
    }
    let reference = read(case["reference"].as_str().unwrap());
    if let Some(known) = expected.get("known_difference") {
        let offset =
            first_difference(rendered.prompt.as_bytes(), &reference).expect("upstream gap closed: remove the known difference and assert reference equality");
        assert_eq!(offset as u64, known["byte"].as_u64().unwrap(), "{name}");
        let observed = read(known["server"].as_str().unwrap());
        assert_bytes(name, rendered.prompt.as_bytes(), &observed);
    } else {
        assert_bytes(name, rendered.prompt.as_bytes(), &reference);
    }
}

macro_rules! cases {
    ($($name:ident),+ $(,)?) => {
        const TESTED_CASES: &[&str] = &[$(stringify!($name)),+];
        $(#[test] fn $name() { compare(stringify!($name)); })+
    };
}

cases!(
    official_1,
    official_2,
    official_3,
    official_4,
    official_5,
    official_5_legacy_tokenizer,
    effort_absent,
    effort_low,
    effort_high,
    effort_xhigh,
    effort_max,
    effort_none,
    effort_1,
    effort_37,
    effort_88,
    effort_100,
    effort_50,
    effort_75,
    thinking_on,
    thinking_off,
    boolean_on,
    boolean_off,
    kwargs_off,
    kwargs_high,
    kwargs_100,
    kwargs_37,
    effort_37_off,
    kwargs_37_off,
    explicit_wins,
    none_explicit_on,
    history_on,
    history_off,
    history_keep,
    tools_parallel,
    tools_parallel_off,
    tool_turn_current,
    tool_turn_completed,
    mid_system,
    mid_system_on,
    pi_like,
);

#[test]
fn every_fixture_has_a_test_and_an_expectation() {
    let tested: std::collections::BTreeSet<_> = TESTED_CASES.iter().copied().collect();
    let fixtures: std::collections::BTreeSet<_> = CASES["cases"].as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap()).collect();
    let expected: std::collections::BTreeSet<_> = EXPECTATIONS.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(tested.len(), TESTED_CASES.len());
    assert_eq!(fixtures.len(), CASES["cases"].as_array().unwrap().len());
    assert_eq!(tested, fixtures);
    assert_eq!(tested, expected);
}

#[test]
fn official_transport_adaptations_are_explicit() {
    // Upstream fixtures are encoder inputs, not all valid OpenAI HTTP bodies.
    for (index, error) in [(1, "missing field `id`"), (3, "missing field `id`"), (5, "Unsupported image_url format")] {
        let raw = read_json(&format!("test_input_{index}.json"));
        let mut body = if raw.is_array() { json!({"messages": raw}) } else { raw };
        let mode = body["thinking_mode"].as_str().unwrap_or("chat");
        body["thinking"] = json!({"type": if mode == "thinking" { "enabled" } else { "disabled" }});
        let actual = convert_chat_request(&serde_json::to_vec(&body).unwrap(), "test-model").unwrap_err();
        assert_eq!(actual.status_code(), 400);
        assert!(actual.to_string().contains(error), "official_{index}: {actual}");
    }
}

#[test]
fn invalid_numeric_effort_is_rejected_without_rounding() {
    for effort in [json!(0), json!(101), json!(-1), json!(true), json!(1.5)] {
        for field in [json!({"reasoning_effort": effort}), json!({"chat_template_kwargs": {"reasoning_effort": effort}})] {
            let mut body = field;
            body["messages"] = json!([{"role": "user", "content": "question"}]);
            let error = convert_chat_request(&serde_json::to_vec(&body).unwrap(), "test-model").unwrap_err();
            assert_eq!(error.status_code(), 400, "{body}");
        }
    }
}

#[test]
fn request_metadata_survives_shared_conversion() {
    let body = json!({"seed": 37, "stream": true, "stream_options": {"include_usage": true},
                      "messages": [{"role": "user", "content": "question"}]});
    let (conv, seed, usage) = convert_chat_request(&serde_json::to_vec(&body).unwrap(), "fallback-model").unwrap();
    assert_eq!(conv.model.as_deref(), Some("fallback-model"));
    assert!(conv.stream && usage);
    assert_eq!(seed, Some(37));
}
