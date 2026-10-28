//! Deliberately ignored: needs exclusive ownership of both GPUs and the full model.
//! This never stops/starts a service itself.
use super::*;
use serde_json::{Value, json};

fn text_tokens(tok: &tokenizers::Tokenizer, label: &str, n: usize) -> Vec<u32> {
    let text = format!(
        "{label}\nThe engineer recorded a numbered observation, checked the result, and kept the full history. "
    );
    let ids = tok.encode(text.as_str(), false).unwrap();
    ids.get_ids().iter().copied().cycle().take(n).collect()
}

fn memory_kb(name: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|l| {
            l.strip_prefix(name)?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
        .unwrap_or(0)
}

fn run(
    r: &mut Real,
    prompt: &[u32],
    label: &str,
    results: &mut Vec<Value>,
    out: &std::path::Path,
) -> Vec<u32> {
    let p = GenParams {
        max_new: 16,
        temperature: 0.0,
        top_p: 1.0,
        min_p: 0.0,
        seed: Some(0),
        stop_token_ids: vec![],
        hesitation: None,
        logprobs: None,
        think: None,
        images: Default::default(),
        client: crate::server::sessions::DEFAULT_CLIENT.into(),
        priority: false,
    };
    let mut ids = vec![];
    let t0 = Instant::now();
    let mut first_ms = None;
    r.generate(prompt, &p, &mut |id| {
        first_ms.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1e3);
        ids.push(id);
        true
    })
    .unwrap();
    assert_eq!(r.gm.pos, r.cached.len());
    assert!(r.sessions.bytes() <= r.sessions.max_bytes());
    assert!(r.sessions.peak_bytes() <= r.sessions.max_bytes());
    assert!(r.sessions.len() <= 1);
    let row = json!({"label": label, "prompt_tokens": prompt.len(), "token_ids": ids,
        "cached_prompt_tokens": r.stats.cached_prompt_tokens,
        "prefill_ms": r.stats.prefill.unwrap().as_secs_f64() * 1e3,
        "first_ms": first_ms, "elapsed_ms": t0.elapsed().as_secs_f64() * 1e3,
        "saved_host_bytes": r.sessions.bytes(), "peak_host_bytes": r.sessions.peak_bytes(),
        "limit_host_bytes": r.sessions.max_bytes(), "rss_kb": memory_kb("VmRSS:"), "locked_kb": memory_kb("VmLck:")});
    eprintln!("SESSION_CHECK {row}");
    results.push(row);
    std::fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(results).unwrap(),
    )
    .unwrap();
    ids
}

fn reset(r: &mut Real, n: usize) {
    r.sessions = SessionCache::new(SessionConfig::new(n, 2.0, 2048).unwrap());
    r.cached.clear();
    r.snap = None;
    r.gm.reset();
    r.live_id = 0;
    r.live_disk = 0;
    r.live_client = crate::server::sessions::DEFAULT_CLIENT.into();
}

#[test]
#[ignore = "exclusive 2x3090, full model; owner-authorized maintenance window only"]
fn gpu_multisession_acceptance() {
    assert_eq!(
        std::env::var("DSX_ADAPT").as_deref(),
        Ok("0"),
        "exactness requires DSX_ADAPT=0"
    );
    assert!(
        std::env::var_os("DSX_DEFER").is_none(),
        "exactness requires no deferred experts"
    );
    assert_eq!(std::env::var("DSX_TEST_EXCLUSIVE").as_deref(), Ok("1"),
        "reserve the test GPUs and set DSX_TEST_EXCLUSIVE=1");
    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_env_filter("info")
        .try_init();
    let model = PathBuf::from(
        std::env::var("DSX_TEST_MODEL").expect("set DSX_TEST_MODEL to the first GGUF shard"),
    );
    let g = crate::gguf::Gguf::open(&model).unwrap();
    let tok: tokenizers::Tokenizer = g.kv["dsx.tokenizer_json"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    drop(g);
    let args = RealEngineArgs {
        model: model.clone(),
        engram: None,
        ctx: 262144,
        chunk: 8192,
        gpu0_gb: -1.5,
        gpu1_gb: -1.0,
        arena_gb: 135.0,
        rank: Some(PathBuf::from(
            std::env::var("DSX_TEST_RANK").expect("set DSX_TEST_RANK"),
        )),
        vision: model
            .parent()
            .map(|p| p.join("vision.safetensors"))
            .filter(|p| p.is_file()),
        sessions: SessionConfig::new(2, 2.0, 2048).unwrap(),
        assistant_token: None,
        user_token: tok.token_to_id("<｜User｜>").unwrap(),
    };
    let out = PathBuf::from(
        std::env::var("DSX_TEST_OUT").unwrap_or_else(|_| "/tmp/dsx-sessions-gpu".into()),
    );
    std::fs::create_dir_all(&out).unwrap();
    let mut results = vec![];
    let mut r = load_real(&args).unwrap();
    let a = text_tokens(&tok, "ALPHA private conversation", 30721); // odd: exercise ratio-2 compressor carry
    let b = text_tokens(&tok, "BETA unrelated conversation", 4097);
    let tail = text_tokens(&tok, "Continue the report with the next observation.", 8);

    if std::env::var_os("DSX_TEST_P2P_AB").is_some() {
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_ne!(r.gm.test_peer_transport(true).unwrap(), 0, "peer transport is required for this A/B test");
        r.gm.test_peer_transport(false).unwrap();
        let pb = r.gm.pb.take().unwrap();
        let reference = bits(r.gm.prefill(&pb, &a).unwrap());
        r.gm.pb = Some(pb);
        let saved = r.gm.save_session(None).unwrap();
        let mut token = 19;
        let mut decode = vec![];
        let mut ids = vec![];
        for _ in 0..32 {
            let logits = r.gm.step(token).unwrap();
            token = logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            ids.push(token);
            decode.push(bits(logits));
        }
        r.gm.load_session(&saved, &mut r.snap).unwrap();
        let pb = r.gm.pb.take().unwrap();
        let verify_ref = r.gm.verify(&pb, &tail).unwrap().into_iter().map(bits).collect::<Vec<_>>();
        r.gm.pb = Some(pb);
        r.gm.test_peer_transport(true).unwrap();
        r.gm.reset();
        let pb = r.gm.pb.take().unwrap();
        let actual = bits(r.gm.prefill(&pb, &a).unwrap());
        r.gm.pb = Some(pb);
        assert_eq!(actual, reference, "P2P prefill logits differ from pinned-host transport");
        token = 19;
        for (i, expected) in decode.into_iter().enumerate() {
            let logits = r.gm.step(token).unwrap();
            token = logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            assert_eq!(token, ids[i]);
            assert_eq!(bits(logits), expected, "P2P decode logits differ at step {i}");
        }
        r.gm.load_session(&saved, &mut r.snap).unwrap();
        let pb = r.gm.pb.take().unwrap();
        let actual = r.gm.verify(&pb, &tail).unwrap().into_iter().map(bits).collect::<Vec<_>>();
        r.gm.pb = Some(pb);
        assert_eq!(actual, verify_ref, "P2P 8-token verify logits differ");
        results.push(json!({"label":"peer-host-bit-equality", "prefill_tokens":a.len(),
            "logits":reference.len(), "decode_steps":32, "verify_tokens":8, "equal":true}));
        reset(&mut r, 2);
    }

    let a_ids = run(&mut r, &a, "a-initial-reference", &mut results, &out);
    let mut ac = a.clone();
    ac.extend_from_slice(&a_ids);
    ac.extend_from_slice(&tail);
    let expected = run(&mut r, &ac, "a-continuation-reference", &mut results, &out);
    assert_eq!(run(&mut r, &a, "a-reprime", &mut results, &out), a_ids);
    run(&mut r, &b, "b-insert", &mut results, &out);
    assert_eq!(
        run(&mut r, &ac, "a-restored", &mut results, &out),
        expected,
        "raw continuation token ids differ"
    );
    assert_eq!(r.stats.cached_prompt_tokens, a.len() + a_ids.len());

    // Stronger check: every f32 logit bit, with identical step shape at exactly the saved position. Overwrite the
    // occupied comp/idx rows with another conversation before restoring, not just a same-conversation rewind.
    {
        let host = r.gm.save_session(r.snap.as_ref()).unwrap();
        assert_eq!(
            host.heap_bytes(),
            r.gm.session_snapshot_bytes(r.snap.as_ref())
        );
        let expected: Vec<u32> =
            r.gm.step(tail[0])
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect();
        r.gm.reset();
        let pb = r.gm.pb.take().unwrap();
        r.gm.prefill(&pb, &b).unwrap();
        r.gm.pb = Some(pb);
        r.gm.load_session(&host, &mut r.snap).unwrap();
        let actual: Vec<u32> =
            r.gm.step(tail[0])
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect();
        assert_eq!(
            actual, expected,
            "logits differ at the bit level after session restoration"
        );
        r.gm.load_session(&host, &mut r.snap).unwrap();
        results.push(
            json!({"label": "logit-bit-equality", "equal": true, "logits": actual.len(),
            "snapshot_heap_bytes": host.heap_bytes(), "memory": host.memory_kind()}),
        );
    }

    // Exercise the saved prompt-boundary rollback when clients re-encode generated assistant text.
    reset(&mut r, 2);
    run(&mut r, &a, "rewind-prime-reference", &mut results, &out);
    let mut rerendered = a.clone();
    rerendered.extend_from_slice(&tail);
    let expected = run(&mut r, &rerendered, "rewind-reference", &mut results, &out);
    reset(&mut r, 2);
    run(&mut r, &a, "rewind-prime", &mut results, &out);
    run(&mut r, &b, "rewind-b-insert", &mut results, &out);
    assert_eq!(
        run(&mut r, &rerendered, "rewind-restored", &mut results, &out),
        expected
    );
    assert_eq!(r.stats.cached_prompt_tokens, a.len() - 1);

    // Exactness uses the same cached continuation without B. Full prefill is only the old-path latency
    // baseline: different chunk shapes and CPU/GPU expert splits need not produce identical rounding.
    let big = text_tokens(&tok, "GAMMA long engineering report", 144051);
    reset(&mut r, 1);
    let big_ids = run(&mut r, &big, "144k-single-prime", &mut results, &out);
    let mut bigc = big.clone();
    bigc.extend_from_slice(&big_ids);
    bigc.extend_from_slice(&tail);
    let big_expected = run(
        &mut r,
        &bigc,
        "144k-continuation-reference",
        &mut results,
        &out,
    );
    assert_eq!(r.stats.cached_prompt_tokens, big.len() + big_ids.len());
    let reference_first_ms = results.last().unwrap()["first_ms"].as_f64().unwrap();
    run(&mut r, &b, "144k-single-b", &mut results, &out);
    run(&mut r, &bigc, "144k-single-return", &mut results, &out);
    assert_eq!(r.stats.cached_prompt_tokens, 0);
    reset(&mut r, 2);
    assert_eq!(
        run(&mut r, &big, "144k-multi-prime", &mut results, &out),
        big_ids
    );
    let b_ids = run(&mut r, &b, "144k-multi-b", &mut results, &out);
    let big_id = r
        .sessions
        .best(&bigc, &r.live_client, &r.cached, r.gm.pos, r.snap.as_ref().map(|s| s.pos))
        .unwrap();
    let default_memory = r.sessions.get(big_id).unwrap().state.as_ref().unwrap().memory_kind();
    assert_eq!(
        run(&mut r, &bigc, "144k-multi-return", &mut results, &out),
        big_expected
    );
    assert_eq!(r.stats.cached_prompt_tokens, big.len() + big_ids.len());
    let default_first_ms = results.last().unwrap()["first_ms"].as_f64().unwrap();
    let switch_overhead_ms = default_first_ms - reference_first_ms;

    // Twenty alternations; every new prompt exactly extends its own earlier live token state.
    let (mut pa, mut pb) = (bigc, b.clone());
    let (mut oa, mut ob) = (big_expected.clone(), b_ids);
    for i in 0..20 {
        let (prompt, previous) = if i % 2 == 0 {
            (&mut pb, &mut ob)
        } else {
            (&mut pa, &mut oa)
        };
        let prior = prompt.len() + previous.len();
        prompt.extend_from_slice(previous);
        prompt.extend_from_slice(&tail);
        *previous = run(
            &mut r,
            prompt,
            &format!("alternation-{i:02}"),
            &mut results,
            &out,
        );
        assert_eq!(r.stats.cached_prompt_tokens, prior);
    }

    // Force the same fallback used when cuMemHostRegister fails, and measure the difference on a 144K return.
    // Only SessionBuffer allocation reads this variable; all inference still runs serially in this test.
    reset(&mut r, 2);
    unsafe {
        std::env::set_var("DSX_SESSION_PAGEABLE", "1");
    }
    assert_eq!(
        run(&mut r, &big, "144k-pageable-prime", &mut results, &out),
        big_ids
    );
    run(&mut r, &b, "144k-pageable-b", &mut results, &out);
    let mut pc = big.clone();
    pc.extend_from_slice(&big_ids);
    pc.extend_from_slice(&tail);
    let id = r
        .sessions
        .best(&pc, &r.live_client, &r.cached, r.gm.pos, r.snap.as_ref().map(|s| s.pos))
        .unwrap();
    assert_eq!(r.sessions.get(id).unwrap().state.as_ref().unwrap().memory_kind(), "pageable");
    assert_eq!(
        run(&mut r, &pc, "144k-pageable-return", &mut results, &out),
        big_expected
    );
    assert_eq!(r.stats.cached_prompt_tokens, big.len() + big_ids.len());
    let pageable_first_ms = results.last().unwrap()["first_ms"].as_f64().unwrap();
    unsafe {
        std::env::remove_var("DSX_SESSION_PAGEABLE");
    }
    results.push(json!({"label": "acceptance", "raw_tokens_equal": true, "logit_bits_equal": true,
        "alternations": 20, "default_memory": default_memory, "default_first_ms": default_first_ms, "pageable_first_ms": pageable_first_ms,
        "reference_first_ms": reference_first_ms, "switch_overhead_ms": switch_overhead_ms,
        "within_2s_plus_new_tokens": switch_overhead_ms <= 2000.0}));
    std::fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .unwrap();
    assert!(
        switch_overhead_ms <= 2000.0,
        "144K return exceeded 2 seconds plus the cached continuation: {switch_overhead_ms} ms overhead"
    );
}


/// One request through the engine, as `run` but for the archive test (any number of saved conversations).
fn run_any(r: &mut Real, prompt: &[u32], label: &str, results: &mut Vec<Value>, out: &std::path::Path) -> Vec<u32> {
    let p = GenParams {
        max_new: 16,
        temperature: 0.0,
        top_p: 1.0,
        min_p: 0.0,
        seed: Some(0),
        stop_token_ids: vec![],
        hesitation: None,
        logprobs: None,
        think: None,
        images: Default::default(),
        client: "owner".into(),
        priority: false,
    };
    let mut ids = vec![];
    let t0 = Instant::now();
    let mut first_ms = None;
    r.generate(prompt, &p, &mut |id| {
        first_ms.get_or_insert_with(|| t0.elapsed().as_secs_f64() * 1e3);
        ids.push(id);
        true
    })
    .unwrap();
    assert_eq!(r.gm.pos, r.cached.len());
    assert!(r.sessions.bytes() <= r.sessions.max_bytes());
    let row = json!({"label": label, "prompt_tokens": prompt.len(), "token_ids": ids,
        "cached_prompt_tokens": r.stats.cached_prompt_tokens, "first_ms": first_ms,
        "elapsed_ms": t0.elapsed().as_secs_f64() * 1e3, "saved": r.sessions.len(),
        "host_bytes": r.sessions.bytes(), "disk_bytes": r.sessions.disk_bytes(), "rss_kb": memory_kb("VmRSS:")});
    eprintln!("ARCHIVE_CHECK {row}");
    results.push(row);
    std::fs::write(out.join("results.json"), serde_json::to_vec_pretty(results).unwrap()).unwrap();
    ids
}

/// A new process as far as the sequences go: GPU state and host memory are gone, the files are adopted.
fn restart(r: &mut Real, cfg: SessionConfig) {
    r.sessions = SessionCache::new(cfg);
    r.cached.clear();
    r.snap = None;
    r.gm.reset();
    r.live_id = 0;
    r.live_disk = 0;
    r.adopt_archive();
}

fn files(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "xss")).collect();
    v.sort();
    v
}

#[test]
#[ignore = "exclusive 2x3090, full model; owner-authorized maintenance window only"]
fn gpu_disk_sessions_acceptance() {
    use crate::server::sessions::DiskConfig;
    assert_eq!(std::env::var("DSX_ADAPT").as_deref(), Ok("0"), "exactness requires DSX_ADAPT=0");
    assert!(std::env::var_os("DSX_DEFER").is_none(), "exactness requires no deferred experts");
    assert_eq!(std::env::var("DSX_TEST_EXCLUSIVE").as_deref(), Ok("1"),
        "reserve the test GPUs and set DSX_TEST_EXCLUSIVE=1");
    let _ = tracing_subscriber::fmt().with_ansi(false).with_env_filter("info").try_init();
    let env = |k: &str| std::env::var(k).ok().map(PathBuf::from);
    let model = env("DSX_TEST_MODEL").expect("set DSX_TEST_MODEL to the first GGUF shard");
    let tok: tokenizers::Tokenizer = match env("DSX_TEST_TOKENIZER") {
        Some(t) => tokenizers::Tokenizer::from_file(t).unwrap(),
        None => crate::gguf::Gguf::open(&model).unwrap().kv["dsx.tokenizer_json"].as_str().unwrap().parse().unwrap(),
    };
    let out = env("DSX_TEST_OUT").unwrap_or_else(|| "/tmp/dsx-archive-gpu".into());
    let dir = out.join("archive");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&out).unwrap();
    let disk = DiskConfig::new(dir.clone(), 64, 20.0, 7.0, Duration::from_secs(1), 0.0).unwrap();
    let cfg = |shelf_gb: f64| SessionConfig::new(4, shelf_gb, 2048).unwrap().with_disk(disk.clone()).unwrap();
    let args = RealEngineArgs {
        model,
        engram: env("DSX_TEST_ENGRAM"),
        ctx: 196608,
        chunk: 8192,
        gpu0_gb: -1.5,
        gpu1_gb: -1.0,
        arena_gb: std::env::var("DSX_TEST_ARENA_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(150.0),
        rank: Some(env("DSX_TEST_RANK").expect("set DSX_TEST_RANK")),
        vision: None,
        sessions: cfg(2.0),
        assistant_token: None,
        user_token: tok.token_to_id("<｜User｜>").unwrap(),
    };
    let mut results = vec![];
    let mut r = load_real(&args).unwrap();
    let a = text_tokens(&tok, "ALPHA private conversation", 30721);
    let b = text_tokens(&tok, "BETA unrelated conversation", 4097);
    let tail = text_tokens(&tok, "Continue the report with the next observation.", 8);

    // the reference: the live continuation on the GPU
    let a_ids = run_any(&mut r, &a, "reference-a", &mut results, &out);
    let mut ac = a.clone();
    ac.extend_from_slice(&a_ids);
    ac.extend_from_slice(&tail);
    let expected = run_any(&mut r, &ac, "reference-continuation", &mut results, &out);
    let reference_first_ms = results.last().unwrap()["first_ms"].as_f64().unwrap();

    // 1) shutdown saves the live conversation; after a restart it continues from its file, token for token
    restart(&mut r, cfg(2.0));
    assert_eq!(run_any(&mut r, &a, "restart-reprime", &mut results, &out), a_ids);
    let t = Instant::now();
    r.shutdown();
    let shutdown_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(files(&dir).len(), 1, "shutdown wrote the live conversation");
    restart(&mut r, cfg(2.0));
    assert_eq!(r.sessions.len(), 1, "the file was adopted");
    assert_eq!(run_any(&mut r, &ac, "restart-continuation", &mut results, &out), expected, "raw token ids differ after a restart");
    assert_eq!(r.stats.cached_prompt_tokens, a.len() + a_ids.len());
    let restart_first_ms = results.last().unwrap()["first_ms"].as_f64().unwrap();

    // 2) a shelf too small for A: switching to B writes A straight to disk, and A comes back from there
    restart(&mut r, cfg(0.01));
    for f in files(&dir) {
        std::fs::remove_file(f).unwrap();
    }
    restart(&mut r, cfg(0.01));
    assert_eq!(run_any(&mut r, &a, "spill-a", &mut results, &out), a_ids);
    run_any(&mut r, &b, "spill-b", &mut results, &out);
    assert_eq!(r.sessions.bytes(), 0, "nothing fits the 10 MB shelf");
    assert_eq!(files(&dir).len(), 1, "A went to disk");
    assert_eq!(run_any(&mut r, &ac, "spill-a-from-disk", &mut results, &out), expected, "raw token ids differ after a spill");
    assert_eq!(r.stats.cached_prompt_tokens, a.len() + a_ids.len());
    let spill_first_ms = results.last().unwrap()["first_ms"].as_f64().unwrap();

    // 3) idle: the shelf is copied to disk, then the live conversation is checkpointed
    for f in files(&dir) {
        std::fs::remove_file(f).unwrap();
    }
    restart(&mut r, cfg(2.0));
    run_any(&mut r, &a, "idle-a", &mut results, &out);
    run_any(&mut r, &b, "idle-b", &mut results, &out);
    assert_eq!(files(&dir).len(), 0);
    while r.idle(Duration::from_secs(5)) {}
    assert_eq!(files(&dir).len(), 2, "A from the shelf and B, the live one");

    // 4) a damaged file: no error, the prompt is read again and the file is dropped
    for f in files(&dir) {
        std::fs::remove_file(f).unwrap();
    }
    restart(&mut r, cfg(2.0));
    run_any(&mut r, &a, "damage-a", &mut results, &out);
    r.shutdown();
    let f = files(&dir).pop().unwrap();
    let mut raw = std::fs::read(&f).unwrap();
    let off = raw.len() - 3 * 4096 + 100;
    raw[off] ^= 0x40;
    std::fs::write(&f, &raw).unwrap();
    restart(&mut r, cfg(2.0));
    run_any(&mut r, &ac, "damaged-file", &mut results, &out);
    assert_eq!(r.stats.cached_prompt_tokens, 0, "a damaged file is not used");
    assert!(!f.exists(), "and it is deleted");

    results.push(json!({"label": "archive-acceptance", "raw_tokens_equal": true, "shutdown_ms": shutdown_ms,
        "reference_first_ms": reference_first_ms, "restart_first_ms": restart_first_ms, "spill_first_ms": spill_first_ms}));
    std::fs::write(out.join("results.json"), serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

/// Shared system prompt, then exact prompt retries through each tier. Run this test alone on the GPU host.
#[test]
#[ignore = "exclusive 2x3090, full model; owner-authorized maintenance window only"]
fn gpu_prefix_retry_acceptance() {
    use crate::server::sessions::DiskConfig;
    let env = |k: &str| std::env::var(k).ok().map(PathBuf::from);
    let Some(model) = env("DSX_TEST_MODEL").filter(|p| p.exists()) else {
        eprintln!("skipping gpu_prefix_retry_acceptance: set DSX_TEST_MODEL");
        return;
    };
    if !std::path::Path::new("/dev/nvidia0").exists() {
        eprintln!("skipping gpu_prefix_retry_acceptance: no NVIDIA GPU");
        return;
    }
    assert_eq!(std::env::var("DSX_ADAPT").as_deref(), Ok("0"), "exactness requires DSX_ADAPT=0");
    assert!(std::env::var_os("DSX_DEFER").is_none(), "exactness requires no deferred experts");
    assert_eq!(std::env::var("DSX_TEST_EXCLUSIVE").as_deref(), Ok("1"),
        "reserve the test GPUs and set DSX_TEST_EXCLUSIVE=1");
    let _ = tracing_subscriber::fmt().with_ansi(false).with_env_filter("info").try_init();
    let tok: tokenizers::Tokenizer = match env("DSX_TEST_TOKENIZER") {
        Some(t) => tokenizers::Tokenizer::from_file(t).unwrap(),
        None => crate::gguf::Gguf::open(&model).unwrap().kv["dsx.tokenizer_json"].as_str().unwrap().parse().unwrap(),
    };
    let out = env("DSX_TEST_OUT").unwrap_or_else(|| "/tmp/dsx-prefix-retry-gpu".into());
    std::fs::create_dir_all(&out).unwrap();
    // A new directory per invocation: never delete an archive supplied by another test or server.
    let dir = out.join(format!("archive-{}-{}", std::process::id(), unix_ms()));
    let disk = DiskConfig::new(dir.clone(), 64, 20.0, 7.0, Duration::from_secs(1), 0.0).unwrap();
    let config = SessionConfig::new(8, 2.0, 2048).unwrap().with_disk(disk).unwrap();
    let user_token = tok.token_to_id("<｜User｜>").unwrap();
    let args = RealEngineArgs {
        model,
        engram: env("DSX_TEST_ENGRAM"),
        ctx: 196608,
        chunk: 8192,
        gpu0_gb: -1.5,
        gpu1_gb: -1.0,
        arena_gb: std::env::var("DSX_TEST_ARENA_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(150.0),
        rank: Some(env("DSX_TEST_RANK").expect("set DSX_TEST_RANK")),
        vision: None,
        sessions: config.clone(),
        assistant_token: None,
        user_token,
    };
    let mut r = load_real(&args).unwrap();
    let system = text_tokens(&tok, "Shared system instructions and tool schemas", 4097);
    assert!(!system.contains(&user_token));
    let prompt = |message: &str| {
        let mut ids = system.clone();
        ids.extend_from_slice(tok.encode(format!("<｜User｜>{message}<｜Assistant｜><think>"), false).unwrap().get_ids());
        assert_eq!(ids.iter().position(|&t| t == user_token), Some(system.len()));
        ids
    };
    let a = prompt("Write eight observations about the ALPHA engineering report.");
    let b = prompt("Write eight observations about the BETA garden experiment.");
    let unrelated = text_tokens(&tok, "Unrelated conversation to overwrite the GPU sequence", 4097);
    let mut results = vec![];
    let mut run8 = |r: &mut Real, ids: &[u32], label: &str, cached: usize| {
        let p = GenParams {
            max_new: 8, temperature: 0.0, top_p: 1.0, min_p: 0.0, seed: Some(0), logprobs: None, think: None, stop_token_ids: vec![],
            hesitation: None, images: Default::default(), client: "owner".into(), priority: false,
        };
        let mut tokens = vec![];
        r.generate(ids, &p, &mut |t| { tokens.push(t); true }).unwrap();
        assert_eq!(tokens.len(), 8);
        assert_eq!(r.stats.cached_prompt_tokens, cached, "{label}");
        assert_eq!(r.gm.pos, r.cached.len());
        assert!(r.sessions.bytes() <= r.sessions.max_bytes());
        assert!(r.sessions.peak_bytes() <= r.sessions.max_bytes());
        let row = json!({"label": label, "cached_prompt_tokens": cached, "token_ids": tokens,
            "top1_logits": r.top1_logits});
        eprintln!("PREFIX_RETRY_CHECK {row}");
        results.push(row);
        (tokens, r.top1_logits.clone())
    };
    run8(&mut r, &a, "system-a", 0);
    let prefix_id = r.sessions.best(&b, "owner", &r.cached, r.gm.pos, r.snap.as_ref().map(|s| s.pos)).unwrap();
    assert!(r.sessions.get(prefix_id).unwrap().prefix);
    let prefix_ptr = r.sessions.get(prefix_id).unwrap().state.as_ref().unwrap().data().as_ptr();
    let shared = run8(&mut r, &b, "system-b-prefix", system.len());
    assert_eq!(r.sessions.get(prefix_id).unwrap().state.as_ref().unwrap().data().as_ptr(), prefix_ptr,
        "the shared snapshot remains on the shelf after loading");
    let live = run8(&mut r, &b, "same-prompt-live", b.len() - 1);
    run8(&mut r, &unrelated, "unrelated-before-shelf", 0);
    let shelf_id = r.sessions.best(&b, "owner", &r.cached, r.gm.pos, r.snap.as_ref().map(|s| s.pos)).unwrap();
    assert!(r.sessions.get(shelf_id).is_some_and(|s| !s.prefix && s.state.is_some()));
    let shelf = run8(&mut r, &b, "same-prompt-shelf", b.len() - 1);
    r.shutdown();
    assert!(!files(&dir).is_empty());
    for path in files(&dir) {
        assert_ne!(r.store.as_ref().unwrap().open_reader(&path).unwrap().info.key, system,
            "prefix entries never reach the archive");
    }
    restart(&mut r, config);
    let disk_id = r.sessions.best(&b, "owner", &[], 0, None).unwrap();
    assert!(r.sessions.get(disk_id).is_some_and(|s| !s.prefix && s.state.is_none() && s.on_disk()));
    let disk = run8(&mut r, &b, "same-prompt-disk-after-restart", b.len() - 1);
    assert!(r.sessions.get(prefix_id).is_none_or(|s| !s.prefix), "restart never adopts prefixes");
    // Fresh sequence and empty shelf, using the same prefix-capture prefill shape as the first request.
    reset(&mut r, 8);
    let fresh = run8(&mut r, &b, "system-b-fresh", 0);
    drop(run8);
    for (label, actual) in [("shared", shared), ("live", live), ("shelf", shelf), ("disk", disk)] {
        assert!(actual.1.iter().chain(&fresh.1).all(|x| x.is_finite()));
        let max_abs = actual.1.iter().zip(&fresh.1).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let row = json!({"label": label, "tokens_equal": actual.0 == fresh.0, "top1_logit_max_abs_diff": max_abs});
        eprintln!("PREFIX_RETRY_CHECK {row}");
        results.push(row);
        assert_eq!(actual.0, fresh.0, "{label}: eight greedy token ids differ from fresh prefill");
    }
    r.assistant_token = tok.token_to_id("<｜Assistant｜>");
    let boundary = a.iter().rposition(|t| Some(*t) == r.assistant_token).unwrap();
    // the rollback point stays a few tokens before the header: the encoder merges a user turn appended after a
    // cancelled one into the previous user message ("...u1\n\nu2"), and BPE changes the end of u1
    let rewind = boundary - rewind_margin();
    let p = GenParams {
        max_new: 8, temperature: 0.0, top_p: 1.0, min_p: 0.0, seed: Some(0), logprobs: None, think: None,
        stop_token_ids: vec![], hesitation: None, images: Default::default(), client: "owner".into(), priority: false,
    };
    // cancel `first` after two tokens, send `second` (the cancelled assistant omitted); the reference prefills the
    // same pieces as the engine did for `first`, because CPU expert placement and BF16 GEMM reduction depend on the
    // batch shape (merging the new turn into an earlier batch is a different numerical experiment)
    let mut cancel_then = |r: &mut Real, first: &[u32], second: &[u32], label: &str| {
        reset(r, 8);
        let mut count = 0;
        let fin = r.generate(first, &p, &mut |_| { count += 1; count < 3 }).unwrap();
        assert_eq!(fin, Finish::Cancelled);
        assert_eq!(r.snap.as_ref().unwrap().pos, rewind, "{label}: the rollback point of the cancelled prompt");
        r.cancelled();
        let mut after = vec![];
        r.generate(second, &p, &mut |id| { after.push(id); true }).unwrap();
        assert_eq!(r.stats.cached_prompt_tokens, rewind, "{label}: cancelled assistant omitted: preserve the user history");
        let reuse = r.stats.cached_prompt_tokens;
        reset(r, 8);
        let pb = r.gm.pb.take().unwrap();
        r.gm.prefill(&pb, &second[..system.len()]).unwrap();
        r.gm.prefill(&pb, &second[system.len()..rewind]).unwrap();
        r.gm.pb = Some(pb);
        r.cached = second[..rewind].to_vec();
        let mut reference = vec![];
        r.generate(second, &p, &mut |id| { reference.push(id); true }).unwrap();
        assert_eq!(after, reference, "{label}: cancelled continuation differs from fresh with the same prefill boundaries");
        json!({"label": label, "cached_prompt_tokens": reuse, "prompt_tokens": second.len(), "tokens_equal": true})
    };
    // (1) token level: the old prompt up to the header, then a new user turn
    let mut next = a[..boundary].to_vec();
    next.extend_from_slice(tok.encode("<｜User｜>Continue after cancellation.<｜Assistant｜><think>", false).unwrap().get_ids());
    results.push(cancel_then(&mut r, &a, &next, "cancelled-assistant-omitted"));
    // (2) as the client sends it: the text of both user turns merged, tokenised as one prompt. BPE moves the first
    // difference before the header (observed 28.10 on the server: common prefix = header - 1)
    let merged = prompt("Write eight observations about the ALPHA engineering report.\n\nAnd mention the budget.");
    let lcp = a.iter().zip(&merged).take_while(|(x, y)| x == y).count();
    eprintln!("PREFIX_RETRY_CHECK merged prompt: common prefix {lcp} of header {boundary}, rollback {rewind}");
    assert!(lcp >= rewind, "tokenisation changed more than {} tokens before the header: common prefix {lcp}", rewind_margin());
    results.push(cancel_then(&mut r, &a, &merged, "cancelled-assistant-omitted-merged-user-turns"));
    std::fs::write(out.join("prefix-retry-results.json"), serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

#[test]
#[ignore = "exclusive full-model context/cancellation acceptance"]
fn gpu_dynamic_context_acceptance() {
    let env = |k: &str| std::env::var(k).ok().map(PathBuf::from);
    let model = env("DSX_TEST_MODEL").expect("DSX_TEST_MODEL");
    let tok = tokenizers::Tokenizer::from_file(env("DSX_TEST_TOKENIZER").unwrap()).unwrap();
    let out = env("DSX_TEST_OUT").unwrap();
    std::fs::create_dir_all(&out).unwrap();
    assert_eq!(std::env::var("XS_DYNAMIC_CTX").as_deref(), Ok("1"));
    assert_eq!(std::env::var("DSX_TEST_EXCLUSIVE").as_deref(), Ok("1"),
        "reserve the test GPUs and set DSX_TEST_EXCLUSIVE=1");
    let _ = tracing_subscriber::fmt().with_ansi(false).with_env_filter("info").try_init();
    let config = SessionConfig::new(8, 2.0, 2048).unwrap().with_disk(
        crate::server::sessions::DiskConfig::new(out.join(format!("archive-{}", unix_ms())), 64, 20.0, 7.0, Duration::from_secs(1), 0.0).unwrap()
    ).unwrap();
    let args = RealEngineArgs {
        model, engram: env("DSX_TEST_ENGRAM"), ctx: 327680, chunk: 4096,
        gpu0_gb: -1.5, gpu1_gb: -1.0, arena_gb: 150.0, rank: env("DSX_TEST_RANK"), vision: None,
        sessions: config.clone(), user_token: tok.token_to_id("<｜User｜>").unwrap(),
        assistant_token: tok.token_to_id("<｜Assistant｜>"),
    };
    let mut r = load_real(&args).unwrap();
    let prompt = text_tokens(&tok, "DYNAMIC context preserved state", 4097);
    let p = GenParams { max_new: 8, temperature: 0.0, top_p: 1.0, min_p: 0.0, seed: Some(0), logprobs: None,
        think: None, stop_token_ids: vec![], hesitation: None, images: Default::default(), client: "owner".into(), priority: false };
    r.generate(&prompt, &p, &mut |_| true).unwrap();
    let state = r.gm.save_session(r.snap.as_ref()).unwrap();
    let original = state.data().to_vec();
    let mut results = vec![];
    for cap in [131072usize, 262144, 327680, 65536] {
        r.gm.ensure_context_capacity(cap, true).unwrap();
        assert_eq!(r.gm.context_memory().0, cap);
        let actual = r.gm.save_session(r.snap.as_ref()).unwrap();
        assert!(actual.data() == original, "resizing changed live state at {cap}, byte {:?}",
            actual.data().iter().zip(&original).position(|(a,b)| a != b));
        results.push(json!({"capacity":cap,"kv_bytes":r.gm.context_memory().1,
            "expert_bytes":r.gm.context_memory().2,"state_equal":true}));
    }
    // A loaded disk conversation must commit enough physical KV before copying any data.
    r.cancelled();
    r.shutdown();
    restart(&mut r, config);
    let mut retry = vec![];
    r.generate(&prompt, &p, &mut |t| { retry.push(t); true }).unwrap();
    assert_eq!(r.stats.cached_prompt_tokens, prompt.len()-1);
    // Old BF16 files use the same on-disk layout even though the live caches are packed.
    if let Some(path) = env("DSX_TEST_ARCHIVE_FILE") {
        let mut reader = r.store.as_ref().unwrap().open_reader(&path).unwrap();
        let meta = SessionMeta::decode(&reader.meta).unwrap();
        let mut bytes = Vec::with_capacity(meta.data_len);
        let mut input = dsx_core::server::session_store::AlignedBuf::new(32 << 20);
        while bytes.len() < meta.data_len {
            let n = reader.read(input.as_mut_slice()).unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&input.as_slice()[..n]);
        }
        reader.finish().unwrap();
        r.load_file(&path).unwrap();
        let restored = r.gm.save_session(r.snap.as_ref()).unwrap();
        assert!(restored.data() == bytes, "old archive roundtrip changed bits at {:?}",
            restored.data().iter().zip(&bytes).position(|(a,b)| a != b));
        results.push(json!({"old_archive_pos":meta.pos,"old_archive_bytes":meta.data_len,"equal":true}));
    }
    std::fs::write(out.join("dynamic-context-results.json"), serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}
