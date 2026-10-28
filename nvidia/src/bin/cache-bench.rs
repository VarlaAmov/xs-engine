//! Fixed-token benchmark; no sampling or HTTP overhead. GPU work must be exclusive.
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use clap::Parser;
    use dsx::{gpumodel::{GpuModel, Placement}, model::Model};
    use std::{io::Write, path::PathBuf, time::Instant};

    #[derive(Parser)]
    struct Args {
        #[arg(long)] model: PathBuf,
        #[arg(long)] rank: PathBuf,
        #[arg(long)] prompt: PathBuf,
        #[arg(long)] teacher: PathBuf,
        #[arg(long)] logits: PathBuf,
        #[arg(long, default_value_t = 200)] warmup: usize,
        #[arg(long, default_value_t = 500)] tokens: usize,
    }
    fn ids(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
        std::fs::read_to_string(path)?.split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty()).map(|s| Ok(s.parse()?)).collect()
    }
    let args = Args::parse();
    let prompt = ids(&args.prompt)?;
    let teacher = ids(&args.teacher)?;
    ensure!(args.tokens > 0 && teacher.len() >= args.warmup + args.tokens, "insufficient teacher tokens");
    let m = Model::load(&args.model, None)?;
    ensure!(prompt.iter().chain(&teacher).all(|&x| (x as usize) < m.cfg.vocab), "invalid token");
    let mut gm = GpuModel::with_placement(&m, 1, 262144, Placement {
        gpu0_gb: 7.2, aux_gb: 22.7, arena_gb: 135.0, prefill_chunk: 8192, rank: Some(args.rank),
    })?;
    let pb = gm.pb.take().unwrap();
    let prefill = Instant::now();
    gm.prefill(&pb, &prompt)?;
    let prefill_s = prefill.elapsed().as_secs_f64();
    for &token in &teacher[..args.warmup] { gm.step(token)?; }
    if let Some(p) = gm.evprof.as_mut() { p.clear(); }
    if let Some(p) = gm.prof.as_mut() { p.clear(); }
    let (p0, c0) = gm.cpu_picks();
    let (busy0, adapt0, swaps0) = (gm.cpu_busy_s(), gm.adapt_s, gm.adapt_swaps);
    let mut outputs = Vec::with_capacity(args.tokens);
    let start = Instant::now();
    for &token in &teacher[args.warmup..args.warmup + args.tokens] {
        outputs.push(gm.step(token)?);
    }
    let elapsed_s = start.elapsed().as_secs_f64();
    let (p1, c1) = gm.cpu_picks();
    ensure!(outputs.iter().flatten().all(|v| v.is_finite()), "non-finite logits");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&args.logits)?);
    for row in &outputs { for v in row { file.write_all(&v.to_le_bytes())?; } }
    file.flush()?;
    println!("{}", serde_json::json!({
        "prompt_tokens": prompt.len(), "warmup": args.warmup, "tokens": args.tokens,
        "seconds": elapsed_s, "tps": args.tokens as f64 / elapsed_s,
        "prefill_tps": prompt.len() as f64 / prefill_s,
        "cpu_pick_pct": 100.0 * (c1 - c0) as f64 / (p1 - p0).max(1) as f64,
        "cpu_busy_ms": (gm.cpu_busy_s() - busy0) * 1e3 / args.tokens as f64,
        "adapt_ms": (gm.adapt_s - adapt0) * 1e3 / args.tokens as f64,
        "swaps": gm.adapt_swaps - swaps0,
        "engram_cache": dsx_core::engram_cache::stats(),
        "prof_ms": gm.prof.as_ref().map(|p| p.iter().map(|(k,v)| (k.to_string(), *v * 1e3 / args.tokens as f64)).collect::<std::collections::BTreeMap<_,_>>()),
        "events_ms": gm.evprof.as_ref().map(|p| p.iter().map(|(k,v)| (*k, *v * 1e3 / args.tokens as f64)).collect::<std::collections::BTreeMap<_,_>>()),
    }));
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() { eprintln!("cache-bench requires --features cuda"); std::process::exit(1); }
