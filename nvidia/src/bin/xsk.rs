//! xs-engine CPU kernel bench: IQ2_XXS in GGUF layout (rows_iq2_xxs) against the xs layout (rows_iq2_xxs_xs).
//! Checks that both give bit-identical outputs, then times them in L2 (1 thread) and streamed from RAM (a pool).
//! XSK_CPUS=0-17 (pool CPUs), XSK_CALLER=19, XSK_EXPERTS=32.

use dsx::cpu::pool::{Ctx, Pool, parse_cpus, pin_to};
use dsx::cpu::qdot::{QAct, dot_rows_sel, dot_rows_sel_xxs_xs, iq2_xxs_from_xs, iq2_xxs_to_xs};
use dsx::gguf::GType;
use half::f16;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

const DIM: usize = 5120;
const INTER: usize = 2304;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn normal(&mut self) -> f32 {
        let a = ((self.u64() >> 40) as f32 / (1u64 << 24) as f32).max(1e-7);
        let b = (self.u64() >> 40) as f32 / (1u64 << 24) as f32;
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn rand_iq2xxs(nr: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let n = GType::Iq2Xxs.row_bytes(cols) * nr;
    let mut v: Vec<u8> = (0..n).map(|_| rng.u64() as u8).collect();
    for blk in v.chunks_exact_mut(66) {
        blk[..2].copy_from_slice(&f16::from_f32(0.02).to_le_bytes());
    }
    v
}

fn to_xs(v: &[u8]) -> Vec<u8> {
    let mut w = v.to_vec();
    w.chunks_exact_mut(66).for_each(iq2_xxs_to_xs);
    w
}

fn main() {
    if std::env::var("XSK_MODE").is_ok_and(|v| v == "mem") {
        return mem_probe();
    }
    let mut rng = Rng(0x2545f4914f6cdd1d);
    // 1. layout round trip and bit-identical outputs (T = 1..8)
    let rows = rand_iq2xxs(64, DIM, &mut rng);
    let xs = to_xs(&rows);
    let mut back = xs.clone();
    back.chunks_exact_mut(66).for_each(iq2_xxs_from_xs);
    assert!(back == rows, "xs layout does not round-trip");
    for t in 1..=8usize {
        let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
        let act = QAct::new(&x, t, DIM, &[GType::Iq2Xxs]);
        let toks: Vec<usize> = (0..t).collect();
        let (mut a, mut b) = (vec![0f32; t * 64], vec![0f32; t * 64]);
        dot_rows_sel(GType::Iq2Xxs, &rows, DIM, &act, &toks, &mut a, 64);
        dot_rows_sel_xxs_xs(&xs, DIM, &act, &toks, &mut b, 64);
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "T={t}: outputs differ");
    }
    println!("check: xs layout round-trips and gives bit-identical outputs for T=1..8");

    // 2. in-cache single thread (64 rows x 5120, ~84 KB)
    let caller: usize = std::env::var("XSK_CALLER").ok().and_then(|v| v.parse().ok()).unwrap_or(19);
    pin_to(0);
    for t in [1usize, 2, 4, 8] {
        let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
        let act = QAct::new(&x, t, DIM, &[GType::Iq2Xxs]);
        let toks: Vec<usize> = (0..t).collect();
        let mut out = vec![0f32; t * 64];
        let reps = 4000 / t;
        let mut line = format!("in L2, 1 thread, T={t}:");
        for (name, w, xsl) in [("gguf", &rows, false), ("xs", &xs, true)] {
            let t0 = Instant::now();
            for _ in 0..reps {
                if xsl {
                    dot_rows_sel_xxs_xs(w, DIM, &act, &toks, &mut out, 64);
                } else {
                    dot_rows_sel(GType::Iq2Xxs, w, DIM, &act, &toks, &mut out, 64);
                }
            }
            let gbs = (w.len() * reps) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            line += &format!("  {name} {gbs:5.2} GB/s");
        }
        println!("{line}");
    }

    // 3. streamed from RAM: matrices of 2304 x 5120 (gate-sized), rows split over the pool in 64-row units
    let cpus = parse_cpus(&std::env::var("XSK_CPUS").unwrap_or("0-17".into())).unwrap();
    let nexp: usize = std::env::var("XSK_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let mats: Vec<Vec<u8>> = (0..3 * nexp).map(|_| rand_iq2xxs(INTER, DIM, &mut rng)).collect();
    let mats_xs: Vec<Vec<u8>> = mats.iter().map(|m| to_xs(m)).collect();
    pin_to(caller);
    for th in [1usize, 6, 12, cpus.len()] {
        let pool = Pool::new(&cpus[..th], Duration::from_millis(20));
        let x: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
        let act = QAct::new(&x, 1, DIM, &[GType::Iq2Xxs]);
        let mut line = format!("from RAM, {th:2} threads, T=1:");
        for (name, ms, xsl) in [("gguf", &mats, false), ("xs", &mats_xs, true)] {
            let rb = GType::Iq2Xxs.row_bytes(DIM);
            let mut outs = vec![0f32; INTER];
            let outp = outs.as_mut_ptr() as usize;
            let run = |m: &Vec<u8>| {
                let next = AtomicUsize::new(0);
                pool.run(&|_c: &Ctx| loop {
                    let u = next.fetch_add(1, Relaxed);
                    if u * 64 >= INTER {
                        break;
                    }
                    let out = unsafe { std::slice::from_raw_parts_mut((outp as *mut f32).add(u * 64), 64) };
                    let w = &m[u * 64 * rb..(u + 1) * 64 * rb];
                    if xsl {
                        dot_rows_sel_xxs_xs(w, DIM, &act, &[0], out, 64);
                    } else {
                        dot_rows_sel(GType::Iq2Xxs, w, DIM, &act, &[0], out, 64);
                    }
                });
            };
            for m in ms.iter() {
                run(m);
            }
            let t0 = Instant::now();
            let passes = if th == 1 { 1 } else { 3 };
            for _ in 0..passes {
                for m in ms.iter() {
                    run(m);
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            let bytes = (ms.iter().map(|m| m.len()).sum::<usize>() * passes) as f64;
            line += &format!("  {name} {:6.1} GB/s ({:.0} us per matrix)", bytes / dt / 1e9, dt * 1e6 / (ms.len() * passes) as f64);
        }
        println!("{line}");
    }
}

/// XSK_MODE=mem: (a) every pool thread runs the kernel on its own L2-resident rows (compute ceiling under full load),
/// (b) pure streaming read of a 4 GB buffer (DRAM ceiling), for 1/6/12/18 threads.
#[allow(dead_code)]
pub fn mem_probe() {
    let cpus = parse_cpus(&std::env::var("XSK_CPUS").unwrap_or("0-17".into())).unwrap();
    let mut rng = Rng(7);
    let caller: usize = std::env::var("XSK_CALLER").ok().and_then(|v| v.parse().ok()).unwrap_or(19);
    pin_to(caller);
    let per: Vec<Vec<u8>> = (0..cpus.len()).map(|_| rand_iq2xxs(64, DIM, &mut rng)).collect();
    let x: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
    let act = QAct::new(&x, 1, DIM, &[GType::Iq2Xxs]);
    let buf: Vec<u64> = (0..(4usize << 30) / 8).map(|i| i as u64).collect();
    for th in [1usize, 6, 12, cpus.len()] {
        let pool = Pool::new(&cpus[..th], Duration::from_millis(20));
        let reps = 400;
        let t0 = Instant::now();
        pool.run(&|c: &Ctx| {
            let mut out = vec![0f32; 64];
            for _ in 0..reps {
                dot_rows_sel(GType::Iq2Xxs, &per[c.id], DIM, &act, &[0], &mut out, 64);
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        let gb = (per[0].len() * reps * th) as f64 / dt / 1e9;
        let sums = AtomicUsize::new(0);
        let t1 = Instant::now();
        let chunk = buf.len() / th;
        pool.run(&|c: &Ctx| {
            let s: u64 = buf[c.id * chunk..(c.id + 1) * chunk].iter().fold(0u64, |a, &v| a.wrapping_add(v));
            sums.fetch_add(s as usize & 1, Relaxed);
        });
        let dt1 = t1.elapsed().as_secs_f64();
        println!("{th:2} threads: kernel in own L2 {gb:6.1} GB/s total ({:.2}/thread); DRAM stream read {:6.1} GB/s", gb / th as f64, (buf.len() * 8) as f64 / dt1 / 1e9);
    }
}
