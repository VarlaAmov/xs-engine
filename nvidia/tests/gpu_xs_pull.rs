#![cfg(feature = "cuda")]
//! Tri-split kernels over pinned GGUF blocks, independently checked against CPU dequantization.
use dsx::formats::dequant_row;
use dsx::gguf::GType;
use dsx::gpu::*;

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn rows(ty: GType, nr: usize, cols: usize, seed: &mut u64) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..nr * ty.row_bytes(cols)).map(|_| rng(seed) as u8).collect();
    for b in bytes.chunks_exact_mut(ty.block().1) {
        let at = if ty == GType::Q2K { 80 } else { 0 };
        b[at..at + 2].copy_from_slice(&half::f16::from_f32(0.005 * (0.5 + (rng(seed) % 100) as f32 * 0.01)).to_le_bytes());
        if ty == GType::Q2K {
            b[82..84].copy_from_slice(&half::f16::from_f32(0.003 * (0.5 + (rng(seed) % 100) as f32 * 0.01)).to_le_bytes());
        }
    }
    bytes
}

fn quantized(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (x, o) in x.chunks_exact(256).zip(out.chunks_exact_mut(256)) {
        let d = x.iter().fold(0f32, |a, &v| a.max(v.abs())) / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (&v, o) in x.iter().zip(o) { *o = (v * id).round_ties_even() * d; }
    }
    out
}

fn dot(ty: GType, row: &[u8], x: &[f32], scratch: &mut [f32]) -> f32 {
    dequant_row(ty, row, scratch);
    scratch.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32
}

fn rel(want: &[f32], got: &[f32]) -> (f64, f64) {
    let den: f64 = want.iter().map(|&v| (v as f64).powi(2)).sum();
    let num: f64 = want.iter().zip(got).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
    let max = want.iter().zip(got).map(|(&a, &b)| (a - b).abs() as f64).fold(0.0, f64::max);
    ((num / den.max(1e-30)).sqrt(), max / (den / want.len() as f64).max(1e-30).sqrt())
}

#[test]
fn pull_slices_match_cpu() {
    let g = Gpu::new(0).unwrap();
    let (dim, inter, ne, k, layer) = (5120usize, 2304usize, 3usize, 3usize, 1usize);
    let mut seed = 0x7a42_4ea1_864a;
    let sel = [2i32, 1, 0];
    let wts = [0.37f32, 0.0, 0.63];
    let mut x: Vec<f32> = (0..dim).map(|_| (rng(&mut seed) % 2001) as f32 * 0.00025 - 0.25).collect();
    x[..256].fill(0.0);
    x[513] *= 8.0;
    let xq = quantized(&x);
    let dx = g.upload(&x).unwrap();
    let (dq, dd) = (g.alloc(dim).unwrap(), g.alloc(dim / 256 * 4).unwrap());
    g.launch(g.kernel("xs_pull_qx").unwrap(), ((dim / 256) as u32, 1, 1), 256, 0,
        &[Arg::P(dx.ptr), Arg::P(dq.ptr), Arg::P(dd.ptr), Arg::P(0)]).unwrap();
    let dsel = g.upload(&sel).unwrap();
    let dwts = g.upload(&wts).unwrap();
    let dout = g.alloc(dim * 4).unwrap();
    let pairs = [
        (GType::Iq1S, GType::Iq2Xxs), (GType::Iq1S, GType::Iq2S),
        (GType::Iq2Xxs, GType::Q2K), (GType::Q2K, GType::Iq1S),
        (GType::Iq2Xs, GType::Iq3Xxs), (GType::Iq2S, GType::Iq2Xs),
        (GType::Iq3Xxs, GType::Iq2S),
    ];
    for (gu, dn) in pairs {
        let mats: Vec<_> = (0..ne).map(|_| (rows(gu, inter, dim, &mut seed), rows(gu, inter, dim, &mut seed), rows(dn, dim, inter, &mut seed))).collect();
        let mut offsets = Vec::new();
        let mut bytes = Vec::new();
        for m in &mats {
            for b in [&m.0, &m.1, &m.2] {
                // The pull's vector loads include alignment padding at both ends of a row slice.
                bytes.resize((bytes.len() + 31).next_multiple_of(16), 0);
                offsets.push(bytes.len());
                bytes.extend_from_slice(b);
            }
        }
        bytes.resize(bytes.len() + 32, 0);
        let mut arena = HostBuf::new(bytes.len()).unwrap();
        arena.as_mut_slice::<u8>()[..bytes.len()].copy_from_slice(&bytes);
        let base = arena.dev_ptr(&g).unwrap();
        let mut ptrs = vec![0u64; (layer + 1) * ne * 3];
        for (i, &off) in offsets.iter().enumerate() { ptrs[layer * ne * 3 + i] = base + off as u64; }
        let dp = g.upload(&ptrs).unwrap();
        for (c0, c1) in [(0usize, 256usize), (256, 1280), (768, 2304)] {
            for miss in [[1u8, 0, 1], [0, 0, 0]] {
                let mut map = vec![0u8; (layer + 1) * ne];
                map[layer * ne..].copy_from_slice(&miss);
                let dm = g.upload(&map).unwrap();
                let initial = vec![f32::NAN; k * inter];
                let dh = g.upload(&initial).unwrap();
                g.launch(g.kernel(&format!("xs_pull_gu{}", gu as u32)).unwrap(), (2, 1, 1), 256, 0,
                    &[Arg::P(dsel.ptr), Arg::P(dwts.ptr), Arg::P(dm.ptr), Arg::P(dp.ptr), Arg::I(layer as i32), Arg::I(ne as i32), Arg::I(k as i32),
                      Arg::P(dq.ptr), Arg::P(dd.ptr), Arg::P(dh.ptr), Arg::I(c0 as i32), Arg::I(c1 as i32), Arg::I(inter as i32), Arg::F(10.0)]).unwrap();
                let h: Vec<f32> = g.download(dh.ptr, k * inter).unwrap();
                let mut want_h = Vec::new();
                let mut got_h = Vec::new();
                let mut want = vec![0f32; dim];
                let mut scratch = vec![0f32; dim.max(inter)];
                for (j, &e) in sel.iter().enumerate() {
                    let e = e as usize;
                    for r in 0..inter {
                        if miss[e] == 0 || r < c0 || r >= c1 { assert!(h[j * inter + r].is_nan(), "unassigned h overwritten"); }
                    }
                    if miss[e] == 0 { continue; }
                    let (gate, up, down) = &mats[e];
                    let rb = gu.row_bytes(dim);
                    for r in c0..c1 {
                        let gv = dot(gu, &gate[r * rb..(r + 1) * rb], &xq, &mut scratch[..dim]).min(10.0);
                        let uv = dot(gu, &up[r * rb..(r + 1) * rb], &xq, &mut scratch[..dim]).clamp(-10.0, 10.0);
                        want_h.push(half::bf16::from_f32(gv / (1.0 + (-gv).exp()) * uv * wts[j]).to_f32());
                        got_h.push(h[j * inter + r]);
                    }
                    // Check down independently from tiny gate/up BF16 rounding differences.
                    let hq = quantized(&h[j * inter + c0..j * inter + c1]);
                    let rs = dn.row_bytes(inter);
                    let (off, rb) = (dn.row_bytes(c0), dn.row_bytes(c1 - c0));
                    for (r, out) in want.iter_mut().enumerate() {
                        *out += dot(dn, &down[r * rs + off..r * rs + off + rb], &hq, &mut scratch[..c1 - c0]);
                    }
                }
                g.launch(g.kernel(&format!("xs_pull_down{}", dn as u32)).unwrap(), (2, 1, 1), 256, 0,
                    &[Arg::P(dsel.ptr), Arg::P(dm.ptr), Arg::P(dp.ptr), Arg::I(layer as i32), Arg::I(ne as i32), Arg::I(k as i32),
                      Arg::P(dh.ptr), Arg::P(dout.ptr), Arg::I(c0 as i32), Arg::I(c1 as i32), Arg::I(inter as i32), Arg::I(dim as i32)]).unwrap();
                let got: Vec<f32> = g.download(dout.ptr, dim).unwrap();
                if want_h.is_empty() { assert!(got.iter().all(|&v| v == 0.0)); }
                else {
                    let (hr, hm) = rel(&want_h, &got_h);
                    let (dr, dm) = rel(&want, &got);
                    eprintln!("pull {gu:?}/{dn:?} {c0}..{c1}: h rms={hr:.2e} max={hm:.2e}, down rms={dr:.2e} max={dm:.2e}");
                    assert!(hr < 2e-3 && hm < 0.012, "gate/up mismatch");
                    assert!(dr < 1e-6 && dm < 1e-5, "down mismatch");
                }
            }
        }
    }
}
