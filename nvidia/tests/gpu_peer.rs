#![cfg(feature = "cuda")]
//! Run with exclusive GPUs: checks actual DMA, cross-device event ordering and mixed fallback sums.
use dsx::gpu::*;

#[test]
fn peer_roundtrip_and_cross_device_events() {
    if Gpu::count().unwrap() < 2 { return; }
    let a = Gpu::new(0).unwrap();
    let b = Gpu::new(1).unwrap();
    assert!(a.enable_peer(&b).unwrap());
    assert!(b.enable_peer(&a).unwrap());
    // Repeated enable is legal (the server may create several models in one process).
    assert!(a.enable_peer(&b).unwrap());
    for n in [32usize, 8 * 5120 * 4, 64 << 20] {
        let input: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(131) ^ (i >> 8)) as u8).collect();
        let src = a.upload(&input).unwrap();
        let dst = b.alloc(n).unwrap();
        let back = a.alloc(n).unwrap();
        let ready = a.event().unwrap();
        let done = b.event().unwrap();
        a.record(ready, false).unwrap();
        b.wait(ready, false).unwrap();
        b.peer_from(dst.ptr, &a, src.ptr, n).unwrap();
        b.peer_to(&a, back.ptr, dst.ptr, n).unwrap();
        b.record(done, false).unwrap();
        a.wait(done, false).unwrap();
        assert_eq!(a.download::<u8>(back.ptr, n).unwrap(), input);
    }
}

#[test]
fn combine_mixed_peer_and_host_is_exact_for_verify_batches() {
    let g = Gpu::new(0).unwrap();
    let d = 16;
    let xch = dsx::handoff::Xch::new(d).unwrap();
    let xdev = xch.buf.dev_ptr(&g).unwrap();
    for t in [1, 8] {
        let a = g.upload(&vec![1f32; t * d]).unwrap();
        let out = g.upload(&vec![2f32; t * d]).unwrap();
        xch.y_cpu(t).fill(3.0);
        // Mask selects aux0 from GPU memory, aux1 from its host fallback slot.
        xch.y_aux(0, t).fill(-999.0);
        xch.y_aux(1, t).fill(5.0);
        let sums = g.upload(&vec![4f32; 2 * 8 * d]).unwrap();
        g.launch(g.kernel("combine_experts").unwrap(), (1, 1, 1), 256, 0,
            &[Arg::P(a.ptr), Arg::P(out.ptr), Arg::P(xdev), Arg::I(d as i32), Arg::I(1),
              Arg::I(2), Arg::I(t as i32), Arg::I(dsx::handoff::y_cpu2(d) as i32),
              Arg::I(dsx::handoff::y_cpu3(d) as i32), Arg::P(sums.ptr), Arg::I(1)]).unwrap();
        assert_eq!(g.download::<f32>(a.ptr, t * d).unwrap(), vec![15.0; t * d]);
    }
}

#[test]
#[ignore = "exclusive GPUs, measures real peer bandwidth"]
fn peer_bandwidth() {
    let gpus = [Gpu::new(0).unwrap(), Gpu::new(1).unwrap()];
    assert!(gpus[0].enable_peer(&gpus[1]).unwrap());
    assert!(gpus[1].enable_peer(&gpus[0]).unwrap());
    let n = 64usize << 20;
    let input: Vec<u8> = (0..n).map(|i| (i ^ (i >> 8) ^ (i >> 20)) as u8).collect();
    for source in 0..2 {
        let (src, dst) = (&gpus[source], &gpus[1 - source]);
        let a = src.upload(&input).unwrap();
        let b = dst.alloc(n).unwrap();
        for _ in 0..5 { dst.peer_from(b.ptr, src, a.ptr, n).unwrap(); }
        dst.sync().unwrap();
        let (begin, end) = (dst.timed_event().unwrap(), dst.timed_event().unwrap());
        dst.record(begin, false).unwrap();
        for _ in 0..128 { dst.peer_from(b.ptr, src, a.ptr, n).unwrap(); }
        dst.record(end, false).unwrap();
        dst.sync().unwrap();
        let ms = dst.elapsed_ms(begin, end).unwrap();
        assert_eq!(dst.download::<u8>(b.ptr, n).unwrap(), input);
        eprintln!("PEER_BANDWIDTH GPU{}->GPU{} {:.2} GB/s (128 x 64 MiB, CUDA events, exact bytes)",
                  src.ordinal, dst.ordinal, 128.0 * n as f64 / (ms as f64 * 1e6));
    }
}
