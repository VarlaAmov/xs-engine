#![cfg(feature = "cuda")]
use dsx::gpu::Gpu;

#[test]
fn virtual_cache_grows_and_shrinks_without_moving_or_changing_prefix() -> anyhow::Result<()> {
    if !std::path::Path::new("/dev/nvidia0").exists() { return Ok(()); }
    let g = Gpu::new(0)?;
    let mut b = g.alloc_virtual(16 << 20, 2 << 20)?;
    let address = b.ptr;
    let input: Vec<u32> = (0..4096).map(|x| x * 7919).collect();
    g.htod_sync(b.ptr, &input)?;
    g.quiesce()?;
    b.resize_mapped(10 << 20)?;
    assert_eq!(b.ptr, address);
    assert_eq!(g.download::<u32>(b.ptr, input.len())?, input);
    g.htod_sync(b.ptr + (8 << 20), &input)?;
    g.quiesce()?;
    b.resize_mapped(2 << 20)?;
    assert_eq!(b.mapped_bytes(), 2 << 20);
    assert_eq!(g.download::<u32>(b.ptr, input.len())?, input);
    assert!(b.resize_mapped(18 << 20).is_err());
    assert_eq!(g.download::<u32>(b.ptr, input.len())?, input);
    Ok(())
}

#[test]
fn packed_cache_matches_existing_fake_fp4_and_bf16_archive_roundtrip() -> anyhow::Result<()> {
    use dsx::gpu::Arg;
    if !std::path::Path::new("/dev/nvidia0").exists() { return Ok(()); }
    let g = Gpu::new(0)?;
    let mut state = 7u64;
    let mut input: Vec<f32> = (0..32768).map(|i| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 32) as i32 as f32) / (i32::MAX as f32);
        if i % 97 == 0 { -0.0 } else { x * 2f32.powi((i / 128 % 70) - 35) }
    }).collect();
    // E4M3 subnormal scale rounds .0033 -> .00390625, max quantized code is 4.
    // Recomputing scale from the archived maximum would choose .001953125 and
    // clip 4*scale to 6*(scale/2). Include signed zero and every small FP4 code.
    for i in 0..16 { input[i] = if i == 0 { 0.018 } else { (i as f32 - 8.) * 0.0016 }; }
    for block in [16usize, 32] {
        let mut expected = input.clone();
        dsx::numerics::fake_fp4(&mut expected, block, block == 16);
        let bits: Vec<u16> = expected.iter().map(|&x| half::bf16::from_f32(x).to_bits()).collect();
        let x = g.upload(&input)?;
        let packed = g.alloc(input.len() / block * (block / 2 + 1))?;
        g.launch(g.kernel("cache_pack_f32")?, ((input.len() / block).div_ceil(4) as u32, 1, 1), 128, 0,
            &[Arg::P(x.ptr), Arg::P(packed.ptr), Arg::I(input.len() as i32), Arg::I(block as i32), Arg::I((block == 16) as i32), Arg::I(0)])?;
        let out = g.alloc(input.len() * 2)?;
        let tag = if block == 16 { 1 } else { 3 };
        g.launch(g.kernel("cache_unpack")?, ((input.len()).div_ceil(256) as u32, 1, 1), 256, 0,
            &[Arg::P(packed.ptr | tag), Arg::P(out.ptr), Arg::I(0), Arg::I(input.len() as i32)])?;
        assert_eq!(g.download::<u16>(out.ptr, bits.len())?, bits, "packed {block}");
        // Reimport the original (unpacked) archive representation and compare every BF16 bit again.
        let raw = g.upload(&bits)?;
        let error = g.alloc_zeroed(4)?;
        g.launch(g.kernel("cache_pack_bf16")?, ((input.len() / block).div_ceil(4) as u32, 1, 1), 128, 0,
            &[Arg::P(raw.ptr), Arg::P(packed.ptr), Arg::I(input.len() as i32), Arg::I(block as i32), Arg::I((block == 16) as i32), Arg::I(0), Arg::P(error.ptr)])?;
        g.launch(g.kernel("cache_unpack")?, ((input.len()).div_ceil(256) as u32, 1, 1), 256, 0,
            &[Arg::P(packed.ptr | tag), Arg::P(out.ptr), Arg::I(0), Arg::I(input.len() as i32)])?;
        assert_eq!(g.download::<u16>(out.ptr, bits.len())?, bits, "archive {block}");
        assert_eq!(g.download::<u32>(error.ptr, 1)?[0], 0);
        let mut invalid = vec![half::bf16::from_f32(1.0).to_bits(); block];
        invalid[1] = half::bf16::from_f32(1.1015625).to_bits();
        let invalid = g.upload(&invalid)?;
        g.launch(g.kernel("cache_pack_bf16")?, (1, 1, 1), 128, 0,
            &[Arg::P(invalid.ptr), Arg::P(packed.ptr), Arg::I(block as i32), Arg::I(block as i32), Arg::I((block == 16) as i32), Arg::I(0), Arg::P(error.ptr)])?;
        assert_eq!(g.download::<u32>(error.ptr, 1)?[0], 1, "off-grid archive values must be rejected");
    }
    Ok(())
}
