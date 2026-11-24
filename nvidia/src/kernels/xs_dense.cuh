// xs-engine: one-token GEMVs for the Q5_K dense weights of DeepSeek XS (q_b, o_a, o_b, shared expert, head).
// Same planes as gemv_q5p (q6.rs from_q5k_row): lo nibbles (k/2; weight g in byte g/2, nibble g%2), 5th bits (k/8;
// bit g%8 of byte g/8), (scale, min) bytes per 32 (k/16), f16 (d, dmin) per 256 (k/128 halves); w = d*s*c - dmin*m.
// Differences from gemv_q5p4: a code becomes an exact float by PRMT + FADD (2^23 + c - 2^23) instead of the
// quarter-rate I2F, LPR lanes split a row's k (32 / 16 / 8 so short k still fills a warp) and each lane group takes
// R rows, so one x chunk serves R rows. fp32 throughout; per 32-weight sub-block d*s*sum(c x) - dmin*m*sum(x) as in
// gemv_q5p, only the order of the float sums differs. One token only (decode); T > 1 stays on gemv_q5p / gemv_q5p4.

__device__ __forceinline__ float xs_cf(unsigned q, int b) {  // byte b of q (a code 0..31) as an exact float
  return __int_as_float(__byte_perm(q, 0x4B000000u, 0x7540 | b)) - 8388608.f;
}
// 5th bits of the 4 even weights (or odd ones, after >> 1) of an 8-weight group -> bit 4 of bytes 0..3; the products
// of the multiply land on distinct or carry-isolated bits, the mask keeps bits 4, 12, 20, 28
__device__ __forceinline__ unsigned xs_spread5(unsigned h8) { return ((h8 & 0x55u) * 0x00410410u) & 0x10101010u; }

// sum over 32 weights of code * x for one row's sub-block (L: 16 lo bytes, H: the 32 5th bits)
__device__ __forceinline__ float xs_q5_dot32(uint4 L, unsigned H, const float4 *xv) {
  float s = 0.f;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    const unsigned lw = w == 0 ? L.x : w == 1 ? L.y : w == 2 ? L.z : L.w;
    const unsigned h8 = H >> (8 * w);
    const unsigned qe = (lw & 0x0f0f0f0fu) | xs_spread5(h8), qo = ((lw >> 4) & 0x0f0f0f0fu) | xs_spread5(h8 >> 1);
    const float4 a = xv[2 * w], b = xv[2 * w + 1];
    s = fmaf(xs_cf(qe, 0), a.x, s);
    s = fmaf(xs_cf(qo, 0), a.y, s);
    s = fmaf(xs_cf(qe, 1), a.z, s);
    s = fmaf(xs_cf(qo, 1), a.w, s);
    s = fmaf(xs_cf(qe, 2), b.x, s);
    s = fmaf(xs_cf(qo, 2), b.y, s);
    s = fmaf(xs_cf(qe, 3), b.z, s);
    s = fmaf(xs_cf(qo, 3), b.w, s);
  }
  return s;
}

// y[r] = W[r] . x[(r / rpg) * xgs ..] (rpg = rows, xgs = 0: plain GEMV; o_a: its groups each read their slice of o).
// round: y rounded to bf16 (o_a's rounding in the epilogue). 8 warps per block; grid ceil(rows / (8 * (32/LPR) * R)).
template <int LPR, int R>
__device__ __forceinline__ void xs_gemv_q5(const uint8_t *__restrict__ lo, const float *__restrict__ x, float *__restrict__ y, int rows,
                                           int k, const uint8_t *__restrict__ hb, const uint8_t *__restrict__ sm,
                                           const __half *__restrict__ dd, int rpg, int xgs, int round) {
  constexpr int G = 32 / LPR;
  const int lane = threadIdx.x & 31, sub = lane % LPR, grp = lane / LPR;
  const int warp = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int r0 = (warp * G + grp) * R;
  if (warp * G * R >= rows) return;
  const bool live = r0 < rows;
  const float *xr = x + (size_t)(min(r0, rows - 1) / rpg) * xgs;
  float acc[R];
#pragma unroll
  for (int rr = 0; rr < R; rr++) acc[rr] = 0.f;
  const size_t k2 = k / 2, k8 = k / 8, k16 = k / 16, k128 = k / 128;
  for (int i = sub * 32; live && i < k; i += 32 * LPR) {
    float4 xv[8];
    const float4 *xp = (const float4 *)(xr + i);
#pragma unroll
    for (int q = 0; q < 8; q++) xv[q] = __ldg(xp + q);
    float sx = 0.f;
#pragma unroll
    for (int q = 0; q < 8; q++) sx += (xv[q].x + xv[q].y) + (xv[q].z + xv[q].w);
    uint4 L[R];
    unsigned H[R];
    uchar2 S[R];
    half2 D[R];
#pragma unroll
    for (int rr = 0; rr < R; rr++) {
      const size_t r = min(r0 + rr, rows - 1);
      L[rr] = __ldg((const uint4 *)(lo + r * k2 + i / 2));
      H[rr] = __ldg((const unsigned *)(hb + r * k8 + i / 8));
      S[rr] = __ldg((const uchar2 *)(sm + r * k16 + i / 16));
      D[rr] = __ldg((const half2 *)(dd + r * k128 + i / 256 * 2));
    }
#pragma unroll
    for (int rr = 0; rr < R; rr++) {
      const float s = xs_q5_dot32(L[rr], H[rr], xv);
      const float a = __low2float(D[rr]) * (float)S[rr].x, b = __high2float(D[rr]) * (float)S[rr].y;
      acc[rr] += a * s - b * sx;
    }
  }
#pragma unroll
  for (int rr = 0; rr < R; rr++) {
    float s = acc[rr];
#pragma unroll
    for (int o = LPR / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (sub == 0 && r0 + rr < rows) y[r0 + rr] = round ? bf16r(s) : s;
  }
}

#define XS_Q5_ARGS                                                                                                      \
  const uint8_t *__restrict__ lo, const float *__restrict__ x, float *__restrict__ y, int rows, int k,                \
      const uint8_t *__restrict__ hb, const uint8_t *__restrict__ sm, const __half *__restrict__ dd, int rpg, int xgs, int round
// block 256; rows per block: 32x4 -> 32, 16x4 -> 64, 8x4 -> 128, 8x2 -> 64
extern "C" __global__ void __launch_bounds__(256) xs_q5_32x4(XS_Q5_ARGS) { xs_gemv_q5<32, 4>(lo, x, y, rows, k, hb, sm, dd, rpg, xgs, round); }
extern "C" __global__ void __launch_bounds__(256) xs_q5_16x4(XS_Q5_ARGS) { xs_gemv_q5<16, 4>(lo, x, y, rows, k, hb, sm, dd, rpg, xgs, round); }
extern "C" __global__ void __launch_bounds__(256) xs_q5_8x4(XS_Q5_ARGS) { xs_gemv_q5<8, 4>(lo, x, y, rows, k, hb, sm, dd, rpg, xgs, round); }
extern "C" __global__ void __launch_bounds__(256) xs_q5_8x2(XS_Q5_ARGS) { xs_gemv_q5<8, 2>(lo, x, y, rows, k, hb, sm, dd, rpg, xgs, round); }

// Shared expert gate + up + SwiGLU in one launch: lane group takes R row indices of both matrices (one x chunk for
// 2R rows), h[r] = bf16(silu(min(g, lim)) * clamp(u, -lim, lim)) exactly as swiglu (weight 1). k % 1024 == 0.
template <int R>
__device__ __forceinline__ void xs_gu_q5(const uint8_t *__restrict__ glo, const uint8_t *__restrict__ ghb, const uint8_t *__restrict__ gsm,
                                         const __half *__restrict__ gdd, const uint8_t *__restrict__ ulo, const uint8_t *__restrict__ uhb,
                                         const uint8_t *__restrict__ usm, const __half *__restrict__ udd, const float *__restrict__ x,
                                         float *__restrict__ h, int rows, int k, float lim) {
  const int lane = threadIdx.x & 31;
  const int r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * R;
  if (r0 >= rows) return;
  float ga[R], ua[R];
#pragma unroll
  for (int rr = 0; rr < R; rr++) ga[rr] = ua[rr] = 0.f;
  const size_t k2 = k / 2, k8 = k / 8, k16 = k / 16, k128 = k / 128;
  for (int i = lane * 32; i < k; i += 1024) {
    float4 xv[8];
    const float4 *xp = (const float4 *)(x + i);
#pragma unroll
    for (int q = 0; q < 8; q++) xv[q] = __ldg(xp + q);
    float sx = 0.f;
#pragma unroll
    for (int q = 0; q < 8; q++) sx += (xv[q].x + xv[q].y) + (xv[q].z + xv[q].w);
#pragma unroll
    for (int rr = 0; rr < R; rr++) {
      const size_t r = min(r0 + rr, rows - 1);
      const uint4 GL = __ldg((const uint4 *)(glo + r * k2 + i / 2)), UL = __ldg((const uint4 *)(ulo + r * k2 + i / 2));
      const unsigned GH = __ldg((const unsigned *)(ghb + r * k8 + i / 8)), UH = __ldg((const unsigned *)(uhb + r * k8 + i / 8));
      const uchar2 GS = __ldg((const uchar2 *)(gsm + r * k16 + i / 16)), US = __ldg((const uchar2 *)(usm + r * k16 + i / 16));
      const half2 GD = __ldg((const half2 *)(gdd + r * k128 + i / 256 * 2)), UD = __ldg((const half2 *)(udd + r * k128 + i / 256 * 2));
      const float gs = xs_q5_dot32(GL, GH, xv), us = xs_q5_dot32(UL, UH, xv);
      ga[rr] += __low2float(GD) * (float)GS.x * gs - __high2float(GD) * (float)GS.y * sx;
      ua[rr] += __low2float(UD) * (float)US.x * us - __high2float(UD) * (float)US.y * sx;
    }
  }
#pragma unroll
  for (int rr = 0; rr < R; rr++) {
    const float g = warp_sum(ga[rr]), u = warp_sum(ua[rr]);
    if (lane == 0 && r0 + rr < rows) {
      const float gv = fminf(g, lim), uv = fminf(fmaxf(u, -lim), lim);
      h[r0 + rr] = bf16r(gv / (1.f + expf(-gv)) * uv * 1.f);
    }
  }
}
// grid ceil(rows / 16), block 256
extern "C" __global__ void __launch_bounds__(256) xs_q5_gu(const uint8_t *__restrict__ glo, const uint8_t *__restrict__ ghb,
                                                          const uint8_t *__restrict__ gsm, const __half *__restrict__ gdd,
                                                          const uint8_t *__restrict__ ulo, const uint8_t *__restrict__ uhb,
                                                          const uint8_t *__restrict__ usm, const __half *__restrict__ udd,
                                                          const float *__restrict__ x, float *__restrict__ h, int rows, int k, float lim) {
  xs_gu_q5<2>(glo, ghb, gsm, gdd, ulo, uhb, usm, udd, x, h, rows, k, lim);
}
