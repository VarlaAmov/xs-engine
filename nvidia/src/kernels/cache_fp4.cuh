// Lossless storage of the model's existing fake-FP4 values. Low pointer bits tag the
// representation: 0 = BF16, 1 = E2M1 + E4M3 scale per 16, 3 = E2M1 + E8M0 per 32.
// Tags are decoded before any address arithmetic; no tagged pointer is dereferenced.
__device__ __forceinline__ float cache_e4(uint8_t x) {
  int e = (x >> 3) & 15, m = x & 7;
  return e == 0 ? (float)m * 0.001953125f : ldexpf(1.f + (float)m * 0.125f, e - 7);
}
__device__ __forceinline__ uint8_t cache_e4_code(float x) {
  if (x < 0.015625f) return (uint8_t)__float2int_rn(x * 512.f);
  uint32_t b = __float_as_uint(x);
  return (uint8_t)((((b >> 23) & 255) - 120) * 8 + ((b >> 20) & 7));
}
__device__ __forceinline__ float cache_e2(uint8_t x) {
  const float v[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return copysignf(v[x & 7], (x & 8) ? -1.f : 1.f);
}
__device__ __forceinline__ uint8_t cache_e2_code(float x) {
  float a = fabsf(x);
  int i = a == 0.f ? 0 : a == .5f ? 1 : a == 1.f ? 2 : a == 1.5f ? 3 : a == 2.f ? 4 : a == 3.f ? 5 : a == 4.f ? 6 : 7;
  return i | ((__float_as_uint(x) >> 28) & 8);
}
__device__ __forceinline__ __nv_bfloat16 cache_read(const __nv_bfloat16 *p, size_t i) {
  uintptr_t tag = (uintptr_t)p & 3;
  if (!tag) return p[i];
  int block = tag == 1 ? 16 : 32, stride = block / 2 + 1;
  const uint8_t *b = (const uint8_t *)((uintptr_t)p & ~(uintptr_t)3) + (i / block) * stride;
  uint8_t sc = b[block / 2];
  float scale = tag == 1 ? cache_e4(sc) : __uint_as_float((uint32_t)sc << 23);
  uint8_t code = (b[(i % block) / 2] >> ((i & 1) * 4)) & 15;
  return __float2bfloat16(cache_e2(code) * scale);
}
template <int BLOCK, bool E4, int N>
__device__ __forceinline__ void cache_read_vec(const __nv_bfloat16 *p, size_t i, __nv_bfloat16 *dst) {
  const uint8_t *b = (const uint8_t *)((uintptr_t)p & ~(uintptr_t)3) + (i / BLOCK) * (BLOCK / 2 + 1);
  uint8_t sc = __ldg(b + BLOCK / 2);
  float scale = E4 ? cache_e4(sc) : __uint_as_float((uint32_t)sc << 23);
#pragma unroll
  for (int j = 0; j < N; j += 2) {
    uint8_t pair = __ldg(b + ((i % BLOCK) + j) / 2);
    dst[j] = __float2bfloat16(cache_e2(pair & 15) * scale);
    dst[j + 1] = __float2bfloat16(cache_e2(pair >> 4) * scale);
  }
}
__device__ __forceinline__ uint4 cache_read8(const __nv_bfloat16 *p, size_t i) {
  uintptr_t tag = (uintptr_t)p & 3;
  if (!tag) return *(const uint4 *)(p + i);
  uint4 v;
  if (tag == 1) cache_read_vec<16, true, 8>(p, i, (__nv_bfloat16 *)&v);
  else cache_read_vec<32, false, 8>(p, i, (__nv_bfloat16 *)&v);
  return v;
}
__device__ __forceinline__ uint2 cache_read4(const __nv_bfloat16 *p, size_t i) {
  uintptr_t tag = (uintptr_t)p & 3;
  if (!tag) return __ldg((const uint2 *)(p + i));
  uint2 v;
  if (tag == 1) cache_read_vec<16, true, 4>(p, i, (__nv_bfloat16 *)&v);
  else cache_read_vec<32, false, 4>(p, i, (__nv_bfloat16 *)&v);
  return v;
}

// Each warp handles a 16- or 32-value block. Compute precisely the old fake_fp4 scale,
// preserve rounded f32 activations, and store code + scale instead of expanded BF16.
extern "C" __global__ void cache_pack_f32(float *x, uint8_t *dst, int n, int block, int e4, int first) {
  int group = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32), lane = threadIdx.x & 31;
  if (group * block >= n) return;
  float v = lane < block ? x[group * block + lane] : 0.f;
  float a = fabsf(v);
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
  float scale = e4 ? e4m3_round(fmaxf(a, 6.f * .001953125f) / 6.f) : pow2_ceil(fmaxf(a, 6.f * 1.1754944e-38f) / 6.f);
  float q = e2m1_round(fminf(fmaxf(v / scale, -6.f), 6.f));
  uint8_t code = cache_e2_code(q);
  uint8_t other = __shfl_down_sync(0xffffffff, (int)code, 1);
  uint8_t *out = dst + (first / block + group) * (block / 2 + 1);
  if (lane < block) {
    x[group * block + lane] = q * scale;
    if (!(lane & 1)) out[lane / 2] = code | (other << 4);
  }
  if (!lane) out[block / 2] = e4 ? cache_e4_code(scale) : (__float_as_uint(scale) >> 23);
}
extern "C" __global__ void cache_unpack(const __nv_bfloat16 *src, __nv_bfloat16 *dst, int first, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = cache_read(src, first + i);
}
// Import old-format BF16 snapshots. The values already lie on the FP4 grid.
extern "C" __global__ void cache_pack_bf16(const __nv_bfloat16 *x, uint8_t *dst, int n, int block, int e4, int first, int *error) {
  int group = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32), lane = threadIdx.x & 31;
  if (group * block >= n) return;
  float v = lane < block ? __bfloat162float(x[group * block + lane]) : 0.f;
  float a = fabsf(v);
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
  float scale = e4 ? e4m3_round(fmaxf(a, 6.f * .001953125f) / 6.f) : pow2_ceil(fmaxf(a, 6.f * 1.1754944e-38f) / 6.f);
  // The archive contains rounded values, not the original pre-quantization maximum.
  // E4M3 subnormal scales can round upward far enough that max(code) is 4, not 6.
  // Recover a scale that represents EVERY archived value exactly; re-quantizing
  // max(abs(x))/6 can silently change such a block. The original scale is one of
  // max(abs(x)) / {6,4,3,2,1.5,1,.5}, so this bounded search always covers it.
  const float maxima[7] = {6.f, 4.f, 3.f, 2.f, 1.5f, 1.f, .5f};
  bool found = a == 0.f;
  if (a > 0.f) {
#pragma unroll
    for (int j = 0; j < 7; ++j) {
      float candidate = a / maxima[j];
      bool valid = e4 ? candidate >= .001953125f && candidate == e4m3_round(candidate)
                      : candidate >= 1.1754944e-38f && candidate == pow2_ceil(candidate);
      float represented = e2m1_round(fminf(fmaxf(v / candidate, -6.f), 6.f)) * candidate;
      if (__all_sync(0xffffffff, valid && __float_as_uint(represented) == __float_as_uint(v))) {
        scale = candidate;
        found = true;
        break;
      }
    }
  }
  if (!found && !lane) atomicExch(error, 1);
  uint8_t code = cache_e2_code(e2m1_round(fminf(fmaxf(v / scale, -6.f), 6.f)));
  uint8_t other = __shfl_down_sync(0xffffffff, (int)code, 1);
  uint8_t *out = dst + (first / block + group) * (block / 2 + 1);
  if (lane < block && !(lane & 1)) out[lane / 2] = code | (other << 4);
  if (!lane) out[block / 2] = e4 ? cache_e4_code(scale) : (__float_as_uint(scale) >> 23);
}
