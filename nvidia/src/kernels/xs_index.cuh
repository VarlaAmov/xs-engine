// xs-engine indexer scores (decode and prefill): score[r][s] = sum_h w[r][h] * relu(q[r][h] . k[s]) for the visible
// keys s < ns[r] (and, with a candidate mask, keys whose 8-position block layer 20 kept), -inf for the others.
// q [rows][32][128] f32, k [n][128] bf16 (the shared index-key cache), w [rows][32] f32 (bf16-rounded, scaled).
// indexer_score spends a warp and 32 shuffle reductions per (row, key); here a thread owns a key, the block's RT
// query rows sit in shared memory (broadcast float4 reads), and the key's 128 values are read once for all RT rows.
// Same products and fp32 accumulation per head as indexer_score; only the order of the sums over dims differs.
// Shapes fixed for DeepSeek V4.1: 32 index heads x 128 dims.

#define XS_IH 32
#define XS_ID 128
#define XS_KT 4  // key tiles of 128 per block

template <int RT>
__device__ __forceinline__ void xs_idx_score_t(const float *__restrict__ q, const __nv_bfloat16 *__restrict__ k, const float *__restrict__ w,
                                               float *__restrict__ score, int lds, const int *__restrict__ ns, int n, int rows,
                                               const uint8_t *__restrict__ mask, int ldm, int bs) {
  __shared__ __align__(16) float sq[RT][XS_IH][XS_ID];
  __shared__ float sw[RT][XS_IH];
  __shared__ int sns[RT];
  const int r0 = blockIdx.y * RT;
  for (int i = threadIdx.x; i < RT * XS_IH * XS_ID / 4; i += blockDim.x) {
    const int r = i / (XS_IH * XS_ID / 4);
    ((float4 *)sq)[i] = r0 + r < rows ? ((const float4 *)(q + (size_t)r0 * XS_IH * XS_ID))[i] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  for (int i = threadIdx.x; i < RT * XS_IH; i += blockDim.x) sw[i / XS_IH][i % XS_IH] = r0 + i / XS_IH < rows ? w[(size_t)r0 * XS_IH + i] : 0.f;
  if (threadIdx.x < RT) sns[threadIdx.x] = r0 + (int)threadIdx.x < rows ? ns[r0 + threadIdx.x] : 0;
  __syncthreads();
  int nmax = 0;
#pragma unroll
  for (int r = 0; r < RT; r++) nmax = max(nmax, sns[r]);
  for (int kt = 0; kt < XS_KT; kt++) {
    const int s = (blockIdx.x * XS_KT + kt) * blockDim.x + threadIdx.x;
    if (s >= n) break;
    bool need[RT];
    bool any = false;
#pragma unroll
    for (int r = 0; r < RT; r++) {
      need[r] = s < sns[r] && (mask == nullptr || mask[(size_t)(r0 + r) * ldm + s / bs] != 0);
      any |= need[r];
    }
    float acc[RT][XS_IH];
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
      for (int h = 0; h < XS_IH; h++) acc[r][h] = 0.f;
    // a warp with no lane needing this key tile skips the math (lanes compute and discard otherwise)
    if (__any_sync(0xffffffffu, any) && s < nmax) {

#pragma unroll 2
      for (int d4 = 0; d4 < XS_ID / 4; d4++) {
        const uint2 kv = cache_read4(k, (size_t)s * XS_ID + d4 * 4);
        const float2 a = __bfloat1622float2(*(const __nv_bfloat162 *)&kv.x), b = __bfloat1622float2(*(const __nv_bfloat162 *)&kv.y);
#pragma unroll
        for (int r = 0; r < RT; r++)
#pragma unroll
          for (int h = 0; h < XS_IH; h++) {
            const float4 qv = *(const float4 *)&sq[r][h][4 * d4];
            float v = acc[r][h];
            v = fmaf(qv.x, a.x, v);
            v = fmaf(qv.y, a.y, v);
            v = fmaf(qv.z, b.x, v);
            v = fmaf(qv.w, b.y, v);
            acc[r][h] = v;
          }
      }
    }
#pragma unroll
    for (int r = 0; r < RT; r++) {
      if (r0 + r >= rows) break;
      float tot = 0.f;
#pragma unroll
      for (int h = 0; h < XS_IH; h++) tot += fmaxf(acc[r][h], 0.f) * sw[r][h];
      score[(size_t)(r0 + r) * lds + s] = need[r] ? tot : -INFINITY;
    }
  }
}

// grid (ceil(n / 512), ceil(rows / RT)), block 128. lds: score row stride; mask: u8 [rows][ldm] kept candidate blocks
// of bs positions (0 = none). n: keys scored (n_max of the rows); ns [rows]: visible keys of each row.
extern "C" __global__ void __launch_bounds__(128) xs_idx_score1(const float *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                                                                const float *__restrict__ w, float *__restrict__ score, int lds,
                                                                const int *__restrict__ ns, int n, int rows,
                                                                const uint8_t *__restrict__ mask, int ldm, int bs) {
  xs_idx_score_t<1>(q, k, w, score, lds, ns, n, rows, mask, ldm, bs);
}
extern "C" __global__ void __launch_bounds__(128) xs_idx_score2(const float *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                                                                const float *__restrict__ w, float *__restrict__ score, int lds,
                                                                const int *__restrict__ ns, int n, int rows,
                                                                const uint8_t *__restrict__ mask, int ldm, int bs) {
  xs_idx_score_t<2>(q, k, w, score, lds, ns, n, rows, mask, ldm, bs);
}
