// xs-engine tri-split decode: a non-resident ("missed") routed expert is computed by the CPU and both GPUs at once,
// each on its own slice of the intermediate dimension (256-row blocks): gate/up rows [c0, c1), SwiGLU, then down's
// columns [c0, c1). The weights stay in the pinned RAM arena in their GGUF blocks; these kernels read them over
// PCIe (zero-copy, ~26 GB/s per link) while the CPU computes its slice from RAM. Partial outputs are summed by the
// combine. Activations are int8 per 256 block exactly as quant_act / the CPU's QAct (d = amax / 127, round to
// nearest even); h = bf16(silu(min(g, lim)) * clamp(u, -lim, lim) * w) as moe_gu_fast. One token (decode).
// Formats: the four of DeepSeek XS's experts, ggml's layouts and tables (iq2grid.cuh): IQ2_XXS 16, IQ2_XS 17,
// IQ2_S 22, IQ3_XXS 18.

#define XS_MAXK 6

__device__ __forceinline__ int xs_bs(int ty) { return ty == 16 ? 66 : ty == 17 ? 74 : ty == 22 ? 82 : 98; }

// 4 sign bits -> 4 bytes of 0xff / 0x00 (bits land on bytes 0..3 through distinct product bits)
__device__ __forceinline__ unsigned xs_smask(unsigned s4) { return ((s4 * 0x00204081u) & 0x01010101u) * 0xffu; }

// sum of 8 weights (magnitudes m0 = bytes 0..3, m1 = bytes 4..7, signs s8 bit i) times x[0..8] (x0, x1 = 4 bytes each)
__device__ __forceinline__ int xs_dot8(unsigned m0, unsigned m1, unsigned s8, int x0, int x1) {
  const unsigned k0 = xs_smask(s8 & 15), k1 = xs_smask(s8 >> 4);
  int a = __dp4a((int)m0, x0, 0) - 2 * __dp4a((int)(m0 & k0), x0, 0);
  return a + __dp4a((int)m1, x1, 0) - 2 * __dp4a((int)(m1 & k1), x1, 0);
}

// shared tables of one format: grid (u64 entries; IQ3_XXS: u32) and ksigns
struct XsTab {
  unsigned long long *g8;
  unsigned *g4;
  uint8_t *ks;
};

__device__ __forceinline__ void xs_tab_fill(int ty, XsTab t) {
  for (int i = threadIdx.x; i < 128; i += blockDim.x) t.ks[i] = ksigns_iq2xs[i];
  if (ty == 16) for (int i = threadIdx.x; i < 256; i += blockDim.x) t.g8[i] = iq2xxs_grid[i];
  else if (ty == 17) for (int i = threadIdx.x; i < 512; i += blockDim.x) t.g8[i] = iq2xs_grid[i];
  else if (ty == 22) for (int i = threadIdx.x; i < 1024; i += blockDim.x) t.g8[i] = iq2s_grid[i];
  else for (int i = threadIdx.x; i < 256; i += blockDim.x) t.g4[i] = iq3xxs_grid[i];
}

// sub-block ib (32 weights) of the block at `b` (GGUF bytes, 2-byte aligned) against int8 x (32 bytes, 4-byte
// aligned): the integer sum times (2 s + 1) of each 16 or 32 weights; the caller scales by d * c (c = 1/8 for the
// IQ2s, 1/4 for IQ3_XXS) and the activation block's d.
__device__ __forceinline__ int xs_sub(int ty, const uint8_t *b, int ib, const int *x, const XsTab &t) {
  auto u16 = [&](int o) { return (unsigned)*(const unsigned short *)(b + o); };
  if (ty == 16) {  // IQ2_XXS: 4 grid bytes, u32 (4 x 7 sign bits, 4-bit scale)
    const unsigned g = u16(2 + 8 * ib) | u16(4 + 8 * ib) << 16, sg = u16(6 + 8 * ib) | u16(8 + 8 * ib) << 16;
    int s = 0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const unsigned long long e = t.g8[(g >> (8 * l)) & 255];
      s += xs_dot8((unsigned)e, (unsigned)(e >> 32), t.ks[(sg >> (7 * l)) & 127], x[2 * l], x[2 * l + 1]);
    }
    return s * (int)(2 * (sg >> 28) + 1);
  } else if (ty == 17) {  // IQ2_XS: u16 per 8 weights (9-bit grid, 7-bit signs); scale nibbles at 66 + ib
    const unsigned sc = b[66 + ib];
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const unsigned q = u16(2 + 8 * ib + 2 * l);
      const unsigned long long e = t.g8[q & 511];
      const int v = xs_dot8((unsigned)e, (unsigned)(e >> 32), t.ks[q >> 9], x[2 * l], x[2 * l + 1]);
      if (l < 2) s0 += v; else s1 += v;
    }
    return s0 * (int)(2 * (sc & 15) + 1) + s1 * (int)(2 * (sc >> 4) + 1);
  } else if (ty == 22) {  // IQ2_S: low index bytes at 2, explicit signs at 34, qh at 66, scales at 74
    const unsigned sc = b[74 + ib], qh = b[66 + ib];
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const unsigned idx = b[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300);
      const unsigned long long e = t.g8[idx];
      const int v = xs_dot8((unsigned)e, (unsigned)(e >> 32), b[34 + 4 * ib + l], x[2 * l], x[2 * l + 1]);
      if (l < 2) s0 += v; else s1 += v;
    }
    return s0 * (int)(2 * (sc & 15) + 1) + s1 * (int)(2 * (sc >> 4) + 1);
  } else {  // IQ3_XXS: 8 grid bytes (u32 entries of 4 weights) at 2 + 8 ib, u32 signs + scale at 66 + 4 ib
    const unsigned sg = u16(66 + 4 * ib) | u16(68 + 4 * ib) << 16;
    int s = 0;
#pragma unroll
    for (int l = 0; l < 4; l++) {
      const unsigned m0 = t.g4[b[2 + 8 * ib + 2 * l]], m1 = t.g4[b[3 + 8 * ib + 2 * l]];
      s += xs_dot8(m0, m1, t.ks[(sg >> (7 * l)) & 127], x[2 * l], x[2 * l + 1]);
    }
    return s * (int)(2 * (sg >> 28) + 1);
  }
}

__device__ __forceinline__ float xs_dfac(int ty) { return ty == 18 ? 0.25f : 0.125f; }
__device__ __forceinline__ float xs_d(const uint8_t *b) { return __half2float(__ushort_as_half(*(const unsigned short *)b)); }

// x [k] -> xq [k] int8 (natural order), xd [k / 256]. grid k / 256, block 256
extern "C" __global__ void xs_pull_qx(const float *__restrict__ x, int8_t *__restrict__ xq, float *__restrict__ xd, float *__restrict__ xc) {
  __shared__ float sh[8];
  const int tid = threadIdx.x;
  const size_t i = (size_t)blockIdx.x * 256 + tid;
  const float v = x[i];
  if (xc) xc[i] = v;  // the pull prefetch's own copy of x (XS_PREFETCH)
  const float a0 = warp_max(fabsf(v));
  if ((tid & 31) == 0) sh[tid >> 5] = a0;
  __syncthreads();
  float a = sh[0];
#pragma unroll
  for (int j = 1; j < 8; j++) a = fmaxf(a, sh[j]);
  const float dd = a / 127.f, id = dd > 0.f ? 1.f / dd : 0.f;
  xq[i] = (int8_t)__float2int_rn(v * id);
  if (tid == 0) xd[blockIdx.x] = dd;
}

// Pull kernels run on a low-priority stream beside the resident experts: one CTA of 8 warps per SM (grid = SMs), so
// each SM keeps room for the resident experts' and the shared expert's CTAs; a warp loops over its (pick, row) items.
// Shared memory: the codebook, the activations and each warp's staged rows. Work is (missed pick, row).

#define XS_PULL_WARPS 8

// the missed picks of this layer, in pick order: picks[0..*np)
__device__ __forceinline__ void xs_missed(const int *sel, const uint8_t *miss, int layer, int E, int K, int *picks, int *np) {
  if (threadIdx.x == 0) {
    int n = 0;
    for (int j = 0; j < K; j++)
      if (miss[(size_t)layer * E + sel[j]]) picks[n++] = j;
    *np = n;
  }
}

// h [j][r] for this GPU's rows r in [c0, c1) of every missed pick j (grid SMs, block 256). k = 5120.
// A warp stages its row pair (gate, up) in shared memory with coalesced 8-byte loads, so each arena byte crosses PCIe
// once (zero-copy reads are not cached), then decodes from shared memory: lane takes 5 of the 160 sub-blocks.
template <int TY>
__device__ __forceinline__ void xs_pull_gu_t(const int *__restrict__ sel, const float *__restrict__ wts, const uint8_t *__restrict__ miss,
                                             const unsigned long long *__restrict__ ptrs, int layer, int E, int K, const int8_t *__restrict__ xq,
                                             const float *__restrict__ xd, float *__restrict__ h, int c0, int c1, int inter, float lim) {
  constexpr int BS = TY == 16 ? 66 : TY == 17 ? 74 : TY == 22 ? 82 : 98, RB = 20 * BS;  // row bytes: a multiple of 8
  __shared__ __align__(16) unsigned long long g8[1024];
  __shared__ __align__(16) uint8_t ks[128];
  __shared__ __align__(16) int8_t sx[5120];
  __shared__ float sxd[20];
  __shared__ int picks[XS_MAXK], np;
  // each row staged as the aligned 16-byte chunks covering it (rows are 8-byte aligned: at most one extra chunk)
  __shared__ __align__(16) uint4 rows16[XS_PULL_WARPS][2][RB / 16 + 2];
  xs_missed(sel, miss, layer, E, K, picks, &np);
  __syncthreads();
  const int n = np;
  if (n == 0) return;
  XsTab t{g8, (unsigned *)g8, ks};
  xs_tab_fill(TY, t);
  for (int i = threadIdx.x; i < 5120 / 16; i += blockDim.x) ((int4 *)sx)[i] = ((const int4 *)xq)[i];
  if (threadIdx.x < 20) sxd[threadIdx.x] = xd[threadIdx.x];
  __syncthreads();
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, rows = c1 - c0;
  const int nw = gridDim.x * XS_PULL_WARPS;
  const float df = xs_dfac(TY);
  for (int it = blockIdx.x * XS_PULL_WARPS + w; it < n * rows; it += nw) {
    const int pj = it / rows, r = c0 + it % rows, j = picks[pj];
    const size_t pe = ((size_t)layer * E + sel[j]) * 3;
    int shift[2];
#pragma unroll
    for (int m = 0; m < 2; m++) {
      const uintptr_t rp = (uintptr_t)(ptrs[pe + m] + (size_t)r * RB), a0 = rp & ~(uintptr_t)15, a1 = (rp + RB + 15) & ~(uintptr_t)15;
      shift[m] = (int)(rp - a0);
      const int nch = (int)((a1 - a0) / 16);
      for (int i = lane; i < nch; i += 32) rows16[w][m][i] = __ldcv((const uint4 *)a0 + i);
    }
    __syncwarp();
    float ga = 0.f, ua = 0.f;
#pragma unroll
    for (int i = 0; i < 5; i++) {
      const int s = lane + 32 * i, b = s >> 3, ib = s & 7;
      const uint8_t *gb = (const uint8_t *)rows16[w][0] + shift[0] + b * BS, *ub = (const uint8_t *)rows16[w][1] + shift[1] + b * BS;
      const int *xs = (const int *)(sx + 32 * s);
      const float sc = sxd[b] * df;
      ga += (float)xs_sub(TY, gb, ib, xs, t) * (xs_d(gb) * sc);
      ua += (float)xs_sub(TY, ub, ib, xs, t) * (xs_d(ub) * sc);
    }
    const float g = warp_sum(ga), u = warp_sum(ua);
    if (lane == 0) {
      const float gv = fminf(g, lim), uv = fminf(fmaxf(u, -lim), lim);
      h[(size_t)j * inter + r] = bf16r(gv / (1.f + expf(-gv)) * uv * wts[j]);
    }
    __syncwarp();  // the next row pair overwrites the staging
  }
}

#define XS_GU_ARGS                                                                                                       \
  const int *__restrict__ sel, const float *__restrict__ wts, const uint8_t *__restrict__ miss, const unsigned long long *__restrict__ ptrs, \
      int layer, int E, int K, const int8_t *__restrict__ xq, const float *__restrict__ xd, float *__restrict__ h, int c0, int c1, int inter, float lim
extern "C" __global__ void __launch_bounds__(256) xs_pull_gu16(XS_GU_ARGS) { xs_pull_gu_t<16>(sel, wts, miss, ptrs, layer, E, K, xq, xd, h, c0, c1, inter, lim); }
extern "C" __global__ void __launch_bounds__(256) xs_pull_gu17(XS_GU_ARGS) { xs_pull_gu_t<17>(sel, wts, miss, ptrs, layer, E, K, xq, xd, h, c0, c1, inter, lim); }
extern "C" __global__ void __launch_bounds__(256) xs_pull_gu22(XS_GU_ARGS) { xs_pull_gu_t<22>(sel, wts, miss, ptrs, layer, E, K, xq, xd, h, c0, c1, inter, lim); }
extern "C" __global__ void __launch_bounds__(256) xs_pull_gu18(XS_GU_ARGS) { xs_pull_gu_t<18>(sel, wts, miss, ptrs, layer, E, K, xq, xd, h, c0, c1, inter, lim); }

// out [dim] = sum over missed picks of down[:, c0..c1) . h[j][c0..c1) (int8 h per 256 block), zeros when nothing is
// missed. grid SMs, block 256; a warp takes 4 rows per step (their slices' loads in flight together).
// Slice width nb = (c1 - c0) / 256 <= 6 blocks: lane l owns sub-blocks l and l + 32 (when < 8 nb).
template <int TY>
__device__ __forceinline__ void xs_pull_down_t(const int *__restrict__ sel, const uint8_t *__restrict__ miss, const unsigned long long *__restrict__ ptrs,
                                               int layer, int E, int K, const float *__restrict__ h, float *__restrict__ out, int c0, int c1,
                                               int inter, int dim) {
  constexpr int BS = TY == 16 ? 66 : TY == 17 ? 74 : TY == 22 ? 82 : 98;
  __shared__ __align__(16) unsigned long long g8[1024];
  __shared__ __align__(16) uint8_t ks[128];
  __shared__ __align__(16) int8_t hq[XS_MAXK][1536];
  __shared__ float hd[XS_MAXK][6];
  __shared__ int picks[XS_MAXK], np;
  __shared__ float sh[XS_PULL_WARPS];
  // each row's slice staged as the aligned 16-byte chunks that cover it (2-byte loads made 32-byte PCIe requests)
  __shared__ __align__(16) uint4 stage[XS_PULL_WARPS][4][(6 * 98 + 30) / 16 + 1];
  xs_missed(sel, miss, layer, E, K, picks, &np);
  __syncthreads();
  const int n = np, nb = (c1 - c0) / 256;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int nw = gridDim.x * XS_PULL_WARPS;
  if (n == 0) {
    for (int o = blockIdx.x * blockDim.x + threadIdx.x; o < dim; o += gridDim.x * blockDim.x) out[o] = 0.f;
    return;
  }
  XsTab t{g8, (unsigned *)g8, ks};
  xs_tab_fill(TY, t);
  // each missed pick's h slice to int8 per 256 block (256 threads: one value each)
  for (int pj = 0; pj < n; pj++)
    for (int b = 0; b < nb; b++) {
      const float v = h[(size_t)picks[pj] * inter + c0 + b * 256 + threadIdx.x];
      const float a0 = warp_max(fabsf(v));
      __syncthreads();
      if (lane == 0) sh[w] = a0;
      __syncthreads();
      float a = sh[0];
#pragma unroll
      for (int i = 1; i < XS_PULL_WARPS; i++) a = fmaxf(a, sh[i]);
      const float dd = a / 127.f, id = dd > 0.f ? 1.f / dd : 0.f;
      hq[pj][b * 256 + threadIdx.x] = (int8_t)__float2int_rn(v * id);
      if (threadIdx.x == 0) hd[pj][b] = dd;
    }
  __syncthreads();
  const int rbd = inter / 256 * BS, off = (c0 / 256) * BS, sb = nb * BS;  // slice bytes per row: even, 2-byte aligned
  const float df = xs_dfac(TY);
  const int nsub = nb * 8;
  for (int o0 = (blockIdx.x * XS_PULL_WARPS + w) * 4; o0 < dim; o0 += nw * 4) {
    float tot[4] = {0.f, 0.f, 0.f, 0.f};
    for (int pj = 0; pj < n; pj++) {
      const uint8_t *D = (const uint8_t *)ptrs[((size_t)layer * E + sel[picks[pj]]) * 3 + 2] + off;
      // the 4 rows' slices as aligned 16-byte chunks, all in flight before the math
      int shift[4];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const uint8_t *rp = D + (size_t)min(o0 + q, dim - 1) * rbd;
        const uintptr_t a0 = (uintptr_t)rp & ~(uintptr_t)15, a1 = ((uintptr_t)rp + sb + 15) & ~(uintptr_t)15;
        shift[q] = (int)((uintptr_t)rp - a0);
        const int nch = (int)((a1 - a0) / 16);
        for (int i = lane; i < nch; i += 32) stage[w][q][i] = __ldcv((const uint4 *)a0 + i);
      }
      __syncwarp();
      float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ss = 0; ss < 2; ss++) {
        const int sbk = lane + 32 * ss;
        if (sbk >= nsub) break;
        const int b = sbk >> 3, ib = sbk & 7;
        const int *hx = (const int *)(hq[pj] + 32 * sbk);
        const float hs = hd[pj][b] * df;
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint8_t *blk = (const uint8_t *)stage[w][q] + shift[q] + b * BS;
          v[q] += (float)xs_sub(TY, blk, ib, hx, t) * (xs_d(blk) * hs);
        }
      }
#pragma unroll
      for (int q = 0; q < 4; q++) tot[q] += warp_sum(v[q]);
      __syncwarp();
    }
    if (lane < 4 && o0 + lane < dim) out[o0 + lane] = lane == 0 ? tot[0] : lane == 1 ? tot[1] : lane == 2 ? tot[2] : tot[3];
  }
}

#define XS_DN_ARGS                                                                                                        \
  const int *__restrict__ sel, const uint8_t *__restrict__ miss, const unsigned long long *__restrict__ ptrs, int layer, int E, int K, \
      const float *__restrict__ h, float *__restrict__ out, int c0, int c1, int inter, int dim
extern "C" __global__ void __launch_bounds__(256) xs_pull_down16(XS_DN_ARGS) { xs_pull_down_t<16>(sel, miss, ptrs, layer, E, K, h, out, c0, c1, inter, dim); }
extern "C" __global__ void __launch_bounds__(256) xs_pull_down17(XS_DN_ARGS) { xs_pull_down_t<17>(sel, miss, ptrs, layer, E, K, h, out, c0, c1, inter, dim); }
extern "C" __global__ void __launch_bounds__(256) xs_pull_down22(XS_DN_ARGS) { xs_pull_down_t<22>(sel, miss, ptrs, layer, E, K, h, out, c0, c1, inter, dim); }
extern "C" __global__ void __launch_bounds__(256) xs_pull_down18(XS_DN_ARGS) { xs_pull_down_t<18>(sel, miss, ptrs, layer, E, K, h, out, c0, c1, inter, dim); }

// y += x, n floats. grid ceil(n / 256), block 256
extern "C" __global__ void xs_add(float *__restrict__ y, const float *__restrict__ x, int n) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) y[i] += x[i];
}
