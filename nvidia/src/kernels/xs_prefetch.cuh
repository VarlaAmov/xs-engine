// xs-engine pull prefetch (XS_PREFETCH=R, decode). A GPU's pull stream is idle while the next layer's attention runs.
// During that time it pulls the slices of the R experts that layer L+1 is predicted to miss (L+1's router applied to
// layer L's expert input, `DSX_PREGATE=1` found 37 / 53 / 60 / 63 % of the real misses in the first 1 / 2 / 3 / 4
// predicted non-resident experts) into small VRAM staging areas and patches the pointer table `ptrs` the pull
// kernels read, so that those kernels find the slices in VRAM. Same bytes, same kernels, same sums: the output is
// bit-identical to the run without prefetch, only the PCIe time moves off the critical path.
//
// Layout contract with xs_pull.cuh: gate / up row r of an expert is at ptrs[..] + r * RB (RB = 20 * block bytes),
// down's row-o slice is at ptrs[..] + off + o * rbd. The staged copies keep exactly those relative addresses (and the
// source's address mod 16), so the pull kernels' own pointer arithmetic and 16-byte chunking work unchanged.

// One thread: give the previous round's tenants their arena addresses back, then take the first R predicted experts
// that are misses (no GPU holds them) and patch their three base pointers to the staging slots.
extern "C" __global__ void xs_pf_pick(const int *__restrict__ pred, const uint8_t *__restrict__ miss, const unsigned long long *__restrict__ host_ptrs,
                                      unsigned long long *ptrs, int *tag, int *psel, int layer, int E, int R, int c0, int RB, int off,
                                      unsigned long long gu_stage, unsigned long long gu_stride, unsigned long long dn_stage, unsigned long long dn_stride,
                                      const int *sel, int K, unsigned long long *stats) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  // statistics of the round that prepared layer `layer - 1` (whose picks `sel` still holds): [0] slices prefetched,
  // [1] of them picked, [2] missed picks of that layer, [3] rounds counted
  if (stats && layer > 1) {
    for (int j = 0; j < R; j++) {
      const int t = tag[j];
      if (t < 0) continue;
      stats[0]++;
      for (int k = 0; k < K; k++)
        if (sel[k] == t % E) { stats[1]++; break; }
    }
    for (int k = 0; k < K; k++)
      if (miss[(size_t)(layer - 1) * E + sel[k]]) stats[2]++;
    stats[3]++;
  }
  for (int j = 0; j < R; j++) {
    const int t = tag[j];
    if (t >= 0)
      for (int m = 0; m < 3; m++) ptrs[(size_t)t * 3 + m] = host_ptrs[(size_t)t * 3 + m];
    tag[j] = -1;
    psel[j] = -1;
  }
  int n = 0;
  for (int i = 0; i < 8 && n < R; i++) {
    const int e = pred[i];
    if (e < 0 || e >= E || !miss[(size_t)layer * E + e]) continue;
    const size_t t = (size_t)layer * E + e;
    const unsigned long long hg = host_ptrs[t * 3], hu = host_ptrs[t * 3 + 1], hd = host_ptrs[t * 3 + 2];
    const unsigned long long rows0 = (unsigned long long)c0 * RB;
    ptrs[t * 3] = gu_stage + (unsigned long long)(2 * n) * gu_stride + ((hg + rows0) & 15) - rows0;
    ptrs[t * 3 + 1] = gu_stage + (unsigned long long)(2 * n + 1) * gu_stride + ((hu + rows0) & 15) - rows0;
    ptrs[t * 3 + 2] = dn_stage + (unsigned long long)n * dn_stride + ((hd + off) & 15) - (unsigned long long)off;
    tag[n] = (int)t;
    psel[n] = e;
    n++;
  }
}

// gate and up rows [c0, c1) of the selected experts: one contiguous byte range each, copied as the aligned 16-byte
// chunks that cover it. grid (CTAs, 2 * R)
extern "C" __global__ void xs_pf_gu(const int *__restrict__ psel, const unsigned long long *__restrict__ host_ptrs, int layer, int E, int c0, int c1,
                                    int RB, unsigned long long gu_stage, unsigned long long gu_stride) {
  const int j = blockIdx.y >> 1, m = blockIdx.y & 1;
  const int e = psel[j];
  if (e < 0) return;
  const unsigned long long h = host_ptrs[((size_t)layer * E + e) * 3 + m] + (unsigned long long)c0 * RB;
  const unsigned long long hend = h + (unsigned long long)(c1 - c0) * RB;
  const unsigned long long a0 = h & ~15ull, a1 = (hend + 15) & ~15ull;
  const uint4 *src = (const uint4 *)a0;
  uint4 *dst = (uint4 *)(gu_stage + (unsigned long long)blockIdx.y * gu_stride);
  const size_t n = (a1 - a0) / 16;
#pragma unroll 4
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] = __ldcv(src + i);
}

// down: every output row's slice [off, off + sb) of its rbd bytes, as the aligned chunks that cover it. grid (CTAs, R)
extern "C" __global__ void xs_pf_down(const int *__restrict__ psel, const unsigned long long *__restrict__ host_ptrs, int layer, int E, int dim, int rbd,
                                      int off, int sb, unsigned long long dn_stage, unsigned long long dn_stride) {
  const int j = blockIdx.y;
  const int e = psel[j];
  if (e < 0) return;
  const unsigned long long hd = host_ptrs[((size_t)layer * E + e) * 3 + 2] + off;
  const unsigned long long sd = dn_stage + (unsigned long long)j * dn_stride + (hd & 15);
  const int nch = (sb + 30) / 16 + 1;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dim * nch; idx += gridDim.x * blockDim.x) {
    const int row = idx / nch, i = idx - row * nch;
    const unsigned long long p = hd + (unsigned long long)row * rbd;
    const unsigned long long a0 = p & ~15ull, a1 = (p + sb + 15) & ~15ull;
    if (a0 + 16ull * i >= a1) continue;
    const unsigned long long q = sd + (unsigned long long)row * rbd;
    *(uint4 *)((q & ~15ull) + 16ull * i) = __ldcv((const uint4 *)(a0 + 16ull * i));
  }
}
