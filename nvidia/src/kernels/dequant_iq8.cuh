// Prefill weight expansion for the IQ codebook formats: one thread per 8-weight codebook group decodes the group's
// scale, grid entry and signs once and writes 8 halves in one 16-byte store (the per-weight dequant_f16 decoded each
// group 8 times, and for repacked IQ1_S / IQ2_XS / IQ2_S rebuilt the whole GGUF block per weight).
// Types: ggml 16 IQ2_XXS, 19 IQ1_S, 17 IQ2_XS, 22 IQ2_S as GGUF blocks, or the same + 2000 as repacked by
// repack_expert (expert_fast.cuh: block bodies [rows][k/256][body], then tails [rows][k/256][tail], then d f16).
// Same arguments as dequant_f16 (scale unused); grid (ceil(k / 8 / 256), rows), block 256; k % 256 == 0.
#pragma once
#include "iq2grid.cuh"

// byte o of GGUF block b (row-major [rows][k/256]) in either layout
struct IqBlk {
  const uint8_t *w;
  size_t b, nbt;  // block index, total blocks (rows * k / 256)
  int bs, body, rp;
  __device__ __forceinline__ unsigned at(int o) const {
    if (!rp || bs == 56) return w[b * bs + o];
    const int tail = bs - 2 - body;
    if (o < 2) return w[nbt * (body + tail) + b * 2 + o];
    if (o - 2 < body) return w[b * body + o - 2];
    return w[nbt * body + b * tail + o - 2 - body];
  }
  __device__ __forceinline__ float d() const { return __half2float(__ushort_as_half((unsigned short)(at(0) | at(1) << 8))); }
};

extern "C" __global__ void dequant_f16_iq8(int t, const uint8_t *w, const uint8_t *scale, __half *out, int rows, int k) {
  const int r = blockIdx.y, c = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= k) return;
  const int rp = t >= 2000, tt = rp ? t - 2000 : t;
  const int bs = tt == 18 ? 98 : tt == 29 ? 56 : tt == 16 ? 66 : tt == 19 ? 50 : tt == 17 ? 74 : 82, body = tt == 18 ? 96 : tt == 19 ? 48 : 64;
  const IqBlk B{w, (size_t)r * (k / 256) + c / 256, (size_t)rows * (k / 256), bs, body, rp};
  const int e = c & 255, ib = e >> 5, l = (e >> 3) & 3;
  float db, add = 0.f;
  unsigned long long grid;
  unsigned signs = 0;
  bool mag = true;  // IQ2 grid bytes are magnitudes with sign bits; IQ1_S grid bytes are signed values
  if (tt == 16) {
    const int p = 2 + ib * 8;
    const unsigned g = B.at(p) | B.at(p + 1) << 8 | B.at(p + 2) << 16 | B.at(p + 3) << 24;
    const unsigned sg = B.at(p + 4) | B.at(p + 5) << 8 | B.at(p + 6) << 16 | B.at(p + 7) << 24;
    db = B.d() * (0.5f + (float)(sg >> 28)) * 0.25f;
    grid = iq2xxs_grid[(g >> (8 * l)) & 255];
    signs = ksigns_iq2xs[(sg >> (7 * l)) & 127];
  } else if (tt == 18) {
    unsigned sg=B.at(66+4*ib)|B.at(67+4*ib)<<8|B.at(68+4*ib)<<16|B.at(69+4*ib)<<24;
    db=B.d()*(0.5f+(float)(sg>>28))*0.5f;
    grid=(unsigned long long)iq3xxs_grid[B.at(2+8*ib+2*l)]|((unsigned long long)iq3xxs_grid[B.at(3+8*ib+2*l)]<<32);
    signs=ksigns_iq2xs[(sg>>(7*l))&127];
  } else if (tt == 29) {
    const uint8_t *p=w+B.b*56;
    unsigned h=(p[32+(4*ib+l)/2]>>(4*(l&1)))&15;
    db=iq1m_d(p)*(float)(2*((iq1m_sc(p,ib/2)>>(6*(ib&1)+3*(l/2)))&7)+1);
    grid=iq1s_grid[p[4*ib+l]|(h&7)<<8];
    add=(h&8)?-0.125f:0.125f;
    mag=false;
  } else if (tt == 19) {
    const unsigned qh = B.at(34 + 2 * ib) | B.at(35 + 2 * ib) << 8;
    db = B.d() * (float)(2 * ((qh >> 12) & 7) + 1);
    grid = iq1s_grid[B.at(2 + 4 * ib + l) | ((qh >> (3 * l)) & 7) << 8];
    add = (qh & 0x8000) ? -0.125f : 0.125f;
    mag = false;
  } else if (tt == 17) {
    const unsigned q = B.at(2 + 2 * (4 * ib + l)) | B.at(3 + 2 * (4 * ib + l)) << 8, sc = B.at(66 + ib);
    db = B.d() * (0.5f + (float)((l < 2 ? sc : sc >> 4) & 15)) * 0.25f;
    grid = iq2xs_grid[q & 511];
    signs = ksigns_iq2xs[q >> 9];
  } else {
    const unsigned sc = B.at(74 + ib);
    db = B.d() * (0.5f + (float)((l < 2 ? sc : sc >> 4) & 15)) * 0.25f;
    grid = iq2s_grid[B.at(2 + 4 * ib + l) | ((B.at(66 + ib) << (8 - 2 * l)) & 0x300)];
    signs = B.at(34 + 4 * ib + l);
  }
  __align__(16) __half h[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    float v;
    if (mag) {
      v = db * (float)((grid >> (8 * j)) & 255);
      if ((signs >> j) & 1) v = -v;
    } else {
      v = db * ((float)(int)(int8_t)((grid >> (8 * j)) & 255) + add);
    }
    h[j] = __float2half(v);
  }
  *(uint4 *)(out + (size_t)r * k + c) = *(const uint4 *)h;
}
