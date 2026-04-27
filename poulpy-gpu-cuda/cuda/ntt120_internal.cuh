// Self-contained NTT120 device helpers.
//
// Extracts pieces from Cheddar's NTTUtils.cuh (butterfly radix routines,
// NTTLaunchConfig, element functions) without including core/NTT.h, which
// drags in core/DeviceVector.h -> Thrust/RMM. Only common/Basic.cuh (which
// depends solely on stdlib) is included from Cheddar.

#pragma once

#include "common/Basic.cuh"

namespace cheddar {

// ── Enums from core/NTT.h ────────────────────────────────────────────────────

enum class NTTType { NTT, INTT };
enum class Phase { Phase1, Phase2 };

// Global compile-time flag from core/Parameter.h (production setting).
constexpr bool kExtendedOT = true;

// ── NTTLaunchConfig from core/NTTUtils.cuh ───────────────────────────────────

template <int log_degree, NTTType type, Phase phase>
struct NTTLaunchConfig {
  __host__ __device__ static constexpr int RadixStages() {
    if ((type == NTTType::NTT   && phase == Phase::Phase1) ||
        (type == NTTType::INTT  && phase == Phase::Phase2)) {
      return (log_degree == 16) ? 7 : log_degree - 9;
    }
    return 9;
  }
  __host__ __device__ static constexpr int StageMerging() {
    if ((type == NTTType::NTT   && phase == Phase::Phase1) ||
        (type == NTTType::INTT  && phase == Phase::Phase2)) {
      if (log_degree == 16) return 4;
    }
    return 3;
  }
  __host__ __device__ static constexpr int LogWarpBatching() {
    if ((type == NTTType::NTT   && phase == Phase::Phase1) ||
        (type == NTTType::INTT  && phase == Phase::Phase2)) {
      if (log_degree == 16) return 4;
    }
    return 0;
  }
  __host__ __device__ static constexpr int LsbSize() { return 32; }
  __host__ __device__ static constexpr bool OFTwiddle() { return true; }
  __host__ static constexpr int BlockDim() {
    return 1 << (RadixStages() - StageMerging() + LogWarpBatching());
  }
};

// ── Kernel helpers from core/NTTUtils.cuh ────────────────────────────────────

namespace kernel {

template <typename word, int size>
__device__ __inline__ void CopyWords(word *dst, const word *src) {
#pragma unroll
  for (int i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

template <typename word>
__device__ __inline__ void ButterflyNTT(
    make_signed_t<word> &a, make_signed_t<word> &b,
    const word w, const word q, const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;
  if (a < 0) a += q;
  signed_word mult = basic::detail::__mult_montgomery_lazy<word>(
      b, static_cast<signed_word>(w), q, q_inv);
  if (mult < 0) mult += q;
  b = a - mult;
  a = (a - q) + mult;
}

template <typename word>
__device__ __inline__ void ButterflyINTT(
    make_signed_t<word> &a, make_signed_t<word> &b,
    const word w, const word q, const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;
  if (a < 0) a += q;
  if (b < 0) b += q;
  signed_word diff = a - b;
  a = (a - q) + b;
  b = basic::detail::__mult_montgomery_lazy<word>(
      diff, static_cast<signed_word>(w), q, q_inv);
}

// Recursive compile-time radix kernels — verbatim from core/NTTUtils.cuh.

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixNTTFirst(
    make_signed_t<word> *local, int tw_idx, const word *w,
    const word q, const make_signed_t<word> q_inv) {
  if constexpr (stage > 1)
    MultiRadixNTTFirst<word, radix, stage - 1>(local, tw_idx, w, q, q_inv);
  constexpr int num_tw = 1 << (stage - 1);
  constexpr int stride = radix / (1 << stage);
  word w_vec[num_tw];
  CopyWords<word, num_tw>(w_vec, w + (tw_idx << (stage - 1)));
#pragma unroll
  for (int i = 0; i < num_tw; i++)
#pragma unroll
    for (int j = 0; j < stride; j++)
      ButterflyNTT<word>(local[i * 2 * stride + j],
                         local[i * 2 * stride + j + stride], w_vec[i], q, q_inv);
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixNTT(
    make_signed_t<word> *local, int tw_idx, const word *w,
    const word q, const make_signed_t<word> q_inv) {
  constexpr int num_tw = radix / (1 << stage);
  constexpr int stride = 1 << (stage - 1);
  word w_vec[num_tw];
  CopyWords<word, num_tw>(w_vec, w + tw_idx);
#pragma unroll
  for (int i = 0; i < num_tw; i++)
#pragma unroll
    for (int j = 0; j < stride; j++)
      ButterflyNTT<word>(local[i * 2 * stride + j],
                         local[i * 2 * stride + j + stride], w_vec[i], q, q_inv);
  if constexpr (stage > 1)
    MultiRadixNTT<word, radix, stage - 1>(local, 2 * tw_idx, w, q, q_inv);
}

template <typename word, int radix, int stage, int lsb_size>
__device__ __inline__ void MultiRadixNTT_OT(
    make_signed_t<word> *local, int tw_idx, const word *w, const word *w_msb,
    const word q, const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;
  int last_tw_idx  = (1 << (stage - 1)) * tw_idx;
  int msbIdx = last_tw_idx / lsb_size;
  int lsbIdx = last_tw_idx % lsb_size;
  constexpr int num_outer_blocks = radix / (1 << stage);
  constexpr int accumed_tw_num   = (1 << stage) - 1;
  word twiddle_factor_set[accumed_tw_num * num_outer_blocks];
  constexpr int num_tw_factor = (1 << (stage - 1)) * num_outer_blocks;
  constexpr int offset = ((1 << (stage - 1)) - 1) * num_outer_blocks;
  if constexpr (kExtendedOT) {
#pragma unroll
    for (int i = 0; i < num_tw_factor; i++)
      twiddle_factor_set[i + offset] =
          basic::detail::__mult_montgomery_lazy<word>(
              w[lsbIdx + i], static_cast<signed_word>(w_msb[msbIdx]), q, q_inv);
#pragma unroll
    for (int curr_stage = stage; curr_stage > 1; curr_stage--) {
      int src_off = ((1 << (curr_stage - 1)) - 1) * num_outer_blocks;
      int dst_off = ((1 << (curr_stage - 2)) - 1) * num_outer_blocks;
      int cnt = ((1 << (curr_stage - 1))) * num_outer_blocks;
#pragma unroll
      for (int i = 0; i < cnt / 2; i++) {
        word op = twiddle_factor_set[src_off + i * 2];
        twiddle_factor_set[dst_off + i] =
            basic::detail::__mult_montgomery_lazy<word>(
                op, static_cast<signed_word>(op), q, q_inv);
      }
    }
#pragma unroll
    for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
      int block_size = 1 << (stage - curr_stage + 1);
      int num_blocks = radix / block_size;
      int tw_off = ((1 << (curr_stage - 1)) - 1) * num_outer_blocks;
#pragma unroll
      for (int b = 0; b < num_blocks; b++) {
        int stride = block_size / 2;
#pragma unroll
        for (int i = 0; i < stride; i++)
          ButterflyNTT<word>(local[b * block_size + i],
                             local[b * block_size + i + stride],
                             twiddle_factor_set[tw_off + b], q, q_inv);
      }
    }
  } else {
#pragma unroll
    for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
      int block_size = 1 << (stage - curr_stage + 1);
      int num_blocks = radix / block_size;
      if (curr_stage == stage) {
        word OT_factors[num_tw_factor];
#pragma unroll
        for (int i = 0; i < num_tw_factor; i++)
          OT_factors[i] = basic::detail::__mult_montgomery_lazy<word>(
              w[lsbIdx + i], static_cast<signed_word>(w_msb[msbIdx]), q, q_inv);
#pragma unroll
        for (int b = 0; b < num_blocks; b++) {
          int stride = block_size / 2;
#pragma unroll
          for (int i = 0; i < stride; i++)
            ButterflyNTT<word>(local[b * block_size + i],
                               local[b * block_size + i + stride],
                               OT_factors[b], q, q_inv);
        }
        continue;
      }
#pragma unroll
      for (int b = 0; b < num_blocks; b++) {
        int stride = block_size / 2;
#pragma unroll
        for (int i = 0; i < stride; i++)
          ButterflyNTT<word>(local[b * block_size + i],
                             local[b * block_size + i + stride],
                             w[(1 << (curr_stage - 1)) * tw_idx + b], q, q_inv);
      }
    }
  }
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixINTTLast(
    make_signed_t<word> *local, int tw_idx, const word *w,
    const word q, const make_signed_t<word> q_inv) {
  constexpr int num_tw = 1 << (stage - 1);
  constexpr int stride = radix / (1 << stage);
  word w_vec[num_tw];
  CopyWords<word, num_tw>(w_vec, w + (tw_idx << (stage - 1)));
#pragma unroll
  for (int i = 0; i < num_tw; i++)
#pragma unroll
    for (int j = 0; j < stride; j++)
      ButterflyINTT<word>(local[i * 2 * stride + j],
                          local[i * 2 * stride + j + stride], w_vec[i], q, q_inv);
  if constexpr (stage > 1)
    MultiRadixINTTLast<word, radix, stage - 1>(local, tw_idx, w, q, q_inv);
}

template <typename word, int radix, int stage>
__device__ __inline__ void MultiRadixINTT(
    make_signed_t<word> *local, int tw_idx, const word *w,
    const word q, const make_signed_t<word> q_inv) {
  if constexpr (stage > 1)
    MultiRadixINTT<word, radix, stage - 1>(local, 2 * tw_idx, w, q, q_inv);
  constexpr int num_tw = radix / (1 << stage);
  constexpr int stride = 1 << (stage - 1);
  word w_vec[num_tw];
  CopyWords<word, num_tw>(w_vec, w + tw_idx);
#pragma unroll
  for (int i = 0; i < num_tw; i++)
#pragma unroll
    for (int j = 0; j < stride; j++)
      ButterflyINTT<word>(local[i * 2 * stride + j],
                          local[i * 2 * stride + j + stride], w_vec[i], q, q_inv);
}

template <typename word, int radix, int stage, int lsb_size>
__device__ __inline__ void MultiRadixINTT_OT(
    make_signed_t<word> *local, int tw_idx, const word *w, const word *w_msb,
    const word q, const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;
  int first_tw_idx = (1 << (stage - 1)) * tw_idx;
  int msbIdx = first_tw_idx / lsb_size;
  int lsbIdx = first_tw_idx % lsb_size;
  constexpr int num_outer_blocks = radix / (1 << stage);
  constexpr int accumed_tw_num   = (1 << stage) - 1;
  word twiddle_factor_set[accumed_tw_num * num_outer_blocks];
  constexpr int num_tw_factor = (1 << (stage - 1)) * num_outer_blocks;
  if constexpr (kExtendedOT) {
#pragma unroll
    for (int i = 0; i < num_tw_factor; i++)
      twiddle_factor_set[i] = basic::detail::__mult_montgomery_lazy<word>(
          w[lsbIdx + i], static_cast<signed_word>(w_msb[msbIdx]), q, q_inv);
    int accum = 0;
#pragma unroll
    for (int curr_stage = 1; curr_stage < stage; curr_stage++) {
      int cnt = num_outer_blocks * (1 << (stage - curr_stage));
#pragma unroll
      for (int i = 0; i < cnt / 2; i++) {
        word op = twiddle_factor_set[accum + i * 2];
        twiddle_factor_set[cnt + accum + i] =
            basic::detail::__mult_montgomery_lazy<word>(
                op, static_cast<signed_word>(op), q, q_inv);
      }
      accum += cnt;
    }
    accum = 0;
#pragma unroll
    for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
      int block_size = 1 << curr_stage;
      int num_blocks = radix / block_size;
#pragma unroll
      for (int b = 0; b < num_blocks; b++) {
        int stride = block_size / 2;
#pragma unroll
        for (int i = 0; i < stride; i++)
          ButterflyINTT<word>(local[b * block_size + i],
                              local[b * block_size + i + stride],
                              twiddle_factor_set[accum + b], q, q_inv);
      }
      accum += num_blocks;
    }
  } else {
#pragma unroll
    for (int curr_stage = 1; curr_stage <= stage; curr_stage++) {
      int block_size = 1 << curr_stage;
      int num_blocks = radix / block_size;
      if (curr_stage == 1) {
        word OT_factors[num_tw_factor];
#pragma unroll
        for (int i = 0; i < num_tw_factor; i++)
          OT_factors[i] = basic::detail::__mult_montgomery_lazy<word>(
              w[lsbIdx + i], static_cast<signed_word>(w_msb[msbIdx]), q, q_inv);
#pragma unroll
        for (int b = 0; b < num_blocks; b++) {
          int stride = block_size / 2;
#pragma unroll
          for (int i = 0; i < stride; i++)
            ButterflyINTT<word>(local[b * block_size + i],
                                local[b * block_size + i + stride],
                                OT_factors[b], q, q_inv);
        }
        continue;
      }
#pragma unroll
      for (int b = 0; b < num_blocks; b++) {
        int stride = block_size / 2;
#pragma unroll
        for (int i = 0; i < stride; i++)
          ButterflyINTT<word>(local[b * block_size + i],
                              local[b * block_size + i + stride],
                              w[(1 << (stage - curr_stage)) * tw_idx + b], q, q_inv);
      }
    }
  }
}

// Element-function templates (used as INTTPhase2 template parameter).

template <typename word>
__device__ __inline__ void NopFunc(
    make_signed_t<word> &, const make_signed_t<word>,
    const word, const word, const make_signed_t<word>) {}

template <typename word>
__device__ __inline__ void MultConstNormalize(
    make_signed_t<word> &result, const make_signed_t<word> a,
    const word b, const word prime, const make_signed_t<word> montgomery) {
  auto temp = basic::detail::__mult_montgomery_lazy<word>(
      a, static_cast<make_signed_t<word>>(b), prime, montgomery);
  if (temp < 0) temp += prime;
  if (temp > (prime >> 1)) temp -= prime;
  result = temp;
}

} // namespace kernel
} // namespace cheddar
