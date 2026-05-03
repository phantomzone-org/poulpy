// NTT120 forward/inverse NTT kernel launchers.
//
// Adapted from Cheddar's NTT.cu with three changes per kernel:
//   1. NTTPhase1 only: input is int64_t*, reduced mod prime inline at load.
//   2. blockIdx.z indexes the batch (col*size+limb); Cheddar has no batch dim.
//   3. InputPtrList<> replaced by direct pointers; always 4 primes, no skip.

#include "ntt120_internal.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace cheddar;
using namespace cheddar::kernel;

// ── Barrett reduction for signed 64-bit input ────────────────────────────────
//
// Fast replacement for `raw % prime` where prime is a 30-bit NTT prime.
// m4 = UINT64_MAX / prime (precomputed once per prime per kernel invocation).
// Returns r in [0, prime) with r ≡ raw (mod prime).
__device__ static __inline__ int32_t barrett_reduce_i64(
    int64_t raw, uint32_t prime, uint64_t m4)
{
    // Two's-complement absolute value — correct even for INT64_MIN.
    uint64_t sign = (uint64_t)((int64_t)raw >> 63);  // all-1s or all-0s
    uint64_t ax   = ((uint64_t)raw ^ sign) - sign;    // |raw| as uint64_t

    // Barrett: q = upper 64 bits of ax * m4 ≈ floor(ax / prime).
    uint64_t q = __umul64hi(ax, m4);
    uint64_t r = ax - q * (uint64_t)prime;
    // m4 underestimates 2^64/prime by at most a few units; two corrections suffice.
    if (r >= (uint64_t)prime) r -= (uint64_t)prime;
    if (r >= (uint64_t)prime) r -= (uint64_t)prime;

    // Negate modulo prime if the original input was negative.
    if (sign != 0 && r != 0) r = (uint64_t)prime - r;
    return (int32_t)r;
}

static void check_cuda_runtime(const char *where) {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
    std::abort();
  }
}

// ── Adapted NTTPhase1: i64 input → i32 DFT output ───────────────────────────
//
// dst: prime-major [batch × 4 × n] int32_t
// src: [batch × n] int64_t
// All other params mirror Cheddar's NTTPhase1<u32, log_degree>.

template <typename word, int log_degree>
__global__ void NTTPhase1_i64(
    make_signed_t<word>  *dst,
    const int64_t        *src,
    const word           *primes,
    const make_signed_t<word> *inv_primes,
    const word           *twiddle_factors,
    int                   n) {
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  using Config = NTTLaunchConfig<log_degree, NTTType::NTT, Phase::Phase1>;
  constexpr int kNumStages      = Config::RadixStages();
  constexpr int kStageMerging   = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages     = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLogWarpBatching = Config::LogWarpBatching();

  int y_idx = blockIdx.y;   // prime index in [0, 4)
  int z_idx = blockIdx.z;   // batch index

  word prime        = basic::StreamingLoadConst(primes     + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  // Barrett multiplier: one 64-bit division per thread, amortized over kPerThreadElems elements.
  uint64_t m4 = UINT64_MAX / (uint64_t)prime;

  // Source: flat i64 layout [batch][n].
  const int64_t *src_limb = src + (int64_t)z_idx * n;
  // Destination: prime-major i32 layout [batch][prime][n].
  signed_word *dst_limb = dst + ((int64_t)z_idx * 4 + y_idx) * n;
  const word  *w        = twiddle_factors + (y_idx << log_degree);

  // Load: strided pattern (same as Cheddar NTTPhase1), but from int64_t.
  // Each element is reduced mod prime via Barrett reduction.
  signed_word local[kPerThreadElems];
  int stage_group_idx = threadIdx.x >> kLogWarpBatching;
  int batch_idx       = threadIdx.x & ((1 << kLogWarpBatching) - 1);
  const int64_t *load_ptr = src_limb + batch_idx
                            + (blockIdx.x << kLogWarpBatching)
                            + (stage_group_idx << (log_degree - kNumStages));
  for (int i = 0; i < kPerThreadElems; i++) {
    int64_t raw = basic::StreamingLoad<int64_t>(
        load_ptr + (i << (log_degree - kStageMerging)));
    local[i] = (signed_word)barrett_reduce_i64(raw, prime, m4);
  }

  // Butterfly stages — identical to Cheddar NTTPhase1 from here.
  int final_tw_idx  = (1 << (kNumStages - kStageMerging)) + stage_group_idx;
  int tw_idx        = final_tw_idx >> (kNumStages - kStageMerging);
  int sm_log_stride = kNumStages - kStageMerging + kLogWarpBatching;

  MultiRadixNTTFirst<word, kPerThreadElems, kTailStages>(
      local, tw_idx, w, prime, inv_prime);
  for (int j = 0; j < kPerThreadElems; j++)
    temp[threadIdx.x + (j << sm_log_stride)] = local[j];
  __syncthreads();
  sm_log_stride -= kTailStages;

  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = num_main_iters - 1; i >= 0; i--) {
    int sm_idx =
        ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
        (threadIdx.x & ((1 << sm_log_stride) - 1));
    for (int j = 0; j < kPerThreadElems; j++)
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    int tw = final_tw_idx >> (kStageMerging * i);
    MultiRadixNTT<word, kPerThreadElems, kStageMerging>(
        local, tw, w, prime, inv_prime);
    if (i == 0) break;
    for (int j = 0; j < kPerThreadElems; j++)
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    __syncthreads();
    sm_log_stride -= kStageMerging;
  }

  int dst_idx = batch_idx
      + (stage_group_idx << ((log_degree - kNumStages) + kStageMerging))
      + (blockIdx.x << kLogWarpBatching);
  for (int i = 0; i < kPerThreadElems; i++)
    dst_limb[dst_idx + (i << (log_degree - kNumStages))] = local[i];
}

// ── Adapted NTTPhase2: i32 input → i32 DFT output ───────────────────────────
//
// src: prime-major [batch × 4 × n] int32_t (Phase1 output)
// dst: prime-major [batch × 4 × n] int32_t (Phase2 output, same buffer ok)

template <typename word, int log_degree>
__global__ void NTTPhase2_u32(
    make_signed_t<word>        *dst,
    const make_signed_t<word>  *src,
    const word                 *primes,
    const make_signed_t<word>  *inv_primes,
    const word                 *twiddle_factors,
    const word                 *twiddle_factors_msb) {
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  using Config = NTTLaunchConfig<log_degree, NTTType::NTT, Phase::Phase2>;
  constexpr int kNumStages      = Config::RadixStages();
  constexpr int kStageMerging   = Config::StageMerging();
  constexpr int kPerThreadElems = 1 << kStageMerging;
  constexpr int kTailStages     = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize        = Config::LsbSize();
  constexpr int kMsbSize        = (1 << log_degree) / kLsbSize;
  constexpr int kOFTwiddle      = Config::OFTwiddle();
  constexpr int kLogWarpBatching = Config::LogWarpBatching();

  int row_idx   = threadIdx.x >> (kNumStages - kStageMerging);
  int batch_idx = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
  temp += row_idx << kNumStages;

  int y_idx = blockIdx.y;
  int z_idx = blockIdx.z;

  word prime        = basic::StreamingLoadConst(primes     + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);

  const signed_word *src_limb = src + ((int64_t)z_idx * 4 + y_idx) * (1 << log_degree);
  signed_word       *dst_limb = dst + ((int64_t)z_idx * 4 + y_idx) * (1 << log_degree);
  const word *w     = twiddle_factors     + (y_idx << log_degree);
  const word *w_msb = twiddle_factors_msb + (y_idx * kMsbSize);

  signed_word local[kPerThreadElems];
  int log_stride = kNumStages - kStageMerging;
  const signed_word *load_ptr =
      src_limb + batch_idx
      + (blockIdx.x << (kNumStages + kLogWarpBatching))
      + (row_idx << kNumStages);
  for (int i = 0; i < kPerThreadElems; i++)
    local[i] = basic::StreamingLoad(load_ptr + (i << log_stride));

  int x_idx        = blockIdx.x * blockDim.x + threadIdx.x;
  int final_tw_idx = (1 << (log_degree - kStageMerging)) + x_idx;
  int tw_idx       = final_tw_idx >> (kNumStages - kStageMerging);
  int sm_log_stride = log_stride;

  MultiRadixNTTFirst<word, kPerThreadElems, kTailStages>(
      local, tw_idx, w, prime, inv_prime);
  for (int j = 0; j < kPerThreadElems; j++)
    temp[batch_idx + (j << sm_log_stride)] = local[j];
  __syncthreads();
  sm_log_stride -= kTailStages;

  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = num_main_iters - 1; i >= 0; i--) {
    int sm_idx =
        ((batch_idx >> sm_log_stride) << (sm_log_stride + kStageMerging)) +
        (batch_idx & ((1 << sm_log_stride) - 1));
    for (int j = 0; j < kPerThreadElems; j++)
      local[j] = temp[sm_idx + (j << sm_log_stride)];
    int tw = final_tw_idx >> (kStageMerging * i);
    if (i == 0) {
      if constexpr (kOFTwiddle)
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw, w, w_msb, prime, inv_prime);
      else
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(
            local, tw, w, prime, inv_prime);
    } else {
      if constexpr (kOFTwiddle & !kExtendedOT)
        MultiRadixNTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw, w, w_msb, prime, inv_prime);
      else
        MultiRadixNTT<word, kPerThreadElems, kStageMerging>(
            local, tw, w, prime, inv_prime);
    }
    if (i == 0) break;
    for (int j = 0; j < kPerThreadElems; j++)
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    __syncthreads();
    sm_log_stride -= kStageMerging;
  }

  for (int i = 0; i < kPerThreadElems; i++)
    if (local[i] < 0) local[i] += prime;

  signed_word *dst_ptr = dst_limb + (x_idx << kStageMerging);
#pragma unroll
  for (int i = 0; i < kPerThreadElems; i++) {
    dst_ptr[i] = local[i];
  }
}

// ── Per-log_n launcher helper ─────────────────────────────────────────────────

template <int log_n>
static void launch_ntt_fwd(
    cudaStream_t    stream,
    int32_t        *dst,
    const int64_t  *src,
    const uint32_t *tw_fwd,
    const uint32_t *tw_fwd_msb,
    const uint32_t *primes,
    const int32_t  *inv_primes,
    int             n,
    int             batch) {
  using C1 = NTTLaunchConfig<log_n, NTTType::NTT, Phase::Phase1>;
  using C2 = NTTLaunchConfig<log_n, NTTType::NTT, Phase::Phase2>;

  // Phase 1: i64 → i32 with inline reduction.
  {
    constexpr int bd    = C1::BlockDim();
    constexpr int sm    = C1::StageMerging();
    dim3 grid(n / (1 << sm) / bd, 4, batch);
    int  shmem = bd * (1 << sm) * sizeof(int32_t);
    NTTPhase1_i64<uint32_t, log_n><<<grid, bd, shmem, stream>>>(
        dst, src, primes, inv_primes, tw_fwd, n);
    check_cuda_runtime("NTTPhase1_i64 launch");
  }

  // Phase 2: i32 → final NTT output (in-place via intermediate buffer = dst).
  {
    constexpr int bd    = C2::BlockDim();
    constexpr int sm    = C2::StageMerging();
    dim3 grid(n / (1 << sm) / bd, 4, batch);
    int  shmem = bd * (1 << sm) * sizeof(int32_t);
    NTTPhase2_u32<uint32_t, log_n><<<grid, bd, shmem, stream>>>(
        dst, dst, primes, inv_primes, tw_fwd, tw_fwd_msb);
    check_cuda_runtime("NTTPhase2_u32 launch");
  }
}

// ── Public C-linkage launchers ────────────────────────────────────────────────

extern "C" void ntt120_ntt_fwd_apply(
    cudaStream_t    stream,
    int32_t        *dst,
    const int64_t  *src,
    const uint32_t *twiddle_fwd,
    const uint32_t *twiddle_fwd_msb,
    const uint32_t *primes,
    const int32_t  *inv_primes,
    int             log_n,
    int             batch) {
  int n = 1 << log_n;
  switch (log_n) {
    case 12: launch_ntt_fwd<12>(stream, dst, src, twiddle_fwd, twiddle_fwd_msb, primes, inv_primes, n, batch); break;
    case 13: launch_ntt_fwd<13>(stream, dst, src, twiddle_fwd, twiddle_fwd_msb, primes, inv_primes, n, batch); break;
    case 14: launch_ntt_fwd<14>(stream, dst, src, twiddle_fwd, twiddle_fwd_msb, primes, inv_primes, n, batch); break;
    case 15: launch_ntt_fwd<15>(stream, dst, src, twiddle_fwd, twiddle_fwd_msb, primes, inv_primes, n, batch); break;
    case 16: launch_ntt_fwd<16>(stream, dst, src, twiddle_fwd, twiddle_fwd_msb, primes, inv_primes, n, batch); break;
    default:
      std::fprintf(stderr,
                   "ntt120_ntt_fwd_apply: unsupported log_n=%d (supported: 12..16)\n",
                   log_n);
      std::abort();
  }
}

// ── Adapted INTTPhase1: i32 NTT-domain → i32 intermediate ───────────────────
//
// Mirrors Cheddar's INTTPhase1 with the batch dimension (blockIdx.z = limb index).
// Each thread loads kPerThreadElems consecutive i32 residues, applies 9 INTT stages
// (with OT-twiddle for the first stage), and stores in stride-64 order.
// Because the __syncthreads() calls between shared-memory passes guarantee all
// global loads complete before any global stores, this kernel is in-place safe
// (src and dst may alias the same buffer).

template <typename word, int log_degree>
__global__ void INTTPhase1_i32(
    make_signed_t<word>        *dst,
    const make_signed_t<word>  *src,
    const word                 *primes,
    const make_signed_t<word>  *inv_primes,
    const word                 *twiddle_inv,
    const word                 *twiddle_inv_msb) {
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  using Config = NTTLaunchConfig<log_degree, NTTType::INTT, Phase::Phase1>;
  constexpr int kNumStages       = Config::RadixStages();       // 9
  constexpr int kStageMerging    = Config::StageMerging();      // 3
  constexpr int kPerThreadElems  = 1 << kStageMerging;          // 8
  constexpr int kTailStages      = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLsbSize         = Config::LsbSize();
  constexpr int kMsbSize         = (1 << log_degree) / kLsbSize;
  constexpr int kOFTwiddle       = Config::OFTwiddle();
  constexpr int kLogWarpBatching = Config::LogWarpBatching();   // 0

  int row_idx   = threadIdx.x >> (kNumStages - kStageMerging);
  int batch_idx = threadIdx.x & ((1 << (kNumStages - kStageMerging)) - 1);
  temp += row_idx << kNumStages;

  int y_idx = blockIdx.y;
  int z_idx = blockIdx.z;

  word prime         = basic::StreamingLoadConst(primes     + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);

  const signed_word *src_limb = src + ((int64_t)z_idx * 4 + y_idx) * (1 << log_degree);
        signed_word *dst_limb = dst + ((int64_t)z_idx * 4 + y_idx) * (1 << log_degree);
  const word *w     = twiddle_inv     + (y_idx << log_degree);
  const word *w_msb = twiddle_inv_msb + (y_idx * kMsbSize);

  signed_word local[kPerThreadElems];
  int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  basic::VectorizedMove<signed_word, kPerThreadElems>(local, src_limb + (x_idx << kStageMerging));

  int tw_idx       = (1 << (log_degree - kStageMerging)) + x_idx;
  int sm_log_stride = 0;
  int sm_idx        = batch_idx << kStageMerging;

  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = 0; i < num_main_iters; i++) {
    if (i == 0) {
      if constexpr (kOFTwiddle)
        MultiRadixINTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      else
        MultiRadixINTT<word, kPerThreadElems, kStageMerging>(
            local, tw_idx, w, prime, inv_prime);
    } else {
      if constexpr (kOFTwiddle & !kExtendedOT)
        MultiRadixINTT_OT<word, kPerThreadElems, kStageMerging, kLsbSize>(
            local, tw_idx, w, w_msb, prime, inv_prime);
      else
        MultiRadixINTT<word, kPerThreadElems, kStageMerging>(
            local, tw_idx, w, prime, inv_prime);
    }

    for (int j = 0; j < kPerThreadElems; j++)
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    __syncthreads();

    if (i == num_main_iters - 1) {
      tw_idx        >>= kTailStages;
      sm_log_stride  += kTailStages;
    } else {
      tw_idx        >>= kStageMerging;
      sm_log_stride  += kStageMerging;
    }
    sm_idx = (batch_idx & ((1 << sm_log_stride) - 1)) +
             ((batch_idx >> sm_log_stride) << (sm_log_stride + kStageMerging));
    for (int j = 0; j < kPerThreadElems; j++)
      local[j] = temp[sm_idx + (j << sm_log_stride)];
  }
  MultiRadixINTTLast<word, kPerThreadElems, kTailStages>(local, tw_idx, w, prime, inv_prime);

  int dst_idx = batch_idx + (blockIdx.x << (kNumStages + kLogWarpBatching)) +
                (row_idx << kNumStages);
  for (int j = 0; j < kPerThreadElems; j++)
    dst_limb[dst_idx + (j << (kNumStages - kStageMerging))] = local[j];
}

// ── Adapted INTTPhase2: i32 intermediate → i32 time-domain (normalized) ──────
//
// Mirrors Cheddar's INTTPhase2 with MultConstNormalize: multiplies each coefficient
// by inv_n_mont[y_idx] in Montgomery form and maps to (-prime/2, prime/2].
// Load/store pattern is identical to NTTPhase1 (strided by n / kNumStages).

template <typename word, int log_degree>
__global__ void INTTPhase2_i32(
    make_signed_t<word>        *dst,
    const make_signed_t<word>  *src,
    const word                 *primes,
    const make_signed_t<word>  *inv_primes,
    const word                 *twiddle_inv,
    const word                 *inv_n_mont) {
  extern __shared__ char shared_mem[];
  using signed_word = make_signed_t<word>;
  signed_word *temp = reinterpret_cast<signed_word *>(shared_mem);

  using Config = NTTLaunchConfig<log_degree, NTTType::INTT, Phase::Phase2>;
  constexpr int kNumStages       = Config::RadixStages();
  constexpr int kStageMerging    = Config::StageMerging();
  constexpr int kPerThreadElems  = 1 << kStageMerging;
  constexpr int kTailStages      = (kNumStages - 1) % kStageMerging + 1;
  constexpr int kLogWarpBatching = Config::LogWarpBatching();

  int y_idx = blockIdx.y;
  int z_idx = blockIdx.z;

  word prime         = basic::StreamingLoadConst(primes     + y_idx);
  signed_word inv_prime = basic::StreamingLoadConst(inv_primes + y_idx);
  word inv_n         = basic::StreamingLoadConst(inv_n_mont + y_idx);

  const signed_word *src_limb = src + ((int64_t)z_idx * 4 + y_idx) * (1 << log_degree);
        signed_word *dst_limb = dst + ((int64_t)z_idx * 4 + y_idx) * (1 << log_degree);
  const word *w = twiddle_inv + (y_idx << log_degree);

  signed_word local[kPerThreadElems];
  constexpr int initial_log_stride = log_degree - kNumStages;
  int stage_group_idx = threadIdx.x >> kLogWarpBatching;
  int batch_idx       = threadIdx.x & ((1 << kLogWarpBatching) - 1);
  const signed_word *load_ptr =
      src_limb + (stage_group_idx << (initial_log_stride + kStageMerging)) +
      batch_idx + (blockIdx.x << kLogWarpBatching);
  for (int i = 0; i < kPerThreadElems; i++)
    local[i] = basic::StreamingLoad(load_ptr + (i << initial_log_stride));

  int tw_idx        = (1 << (kNumStages - kStageMerging)) + stage_group_idx;
  int sm_log_stride = kLogWarpBatching;
  int sm_idx        = (threadIdx.x & ((1 << sm_log_stride) - 1)) +
                      ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging));

  constexpr int num_main_iters = (kNumStages - kTailStages) / kStageMerging;
#pragma unroll
  for (int i = 0; i < num_main_iters; i++) {
    MultiRadixINTT<word, kPerThreadElems, kStageMerging>(local, tw_idx, w, prime, inv_prime);
    for (int j = 0; j < kPerThreadElems; j++)
      temp[sm_idx + (j << sm_log_stride)] = local[j];
    __syncthreads();

    if (i == num_main_iters - 1) {
      tw_idx        >>= kTailStages;
      sm_log_stride  += kTailStages;
    } else {
      tw_idx        >>= kStageMerging;
      sm_log_stride  += kStageMerging;
    }
    sm_idx = (threadIdx.x & ((1 << sm_log_stride) - 1)) +
             ((threadIdx.x >> sm_log_stride) << (sm_log_stride + kStageMerging));
    for (int j = 0; j < kPerThreadElems; j++)
      local[j] = temp[sm_idx + (j << sm_log_stride)];
  }
  MultiRadixINTTLast<word, kPerThreadElems, kTailStages>(local, tw_idx, w, prime, inv_prime);

  int dst_idx = batch_idx + (stage_group_idx << initial_log_stride) +
                (blockIdx.x << kLogWarpBatching);
  for (int j = 0; j < kPerThreadElems; j++) {
    MultConstNormalize<word>(local[j], local[j], inv_n, prime, inv_prime);
    dst_limb[dst_idx + (j << (log_degree - kStageMerging))] = local[j];
  }
}

// ── Per-log_n inverse launcher ────────────────────────────────────────────────

template <int log_n>
static void launch_ntt_inv(
    cudaStream_t    stream,
    int32_t        *dst,
    const int32_t  *src,
    const uint32_t *tw_inv,
    const uint32_t *tw_inv_msb,
    const uint32_t *primes,
    const int32_t  *inv_primes,
    const uint32_t *inv_n_mont,
    int             n,
    int             batch) {
  using C1 = NTTLaunchConfig<log_n, NTTType::INTT, Phase::Phase1>;
  using C2 = NTTLaunchConfig<log_n, NTTType::INTT, Phase::Phase2>;

  // Phase 1: 9-stage INTT with OT-twiddle.
  {
    constexpr int bd   = C1::BlockDim();
    constexpr int sm   = C1::StageMerging();
    dim3 grid(n / (1 << sm) / bd, 4, batch);
    int  shmem = bd * (1 << sm) * sizeof(int32_t);
    INTTPhase1_i32<uint32_t, log_n><<<grid, bd, shmem, stream>>>(
        dst, src, primes, inv_primes, tw_inv, tw_inv_msb);
    check_cuda_runtime("INTTPhase1_i32 launch");
  }

  // Phase 2: (log_n-9)-stage INTT with n^{-1} normalization (in-place on dst).
  {
    constexpr int bd   = C2::BlockDim();
    constexpr int sm   = C2::StageMerging();
    dim3 grid(n / (1 << sm) / bd, 4, batch);
    int  shmem = bd * (1 << sm) * sizeof(int32_t);
    INTTPhase2_i32<uint32_t, log_n><<<grid, bd, shmem, stream>>>(
        dst, dst, primes, inv_primes, tw_inv, inv_n_mont);
    check_cuda_runtime("INTTPhase2_i32 launch");
  }
}

extern "C" void ntt120_ntt_inv_apply(
    cudaStream_t    stream,
    int32_t        *scratch,
    const int32_t  *src,
    const uint32_t *twiddle_inv,
    const uint32_t *twiddle_inv_msb,
    const uint32_t *primes,
    const int32_t  *inv_primes,
    const uint32_t *inv_n_mont,
    int             log_n,
    int             batch) {
  int n = 1 << log_n;
  switch (log_n) {
    case 12: launch_ntt_inv<12>(stream, scratch, src, twiddle_inv, twiddle_inv_msb, primes, inv_primes, inv_n_mont, n, batch); break;
    case 13: launch_ntt_inv<13>(stream, scratch, src, twiddle_inv, twiddle_inv_msb, primes, inv_primes, inv_n_mont, n, batch); break;
    case 14: launch_ntt_inv<14>(stream, scratch, src, twiddle_inv, twiddle_inv_msb, primes, inv_primes, inv_n_mont, n, batch); break;
    case 15: launch_ntt_inv<15>(stream, scratch, src, twiddle_inv, twiddle_inv_msb, primes, inv_primes, inv_n_mont, n, batch); break;
    case 16: launch_ntt_inv<16>(stream, scratch, src, twiddle_inv, twiddle_inv_msb, primes, inv_primes, inv_n_mont, n, batch); break;
    default:
      std::fprintf(stderr,
                   "ntt120_ntt_inv_apply: unsupported log_n=%d (supported: 12..16)\n",
                   log_n);
      std::abort();
  }
}
