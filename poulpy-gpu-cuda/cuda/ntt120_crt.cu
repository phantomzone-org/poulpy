// NTT120 CRT reconstruction and normalize kernels (Primes30).
//
// After INTT + MultConstNormalize, residues a[k] are in (-Q[k]/2, Q[k]/2] as
// signed i32.  Lagrange interpolation (same algorithm as compact_all_blocks_scalar
// in poulpy-cpu-ref/src/reference/ntt120/vec_znx_dft.rs):
//
//   r[k] = a[k] < 0 ? a[k] + Q[k] : a[k]           → [0, Q[k])
//   t[k] = r[k] * CRT_CST[k] % Q[k]                 → [0, Q[k])
//   v    = Σ_k t[k] * (Q / Q[k])     (__uint128_t, in [0, 4Q))
//   reduce v mod Q via q_approx = (v >> 120) trick
//   val  = v > floor(Q/2) ? v - Q : v                → (-Q/2, Q/2]
//
// Output is Big32 = [u32; 4] little-endian two's-complement i128.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void check_crt(const char *where) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
        std::abort();
    }
}

// ── Device helpers ────────────────────────────────────────────────────────────

// Reconstruct one CRT coefficient from 4 signed i32 residues.
// src_base[k * stride] is the residue for prime k (k = 0..3).
// Constants are Primes30 — must match primes.rs :: Primes30.
__device__ static __inline__ __int128
crt_reconstruct(const int32_t *src_base, int stride) {
    // Primes30: Q[k] = (1<<30) - c_k*(1<<17) + 1
    constexpr uint32_t kQ[4] = {1073479681u, 1071513601u, 1070727169u, 1068236801u};
    // CRT_CST[k] = (Q/Q[k])^{-1} mod Q[k]
    constexpr uint32_t kCrtCst[4] = {43599465u, 292938863u, 594011630u, 140177212u};

    // qm[k] = Q / Q[k] = product of the other three primes (~90 bits).
    const __uint128_t qm0 = (__uint128_t)kQ[1] * kQ[2] * kQ[3];
    const __uint128_t qm1 = (__uint128_t)kQ[0] * kQ[2] * kQ[3];
    const __uint128_t qm2 = (__uint128_t)kQ[0] * kQ[1] * kQ[3];
    const __uint128_t qm3 = (__uint128_t)kQ[0] * kQ[1] * kQ[2];
    const __uint128_t Q_total = (__uint128_t)kQ[0] * kQ[1] * kQ[2] * kQ[3];

    __uint128_t v = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t q = kQ[k], crt = kCrtCst[k];
        int32_t a = src_base[k * stride];
        uint32_t r = (uint32_t)(a < 0 ? a + (int32_t)q : a);
        uint32_t t = (uint32_t)((uint64_t)r * crt % q);
        __uint128_t qm = (k == 0) ? qm0 : (k == 1) ? qm1 : (k == 2) ? qm2 : qm3;
        v += (__uint128_t)t * qm;
    }

    // Reduce v mod Q_total.  Because t[k] < Q[k], v < 4*Q_total < 2^122, so
    // q_approx = (v >> 120) is in {0,1,2,3} and approximates floor(v / Q_total).
    int q_approx = (int)(v >> 120);
    if (q_approx == 1) v -= Q_total;
    else if (q_approx == 2) v -= 2 * Q_total;
    else if (q_approx == 3) v -= 3 * Q_total;
    if (v >= Q_total) v -= Q_total;

    // Signed correction: map [0, Q) → (-Q/2, Q/2].
    // half_Q = floor(Q/2): condition v > half_Q ↔ v >= ceil(Q/2), matching CPU.
    const __uint128_t half_Q = Q_total >> 1;
    return (v > half_Q) ? (__int128)v - (__int128)Q_total : (__int128)v;
}

// Store __int128 as Big32 (little-endian u32[4]).
__device__ static __inline__ void store_i128(uint32_t *out, __int128 val) {
    out[0] = (uint32_t)val;
    out[1] = (uint32_t)(val >> 32);
    out[2] = (uint32_t)(val >> 64);
    out[3] = (uint32_t)(val >> 96);
}

// Load Big32 as __int128.
__device__ static __inline__ __int128 load_i128(const uint32_t *p) {
    __uint128_t bits =
        ((__uint128_t)p[0])
        | ((__uint128_t)p[1] << 32)
        | ((__uint128_t)p[2] << 64)
        | ((__uint128_t)p[3] << 96);
    return (__int128)bits;
}

// ── CRT kernel: prime-major i32 → Big32 ──────────────────────────────────────
//
// src: [cols × 4 × n] i32 (prime-major)
// dst: [cols × n × 4] u32 (coefficient-major Big32)

__global__ static void crt_kernel(uint32_t *dst, const int32_t *src, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y;  // batch (limb) index
    if (j >= n) return;
    // prime k at: src[(z*4+k)*n + j] = src[z*4*n + j + k*n]
    __int128 val = crt_reconstruct(src + (int64_t)z * 4 * n + j, n);
    store_i128(dst + ((int64_t)z * n + j) * 4, val);
}

// ── CRT-accumulate kernel: add new CRT result into Big32 accumulator ──────────
//
// acc: [cols × n × 4] u32 Big32 (in/out)
// src: [cols × 4 × n] i32 (prime-major)

__global__ static void crt_accum_kernel(uint32_t *acc, const int32_t *src, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y;
    if (j >= n) return;
    __int128 val = crt_reconstruct(src + (int64_t)z * 4 * n + j, n);
    uint32_t *p = acc + ((int64_t)z * n + j) * 4;
    store_i128(p, load_i128(p) + val);
}

// ── Normalize kernel: extract i64 limb from Big32 ────────────────────────────
//
// dst: [batch × n] i64
// src: [batch × n × 4] u32 Big32
// result: dst[z*n+j] = (int64_t)(Big32[z,j] >> shift)

__global__ static void normalize_kernel(
    int64_t *dst, const uint32_t *src, int n, int shift) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y;
    if (j >= n) return;
    __int128 val = load_i128(src + ((int64_t)z * n + j) * 4);
    dst[(int64_t)z * n + j] = (int64_t)(val >> shift);
}

// ── Big normalize kernel: VecZnxBig [a_size×n×4 u32] → VecZnx [res_size×n i64]
//
// Implements ntt120_vec_znx_big_normalize_inter (a_base2k == res_base2k == base2k).
// One thread handles all limbs for one coefficient j; carry is register-local.

__device__ static __inline__ __int128 get_digit(__int128 x, int base2k) {
    int shift = 128 - base2k;
    return (x << shift) >> shift;
}

__device__ static __inline__ __int128 get_carry(__int128 x, __int128 digit, int base2k) {
    return (x - digit) >> base2k;
}

__global__ static void big_normalize_kernel(
    int64_t        *dst,
    const uint32_t *src,
    int n, int a_size, int res_size, int base2k, int64_t res_offset)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    // Decompose res_offset → (limbs_offset, lsh) matching the CPU reference.
    int lsh = (int)(res_offset % (int64_t)base2k);
    int limbs_offset = (int)(res_offset / (int64_t)base2k);
    if (res_offset < 0 && lsh != 0) {
        lsh = (lsh + base2k) % base2k;
        limbs_offset -= 1;
    }
    int base2k_lsh = base2k - lsh;

    // Clamped index ranges.
    int res_end   = min(max(-limbs_offset, 0), res_size);
    int res_start = min(max(a_size - limbs_offset, 0), res_size);
    int a_end     = min(max(limbs_offset, 0), a_size);
    int a_start   = min(max(res_size + limbs_offset, 0), a_size);
    int a_out_range = a_size - a_start;
    int mid_range   = a_start - a_end;

    __int128 carry = 0;

    // Step 1: carry-only for discarded high a limbs (beyond res precision).
    for (int k = 0; k < a_out_range; k++) {
        int limb = a_size - k - 1;
        __int128 ai = load_i128(src + ((int64_t)limb * n + j) * 4);
        if (k == 0) {
            if (lsh == 0) {
                __int128 d = get_digit(ai, base2k);
                carry = get_carry(ai, d, base2k);
            } else {
                __int128 d = get_digit(ai, base2k_lsh);
                carry = get_carry(ai, d, base2k_lsh);
            }
        } else {
            if (lsh == 0) {
                __int128 d  = get_digit(ai, base2k);
                __int128 co = get_carry(ai, d, base2k);
                __int128 dc = d + carry;
                carry = co + get_carry(dc, get_digit(dc, base2k), base2k);
            } else {
                __int128 d  = get_digit(ai, base2k_lsh);
                __int128 co = get_carry(ai, d, base2k_lsh);
                __int128 dc = (d << lsh) + carry;
                carry = co + get_carry(dc, get_digit(dc, base2k), base2k);
            }
        }
    }
    if (a_out_range == 0) carry = 0;

    // Zero res limbs that have no matching a limb.
    for (int k = res_start; k < res_size; k++)
        dst[(int64_t)k * n + j] = 0;

    // Step 2: normalize overlapping a limbs into res.
    for (int k = 0; k < mid_range; k++) {
        int a_limb = a_start - k - 1;
        int r_limb = res_start - k - 1;
        __int128 ai = load_i128(src + ((int64_t)a_limb * n + j) * 4);
        __int128 out128;
        if (lsh == 0) {
            __int128 d  = get_digit(ai, base2k);
            __int128 co = get_carry(ai, d, base2k);
            __int128 dc = d + carry;
            out128 = get_digit(dc, base2k);
            carry  = co + get_carry(dc, out128, base2k);
        } else {
            __int128 d  = get_digit(ai, base2k_lsh);
            __int128 co = get_carry(ai, d, base2k_lsh);
            __int128 dc = (d << lsh) + carry;
            out128 = get_digit(dc, base2k);
            carry  = co + get_carry(dc, out128, base2k);
        }
        dst[(int64_t)r_limb * n + j] = (int64_t)out128;
    }

    // Step 3: flush carry into bottom res limbs (all were zeroed above, so ri = 0
    // and each step simplifies to: extract digit from carry).
    for (int k = 0; k < res_end; k++) {
        int r_limb = res_end - k - 1;
        if (k == res_end - 1) {
            dst[(int64_t)r_limb * n + j] = (int64_t)get_digit(carry, base2k);
        } else {
            __int128 out128 = get_digit(carry, base2k);
            dst[(int64_t)r_limb * n + j] = (int64_t)out128;
            carry = get_carry(carry, out128, base2k);
        }
    }
}

// ── SVP prepare: NTT output i32 → q120c u32 (c_from_b) ──────────────────────
//
// src: [cols × 4 × n] i32, prime-major NTT output (values in [0, prime)).
// dst: SvpPPol [cols × 4 × n × 2] u32, prime-major layout.
// Lane 0: r.  Lane 1: r * 2^32 mod Q[k].
// Requires n >= 256 (blockDim.x) so every block covers a single prime.

__global__ static void c_from_b_kernel(
    uint32_t *dst, const int32_t *src,
    int n, int cols, const uint32_t *primes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // in [0, 4*n)
    int z   = blockIdx.y;                              // col index
    if (idx >= 4 * n || z >= cols) return;

    int k = idx / n;  // prime index
    int j = idx % n;  // coeff index

    uint32_t prime = primes[k];
    int32_t  raw   = src[(int64_t)(z * 4 + k) * n + j];
    uint32_t r     = (uint32_t)(raw < 0 ? raw + (int32_t)prime : raw);
    uint32_t r_shl = (uint32_t)(((uint64_t)r << 32) % (uint64_t)prime);

    int64_t out_base = ((int64_t)(z * 4 + k) * n + j) * 2;
    dst[out_base]     = r;
    dst[out_base + 1] = r_shl;
}

// ── SVP multiply: VecZnxDft × SvpPPol (one col) → VecZnxDft ────────────────
//
// a: SvpPPol column [4 × n × 2] u32 (prime-major; pointer already at col start).
// b: VecZnxDft [size × 4 × n] i32.  May alias res (in-place is safe).
// res: VecZnxDft [size × 4 × n] i32.
// Requires n >= 256 so that all threads in a block share the same prime.

__global__ static void svp_mul_dft_kernel(
    int32_t *res, const uint32_t *a, const int32_t *b,
    int n, int size, const uint32_t *primes)
{
    __shared__ uint32_t s_prime;
    __shared__ uint64_t s_m4;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // in [0, 4*n)
    int z   = blockIdx.y;                              // limb index
    if (idx >= 4 * n || z >= size) return;

    int k = idx / n;
    int j = idx % n;

    // All threads in a block share the same prime (n >= blockDim.x guaranteed).
    if (threadIdx.x == 0) {
        s_prime = primes[k];
        s_m4    = UINT64_MAX / (uint64_t)s_prime;
    }
    __syncthreads();

    uint32_t prime = s_prime;
    uint64_t m4    = s_m4;
    uint32_t r     = a[(k * n + j) * 2];             // lane 0
    uint32_t x     = (uint32_t)b[(int64_t)(z * 4 + k) * n + j];

    uint64_t prod   = (uint64_t)x * (uint64_t)r;
    uint64_t q      = __umul64hi(prod, m4);
    uint64_t result = prod - q * (uint64_t)prime;
    if (result >= (uint64_t)prime) result -= (uint64_t)prime;

    res[(int64_t)(z * 4 + k) * n + j] = (int32_t)result;
}

// ── DFT-domain pointwise modular add ─────────────────────────────────────────
//
// dst: [size × 4 × n] i32, prime-major layout (same as NTT kernel output).
// src: same layout.
// Values are in [0, prime[k]) after forward NTT; their sum fits in int32_t
// since prime[k] < 2^30, so sum < 2^31.

__global__ static void dft_add_assign_kernel(
    int32_t *res, const int32_t *a,
    int64_t total, int log_n, const uint32_t *primes)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint32_t prime = primes[(i >> log_n) & 3];
    int32_t r = res[i] + a[i];
    if ((uint32_t)r >= prime) r -= (int32_t)prime;
    res[i] = r;
}

// ── C-linkage launchers ───────────────────────────────────────────────────────

static constexpr int kCrtBlockDim = 256;

extern "C" void ntt120_crt(
    cudaStream_t    stream,
    uint32_t       *dst,
    const int32_t  *src,
    int             n,
    int             cols) {
    dim3 grid((n + kCrtBlockDim - 1) / kCrtBlockDim, cols);
    crt_kernel<<<grid, kCrtBlockDim, 0, stream>>>(dst, src, n);
    check_crt("ntt120_crt launch");
}

extern "C" void ntt120_crt_accum(
    cudaStream_t    stream,
    uint32_t       *acc,
    const int32_t  *src,
    int             n,
    int             cols) {
    dim3 grid((n + kCrtBlockDim - 1) / kCrtBlockDim, cols);
    crt_accum_kernel<<<grid, kCrtBlockDim, 0, stream>>>(acc, src, n);
    check_crt("ntt120_crt_accum launch");
}

extern "C" void ntt120_normalize(
    cudaStream_t    stream,
    int64_t        *dst,
    const uint32_t *src,
    int             n,
    int             batch,
    int             base2k,
    int             limb_idx) {
    dim3 grid((n + kCrtBlockDim - 1) / kCrtBlockDim, batch);
    normalize_kernel<<<grid, kCrtBlockDim, 0, stream>>>(dst, src, n, base2k * limb_idx);
    check_crt("ntt120_normalize launch");
}

extern "C" void ntt120_big_normalize(
    cudaStream_t    stream,
    int64_t        *dst,
    const uint32_t *src,
    int             n,
    int             a_size,
    int             res_size,
    int             base2k,
    int64_t         res_offset) {
    if (n == 0) return;
    dim3 grid((n + kCrtBlockDim - 1) / kCrtBlockDim);
    big_normalize_kernel<<<grid, kCrtBlockDim, 0, stream>>>(
        dst, src, n, a_size, res_size, base2k, res_offset);
    check_crt("ntt120_big_normalize launch");
}

// size: number of limbs; total elements = size * 4 * n.
// primes: device pointer to the 4 NTT primes in prime-major order.
extern "C" void ntt120_dft_add_assign(
    cudaStream_t    stream,
    int32_t        *res,
    const int32_t  *a,
    int             n,
    int             size,
    const uint32_t *primes) {
    if (size == 0) return;
    int log_n = __builtin_ctz((unsigned)n);
    int64_t total = (int64_t)size * 4 * n;
    dim3 grid((int)((total + kCrtBlockDim - 1) / kCrtBlockDim));
    dft_add_assign_kernel<<<grid, kCrtBlockDim, 0, stream>>>(res, a, total, log_n, primes);
    check_crt("ntt120_dft_add_assign launch");
}

// src: [cols × 4 × n] i32 NTT output (prime-major).
// dst: SvpPPol [cols × 4 × n × 2] u32 (prime-major, 2 Montgomery lanes).
extern "C" void ntt120_svp_c_from_b(
    cudaStream_t    stream,
    uint32_t       *dst,
    const int32_t  *src,
    int             n,
    int             cols,
    const uint32_t *primes) {
    if (cols == 0) return;
    dim3 grid((4 * n + kCrtBlockDim - 1) / kCrtBlockDim, cols);
    c_from_b_kernel<<<grid, kCrtBlockDim, 0, stream>>>(dst, src, n, cols, primes);
    check_crt("ntt120_svp_c_from_b launch");
}

// a: SvpPPol column [4 × n × 2] u32 (prime-major; pointer at col start).
// b: VecZnxDft [size × 4 × n] i32.  res may alias b (in-place).
extern "C" void ntt120_svp_mul_dft(
    cudaStream_t    stream,
    int32_t        *res,
    const uint32_t *a,
    const int32_t  *b,
    int             n,
    int             size,
    const uint32_t *primes) {
    if (size == 0) return;
    dim3 grid((4 * n + kCrtBlockDim - 1) / kCrtBlockDim, size);
    svp_mul_dft_kernel<<<grid, kCrtBlockDim, 0, stream>>>(res, a, b, n, size, primes);
    check_crt("ntt120_svp_mul_dft launch");
}

// ── VmpPMat prepare: NTT i32 output → VmpPMat u32 layout ────────────────────
//
// Rearranges `in_rows * out_vecs` NTT polynomials from the flat prime-major
// NTT output into the VmpPMat block-interleaved GPU layout:
//   pmat_dst[blk_j][ov][ir][k*2+pair] : u32
// where blk_j = j/2, pair = j%2, k = prime index.
//
// ntt_src: [io_pair × 4 × n] i32, io_pair = ir * out_vecs + ov.
// pmat_dst: [n/2 × out_vecs × in_rows × 8] u32.
// Values in ntt_src are in [0, Q[k]) (forward NTT output).

__global__ static void vmp_pmat_pack_kernel(
    uint32_t       *pmat_dst,
    const int32_t  *ntt_src,
    int             n,
    int             in_rows,
    int             out_vecs)
{
    int flat_kj  = blockIdx.x * blockDim.x + threadIdx.x;  // [0, 4*n)
    int io_pair  = blockIdx.y;                              // [0, in_rows*out_vecs)

    if (flat_kj >= 4 * n) return;

    int k    = flat_kj / n;
    int j    = flat_kj % n;
    int ir   = io_pair / out_vecs;
    int ov   = io_pair % out_vecs;
    int blk  = j / 2;
    int pair = j & 1;

    int32_t  src_val  = ntt_src[(int64_t)io_pair * 4 * n + flat_kj];
    uint32_t residue  = (uint32_t)src_val;  // forward NTT output is in [0, Q[k])

    int64_t dst_idx = ((int64_t)blk * out_vecs * in_rows
                       + (int64_t)ov * in_rows + ir) * 8
                      + k * 2 + pair;
    pmat_dst[dst_idx] = residue;
}

extern "C" void ntt120_vmp_pmat_pack(
    cudaStream_t    stream,
    uint32_t       *pmat_dst,
    const int32_t  *ntt_src,
    int             n,
    int             in_rows,
    int             out_vecs) {
    int io_pairs = in_rows * out_vecs;
    if (io_pairs == 0) return;
    dim3 grid((4 * n + kCrtBlockDim - 1) / kCrtBlockDim, io_pairs);
    vmp_pmat_pack_kernel<<<grid, kCrtBlockDim, 0, stream>>>(pmat_dst, ntt_src, n, in_rows, out_vecs);
    check_crt("ntt120_vmp_pmat_pack launch");
}

// ── VmpPMat apply: VecZnxDft × VmpPMat → VecZnxDft ─────────────────────────
//
// Computes the NTT-domain vector-matrix product (overwrite mode):
//   res[ov][k][j] = Σ_{ir} a[ir][k][j] * pmat[blk_j][ov+col_off][ir][k*2+pair] mod Q[k]
//
// a:    [a_row_max × 4 × n] i32, prime-major.
// pmat: [n/2 × out_vecs × in_rows × 8] u32.
// res:  [active_ovs × 4 × n] i32.
//
// Requires n >= kVmpBlockDim so every thread block covers a single prime.

__global__ static void vmp_apply_kernel(
    int32_t        *res,
    const int32_t  *a,
    const uint32_t *pmat,
    int             n,
    int             a_row_max,
    int             in_rows,
    int             out_vecs,
    int64_t         pmat_col_off,
    const uint32_t *primes)
{
    __shared__ uint32_t s_prime;
    __shared__ uint64_t s_m4;

    int flat_kj = blockIdx.x * blockDim.x + threadIdx.x;  // [0, 4*n)
    int ov      = blockIdx.y;                              // output limb index

    if (flat_kj >= 4 * n) return;

    int k    = flat_kj / n;
    int j    = flat_kj % n;
    int blk  = j / 2;
    int pair = j & 1;

    // All threads in a block share the same prime (n >= blockDim.x guaranteed).
    if (threadIdx.x == 0) {
        s_prime = primes[k];
        s_m4    = UINT64_MAX / (uint64_t)s_prime;
    }
    __syncthreads();

    uint32_t prime  = s_prime;
    uint64_t m4     = s_m4;
    int64_t pmat_col = (int64_t)ov + pmat_col_off;

    // Accumulate over input rows.  Each product < prime < 2^30, and
    // in_rows < 2^30, so acc < 2^60 fits in uint64_t.
    uint64_t acc = 0;
    for (int ir = 0; ir < a_row_max; ir++) {
        uint32_t a_val = (uint32_t)a[(int64_t)(ir * 4 + k) * n + j];
        int64_t  pmat_idx = (((int64_t)blk * out_vecs + pmat_col) * in_rows + ir) * 8
                            + k * 2 + pair;
        uint32_t m_val = pmat[pmat_idx];

        uint64_t prod = (uint64_t)a_val * (uint64_t)m_val;
        uint64_t q    = __umul64hi(prod, m4);
        uint64_t r    = prod - q * (uint64_t)prime;
        if (r >= (uint64_t)prime) r -= (uint64_t)prime;
        acc += r;
    }

    // Final Barrett reduction (acc may be up to in_rows * prime < 2^60).
    uint64_t qa = __umul64hi(acc, m4);
    uint64_t ra = acc - qa * (uint64_t)prime;
    if (ra >= (uint64_t)prime) ra -= (uint64_t)prime;
    if (ra >= (uint64_t)prime) ra -= (uint64_t)prime;

    res[(int64_t)(ov * 4 + k) * n + j] = (int32_t)ra;
}

// active_ovs:      number of output limbs to write (min(res_size, ncols - pmat_col_off)).
// pmat_col_off:    offset into pmat's ov dimension (= limb_offset * cols_out).
extern "C" void ntt120_vmp_apply(
    cudaStream_t    stream,
    int32_t        *res,
    const int32_t  *a,
    const uint32_t *pmat,
    int             n,
    int             a_row_max,
    int             in_rows,
    int             out_vecs,
    int             active_ovs,
    int64_t         pmat_col_off,
    const uint32_t *primes) {
    if (active_ovs == 0 || a_row_max == 0) return;
    dim3 grid((4 * n + kCrtBlockDim - 1) / kCrtBlockDim, active_ovs);
    vmp_apply_kernel<<<grid, kCrtBlockDim, 0, stream>>>(
        res, a, pmat, n, a_row_max, in_rows, out_vecs, pmat_col_off, primes);
    check_crt("ntt120_vmp_apply launch");
}
