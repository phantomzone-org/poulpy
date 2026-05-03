// NTT120 bivariate convolution kernels.
//
// Memory layouts used here:
//
//   CnvPVecL  — same as VecZnxDft: [batch × 4 × n] i32 (prime-major).
//               bytes_of_cnv_pvec_left(n, cols, size) = 4*n*cols*size*sizeof(i32).
//
//   CnvPVecR  — same as SvpPPol (effective_cols = cols*size):
//               [batch × 4 × n × 2] u32 (prime-major, 2 Montgomery lanes per coeff).
//               bytes_of_cnv_pvec_right(n, cols, size) = 8*n*cols*size*sizeof(u32).
//
//   VecZnxDft — [batch × 4 × n] i32 (prime-major, NTT output).
//   VecZnx    — [cols × size × n] i64 (coefficient domain, column-major).
//   VecZnxBig — [batch × n × 4] u32  (Big32, little-endian i128 per coefficient).

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

static constexpr int kCnvBlock = 256;

static void check_cnv(const char *where) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
        std::abort();
    }
}

// ── Mask kernel ───────────────────────────────────────────────────────────────
//
// dst[j] = src[j] & mask  for j in [0, n).
// Used to apply the coefficient precision mask to the last active limb before
// the forward NTT.

__global__ static void apply_mask_i64_kernel(
    int64_t       *dst,
    const int64_t *src,
    int64_t        mask,
    int            n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) dst[j] = src[j] & mask;
}

extern "C" void ntt120_apply_mask_i64(
    void          *stream,
    int64_t       *dst,
    const int64_t *src,
    int64_t        mask,
    int            n)
{
    if (n == 0) return;
    int grid = (n + kCnvBlock - 1) / kCnvBlock;
    apply_mask_i64_kernel<<<grid, kCnvBlock, 0, (cudaStream_t)stream>>>(dst, src, mask, n);
    check_cnv("ntt120_apply_mask_i64");
}

// ── Convolution apply DFT kernel ──────────────────────────────────────────────
//
// Computes the NTT-domain bbc convolution:
//   res[out_k][k][j] = Σ_{bj=j_min}^{j_max-1} a[k_abs-bj][k][j] * b[bj][k][j] mod Q[k]
//
// where k_abs = out_k + cnv_offset.
//
// a:   CnvPVecL column [a_size × 4 × n] i32  (prime-major NTT residues, values in [0,Q))
// b:   CnvPVecR column [b_size × 4 × n × 2] u32  (prime-major; lane 0 = r in [0,Q))
// res: VecZnxDft column [active_ovs × 4 × n] i32
//
// One thread per NTT position (prime k, coeff j).  Each thread loops over the
// accumulation index bj.

__global__ static void cnv_apply_dft_kernel(
    int32_t        *res,
    const int32_t  *a,
    const uint32_t *b,
    int             n,
    int             a_size,
    int             b_size,
    int             active_ovs,
    int             cnv_offset,
    const uint32_t *primes)
{
    int flat_kj = blockIdx.x * blockDim.x + threadIdx.x;  // [0, 4*n)
    int out_k   = blockIdx.y;                              // output limb index

    if (flat_kj >= 4 * n || out_k >= active_ovs) return;

    int k = flat_kj / n;
    int j = flat_kj % n;

    uint32_t prime = primes[k];
    uint64_t m4    = UINT64_MAX / (uint64_t)prime;

    int k_abs = out_k + cnv_offset;
    int j_min = (k_abs >= a_size) ? k_abs - a_size + 1 : 0;
    int j_max = (k_abs + 1 < b_size) ? k_abs + 1 : b_size;

    uint64_t acc = 0;

    for (int bj = j_min; bj < j_max; bj++) {
        int aj = k_abs - bj;

        // a[aj] at (prime k, coeff j): prime-major [aj × 4 × n]
        uint32_t a_val = (uint32_t)a[(int64_t)(aj * 4 + k) * n + j];

        // b[bj] lane 0 at (prime k, coeff j): [bj × 4 × n × 2], lane 0
        uint32_t b_val = b[((int64_t)(bj * 4 + k) * n + j) * 2];

        // Barrett: a_val * b_val mod prime
        uint64_t prod = (uint64_t)a_val * (uint64_t)b_val;
        uint64_t q    = __umul64hi(prod, m4);
        uint64_t r    = prod - q * (uint64_t)prime;
        if (r >= (uint64_t)prime) r -= (uint64_t)prime;
        acc += r;
    }

    // Final Barrett reduction of acc (sum of at most min(a_size,b_size) terms < prime each).
    if (acc >= (uint64_t)prime) {
        uint64_t qa = __umul64hi(acc, m4);
        uint64_t ra = acc - qa * (uint64_t)prime;
        if (ra >= (uint64_t)prime) ra -= (uint64_t)prime;
        if (ra >= (uint64_t)prime) ra -= (uint64_t)prime;
        acc = ra;
    }

    res[(int64_t)(out_k * 4 + k) * n + j] = (int32_t)acc;
}

extern "C" void ntt120_cnv_apply_dft(
    void           *stream,
    int32_t        *res,
    const int32_t  *a,
    const uint32_t *b,
    int             n,
    int             a_size,
    int             b_size,
    int             active_ovs,
    int             cnv_offset,
    const uint32_t *primes)
{
    if (active_ovs == 0 || a_size == 0 || b_size == 0) return;
    dim3 grid((4 * n + kCnvBlock - 1) / kCnvBlock, active_ovs);
    cnv_apply_dft_kernel<<<grid, kCnvBlock, 0, (cudaStream_t)stream>>>(
        res, a, b, n, a_size, b_size, active_ovs, cnv_offset, primes);
    check_cnv("ntt120_cnv_apply_dft");
}

// ── Pairwise convolution apply DFT kernel ────────────────────────────────────
//
// Computes:
//   res[out_k][k][j] = Σ_bj (a_i[aj][k][j] + a_j[aj][k][j])
//                           * (b_i[bj][k][j] + b_j[bj][k][j]) mod Q[k]
//
// Cross-products between the two pairs of columns are included by design.
//
// a_i, a_j: CnvPVecL columns [a_size × 4 × n] i32
// b_i, b_j: CnvPVecR columns [b_size × 4 × n × 2] u32
// res:       VecZnxDft column [active_ovs × 4 × n] i32

__global__ static void cnv_pairwise_apply_dft_kernel(
    int32_t        *res,
    const int32_t  *a_i,
    const int32_t  *a_j,
    const uint32_t *b_i,
    const uint32_t *b_j,
    int             n,
    int             a_size,
    int             b_size,
    int             active_ovs,
    int             cnv_offset,
    const uint32_t *primes)
{
    int flat_kj = blockIdx.x * blockDim.x + threadIdx.x;
    int out_k   = blockIdx.y;

    if (flat_kj >= 4 * n || out_k >= active_ovs) return;

    int k = flat_kj / n;
    int j = flat_kj % n;

    uint32_t prime = primes[k];
    uint64_t m4    = UINT64_MAX / (uint64_t)prime;

    int k_abs = out_k + cnv_offset;
    int j_min = (k_abs >= a_size) ? k_abs - a_size + 1 : 0;
    int j_max = (k_abs + 1 < b_size) ? k_abs + 1 : b_size;

    uint64_t acc = 0;

    for (int bj = j_min; bj < j_max; bj++) {
        int aj = k_abs - bj;
        int64_t a_pos = (int64_t)(aj * 4 + k) * n + j;
        int64_t b_pos = ((int64_t)(bj * 4 + k) * n + j) * 2;

        // a_sum = (a_i[aj] + a_j[aj]) mod prime
        uint32_t av_i = (uint32_t)a_i[a_pos];
        uint32_t av_j = (uint32_t)a_j[a_pos];
        uint32_t a_sum = av_i + av_j;
        if (a_sum >= prime) a_sum -= prime;

        // b_sum = (b_i[bj] + b_j[bj]) mod prime  (lane 0 only)
        uint32_t bv_i = b_i[b_pos];
        uint32_t bv_j = b_j[b_pos];
        uint32_t b_sum = bv_i + bv_j;
        if (b_sum >= prime) b_sum -= prime;

        // Barrett: a_sum * b_sum mod prime
        uint64_t prod = (uint64_t)a_sum * (uint64_t)b_sum;
        uint64_t q    = __umul64hi(prod, m4);
        uint64_t r    = prod - q * (uint64_t)prime;
        if (r >= (uint64_t)prime) r -= (uint64_t)prime;
        acc += r;
    }

    if (acc >= (uint64_t)prime) {
        uint64_t qa = __umul64hi(acc, m4);
        uint64_t ra = acc - qa * (uint64_t)prime;
        if (ra >= (uint64_t)prime) ra -= (uint64_t)prime;
        if (ra >= (uint64_t)prime) ra -= (uint64_t)prime;
        acc = ra;
    }

    res[(int64_t)(out_k * 4 + k) * n + j] = (int32_t)acc;
}

extern "C" void ntt120_cnv_pairwise_apply_dft(
    void           *stream,
    int32_t        *res,
    const int32_t  *a_i,
    const int32_t  *a_j,
    const uint32_t *b_i,
    const uint32_t *b_j,
    int             n,
    int             a_size,
    int             b_size,
    int             active_ovs,
    int             cnv_offset,
    const uint32_t *primes)
{
    if (active_ovs == 0 || a_size == 0 || b_size == 0) return;
    dim3 grid((4 * n + kCnvBlock - 1) / kCnvBlock, active_ovs);
    cnv_pairwise_apply_dft_kernel<<<grid, kCnvBlock, 0, (cudaStream_t)stream>>>(
        res, a_i, a_j, b_i, b_j, n, a_size, b_size, active_ovs, cnv_offset, primes);
    check_cnv("ntt120_cnv_pairwise_apply_dft");
}

// ── By-const apply kernel ─────────────────────────────────────────────────────
//
// Coefficient-domain negacyclic inner product into i128 accumulators:
//   res[out_k][j] = Σ_{bj=j_min}^{j_max-1} a[k_abs - bj][j] * b[bj]
//
// a:   VecZnx column [a_size × n] i64 (coefficient domain, limb-major)
// b:   constant array [b_size] i64 (device pointer)
// res: VecZnxBig column [active_ovs × n × 4] u32 (Big32 = little-endian i128)

__global__ static void cnv_by_const_apply_kernel(
    uint32_t       *res,
    const int64_t  *a,
    const int64_t  *b,
    int             n,
    int             a_size,
    int             b_size,
    int             active_ovs,
    int             cnv_offset)
{
    int j     = blockIdx.x * blockDim.x + threadIdx.x;  // coefficient index
    int out_k = blockIdx.y;                              // output limb

    if (j >= n || out_k >= active_ovs) return;

    int k_abs = out_k + cnv_offset;
    int j_min = (k_abs >= a_size) ? k_abs - a_size + 1 : 0;
    int j_max = (k_abs + 1 < b_size) ? k_abs + 1 : b_size;

    __int128 acc = 0;

    for (int bj = j_min; bj < j_max; bj++) {
        int aj = k_abs - bj;
        acc += (__int128)a[(int64_t)aj * n + j] * (__int128)b[bj];
    }

    // Store as Big32 (4 × u32 little-endian).
    uint32_t *out = res + ((int64_t)out_k * n + j) * 4;
    out[0] = (uint32_t)acc;
    out[1] = (uint32_t)((unsigned __int128)acc >> 32);
    out[2] = (uint32_t)((unsigned __int128)acc >> 64);
    out[3] = (uint32_t)((unsigned __int128)acc >> 96);
}

extern "C" void ntt120_cnv_by_const_apply(
    void           *stream,
    uint32_t       *res,
    const int64_t  *a,
    const int64_t  *b,
    int             n,
    int             a_size,
    int             b_size,
    int             active_ovs,
    int             cnv_offset)
{
    if (active_ovs == 0 || a_size == 0 || b_size == 0) return;
    dim3 grid((n + kCnvBlock - 1) / kCnvBlock, active_ovs);
    cnv_by_const_apply_kernel<<<grid, kCnvBlock, 0, (cudaStream_t)stream>>>(
        res, a, b, n, a_size, b_size, active_ovs, cnv_offset);
    check_cnv("ntt120_cnv_by_const_apply");
}
