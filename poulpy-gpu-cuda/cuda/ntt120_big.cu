// VecZnxBig element-wise arithmetic kernels for CudaNtt120Backend.
//
// VecZnxBig layout: [cols][size][n] Big32, where Big32 = [u32; 4] little-endian i128.
// Within a column, element at limb j, coeff i starts at byte
//   (j * n + i) * 4 * sizeof(u32) from the column base pointer.
//
// All kernels below receive pre-computed column pointers and operate over
// `len` Big32 elements (i.e., len = size * n elements in column-major).

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

static constexpr int kBigBlock = 256;

static void check_big(const char *where) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
        std::abort();
    }
}

// ── Big32 helpers ─────────────────────────────────────────────────────────────

__device__ static __inline__ __int128 load_i128(const uint32_t *p) {
    __uint128_t bits =
        ((__uint128_t)p[0])
        | ((__uint128_t)p[1] << 32)
        | ((__uint128_t)p[2] << 64)
        | ((__uint128_t)p[3] << 96);
    return (__int128)bits;
}

__device__ static __inline__ void store_i128(uint32_t *out, __int128 val) {
    out[0] = (uint32_t)val;
    out[1] = (uint32_t)(val >> 32);
    out[2] = (uint32_t)(val >> 64);
    out[3] = (uint32_t)(val >> 96);
}

// ── Element-wise kernels ──────────────────────────────────────────────────────

// res[i] = a[i] + b[i]   (i128 wrapping add)
__global__ static void big_add_into_kernel(uint32_t *res, const uint32_t *a, const uint32_t *b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(a + 4 * i) + load_i128(b + 4 * i));
}

// res[i] += a[i]
__global__ static void big_add_assign_kernel(uint32_t *res, const uint32_t *a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(res + 4 * i) + load_i128(a + 4 * i));
}

// res[i] = a[i] + (int128)b[i]   (a: Big32, b: i64)
__global__ static void big_add_small_into_kernel(uint32_t *res, const uint32_t *a, const int64_t *b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(a + 4 * i) + (__int128)b[i]);
}

// res[i] += (int128)a[i]   (a: i64)
__global__ static void big_add_small_assign_kernel(uint32_t *res, const int64_t *a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(res + 4 * i) + (__int128)a[i]);
}

// res[i] = a[i] - b[i]
__global__ static void big_sub_into_kernel(uint32_t *res, const uint32_t *a, const uint32_t *b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(a + 4 * i) - load_i128(b + 4 * i));
}

// res[i] -= a[i]
__global__ static void big_sub_assign_kernel(uint32_t *res, const uint32_t *a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(res + 4 * i) - load_i128(a + 4 * i));
}

// res[i] = a[i] - res[i]
__global__ static void big_sub_negate_assign_kernel(uint32_t *res, const uint32_t *a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(a + 4 * i) - load_i128(res + 4 * i));
}

// res[i] = -a[i]
__global__ static void big_negate_into_kernel(uint32_t *res, const uint32_t *a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, -load_i128(a + 4 * i));
}

// res[i] = -res[i]
__global__ static void big_negate_assign_kernel(uint32_t *res, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, -load_i128(res + 4 * i));
}

// res[i] = (int128)a[i] - b[i]   (a: i64, b: Big32)
__global__ static void big_sub_small_a_kernel(uint32_t *res, const int64_t *a, const uint32_t *b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, (__int128)a[i] - load_i128(b + 4 * i));
}

// res[i] = a[i] - (int128)b[i]   (a: Big32, b: i64)
__global__ static void big_sub_small_b_kernel(uint32_t *res, const uint32_t *a, const int64_t *b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(a + 4 * i) - (__int128)b[i]);
}

// res[i] -= (int128)a[i]   (a: i64)
__global__ static void big_sub_small_assign_kernel(uint32_t *res, const int64_t *a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, load_i128(res + 4 * i) - (__int128)a[i]);
}

// res[i] = (int128)a[i] - res[i]   (a: i64)
__global__ static void big_sub_small_negate_assign_kernel(uint32_t *res, const int64_t *a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, (__int128)a[i] - load_i128(res + 4 * i));
}

// res[i] = (int128)a[i]   (sign-extend i64 → i128)
__global__ static void big_from_small_kernel(uint32_t *res, const int64_t *a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    store_i128(res + 4 * i, (__int128)a[i]);
}

// ── Galois automorphism σ_p on Big32 ─────────────────────────────────────────
//
// For each input (limb l, coeff j): dst index k = j*p mod 2n.
//   if k < n:  res[l*n + k]   = a[l*n + j]
//   if k >= n: res[l*n + k-n] = -a[l*n + j]
//
// Caller must guarantee gcd(p, 2n) = 1.

__global__ static void big_automorphism_kernel(
    uint32_t *res, const uint32_t *a, int n, int nlimbs, int64_t p)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nlimbs * n) return;
    int l = tid / n;
    int j = tid % n;

    unsigned long long mask_2n = (unsigned long long)(2 * n - 1);
    unsigned long long p_2n    = (unsigned long long)(p & (int64_t)mask_2n);
    unsigned long long k       = ((unsigned long long)j * p_2n) & mask_2n;

    __int128 val = load_i128(a + (l * n + j) * 4);
    if (k < (unsigned long long)n) {
        store_i128(res + (l * n + (int)k) * 4, val);
    } else {
        store_i128(res + (l * n + (int)(k - (unsigned long long)n)) * 4, -val);
    }
}

// ── Extern "C" launchers ─────────────────────────────────────────────────────

extern "C" {

void ntt120_big_add_into(void *stream, uint32_t *res, const uint32_t *a, const uint32_t *b, int len) {
    if (len <= 0) return;
    big_add_into_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, b, len);
    check_big("ntt120_big_add_into");
}

void ntt120_big_add_assign(void *stream, uint32_t *res, const uint32_t *a, int len) {
    if (len <= 0) return;
    big_add_assign_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, len);
    check_big("ntt120_big_add_assign");
}

void ntt120_big_add_small_into(void *stream, uint32_t *res, const uint32_t *a, const int64_t *b, int len) {
    if (len <= 0) return;
    big_add_small_into_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, b, len);
    check_big("ntt120_big_add_small_into");
}

void ntt120_big_add_small_assign(void *stream, uint32_t *res, const int64_t *a, int len) {
    if (len <= 0) return;
    big_add_small_assign_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, len);
    check_big("ntt120_big_add_small_assign");
}

void ntt120_big_sub_into(void *stream, uint32_t *res, const uint32_t *a, const uint32_t *b, int len) {
    if (len <= 0) return;
    big_sub_into_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, b, len);
    check_big("ntt120_big_sub_into");
}

void ntt120_big_sub_assign(void *stream, uint32_t *res, const uint32_t *a, int len) {
    if (len <= 0) return;
    big_sub_assign_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, len);
    check_big("ntt120_big_sub_assign");
}

void ntt120_big_sub_negate_assign(void *stream, uint32_t *res, const uint32_t *a, int len) {
    if (len <= 0) return;
    big_sub_negate_assign_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, len);
    check_big("ntt120_big_sub_negate_assign");
}

void ntt120_big_negate_into(void *stream, uint32_t *res, const uint32_t *a, int len) {
    if (len <= 0) return;
    big_negate_into_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, len);
    check_big("ntt120_big_negate_into");
}

void ntt120_big_negate_assign(void *stream, uint32_t *res, int len) {
    if (len <= 0) return;
    big_negate_assign_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, len);
    check_big("ntt120_big_negate_assign");
}

void ntt120_big_sub_small_a(void *stream, uint32_t *res, const int64_t *a, const uint32_t *b, int len) {
    if (len <= 0) return;
    big_sub_small_a_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, b, len);
    check_big("ntt120_big_sub_small_a");
}

void ntt120_big_sub_small_b(void *stream, uint32_t *res, const uint32_t *a, const int64_t *b, int len) {
    if (len <= 0) return;
    big_sub_small_b_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, b, len);
    check_big("ntt120_big_sub_small_b");
}

void ntt120_big_sub_small_assign(void *stream, uint32_t *res, const int64_t *a, int len) {
    if (len <= 0) return;
    big_sub_small_assign_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, len);
    check_big("ntt120_big_sub_small_assign");
}

void ntt120_big_sub_small_negate_assign(void *stream, uint32_t *res, const int64_t *a, int len) {
    if (len <= 0) return;
    big_sub_small_negate_assign_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, len);
    check_big("ntt120_big_sub_small_negate_assign");
}

void ntt120_big_from_small(void *stream, uint32_t *res, const int64_t *a, int len) {
    if (len <= 0) return;
    big_from_small_kernel<<<(len + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(res, a, len);
    check_big("ntt120_big_from_small");
}

void ntt120_big_automorphism(void *stream, uint32_t *res, const uint32_t *a, int n, int nlimbs, int64_t p) {
    int total = nlimbs * n;
    if (total <= 0) return;
    big_automorphism_kernel<<<(total + kBigBlock - 1) / kBigBlock, kBigBlock, 0, (cudaStream_t)stream>>>(
        res, a, n, nlimbs, p);
    check_big("ntt120_big_automorphism");
}

} // extern "C"
