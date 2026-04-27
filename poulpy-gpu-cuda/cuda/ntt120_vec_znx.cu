// VecZnx coefficient-domain kernels for CudaNtt120Backend.
//
// GPU VecZnx layout (column-major): [cols][size][n] i64.
// Column `col` of a VecZnx with `size` limbs starts at byte offset
//   col * size * n * sizeof(int64_t)
// from the buffer base.
//
// All operations below take pointers that already point to the start of the
// relevant column (i.e. the Rust side computes the column offset before
// calling the extern "C" launcher).

#include <stdint.h>

// ── Block size ───────────────────────────────────────────────────────────────

static constexpr int BLOCK = 256;

// ── Element-wise kernels on flat int64_t arrays ─────────────────────────────

__global__ void vec_znx_add_into_kernel(int64_t* res, const int64_t* a, const int64_t* b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) res[i] = a[i] + b[i];
}

__global__ void vec_znx_add_assign_kernel(int64_t* res, const int64_t* a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) res[i] += a[i];
}

__global__ void vec_znx_sub_into_kernel(int64_t* res, const int64_t* a, const int64_t* b, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) res[i] = a[i] - b[i];
}

// res -= a
__global__ void vec_znx_sub_inplace_kernel(int64_t* res, const int64_t* a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) res[i] -= a[i];
}

// res = a - res
__global__ void vec_znx_sub_negate_inplace_kernel(int64_t* res, const int64_t* a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) res[i] = a[i] - res[i];
}

// res = -a
__global__ void vec_znx_negate_kernel(int64_t* res, const int64_t* a, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) res[i] = -a[i];
}

// res = -res
__global__ void vec_znx_negate_inplace_kernel(int64_t* res, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) res[i] = -res[i];
}

// res += a  (ScalarZnx add: a and res are single polynomials of n i64)
__global__ void vec_znx_add_scalar_assign_kernel(int64_t* res, const int64_t* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] += a[i];
}

// res = a + b  (ScalarZnx add into limb: a is ScalarZnx, b is one limb of VecZnx)
__global__ void vec_znx_add_scalar_into_kernel(int64_t* res, const int64_t* a, const int64_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = a[i] + b[i];
}

// res = a - b  (ScalarZnx sub into limb: a is ScalarZnx, b is one limb of VecZnx)
__global__ void vec_znx_sub_scalar_into_kernel(int64_t* res, const int64_t* a, const int64_t* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = a[i] - b[i];
}

// res[res_limb] -= a  (ScalarZnx sub inplace: a is ScalarZnx, res is one limb)
__global__ void vec_znx_sub_scalar_inplace_kernel(int64_t* res, const int64_t* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] -= a[i];
}

// ── Polynomial-domain kernels ────────────────────────────────────────────────

// Multiply each of `nlimbs` consecutive polynomials (each of degree n) by X^p
// in Z[X]/(X^n + 1).
//
// Both res and a point to `nlimbs * n` consecutive int64_t values.
// For limb l, coefficient j: res[l*n + j] = ±a[l*n + src_j]
__global__ void vec_znx_rotate_kernel(int64_t* res, const int64_t* a, int n, int nlimbs, int64_t p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nlimbs * n) return;
    int l   = tid / n;
    int j   = tid % n;

    // p mod 2n using bitmask (safe for negative p in two's complement).
    int mask_2n = 2 * n - 1;
    int mp_2n   = (int)(p & (int64_t)mask_2n);
    int mp_1n   = mp_2n & (n - 1);
    bool neg_first = (mp_2n < n);

    int src_j, do_negate;
    if (j < mp_1n) {
        src_j     = j + n - mp_1n;
        do_negate = (int)neg_first;
    } else {
        src_j     = j - mp_1n;
        do_negate = (int)(!neg_first);
    }

    int64_t val = a[l * n + src_j];
    res[l * n + j] = do_negate ? -val : val;
}

// res = (X^p - 1) * a, i.e. res = rotate(a, p) - a, applied to all nlimbs.
__global__ void vec_znx_mul_xp_minus_one_kernel(int64_t* res, const int64_t* a, int n, int nlimbs, int64_t p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nlimbs * n) return;
    int l = tid / n;
    int j = tid % n;

    int mask_2n = 2 * n - 1;
    int mp_2n   = (int)(p & (int64_t)mask_2n);
    int mp_1n   = mp_2n & (n - 1);
    bool neg_first = (mp_2n < n);

    int src_j, do_negate;
    if (j < mp_1n) {
        src_j     = j + n - mp_1n;
        do_negate = (int)neg_first;
    } else {
        src_j     = j - mp_1n;
        do_negate = (int)(!neg_first);
    }

    int64_t orig    = a[l * n + j];
    int64_t rotated = a[l * n + src_j];
    res[l * n + j] = (do_negate ? -rotated : rotated) - orig;
}

// Galois automorphism σ_p: f(X) → f(X^p), applied to all nlimbs.
//
// For each input coefficient (limb l, index j): output index k = j*p mod 2n.
// If k < n: res[l*n + k] = a[l*n + j].
// If k >= n: res[l*n + k-n] = -a[l*n + j].
//
// This is a scatter — caller must guarantee gcd(p, 2n) = 1 (bijection).
__global__ void vec_znx_automorphism_kernel(int64_t* res, const int64_t* a, int n, int nlimbs, int64_t p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nlimbs * n) return;
    int l = tid / n;
    int j = tid % n;

    unsigned long long mask_2n = (unsigned long long)(2 * n - 1);
    unsigned long long p_2n    = (unsigned long long)(p & (int64_t)mask_2n);
    unsigned long long k       = ((unsigned long long)j * p_2n) & mask_2n;

    int64_t val = a[l * n + j];
    if (k < (unsigned long long)n) {
        res[l * n + (int)k] = val;
    } else {
        res[l * n + (int)(k - (unsigned long long)n)] = -val;
    }
}

// ── Extern "C" launchers ─────────────────────────────────────────────────────

extern "C" {

void ntt120_vec_znx_add_into(
    void* stream, int64_t* res, const int64_t* a, const int64_t* b, int len)
{
    if (len <= 0) return;
    int grid = (len + BLOCK - 1) / BLOCK;
    vec_znx_add_into_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, b, len);
}

void ntt120_vec_znx_add_assign(
    void* stream, int64_t* res, const int64_t* a, int len)
{
    if (len <= 0) return;
    int grid = (len + BLOCK - 1) / BLOCK;
    vec_znx_add_assign_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, len);
}

void ntt120_vec_znx_sub_into(
    void* stream, int64_t* res, const int64_t* a, const int64_t* b, int len)
{
    if (len <= 0) return;
    int grid = (len + BLOCK - 1) / BLOCK;
    vec_znx_sub_into_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, b, len);
}

void ntt120_vec_znx_sub_inplace(
    void* stream, int64_t* res, const int64_t* a, int len)
{
    if (len <= 0) return;
    int grid = (len + BLOCK - 1) / BLOCK;
    vec_znx_sub_inplace_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, len);
}

void ntt120_vec_znx_sub_negate_inplace(
    void* stream, int64_t* res, const int64_t* a, int len)
{
    if (len <= 0) return;
    int grid = (len + BLOCK - 1) / BLOCK;
    vec_znx_sub_negate_inplace_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, len);
}

void ntt120_vec_znx_negate(
    void* stream, int64_t* res, const int64_t* a, int len)
{
    if (len <= 0) return;
    int grid = (len + BLOCK - 1) / BLOCK;
    vec_znx_negate_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, len);
}

void ntt120_vec_znx_negate_inplace(
    void* stream, int64_t* res, int len)
{
    if (len <= 0) return;
    int grid = (len + BLOCK - 1) / BLOCK;
    vec_znx_negate_inplace_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, len);
}

void ntt120_vec_znx_add_scalar_assign(
    void* stream, int64_t* res, const int64_t* a, int n)
{
    if (n <= 0) return;
    int grid = (n + BLOCK - 1) / BLOCK;
    vec_znx_add_scalar_assign_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, n);
}

void ntt120_vec_znx_add_scalar_into(
    void* stream, int64_t* res, const int64_t* a, const int64_t* b, int n)
{
    if (n <= 0) return;
    int grid = (n + BLOCK - 1) / BLOCK;
    vec_znx_add_scalar_into_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, b, n);
}

void ntt120_vec_znx_sub_scalar_into(
    void* stream, int64_t* res, const int64_t* a, const int64_t* b, int n)
{
    if (n <= 0) return;
    int grid = (n + BLOCK - 1) / BLOCK;
    vec_znx_sub_scalar_into_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, b, n);
}

void ntt120_vec_znx_sub_scalar_inplace(
    void* stream, int64_t* res, const int64_t* a, int n)
{
    if (n <= 0) return;
    int grid = (n + BLOCK - 1) / BLOCK;
    vec_znx_sub_scalar_inplace_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, n);
}

void ntt120_vec_znx_rotate(
    void* stream, int64_t* res, const int64_t* a, int n, int nlimbs, int64_t p)
{
    int total = nlimbs * n;
    if (total <= 0) return;
    int grid = (total + BLOCK - 1) / BLOCK;
    vec_znx_rotate_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, n, nlimbs, p);
}

void ntt120_vec_znx_mul_xp_minus_one(
    void* stream, int64_t* res, const int64_t* a, int n, int nlimbs, int64_t p)
{
    int total = nlimbs * n;
    if (total <= 0) return;
    int grid = (total + BLOCK - 1) / BLOCK;
    vec_znx_mul_xp_minus_one_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, n, nlimbs, p);
}

void ntt120_vec_znx_automorphism(
    void* stream, int64_t* res, const int64_t* a, int n, int nlimbs, int64_t p)
{
    int total = nlimbs * n;
    if (total <= 0) return;
    int grid = (total + BLOCK - 1) / BLOCK;
    vec_znx_automorphism_kernel<<<grid, BLOCK, 0, (cudaStream_t)stream>>>(res, a, n, nlimbs, p);
}

// ── Dot-product reduction ─────────────────────────────────────────────────────
//
// Computes res_ptr[0] -= sum_j(a[j] * b[j]) for j in 0..len.
// Launched as a single block; len <= n which is at most a few thousand.

__global__ static void vec_znx_sub_dot_kernel(int64_t *res_ptr, const int64_t *a, const int64_t *b, int len) {
    __shared__ int64_t sdata[BLOCK];
    int tid = threadIdx.x;
    int64_t acc = 0;
    for (int i = tid; i < len; i += blockDim.x)
        acc += a[i] * b[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *res_ptr -= sdata[0];
}

void ntt120_vec_znx_sub_dot(void *stream, int64_t *res_ptr, const int64_t *a, const int64_t *b, int len) {
    if (len <= 0) return;
    vec_znx_sub_dot_kernel<<<1, BLOCK, 0, (cudaStream_t)stream>>>(res_ptr, a, b, len);
}

} // extern "C"
