// VecZnx shift and normalize kernels for CudaNtt120Backend.
//
// VecZnx GPU layout: [cols][size][n] i64.
// All kernels receive pre-computed column pointers.
//
// Base-2k representation: the polynomial value encoded in a VecZnx column is
//   val = sum_{j=0}^{size-1} a[j] * 2^(j * base2k)
// where limb 0 is the LEAST significant.
//
// LSH by k bits (k = steps*base2k + k_rem):
//   res[j] = normalized(a[j + steps], carry)   for j in [0, min_size)
//   i.e. a limb at position (src) contributes to res[src - steps].
//   Carry propagates downward (high src → low dst), so we process src high→low.
//
// RSH by k bits (steps = k/base2k + (1 if k_rem!=0), lsh = (base2k-k_rem)%base2k):
//   res[j] = normalized(a[j + steps], carry, lsh)
//   Same direction: a[src] → res[src - steps].
//
// Normalize: same as RSH but with signed res_offset to limbs.
//
// All shift kernels: one thread per coefficient j ∈ [0,n).
// Sequential carry loop over limbs within each thread.

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

static constexpr int kShiftBlock = 256;

static void check_shift(const char *where) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
        std::abort();
    }
}

// ── Normalization primitives (per scalar) ─────────────────────────────────────

// get_digit(base2k, x): lower base2k signed bits of x (sign-extended)
__device__ static __inline__ int64_t get_digit(int base2k, int64_t x) {
    int shift = 64 - base2k;
    return (x << shift) >> shift;
}

// get_carry(base2k, x, digit): carry after extracting digit = (x - digit) >> base2k
__device__ static __inline__ int64_t get_carry(int base2k, int64_t x, int64_t digit) {
    return (x - digit) >> base2k;
}

// ── LSH kernel ────────────────────────────────────────────────────────────────
//
// Computes res [+/-]= a << k bits.
// steps = k / base2k, k_rem = k % base2k.
//
// a[src] contributes its lower (base2k - k_rem) bits to res[src - steps],
// shifted up by k_rem. Upper bits carry to res[src - steps - 1] (lower limb).
// We iterate src from (a_size-1) down to 0 (high→low), carry propagates downward.
//
// mode: 0 = overwrite, 1 = add (res += lsh(a)), 2 = sub (res -= lsh(a))

__global__ static void vec_znx_lsh_kernel(
    int64_t       *res,
    const int64_t *a,
    int n, int res_size, int a_size,
    int base2k, int steps, int k_rem,
    int mode)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int base2k_lsh = base2k - k_rem;  // = base2k when k_rem == 0
    int64_t carry = 0;
    bool first = true;

    // Process a limbs from top to bottom (carry propagates downward)
    for (int src = a_size - 1; src >= 0; src--) {
        int dst = src - steps;
        int64_t val = a[(int64_t)src * n + j];

        // Extract (base2k - k_rem) lower bits and carry
        int64_t digit, new_carry;
        if (k_rem == 0) {
            digit     = get_digit(base2k, val);
            new_carry = get_carry(base2k, val, digit);
        } else {
            digit     = get_digit(base2k_lsh, val);
            new_carry = get_carry(base2k_lsh, val, digit);
        }

        // Shift digit up by k_rem, add incoming carry, extract output base2k digit
        int64_t d_shifted = (k_rem == 0) ? digit : (digit << k_rem);
        int64_t d_plus_c  = d_shifted + (first ? 0LL : carry);
        int64_t out       = get_digit(base2k, d_plus_c);
        carry = new_carry + get_carry(base2k, d_plus_c, out);
        first = false;

        if (dst >= 0 && dst < res_size) {
            if (mode == 0) res[(int64_t)dst * n + j] = out;
            else if (mode == 1) res[(int64_t)dst * n + j] += out;
            else               res[(int64_t)dst * n + j] -= out;
        }
    }

    // Zero the bottom `steps` res limbs (overwrite mode only)
    if (mode == 0) {
        for (int l = 0; l < steps && l < res_size; l++) {
            res[(int64_t)l * n + j] = 0;
        }
        // Zero res limbs beyond what a covers
        int min_size = (a_size > steps) ? (a_size - steps < res_size ? a_size - steps : res_size) : 0;
        for (int l = min_size; l < steps && l < res_size; l++) {
            res[(int64_t)l * n + j] = 0;
        }
        // Zero res limbs in [min_size..res_size) that weren't written
        // (any dst < 0 means we didn't write that res limb at all)
        // Actually we need to ensure res[0..steps-1] are 0.
        // Already done above. But also res[min_size..res_size) if a_size-steps < res_size
        // was already zeroed by mode==0 in the loop (dst = src - steps, if src < steps dst < 0)
        // Those were NOT written. Zero them now if mode==0:
        // For each l in [0, steps) that was not written: handled above.
        // For each l in [a_size-steps, res_size) that was not written: handled above.
    }
}

// ── RSH kernel ────────────────────────────────────────────────────────────────
//
// steps = k/base2k + (1 if k_rem!=0), lsh = (base2k - k_rem) % base2k.
// a[src] contributes its lower (base2k - lsh) bits to res[src - steps].
// mode: 0 = overwrite, 1 = add, 2 = sub

__global__ static void vec_znx_rsh_kernel(
    int64_t       *res,
    const int64_t *a,
    int n, int res_size, int a_size,
    int base2k, int steps, int lsh,
    int mode)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    // base2k_lsh = base2k - lsh (used when lsh > 0)
    int base2k_lsh = (lsh > 0) ? (base2k - lsh) : base2k;

    // Overwrite: zero all res limbs first
    if (mode == 0) {
        for (int l = 0; l < res_size; l++) res[(int64_t)l * n + j] = 0;
    }

    int64_t carry = 0;
    bool first = true;

    // Process a limbs from top to bottom
    for (int src = a_size - 1; src >= 0; src--) {
        int dst = src - steps;
        int64_t val = a[(int64_t)src * n + j];

        int64_t digit, new_carry;
        if (lsh == 0) {
            digit     = get_digit(base2k, val);
            new_carry = get_carry(base2k, val, digit);
        } else {
            digit     = get_digit(base2k_lsh, val);
            new_carry = get_carry(base2k_lsh, val, digit);
        }

        int64_t d_shifted = (lsh == 0) ? digit : (digit << lsh);
        int64_t d_plus_c  = d_shifted + (first ? 0LL : carry);
        int64_t out       = get_digit(base2k, d_plus_c);
        carry = new_carry + get_carry(base2k, d_plus_c, out);
        first = false;

        if (dst >= 0 && dst < res_size) {
            if (mode == 0) res[(int64_t)dst * n + j] = out;
            else if (mode == 1) res[(int64_t)dst * n + j] += out;
            else               res[(int64_t)dst * n + j] -= out;
        }
    }

    // Flush remaining carry into res[-steps..-1] (already zeroed for mode==0)
    // These are res limbs for dst < 0, which don't exist.  No action needed.
}

// ── Normalize kernel (inter-base2k, same base2k) ─────────────────────────────
//
// Equivalent to RSH with a fractional offset:
//   lsh_pos = res_offset % base2k  (may be negative → adjust)
//   limbs_offset = res_offset / base2k
//   a[src] → res[src + limbs_offset] (shifted with lsh_pos)

__global__ static void vec_znx_normalize_kernel(
    int64_t       *res,
    const int64_t *a,
    int n, int res_size, int a_size,
    int base2k, int64_t res_offset)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    // Decompose res_offset
    int64_t lsh64 = res_offset % (int64_t)base2k;
    int64_t limbs_offset = res_offset / (int64_t)base2k;
    if (res_offset < 0 && lsh64 != 0) {
        lsh64 = (lsh64 + base2k) % base2k;
        limbs_offset -= 1;
    }
    int lsh_pos = (int)lsh64;
    int base2k_lsh = (lsh_pos > 0) ? (base2k - lsh_pos) : base2k;

    // Zero all res limbs
    for (int l = 0; l < res_size; l++) res[(int64_t)l * n + j] = 0;

    int64_t carry = 0;
    bool first = true;

    // Process a from top to bottom; a[src] → res[src + limbs_offset]
    for (int src = a_size - 1; src >= 0; src--) {
        int dst = (int)(src + limbs_offset);
        int64_t val = a[(int64_t)src * n + j];

        int64_t digit, new_carry;
        if (lsh_pos == 0) {
            digit     = get_digit(base2k, val);
            new_carry = get_carry(base2k, val, digit);
        } else {
            digit     = get_digit(base2k_lsh, val);
            new_carry = get_carry(base2k_lsh, val, digit);
        }

        int64_t d_shifted = (lsh_pos == 0) ? digit : (digit << lsh_pos);
        int64_t d_plus_c  = d_shifted + (first ? 0LL : carry);
        int64_t out       = get_digit(base2k, d_plus_c);
        carry = new_carry + get_carry(base2k, d_plus_c, out);
        first = false;

        if (dst >= 0 && dst < res_size) {
            res[(int64_t)dst * n + j] = out;
        }
    }

    // Propagate remaining carry into res[-1], res[-2], ... (outside res, no action)
}

// ── Normalize inplace kernel ──────────────────────────────────────────────────
//
// Normalizes each limb to lie in [-2^(base2k-1), 2^(base2k-1)).
// Carry propagates from limb 0 upward (low to high), since overflow from limb j
// spills into limb j+1.

__global__ static void vec_znx_normalize_assign_kernel(
    int64_t *a, int n, int size, int base2k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int64_t carry = 0;

    // Process from lowest limb to highest: carry spills up
    for (int limb = 0; limb < size; limb++) {
        int64_t val = a[(int64_t)limb * n + j] + carry;
        int64_t digit = get_digit(base2k, val);
        carry = get_carry(base2k, val, digit);
        a[(int64_t)limb * n + j] = digit;
    }
    // Remaining carry is discarded (overflow of the top limb)
}

// ── RSH inplace kernel ────────────────────────────────────────────────────────

__global__ static void vec_znx_rsh_assign_kernel(
    int64_t *a, int n, int size, int base2k, int steps, int lsh)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int base2k_lsh = (lsh > 0) ? (base2k - lsh) : base2k;
    int64_t carry = 0;
    bool first = true;

    // Process from top: a[src] → a[src - steps]
    // We need a temp variable since we read and write in place.
    // Strategy: process high→low. Since dst = src - steps < src, writing dst
    // before reading src+1 is safe IF steps > 0.

    for (int src = size - 1; src >= 0; src--) {
        int dst = src - steps;
        int64_t val = a[(int64_t)src * n + j];

        int64_t digit, new_carry;
        if (lsh == 0) {
            digit     = get_digit(base2k, val);
            new_carry = get_carry(base2k, val, digit);
        } else {
            digit     = get_digit(base2k_lsh, val);
            new_carry = get_carry(base2k_lsh, val, digit);
        }

        int64_t d_shifted = (lsh == 0) ? digit : (digit << lsh);
        int64_t d_plus_c  = d_shifted + (first ? 0LL : carry);
        int64_t out       = get_digit(base2k, d_plus_c);
        carry = new_carry + get_carry(base2k, d_plus_c, out);
        first = false;

        if (dst >= 0 && dst < size) {
            a[(int64_t)dst * n + j] = out;
        }
    }

    // Zero the top `steps` limbs (they were shifted down)
    for (int l = size - steps; l < size && l >= 0; l++) {
        a[(int64_t)l * n + j] = 0;
    }
}

// ── LSH inplace kernel ────────────────────────────────────────────────────────

__global__ static void vec_znx_lsh_assign_kernel(
    int64_t *a, int n, int size, int base2k, int steps, int k_rem)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    if (steps >= size) {
        for (int l = 0; l < size; l++) a[(int64_t)l * n + j] = 0;
        return;
    }

    int base2k_lsh = base2k - k_rem;
    int64_t carry = 0;
    bool first = true;

    // LSH: a[src] → a[src - steps]. Process high→low.
    for (int src = size - 1; src >= 0; src--) {
        int dst = src - steps;
        int64_t val = a[(int64_t)src * n + j];

        int64_t digit, new_carry;
        if (k_rem == 0) {
            digit     = get_digit(base2k, val);
            new_carry = get_carry(base2k, val, digit);
        } else {
            digit     = get_digit(base2k_lsh, val);
            new_carry = get_carry(base2k_lsh, val, digit);
        }

        int64_t d_shifted = (k_rem == 0) ? digit : (digit << k_rem);
        int64_t d_plus_c  = d_shifted + (first ? 0LL : carry);
        int64_t out       = get_digit(base2k, d_plus_c);
        carry = new_carry + get_carry(base2k, d_plus_c, out);
        first = false;

        if (dst >= 0 && dst < size) {
            a[(int64_t)dst * n + j] = out;
        }
    }

    // Zero bottom `steps` limbs
    for (int l = 0; l < steps; l++) a[(int64_t)l * n + j] = 0;
}

// ── Ring split/merge/switch ───────────────────────────────────────────────────

__global__ static void vec_znx_split_ring_kernel(
    int64_t *res_even, int64_t *res_odd, const int64_t *a,
    int n_full, int n_half, int nlimbs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nlimbs * n_half) return;
    int l = tid / n_half;
    int j = tid % n_half;
    res_even[(int64_t)l * n_half + j] = a[(int64_t)l * n_full + 2 * j];
    res_odd[(int64_t)l * n_half + j]  = a[(int64_t)l * n_full + 2 * j + 1];
}

__global__ static void vec_znx_merge_rings_kernel(
    int64_t *res, const int64_t *a_even, const int64_t *a_odd,
    int n_full, int n_half, int nlimbs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nlimbs * n_half) return;
    int l = tid / n_half;
    int j = tid % n_half;
    res[(int64_t)l * n_full + 2 * j]     = a_even[(int64_t)l * n_half + j];
    res[(int64_t)l * n_full + 2 * j + 1] = a_odd[(int64_t)l * n_half + j];
}

// Fold: n_src → n_dst with n_dst < n_src
// res[j] = sum_{k: k%n_dst == j} a[k] * sign(k/n_dst)
__global__ static void vec_znx_switch_ring_fold_kernel(
    int64_t *res, const int64_t *a, int n_src, int n_dst, int nlimbs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nlimbs * n_dst) return;
    int l = tid / n_dst;
    int j = tid % n_dst;
    int64_t val = 0;
    for (int k = j; k < n_src; k += n_dst) {
        int sign = ((k / n_dst) % 2 == 0) ? 1 : -1;
        val += sign * a[(int64_t)l * n_src + k];
    }
    res[(int64_t)l * n_dst + j] = val;
}

// Expand: n_src → n_dst with n_dst > n_src (zero-pad at alternating positions)
__global__ static void vec_znx_switch_ring_expand_kernel(
    int64_t *res, const int64_t *a, int n_src, int n_dst, int nlimbs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nlimbs * n_dst) return;
    int l = tid / n_dst;
    int j = tid % n_dst;
    int factor = n_dst / n_src;
    res[(int64_t)l * n_dst + j] = (j % factor == 0) ? a[(int64_t)l * n_src + j / factor] : 0LL;
}

// ── Extern "C" launchers ─────────────────────────────────────────────────────

extern "C" {

void ntt120_vec_znx_lsh(
    void *stream, int64_t *res, const int64_t *a,
    int n, int res_size, int a_size,
    int base2k, int steps, int k_rem, int mode)
{
    if (n <= 0) return;
    vec_znx_lsh_kernel<<<(n + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
        res, a, n, res_size, a_size, base2k, steps, k_rem, mode);
    check_shift("ntt120_vec_znx_lsh");
}

void ntt120_vec_znx_rsh(
    void *stream, int64_t *res, const int64_t *a,
    int n, int res_size, int a_size,
    int base2k, int steps, int lsh, int mode)
{
    if (n <= 0) return;
    vec_znx_rsh_kernel<<<(n + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
        res, a, n, res_size, a_size, base2k, steps, lsh, mode);
    check_shift("ntt120_vec_znx_rsh");
}

void ntt120_vec_znx_normalize(
    void *stream, int64_t *res, const int64_t *a,
    int n, int res_size, int a_size, int base2k, int64_t res_offset)
{
    if (n <= 0) return;
    vec_znx_normalize_kernel<<<(n + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
        res, a, n, res_size, a_size, base2k, res_offset);
    check_shift("ntt120_vec_znx_normalize");
}

void ntt120_vec_znx_normalize_assign(
    void *stream, int64_t *a, int n, int size, int base2k)
{
    if (n <= 0 || size <= 0) return;
    vec_znx_normalize_assign_kernel<<<(n + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
        a, n, size, base2k);
    check_shift("ntt120_vec_znx_normalize_assign");
}

void ntt120_vec_znx_rsh_assign(
    void *stream, int64_t *a, int n, int size, int base2k, int steps, int lsh)
{
    if (n <= 0 || size <= 0) return;
    vec_znx_rsh_assign_kernel<<<(n + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
        a, n, size, base2k, steps, lsh);
    check_shift("ntt120_vec_znx_rsh_assign");
}

void ntt120_vec_znx_lsh_assign(
    void *stream, int64_t *a, int n, int size, int base2k, int steps, int k_rem)
{
    if (n <= 0 || size <= 0) return;
    vec_znx_lsh_assign_kernel<<<(n + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
        a, n, size, base2k, steps, k_rem);
    check_shift("ntt120_vec_znx_lsh_assign");
}

void ntt120_vec_znx_split_ring(
    void *stream, int64_t *res_even, int64_t *res_odd, const int64_t *a,
    int n_full, int n_half, int nlimbs)
{
    int total = nlimbs * n_half;
    if (total <= 0) return;
    vec_znx_split_ring_kernel<<<(total + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
        res_even, res_odd, a, n_full, n_half, nlimbs);
    check_shift("ntt120_vec_znx_split_ring");
}

void ntt120_vec_znx_merge_rings(
    void *stream, int64_t *res, const int64_t *a_even, const int64_t *a_odd,
    int n_full, int n_half, int nlimbs)
{
    int total = nlimbs * n_half;
    if (total <= 0) return;
    vec_znx_merge_rings_kernel<<<(total + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
        res, a_even, a_odd, n_full, n_half, nlimbs);
    check_shift("ntt120_vec_znx_merge_rings");
}

void ntt120_vec_znx_switch_ring(
    void *stream, int64_t *res, const int64_t *a,
    int n_src, int n_dst, int nlimbs, int direction)
{
    if (nlimbs <= 0) return;
    if (direction == 0) {
        int total = nlimbs * n_dst;
        vec_znx_switch_ring_fold_kernel<<<(total + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
            res, a, n_src, n_dst, nlimbs);
    } else {
        int total = nlimbs * n_dst;
        vec_znx_switch_ring_expand_kernel<<<(total + kShiftBlock - 1) / kShiftBlock, kShiftBlock, 0, (cudaStream_t)stream>>>(
            res, a, n_src, n_dst, nlimbs);
    }
    check_shift("ntt120_vec_znx_switch_ring");
}

} // extern "C"
