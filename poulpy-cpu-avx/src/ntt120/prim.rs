//! Trait implementations for [`NTT120Avx`](super::NTT120Avx) — primitive NTT-domain operations.
//!
//! Implements all `Ntt*` traits from [`poulpy_hal::reference::ntt120`] for
//! [`NTT120Avx`](super::NTT120Avx).
//!
//! NTT forward/inverse execution uses the AVX2-accelerated kernels from
//! [`super::ntt`].  BBC mat-vec products use the AVX2-accelerated kernels
//! from [`super::mat_vec_avx`].  All other primitives (domain conversion,
//! ring arithmetic) delegate to the scalar reference implementations.

use poulpy_hal::reference::ntt120::{
    NttAdd, NttAddInplace, NttCFromB, NttCopy, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbb, NttMulBbc,
    NttMulBbc1ColX2, NttMulBbc2ColsX2, NttNegate, NttNegateInplace, NttSub, NttSubInplace, NttSubNegateInplace, NttToZnx128,
    NttZero,
    arithmetic::{add_bbb_ref, b_from_znx64_ref, b_to_znx128_ref, c_from_b_ref},
    mat_vec::{BbbMeta, BbcMeta, extract_1blk_from_contiguous_q120b_ref, vec_mat1col_product_bbb_ref},
    ntt::{NttTable, NttTableInv},
    primes::{PrimeSet, Primes30},
};

use super::mat_vec_avx::{vec_mat1col_product_bbc_avx2, vec_mat1col_product_x2_bbc_avx2, vec_mat2cols_product_x2_bbc_avx2};
use super::ntt::{intt_avx2, ntt_avx2};

use super::NTT120Avx;

// Lazy-reduction bound: Q[k] << 33 for each prime.
const Q_SHIFTED: [u64; 4] = [
    (Primes30::Q[0] as u64) << 33,
    (Primes30::Q[1] as u64) << 33,
    (Primes30::Q[2] as u64) << 33,
    (Primes30::Q[3] as u64) << 33,
];

// ──────────────────────────────────────────────────────────────────────────────
// NTT execution — AVX2 butterfly
// ──────────────────────────────────────────────────────────────────────────────

impl NttDFTExecute<NttTable<Primes30>> for NTT120Avx {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTable<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { ntt_avx2::<Primes30>(table, data) }
    }
}

impl NttDFTExecute<NttTableInv<Primes30>> for NTT120Avx {
    #[inline(always)]
    fn ntt_dft_execute(table: &NttTableInv<Primes30>, data: &mut [u64]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { intt_avx2::<Primes30>(table, data) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttFromZnx64 for NTT120Avx {
    #[inline(always)]
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]) {
        b_from_znx64_ref::<Primes30>(a.len(), res, a);
    }
}

impl NttToZnx128 for NTT120Avx {
    #[inline(always)]
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        b_to_znx128_ref::<Primes30>(divisor_is_n, res, a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Addition / subtraction / negation / copy / zero
// ──────────────────────────────────────────────────────────────────────────────

impl NttAdd for NTT120Avx {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        add_bbb_ref::<Primes30>(res.len() / 4, res, a, b);
    }
}

impl NttAddInplace for NTT120Avx {
    #[inline(always)]
    fn ntt_add_inplace(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = res[idx] % q_s + a[idx] % q_s;
            }
        }
    }
}

impl NttSub for NTT120Avx {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = a[idx] % q_s + (q_s - b[idx] % q_s);
            }
        }
    }
}

impl NttSubInplace for NTT120Avx {
    #[inline(always)]
    fn ntt_sub_inplace(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = res[idx] % q_s + (q_s - a[idx] % q_s);
            }
        }
    }
}

impl NttSubNegateInplace for NTT120Avx {
    #[inline(always)]
    fn ntt_sub_negate_inplace(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = a[idx] % q_s + (q_s - res[idx] % q_s);
            }
        }
    }
}

impl NttNegate for NTT120Avx {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = q_s - a[idx] % q_s;
            }
        }
    }
}

impl NttNegateInplace for NTT120Avx {
    #[inline(always)]
    fn ntt_negate_inplace(res: &mut [u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q_s) in Q_SHIFTED.iter().enumerate() {
                let idx = 4 * j + k;
                res[idx] = q_s - res[idx] % q_s;
            }
        }
    }
}

impl NttZero for NTT120Avx {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTT120Avx {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl NttMulBbb for NTT120Avx {
    #[inline(always)]
    fn ntt_mul_bbb(ell: usize, res: &mut [u64], a: &[u64], b: &[u64]) {
        let meta = BbbMeta::<Primes30>::new();
        vec_mat1col_product_bbb_ref::<Primes30>(&meta, ell, res, a, b);
    }
}

impl NttMulBbc for NTT120Avx {
    #[inline(always)]
    fn ntt_mul_bbc(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { vec_mat1col_product_bbc_avx2(meta, ell, res, a, b) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// q120b → q120c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttCFromB for NTT120Avx {
    #[inline(always)]
    fn ntt_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        c_from_b_ref::<Primes30>(n, res, a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// VMP x2-block kernels
// ──────────────────────────────────────────────────────────────────────────────

impl NttMulBbc1ColX2 for NTT120Avx {
    #[inline(always)]
    fn ntt_mul_bbc_1col_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { vec_mat1col_product_x2_bbc_avx2(meta, ell, res, a, b) }
    }
}

impl NttMulBbc2ColsX2 for NTT120Avx {
    #[inline(always)]
    fn ntt_mul_bbc_2cols_x2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        // SAFETY: NTT120Avx::new() verifies AVX2 availability at construction time.
        unsafe { vec_mat2cols_product_x2_bbc_avx2(meta, ell, res, a, b) }
    }
}

impl NttExtract1BlkContiguous for NTT120Avx {
    #[inline(always)]
    fn ntt_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
        extract_1blk_from_contiguous_q120b_ref(n, row_max, blk, dst, src);
    }
}
