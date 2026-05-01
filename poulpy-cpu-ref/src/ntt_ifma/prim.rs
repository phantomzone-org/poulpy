//! Trait implementations for [`NTTIfmaRef`] -- primitive NTT-domain operations.
//!
//! Implements:
//! - All `NttIfma*` traits from [`poulpy_hal::reference::ntt_ifma`] for IFMA-specific
//!   operations (NTT execution, domain conversion, multiply-accumulate).
//! - The `Ntt{Add,AddInplace,Sub,SubInplace,SubNegateInplace,Negate,NegateInplace,
//!   Zero,Copy}` traits from [`poulpy_hal::reference::ntt120`] for DFT-domain
//!   arithmetic reuse. The ntt120 `ntt120_vec_znx_dft_*` generic functions require
//!   these traits, and since the u64 slice layout (4 x u64 per coefficient, 3 active
//!   lanes + 1 padding) is the same, the implementations are identical.

use crate::reference::ntt_ifma::{
    NttIfmaAdd, NttIfmaAddAssign, NttIfmaCFromB, NttIfmaCopy, NttIfmaDFTExecute, NttIfmaExtract1BlkContiguous, NttIfmaFromZnx64,
    NttIfmaMulBbc, NttIfmaMulBbc1ColX2, NttIfmaMulBbc2ColsX2, NttIfmaNegate, NttIfmaNegateAssign, NttIfmaSub, NttIfmaSubAssign,
    NttIfmaSubNegateAssign, NttIfmaToZnx128, NttIfmaZero,
    arithmetic::{b_ifma_from_znx64_ref, b_ifma_to_znx128_ref, c_ifma_from_b_ref},
    mat_vec::{
        BbcIfmaMeta, extract_1blk_from_contiguous_ifma_ref, vec_mat1col_product_bbc_ifma_ref,
        vec_mat1col_product_x2_bbc_ifma_ref, vec_mat2cols_product_x2_bbc_ifma_ref,
    },
    ntt::{NttIfmaTable, NttIfmaTableInv, intt_ifma_ref, ntt_ifma_ref},
    primes::Primes42,
    types::Q_SHIFTED_IFMA,
};

use crate::reference::ntt120::{
    NttAdd, NttAddAssign, NttCopy, NttNegate, NttNegateAssign, NttSub, NttSubAssign, NttSubNegateAssign, NttZero,
};

use crate::NTTIfmaRef;

// ──────────────────────────────────────────────────────────────────────────────
// IFMA NTT execution
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaDFTExecute<NttIfmaTable<Primes42>> for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_dft_execute(table: &NttIfmaTable<Primes42>, data: &mut [u64]) {
        ntt_ifma_ref::<Primes42>(table, data);
    }
}

impl NttIfmaDFTExecute<NttIfmaTableInv<Primes42>> for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_dft_execute(table: &NttIfmaTableInv<Primes42>, data: &mut [u64]) {
        intt_ifma_ref::<Primes42>(table, data);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Domain conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaFromZnx64 for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_from_znx64(res: &mut [u64], a: &[i64]) {
        b_ifma_from_znx64_ref(a.len(), res, a);
    }
}

impl NttIfmaToZnx128 for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]) {
        b_ifma_to_znx128_ref(divisor_is_n, res, a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA-specific addition / subtraction / negation / copy / zero
// ──────────────────────────────────────────────────────────────────────────────

// Helper: conditional subtract of 2q to keep values in [0, 2q).
#[inline(always)]
fn cond_sub(x: u64, q2: u64) -> u64 {
    if x >= q2 { x - q2 } else { x }
}

impl NttIfmaAdd for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                // a, b ∈ [0, 2q), sum ∈ [0, 4q), one cond_sub → [0, 2q)
                res[idx] = cond_sub(a[idx] + b[idx], q2);
            }
            res[4 * j + 3] = 0;
        }
    }
}

impl NttIfmaAddAssign for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_add_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(res[idx] + a[idx], q2);
            }
        }
    }
}

impl NttIfmaSub for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                // a + 2q - b ∈ [1, 4q), one cond_sub → [0, 2q)
                res[idx] = cond_sub(a[idx] + q2 - b[idx], q2);
            }
            res[4 * j + 3] = 0;
        }
    }
}

impl NttIfmaSubAssign for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_sub_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(res[idx] + q2 - a[idx], q2);
            }
        }
    }
}

impl NttIfmaSubNegateAssign for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(a[idx] + q2 - res[idx], q2);
            }
        }
    }
}

impl NttIfmaNegate for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_negate(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                // 2q - a ∈ (0, 2q] → cond_sub handles the 2q case
                res[idx] = cond_sub(q2 - a[idx], q2);
            }
            res[4 * j + 3] = 0;
        }
    }
}

impl NttIfmaNegateAssign for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_negate_assign(res: &mut [u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(q2 - res[idx], q2);
            }
        }
    }
}

impl NttIfmaZero for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttIfmaCopy for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IFMA multiply-accumulate
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaMulBbc for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_mul_bbc(meta: &BbcIfmaMeta<Primes42>, ell: usize, res: &mut [u64], ntt_coeff: &[u32], prepared: &[u32]) {
        vec_mat1col_product_bbc_ifma_ref(meta, ell, res, ntt_coeff, prepared);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// b -> c conversion
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaCFromB for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_c_from_b(n: usize, res: &mut [u32], a: &[u64]) {
        c_ifma_from_b_ref(n, res, a);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// VMP x2-block kernels
// ──────────────────────────────────────────────────────────────────────────────

impl NttIfmaMulBbc1ColX2 for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_mul_bbc_1col_x2(meta: &BbcIfmaMeta<Primes42>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        vec_mat1col_product_x2_bbc_ifma_ref(meta, ell, res, a, b);
    }
}

impl NttIfmaMulBbc2ColsX2 for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_mul_bbc_2cols_x2(meta: &BbcIfmaMeta<Primes42>, ell: usize, res: &mut [u64], a: &[u32], b: &[u32]) {
        vec_mat2cols_product_x2_bbc_ifma_ref(meta, ell, res, a, b);
    }
}

impl NttIfmaExtract1BlkContiguous for NTTIfmaRef {
    #[inline(always)]
    fn ntt_ifma_extract_1blk_contiguous(n: usize, row_max: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
        extract_1blk_from_contiguous_ifma_ref(n, row_max, blk, dst, src);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NTT120 Ntt* traits (DFT-domain arithmetic reuse)
//
// The ntt120 generic functions (ntt120_vec_znx_dft_add, etc.) require these
// traits. Since the 3-prime IFMA layout uses the same 4 x u64 per coefficient
// representation (with lane 3 as padding), the implementation is identical
// to the NTT120 version but uses Q_SHIFTED_IFMA for the 3 active lanes.
// ──────────────────────────────────────────────────────────────────────────────

impl NttAdd for NTTIfmaRef {
    #[inline(always)]
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(a[idx] + b[idx], q2);
            }
            res[4 * j + 3] = 0;
        }
    }
}

impl NttAddAssign for NTTIfmaRef {
    #[inline(always)]
    fn ntt_add_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(res[idx] + a[idx], q2);
            }
        }
    }
}

impl NttSub for NTTIfmaRef {
    #[inline(always)]
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(a[idx] + q2 - b[idx], q2);
            }
            res[4 * j + 3] = 0;
        }
    }
}

impl NttSubAssign for NTTIfmaRef {
    #[inline(always)]
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(res[idx] + q2 - a[idx], q2);
            }
        }
    }
}

impl NttSubNegateAssign for NTTIfmaRef {
    #[inline(always)]
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(a[idx] + q2 - res[idx], q2);
            }
        }
    }
}

impl NttNegate for NTTIfmaRef {
    #[inline(always)]
    fn ntt_negate(res: &mut [u64], a: &[u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(q2 - a[idx], q2);
            }
            res[4 * j + 3] = 0;
        }
    }
}

impl NttNegateAssign for NTTIfmaRef {
    #[inline(always)]
    fn ntt_negate_assign(res: &mut [u64]) {
        let n = res.len() / 4;
        for j in 0..n {
            for (k, &q2) in Q_SHIFTED_IFMA.iter().enumerate().take(3) {
                let idx = 4 * j + k;
                res[idx] = cond_sub(q2 - res[idx], q2);
            }
        }
    }
}

impl NttZero for NTTIfmaRef {
    #[inline(always)]
    fn ntt_zero(res: &mut [u64]) {
        res.fill(0);
    }
}

impl NttCopy for NTTIfmaRef {
    #[inline(always)]
    fn ntt_copy(res: &mut [u64], a: &[u64]) {
        res.copy_from_slice(a);
    }
}
