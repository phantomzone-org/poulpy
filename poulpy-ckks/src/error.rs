use std::{error::Error, fmt};

use anyhow::Result;

/// CKKS composition and alignment errors returned by high-level operations.
///
/// These errors describe semantic failures such as insufficient precision,
/// incompatible plaintext/ciphertext layouts, or metadata that cannot fit in
/// the requested output storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CKKSCompositionError {
    /// Shrinking a ciphertext buffer would drop required semantic bits.
    LimbReallocationShrinksBelowMetadata {
        max_k: usize,
        log_decimal: usize,
        base2k: usize,
        requested_limbs: usize,
    },
    /// An operation requires more `log_hom_rem` than is still available.
    InsufficientHomomorphicCapacity {
        op: &'static str,
        available_log_hom_rem: usize,
        required_bits: usize,
    },
    /// A plaintext and ciphertext use different limb radices.
    PlaintextBase2KMismatch {
        op: &'static str,
        ct_base2k: usize,
        pt_base2k: usize,
    },
    /// A plaintext cannot be aligned into the requested destination precision.
    PlaintextAlignmentImpossible {
        op: &'static str,
        ct_log_hom_rem: usize,
        pt_log_decimal: usize,
        pt_max_k: usize,
    },
    /// A multiplication would consume more semantic precision than available.
    MultiplicationPrecisionUnderflow {
        op: &'static str,
        lhs_log_hom_rem: usize,
        rhs_log_hom_rem: usize,
        lhs_log_decimal: usize,
        rhs_log_decimal: usize,
    },
}

impl fmt::Display for CKKSCompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimbReallocationShrinksBelowMetadata {
                max_k,
                log_decimal,
                base2k,
                requested_limbs,
            } => write!(
                f,
                "cannot reallocate to {requested_limbs} limbs: requested capacity is {} bits but ciphertext needs at least {} bits to preserve metadata (max_k={max_k}, log_decimal={log_decimal}, base2k={base2k})",
                requested_limbs * base2k,
                max_k.saturating_sub(*log_decimal)
            ),
            Self::InsufficientHomomorphicCapacity {
                op,
                available_log_hom_rem,
                required_bits,
            } => write!(
                f,
                "{op} cannot consume {required_bits} bits of log_hom_rem: only {available_log_hom_rem} bits remain"
            ),
            Self::PlaintextBase2KMismatch {
                op,
                ct_base2k,
                pt_base2k,
            } => write!(
                f,
                "{op} requires matching base2k values, got ciphertext base2k={ct_base2k} and plaintext base2k={pt_base2k}"
            ),
            Self::PlaintextAlignmentImpossible {
                op,
                ct_log_hom_rem,
                pt_log_decimal,
                pt_max_k,
            } => write!(
                f,
                "{op} cannot align plaintext with ciphertext: ct.log_hom_rem + pt.log_decimal = {} but pt.max_k = {pt_max_k} (ct.log_hom_rem={ct_log_hom_rem}, pt.log_decimal={pt_log_decimal})",
                ct_log_hom_rem + pt_log_decimal
            ),
            Self::MultiplicationPrecisionUnderflow {
                op,
                lhs_log_hom_rem,
                rhs_log_hom_rem,
                lhs_log_decimal,
                rhs_log_decimal,
            } => write!(
                f,
                "{op} cannot compose inputs: min(log_hom_rem)={} is smaller than min(log_decimal)={} (lhs: log_hom_rem={lhs_log_hom_rem}, log_decimal={lhs_log_decimal}; rhs: log_hom_rem={rhs_log_hom_rem}, log_decimal={rhs_log_decimal})",
                lhs_log_hom_rem.min(rhs_log_hom_rem),
                lhs_log_decimal.min(rhs_log_decimal)
            ),
        }
    }
}

impl Error for CKKSCompositionError {}

pub(crate) fn ensure_limb_count_fits(max_k: usize, log_decimal: usize, base2k: usize, requested_limbs: usize) -> Result<()> {
    if max_k.saturating_sub(log_decimal) < requested_limbs * base2k {
        return Err(CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
            max_k,
            log_decimal,
            base2k,
            requested_limbs,
        }
        .into());
    }
    Ok(())
}

pub(crate) fn checked_log_hom_rem_sub(op: &'static str, available_log_hom_rem: usize, required_bits: usize) -> Result<usize> {
    available_log_hom_rem.checked_sub(required_bits).ok_or_else(|| {
        CKKSCompositionError::InsufficientHomomorphicCapacity {
            op,
            available_log_hom_rem,
            required_bits,
        }
        .into()
    })
}

pub(crate) fn ensure_base2k_match(op: &'static str, ct_base2k: usize, pt_base2k: usize) -> Result<()> {
    if ct_base2k != pt_base2k {
        return Err(CKKSCompositionError::PlaintextBase2KMismatch {
            op,
            ct_base2k,
            pt_base2k,
        }
        .into());
    }
    Ok(())
}

pub(crate) fn ensure_plaintext_alignment(
    op: &'static str,
    ct_log_hom_rem: usize,
    pt_log_decimal: usize,
    pt_max_k: usize,
) -> Result<usize> {
    let available = ct_log_hom_rem + pt_log_decimal;
    if available < pt_max_k {
        return Err(CKKSCompositionError::PlaintextAlignmentImpossible {
            op,
            ct_log_hom_rem,
            pt_log_decimal,
            pt_max_k,
        }
        .into());
    }
    Ok(available - pt_max_k)
}

pub(crate) fn checked_mul_ct_log_hom_rem(
    op: &'static str,
    lhs_log_hom_rem: usize,
    rhs_log_hom_rem: usize,
    lhs_log_decimal: usize,
    rhs_log_decimal: usize,
) -> Result<usize> {
    lhs_log_hom_rem
        .min(rhs_log_hom_rem)
        .checked_sub(lhs_log_decimal.min(rhs_log_decimal))
        .ok_or_else(|| {
            CKKSCompositionError::MultiplicationPrecisionUnderflow {
                op,
                lhs_log_hom_rem,
                rhs_log_hom_rem,
                lhs_log_decimal,
                rhs_log_decimal,
            }
            .into()
        })
}

pub(crate) fn checked_mul_pt_log_hom_rem(
    op: &'static str,
    lhs_log_hom_rem: usize,
    rhs_log_hom_rem: usize,
    lhs_log_decimal: usize,
    rhs_log_decimal: usize,
) -> Result<usize> {
    lhs_log_hom_rem.checked_sub(rhs_log_decimal).ok_or_else(|| {
        CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op,
            lhs_log_hom_rem,
            rhs_log_hom_rem,
            lhs_log_decimal,
            rhs_log_decimal,
        }
        .into()
    })
}
