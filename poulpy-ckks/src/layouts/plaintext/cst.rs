use anyhow::Result;
use poulpy_core::layouts::{Base2K, LWEInfos, LWEPlaintext};
use poulpy_hal::layouts::ZnxView;

use crate::{CKKS, CKKSInfos};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CKKSPlaintextCstRnx<F> {
    re: Option<F>,
    im: Option<F>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CKKSPlaintextCstZnx {
    re: Option<Vec<i64>>,
    im: Option<Vec<i64>>,
    meta: CKKS,
}

impl<F> CKKSPlaintextCstRnx<F> {
    pub fn new(re: Option<F>, im: Option<F>) -> Self {
        Self { re, im }
    }

    pub fn re(&self) -> Option<&F> {
        self.re.as_ref()
    }

    pub fn im(&self) -> Option<&F> {
        self.im.as_ref()
    }

    pub fn into_parts(self) -> (Option<F>, Option<F>) {
        (self.re, self.im)
    }
}

impl CKKSPlaintextCstZnx {
    pub fn new(re: Option<Vec<i64>>, im: Option<Vec<i64>>, meta: CKKS) -> Self {
        Self { re, im, meta }
    }

    pub fn re(&self) -> Option<&[i64]> {
        self.re.as_deref()
    }

    pub fn im(&self) -> Option<&[i64]> {
        self.im.as_deref()
    }

    pub fn into_parts(self) -> (Option<Vec<i64>>, Option<Vec<i64>>) {
        (self.re, self.im)
    }
}

pub trait CKKSConstPlaintextConversion {
    const MAX_LOG_DECIMAL_PREC: usize;

    /// Encodes a constant RNX plaintext into its default ZNX representation.
    ///
    /// This uses the natural plaintext precision for `prec`, namely
    /// `prec.min_k(base2k)`. This is the right encoding for operations such as
    /// `mul_const`, where the constant is consumed through the generic
    /// convolution path and does not need to be pre-aligned to a ciphertext's
    /// current remaining homomorphic capacity.
    fn to_znx(&self, base2k: Base2K, prec: CKKS) -> Result<CKKSPlaintextCstZnx>;

    /// Encodes a constant RNX plaintext into a ZNX representation with an
    /// explicit effective torus precision `k`.
    ///
    /// This exists for direct in-ciphertext coefficient injection paths such as
    /// `add_const`. In that case the encoded digits are written straight into
    /// the ciphertext body, so they must already be aligned to the destination
    /// ciphertext precision, typically:
    ///
    /// `k = dst.log_hom_rem() + prec.log_decimal`
    ///
    /// Using `prec.min_k(base2k)` there would place the constant at the wrong
    /// bit position.
    fn to_znx_at_k(&self, base2k: Base2K, k: usize, log_decimal: usize) -> Result<CKKSPlaintextCstZnx>;
}

impl CKKSConstPlaintextConversion for CKKSPlaintextCstRnx<f64> {
    const MAX_LOG_DECIMAL_PREC: usize = 53;

    fn to_znx(&self, base2k: Base2K, prec: CKKS) -> Result<CKKSPlaintextCstZnx> {
        anyhow::ensure!(prec.log_decimal <= Self::MAX_LOG_DECIMAL_PREC);

        let k = prec.min_k(base2k).as_usize();
        let scale = (prec.log_decimal as f64).exp2();
        let re = self
            .re
            .map(|re| encode_const_coeff_i64(base2k, k, (re * scale).round() as i64));
        let im = self
            .im
            .map(|im| encode_const_coeff_i64(base2k, k, (im * scale).round() as i64));

        Ok(CKKSPlaintextCstZnx::new(re, im, prec))
    }

    fn to_znx_at_k(&self, base2k: Base2K, k: usize, log_decimal: usize) -> Result<CKKSPlaintextCstZnx> {
        anyhow::ensure!(log_decimal <= Self::MAX_LOG_DECIMAL_PREC);

        let scale = (log_decimal as f64).exp2();
        let re = self
            .re
            .map(|re| encode_const_coeff_i64(base2k, k, (re * scale).round() as i64));
        let im = self
            .im
            .map(|im| encode_const_coeff_i64(base2k, k, (im * scale).round() as i64));

        Ok(CKKSPlaintextCstZnx::new(
            re,
            im,
            CKKS {
                log_decimal,
                log_hom_rem: k.saturating_sub(log_decimal),
            },
        ))
    }
}

fn encode_const_coeff_i64(base2k: Base2K, k: usize, value: i64) -> Vec<i64> {
    let mut pt = LWEPlaintext::alloc(base2k, k.into());
    pt.encode_i64(value, k.into());
    (0..pt.size()).map(|limb| pt.data().at(0, limb)[0]).collect()
}

impl CKKSInfos for CKKSPlaintextCstZnx {
    fn meta(&self) -> CKKS {
        self.meta
    }

    fn log_decimal(&self) -> usize {
        self.meta.log_decimal
    }

    fn log_hom_rem(&self) -> usize {
        self.meta.log_hom_rem
    }
}
