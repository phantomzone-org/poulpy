use std::{
    fmt::{self, Debug},
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::layouts::{
    Base2K, Degree, GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextToMut, GLWEPlaintextToRef, GLWEToMut, GLWEToRef, LWEInfos, Rank,
    SetLWEInfos,
};
use poulpy_hal::layouts::{Data, DataMut, DataRef};
use rand_distr::num_traits::Zero;

use crate::{CKKSInfos, CKKSMeta};

#[derive(Debug, Clone)]
/// CKKS vector plaintext in the RNX domain.
///
/// This form stores floating-point coefficients before quantization into torus
/// digits. It is typically produced/consumed by the slot encoder.
pub struct CKKSPlaintextVecRnx<F>(Vec<F>);

/// CKKS plaintext in the ZNX (torus) domain.
pub struct CKKSPlaintextVecZnx<D: Data> {
    /// Raw GLWE plaintext limb storage.
    pub inner: GLWEPlaintext<D>,
    /// Semantic CKKS metadata associated with `inner`.
    pub meta: CKKSMeta,
}

impl<D: Data> CKKSPlaintextVecZnx<D> {
    pub(crate) fn from_inner(inner: GLWEPlaintext<D>, meta: CKKSMeta) -> Self {
        Self { inner, meta }
    }

    pub(crate) fn from_plaintext_with_meta(pt: GLWEPlaintext<D>, meta: CKKSMeta) -> Self {
        Self::from_inner(
            GLWEPlaintext {
                data: pt.data,
                base2k: pt.base2k,
            },
            meta,
        )
    }
}

impl CKKSPlaintextVecZnx<Vec<u8>> {
    /// Allocates an owned ZNX plaintext using the minimum storage implied by
    /// `meta`.
    ///
    /// Inputs:
    /// - `n`: polynomial degree
    /// - `base2k`: limb radix
    /// - `meta`: semantic plaintext precision
    ///
    /// Output:
    /// - a zeroed plaintext buffer sized to `meta.min_k(base2k)`
    pub fn alloc(n: Degree, base2k: Base2K, meta: CKKSMeta) -> Self {
        Self::from_inner(GLWEPlaintext::alloc_with_meta(n, base2k, meta.min_k(base2k)), meta)
    }

    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos + CKKSInfos,
    {
        Self::from_inner(GLWEPlaintext::alloc_from_infos(infos), infos.meta())
    }
}

/// Allocates an owned CKKS vector plaintext in ZNX form.
pub fn alloc_pt_vec_znx(n: Degree, base2k: Base2K, prec: CKKSMeta) -> CKKSPlaintextVecZnx<Vec<u8>> {
    CKKSPlaintextVecZnx::alloc(n, base2k, prec)
}

/// Allocates an owned CKKS plaintext in ZNX form.
///
/// This is the conventional alias used throughout the crate.
pub fn alloc_pt_znx(n: Degree, base2k: Base2K, prec: CKKSMeta) -> CKKSPlaintextVecZnx<Vec<u8>> {
    alloc_pt_vec_znx(n, base2k, prec)
}

impl<D: Data> Deref for CKKSPlaintextVecZnx<D> {
    type Target = GLWEPlaintext<D>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data> DerefMut for CKKSPlaintextVecZnx<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data> LWEInfos for CKKSPlaintextVecZnx<D> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }
}

impl<D: Data> GLWEInfos for CKKSPlaintextVecZnx<D> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: DataRef> GLWEToRef for CKKSPlaintextVecZnx<D> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        GLWEToRef::to_ref(&self.inner)
    }
}

impl<D: DataRef> GLWEPlaintextToRef for CKKSPlaintextVecZnx<D> {
    fn to_ref(&self) -> GLWEPlaintext<&[u8]> {
        GLWEPlaintextToRef::to_ref(&self.inner)
    }
}

impl<D: DataMut> GLWEToMut for CKKSPlaintextVecZnx<D> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        GLWEToMut::to_mut(&mut self.inner)
    }
}

impl<D: DataMut> GLWEPlaintextToMut for CKKSPlaintextVecZnx<D> {
    fn to_mut(&mut self) -> GLWEPlaintext<&mut [u8]> {
        GLWEPlaintextToMut::to_mut(&mut self.inner)
    }
}

impl<D: DataMut> SetLWEInfos for CKKSPlaintextVecZnx<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.inner.set_base2k(base2k);
    }
}

impl<D: DataRef> fmt::Display for CKKSPlaintextVecZnx<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Conversion between RNX floating-point plaintexts and ZNX torus plaintexts.
pub trait CKKSPlaintextConversion {
    /// Maximum supported `log_decimal` for the conversion implementation.
    const MAX_LOG_DECIMAL_PREC: usize;

    /// Quantizes an RNX plaintext into a ZNX plaintext buffer.
    ///
    /// Inputs:
    /// - `self`: source RNX plaintext
    /// - `other`: destination ZNX plaintext whose metadata controls scaling
    ///
    /// Output:
    /// - fills `other` with torus digits representing the quantized plaintext
    ///
    /// Errors:
    /// - returns an error if the destination precision exceeds the supported
    ///   decimal precision or if the plaintext sizes do not match
    fn to_znx(&self, other: &mut CKKSPlaintextVecZnx<impl DataMut>) -> Result<()>;

    /// Decodes a quantized ZNX plaintext back into RNX floating-point form.
    ///
    /// Inputs:
    /// - `self`: destination RNX plaintext
    /// - `other`: source ZNX plaintext
    ///
    /// Output:
    /// - overwrites `self` with the decoded floating-point coefficients
    ///
    /// Errors:
    /// - returns an error if sizes do not match or if the requested semantic
    ///   precision exceeds the supported numeric path
    fn decode_from_znx(&mut self, other: &CKKSPlaintextVecZnx<impl DataRef>) -> Result<()>;
}

impl<F: Zero + Clone> CKKSPlaintextVecRnx<F> {
    /// Allocates an RNX plaintext of degree `n`.
    ///
    /// Errors:
    /// - returns an error if `n` is not a power of two
    pub fn alloc(n: usize) -> Result<Self> {
        anyhow::ensure!(n.is_power_of_two(), "n must be a power of two, got {n}");
        Ok(Self(vec![F::zero(); n]))
    }
}

impl<F> CKKSPlaintextVecRnx<F> {
    /// Returns the polynomial degree.
    pub fn n(&self) -> usize {
        self.0.len()
    }

    /// Returns the underlying coefficient slice.
    pub fn data(&self) -> &[F] {
        &self.0
    }

    /// Returns a mutable view of the underlying coefficient slice.
    pub fn data_mut(&mut self) -> &mut [F] {
        &mut self.0
    }
}

/// NOTE: only `f64` conversion is currently supported.
impl CKKSPlaintextConversion for CKKSPlaintextVecRnx<f64> {
    const MAX_LOG_DECIMAL_PREC: usize = 53;

    /// TODO: use buffers internally instead of allocating.
    fn decode_from_znx(&mut self, other: &CKKSPlaintextVecZnx<impl DataRef>) -> Result<()> {
        let log_decimal = other.log_decimal();
        let log_hom_rem = other.log_hom_rem();
        let n = other.n().as_usize();

        anyhow::ensure!(log_decimal <= Self::MAX_LOG_DECIMAL_PREC);
        anyhow::ensure!(self.0.len() == other.n().as_usize());
        anyhow::ensure!(log_decimal + log_hom_rem <= 127);

        let scale = (-(log_decimal as f64)).exp2();
        let k = other.max_k();
        if log_decimal + log_hom_rem <= 63 {
            let mut data = vec![0i64; n];
            other.decode_vec_i64(&mut data, k);
            self.0.iter_mut().zip(data.iter()).for_each(|(f, i)| *f = (*i as f64) * scale);
        } else {
            let mut data = vec![0i128; n];
            other.decode_vec_i128(&mut data, k);
            self.0.iter_mut().zip(data.iter()).for_each(|(f, i)| *f = (*i as f64) * scale);
        }

        Ok(())
    }

    /// TODO: use buffers internally instead of allocating.
    fn to_znx(&self, other: &mut CKKSPlaintextVecZnx<impl DataMut>) -> Result<()> {
        let log_decimal = other.log_decimal();
        let log_hom_rem = other.log_hom_rem();

        anyhow::ensure!(log_decimal <= Self::MAX_LOG_DECIMAL_PREC);
        anyhow::ensure!(self.0.len() == other.n().as_usize());

        let scale = (log_decimal as f64).exp2();
        let k = other.max_k();
        if log_decimal + log_hom_rem <= 63 {
            let data: Vec<i64> = self.0.iter().map(|&x| (x * scale).round() as i64).collect();
            other.encode_vec_i64(&data, k);
        } else {
            let data: Vec<i128> = self.0.iter().map(|&x| (x * scale).round() as i128).collect();
            other.encode_vec_i128(&data, k);
        }

        Ok(())
    }
}

impl<D: Data> CKKSInfos for CKKSPlaintextVecZnx<D> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }

    fn log_decimal(&self) -> usize {
        self.meta.log_decimal()
    }

    fn log_hom_rem(&self) -> usize {
        self.meta.log_hom_rem()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leveled::operations::pt_znx::CKKSPlaintextZnxOps;
    use poulpy_cpu_ref::NTT120Ref;
    use poulpy_hal::{
        api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeTmpBytes},
        layouts::{Module, ScratchOwned},
    };

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    }

    fn roundtrip_f64(base2k: usize, prec: CKKSMeta) {
        let n = 16usize;
        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        let mut rnx = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
        rnx.0.copy_from_slice(&values);

        let mut znx = alloc_pt_znx(n.into(), base2k.into(), prec);
        rnx.to_znx(&mut znx).unwrap();

        let mut rnx_out = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
        rnx_out.decode_from_znx(&znx).unwrap();

        let err = max_err(&values, &rnx_out.0);
        let bound = (prec.log_decimal as f64).exp2().recip();
        assert!(err < bound, "max_err={err:.2e} exceeds bound={bound:.2e}");
    }

    #[test]
    fn rnx_to_znx_roundtrip_i64_path() {
        roundtrip_f64(
            16,
            CKKSMeta {
                log_hom_rem: 10,
                log_decimal: 40,
            },
        );
    }

    #[test]
    fn rnx_to_znx_roundtrip_i128_path() {
        roundtrip_f64(
            16,
            CKKSMeta {
                log_hom_rem: 30,
                log_decimal: 40,
            },
        );
    }

    #[test]
    fn add_extract_roundtrip() {
        let n = 16usize;
        let prec = CKKSMeta {
            log_hom_rem: 12,
            log_decimal: 40,
        };
        let base2k: usize = 52;

        let module = Module::<NTT120Ref>::new(n as u64);
        let mut scratch = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        let mut rnx = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
        rnx.0.copy_from_slice(&values);
        let mut full_pt = alloc_pt_znx(n.into(), base2k.into(), prec);
        rnx.to_znx(&mut full_pt).unwrap();

        let mut pt_out = alloc_pt_znx(n.into(), base2k.into(), prec);
        module
            .ckks_extract_pt_znx(&mut pt_out, &full_pt.inner, &prec, scratch.borrow())
            .unwrap();

        let mut rnx_out = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
        rnx_out.decode_from_znx(&pt_out).unwrap();

        let err = max_err(&values, &rnx_out.0);
        let bound = (prec.log_decimal as f64).exp2().recip();
        assert!(err < bound, "max_err={err:.2e} exceeds bound={bound:.2e}");
    }
}
