use std::{
    fmt::{self, Debug},
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::layouts::{
    Base2K, Degree, GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextToMut, GLWEPlaintextToRef, GLWEToMut, GLWEToRef, LWEInfos, Rank,
    SetLWEInfos,
};
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef};
use rand_distr::num_traits::Zero;

use crate::{CKKS, CKKSInfos};

#[derive(Debug, Clone)]
pub struct CKKSPlaintextVecRnx<F>(Vec<F>);

/// CKKS plaintext in the ZNX (torus) domain.
pub struct CKKSPlaintextVecZnx<D: Data> {
    inner: GLWEPlaintext<D, CKKS>,
}

impl<D: Data> CKKSPlaintextVecZnx<D> {
    pub(crate) fn from_inner(inner: GLWEPlaintext<D, CKKS>) -> Self {
        Self { inner }
    }

    pub(crate) fn from_plaintext_with_meta(pt: GLWEPlaintext<D, ()>, meta: CKKS) -> Self {
        Self::from_inner(GLWEPlaintext {
            data: pt.data,
            base2k: pt.base2k,
            meta,
        })
    }
}

impl CKKSPlaintextVecZnx<Vec<u8>> {
    pub fn alloc(n: Degree, base2k: Base2K, prec: CKKS) -> Self {
        Self::from_inner(GLWEPlaintext::alloc_with_meta(n, base2k, prec.min_k(base2k), prec))
    }
}

pub fn alloc_pt_vec_znx(n: Degree, base2k: Base2K, prec: CKKS) -> CKKSPlaintextVecZnx<Vec<u8>> {
    CKKSPlaintextVecZnx::alloc(n, base2k, prec)
}

pub fn alloc_pt_znx(n: Degree, base2k: Base2K, prec: CKKS) -> CKKSPlaintextVecZnx<Vec<u8>> {
    alloc_pt_vec_znx(n, base2k, prec)
}

impl<D: Data> Deref for CKKSPlaintextVecZnx<D> {
    type Target = GLWEPlaintext<D, CKKS>;

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

pub trait CKKSPlaintextConversion {
    const MAX_LOG_DECIMAL_PREC: usize;
    fn to_znx<BE>(&self, other: &mut CKKSPlaintextVecZnx<impl DataMut>) -> Result<()>
    where
        BE: Backend;
    fn decode_from_znx<BE>(&mut self, other: &CKKSPlaintextVecZnx<impl DataRef>) -> Result<()>
    where
        BE: Backend;
}

impl<F: Zero + Clone> CKKSPlaintextVecRnx<F> {
    pub fn alloc(n: usize) -> Result<Self> {
        anyhow::ensure!(n.is_power_of_two(), "n must be a power of two, got {n}");
        Ok(Self(vec![F::zero(); n]))
    }
}

impl<F> CKKSPlaintextVecRnx<F> {
    pub fn n(&self) -> usize {
        self.0.len()
    }

    pub fn data(&self) -> &[F] {
        &self.0
    }

    pub fn data_mut(&mut self) -> &mut [F] {
        &mut self.0
    }
}

/// NOTE: only `f64` conversion is currently supported.
impl CKKSPlaintextConversion for CKKSPlaintextVecRnx<f64> {
    const MAX_LOG_DECIMAL_PREC: usize = 53;

    /// TODO: use buffers internally instead of allocating.
    fn decode_from_znx<BE>(&mut self, other: &CKKSPlaintextVecZnx<impl DataRef>) -> Result<()>
    where
        BE: Backend,
    {
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
    fn to_znx<BE>(&self, other: &mut CKKSPlaintextVecZnx<impl DataMut>) -> Result<()>
    where
        BE: Backend,
    {
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

impl<D: Data> CKKSInfos for GLWEPlaintext<D, CKKS> {
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

impl<D: Data> CKKSInfos for CKKSPlaintextVecZnx<D> {
    fn meta(&self) -> CKKS {
        self.inner.meta()
    }

    fn log_decimal(&self) -> usize {
        self.inner.log_decimal()
    }

    fn log_hom_rem(&self) -> usize {
        self.inner.log_hom_rem()
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

    fn roundtrip_f64(base2k: usize, prec: CKKS) {
        let n = 16usize;
        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        let mut rnx = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
        rnx.0.copy_from_slice(&values);

        let mut znx = alloc_pt_znx(n.into(), base2k.into(), prec);
        rnx.to_znx::<NTT120Ref>(&mut znx).unwrap();

        let mut rnx_out = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
        rnx_out.decode_from_znx::<NTT120Ref>(&znx).unwrap();

        let err = max_err(&values, &rnx_out.0);
        let bound = (prec.log_decimal as f64).exp2().recip();
        assert!(err < bound, "max_err={err:.2e} exceeds bound={bound:.2e}");
    }

    #[test]
    fn rnx_to_znx_roundtrip_i64_path() {
        roundtrip_f64(
            16,
            CKKS {
                log_hom_rem: 10,
                log_decimal: 40,
            },
        );
    }

    #[test]
    fn rnx_to_znx_roundtrip_i128_path() {
        roundtrip_f64(
            16,
            CKKS {
                log_hom_rem: 30,
                log_decimal: 40,
            },
        );
    }

    #[test]
    fn add_extract_roundtrip() {
        let n = 16usize;
        let prec = CKKS {
            log_hom_rem: 12,
            log_decimal: 40,
        };
        let base2k: usize = 52;

        let module = Module::<NTT120Ref>::new(n as u64);
        let mut scratch = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        let mut rnx = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
        rnx.0.copy_from_slice(&values);
        let full_prec = CKKS {
            log_hom_rem: 64,
            log_decimal: prec.log_decimal,
        };
        let mut full_pt = alloc_pt_znx(n.into(), base2k.into(), full_prec);
        rnx.to_znx::<NTT120Ref>(&mut full_pt).unwrap();

        let mut pt_out = alloc_pt_znx(n.into(), base2k.into(), prec);
        module.ckks_extract_pt_znx(&mut pt_out, &full_pt, scratch.borrow()).unwrap();

        let mut rnx_out = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
        rnx_out.decode_from_znx::<NTT120Ref>(&pt_out).unwrap();

        let err = max_err(&values, &rnx_out.0);
        let bound = (prec.log_decimal as f64).exp2().recip();
        assert!(err < bound, "max_err={err:.2e} exceeds bound={bound:.2e}");
    }
}
