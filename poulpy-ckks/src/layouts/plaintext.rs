use std::{
    fmt::{self},
    ops::{Deref, DerefMut},
};

use anyhow::Result;
use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWEPlaintext, GLWEPlaintextToBackendMut, GLWEPlaintextToBackendRef, LWEInfos, Rank, SetLWEInfos,
};
use poulpy_hal::layouts::{Backend, Data, HostDataMut, HostDataRef};
use rand_distr::num_traits::{Float, ToPrimitive};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos};

use super::CKKSRnxScalar;

/// CKKS plaintext in the ZNX (torus) domain.
pub struct CKKSPlaintext<D: Data = Vec<u8>> {
    /// Raw GLWE plaintext limb storage.
    pub(crate) inner: GLWEPlaintext<D>,
    /// Semantic CKKS metadata associated with `inner`.
    pub(crate) meta: CKKSMeta,
}

impl<D: Data> CKKSPlaintext<D> {
    pub(crate) fn from_inner(inner: GLWEPlaintext<D>, meta: CKKSMeta) -> Self {
        Self { inner, meta }
    }

    /// Replaces the semantic metadata after checking that the current storage
    /// can represent it.
    ///
    /// This is intended for callers that build plaintext buffers manually.
    /// Normal CKKS operations update metadata themselves.
    pub fn set_meta_checked(&mut self, meta: CKKSMeta) -> Result<()> {
        anyhow::ensure!(
            meta.effective_k() <= self.max_k().as_usize(),
            crate::CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
                max_k: self.max_k().as_usize(),
                log_delta: meta.log_delta(),
                base2k: self.base2k().as_usize(),
                requested_limbs: self.size(),
            }
        );
        self.meta = meta;
        Ok(())
    }
}

impl<BE: Backend, D: Data> GLWEPlaintextToBackendRef<BE> for CKKSPlaintext<D>
where
    GLWEPlaintext<D>: GLWEPlaintextToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEPlaintext<BE::BufRef<'_>> {
        GLWEPlaintextToBackendRef::to_backend_ref(&self.inner)
    }
}

impl<BE: Backend, D: Data> GLWEPlaintextToBackendMut<BE> for CKKSPlaintext<D>
where
    GLWEPlaintext<D>: GLWEPlaintextToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWEPlaintext<BE::BufMut<'_>> {
        GLWEPlaintextToBackendMut::to_backend_mut(&mut self.inner)
    }
}

impl<D: Data> Deref for CKKSPlaintext<D> {
    type Target = GLWEPlaintext<D>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Data> DerefMut for CKKSPlaintext<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Data> LWEInfos for CKKSPlaintext<D> {
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

impl<D: Data> GLWEInfos for CKKSPlaintext<D> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<D: Data> SetCKKSInfos for CKKSPlaintext<D> {
    fn set_meta(&mut self, meta: CKKSMeta) {
        self.meta = meta;
    }
}

impl<D: HostDataMut> SetLWEInfos for CKKSPlaintext<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.inner.set_base2k(base2k);
    }
}

impl<D: HostDataRef> fmt::Display for CKKSPlaintext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<D: Data> CKKSInfos for CKKSPlaintext<D> {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }

    fn log_delta(&self) -> usize {
        self.meta.log_delta()
    }

    fn log_budget(&self) -> usize {
        self.meta.log_budget()
    }
}

pub trait CKKSPlaintextVecHostCodec<F: CKKSRnxScalar>: CKKSInfos + LWEInfos {
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()>;
    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()>;
}

impl<F: CKKSRnxScalar, D: HostDataMut + HostDataRef> CKKSPlaintextVecHostCodec<F> for CKKSPlaintext<D> {
    fn encode_host_floats(&mut self, coeffs: &[F]) -> Result<()> {
        let log_delta = self.log_delta();
        let log_budget = self.log_budget();
        anyhow::ensure!(coeffs.len() == self.n().as_usize());
        anyhow::ensure!(log_delta <= max_log_delta_prec_for::<F>());

        let scale = F::from_usize(log_delta).unwrap().exp2();
        let k = self.max_k();
        if log_delta + log_budget <= 63 {
            let data: Vec<i64> = coeffs.iter().map(|&x| (x * scale).round().to_i64().unwrap()).collect();
            self.encode_vec_i64(&data, k);
        } else {
            let data: Vec<i128> = coeffs.iter().map(|&x| (x * scale).round().to_i128().unwrap()).collect();
            self.encode_vec_i128(&data, k);
        }
        Ok(())
    }

    fn decode_host_floats(&self, coeffs: &mut [F]) -> Result<()> {
        let log_delta = self.log_delta();
        let log_budget = self.log_budget();
        anyhow::ensure!(coeffs.len() == self.n().as_usize());
        anyhow::ensure!(log_delta <= max_log_delta_prec_for::<F>());
        anyhow::ensure!(log_delta + log_budget <= 127);

        let scale = (-F::from_usize(log_delta).unwrap()).exp2();
        let k = self.max_k();
        if log_delta + log_budget <= 63 {
            let mut data = vec![0i64; coeffs.len()];
            self.decode_vec_i64(&mut data, k);
            coeffs
                .iter_mut()
                .zip(data.iter())
                .for_each(|(f, i)| *f = F::from_i64(*i).unwrap() * scale);
        } else {
            let mut data = vec![0i128; coeffs.len()];
            self.decode_vec_i128(&mut data, k);
            coeffs
                .iter_mut()
                .zip(data.iter())
                .for_each(|(f, i)| *f = F::from_i128(*i).unwrap() * scale);
        }
        Ok(())
    }
}

fn max_log_delta_prec_for<F>() -> usize
where
    F: Float + ToPrimitive,
{
    ((-F::epsilon().log2()).round().to_usize().unwrap()) + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encoding::Encoder, layouts::CKKSModuleAlloc, leveled::api::CKKSPlaintextVecOps};
    use poulpy_core::api::ModuleTransfer;
    use poulpy_cpu_ref::NTT120Ref;
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
        layouts::{HostBytesBackend, Module, ScratchOwned},
    };

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn add_extract_roundtrip() {
        let n = 16usize;
        let m = n / 2;
        let prec = CKKSMeta {
            log_budget: 12,
            log_delta: 40,
        };
        let base2k: usize = 52;

        let module = Module::<NTT120Ref>::new(n as u64);
        let host_module = Module::<HostBytesBackend>::new(n as u64);
        let encoder = Encoder::<f64>::new(m).unwrap();
        let mut scratch = ScratchOwned::alloc(module.ckks_extract_pt_tmp_bytes());

        let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64)).collect();
        let im_in: Vec<f64> = (0..m).map(|i| -((i as f64) / (m as f64))).collect();

        let mut full_pt = host_module.ckks_pt_vec_znx_alloc(base2k.into(), prec);
        encoder.encode_reim(&mut full_pt, &re_in, &im_in).unwrap();
        let full_pt_backend = module.upload_glwe_plaintext::<HostBytesBackend>(&full_pt.inner);

        let mut pt_out = module.ckks_pt_vec_znx_alloc(base2k.into(), prec);
        module
            .ckks_extract_pt(&mut pt_out, &full_pt_backend, &prec, &mut scratch.borrow())
            .unwrap();

        let mut re_out = vec![0.0f64; m];
        let mut im_out = vec![0.0f64; m];
        let downloaded = host_module.download_glwe_plaintext::<NTT120Ref>(&pt_out.inner);
        encoder
            .decode_reim(
                &CKKSPlaintext::from_inner(downloaded, pt_out.meta()),
                &mut re_out,
                &mut im_out,
            )
            .unwrap();

        let bound = 2.0 * (prec.log_delta as f64).exp2().recip();
        let err_re = max_err(&re_in, &re_out);
        let err_im = max_err(&im_in, &im_out);
        assert!(err_re < bound, "re err={err_re:.3e}, bound={bound:.3e}, out={re_out:?}");
        assert!(err_im < bound, "im err={err_im:.3e}, bound={bound:.3e}, out={im_out:?}");
    }
}
