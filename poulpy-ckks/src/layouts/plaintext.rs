use std::fmt::Debug;

use anyhow::Result;
use poulpy_core::layouts::{Base2K, Degree, GLWEPlaintext, LWEInfos};
use poulpy_cpu_ref::fft64::FFTModuleHandle;
use poulpy_hal::{
    GALOISGENERATOR,
    api::ModuleNew,
    layouts::{Backend, Data, DataMut, DataRef, Module},
};
use rand_distr::num_traits::{Float, FloatConst, NumCast, Zero};

use crate::{CKKS, CKKSInfos, ensure_log_decimal_fits, ensure_log_hom_rem_fits};

#[derive(Debug, Clone)]
pub struct CKKSPlaintextRnx<F>(Vec<F>);

/// CKKS plaintext in the ZNX (torus) domain.
pub type CKKSPlaintextZnx<D> = GLWEPlaintext<D, CKKS>;

pub fn alloc_pt_znx(n: Degree, base2k: Base2K, prec: CKKS) -> CKKSPlaintextZnx<Vec<u8>> {
    GLWEPlaintext::alloc_with_meta(n, base2k, prec.min_k(base2k), prec)
}

pub(crate) fn attach_meta<D: Data>(pt: GLWEPlaintext<D, ()>, meta: CKKS) -> GLWEPlaintext<D, CKKS> {
    GLWEPlaintext {
        data: pt.data,
        base2k: pt.base2k,
        meta,
    }
}

pub trait CKKSPlaintextConversion {
    const MAX_LOG_DECIMAL_PREC: usize;
    fn to_znx<BE>(&self, other: &mut CKKSPlaintextZnx<impl DataMut>) -> Result<()>
    where
        BE: Backend;
    fn decode_from_znx<BE>(&mut self, other: &CKKSPlaintextZnx<impl DataRef>) -> Result<()>
    where
        BE: Backend;
}

pub struct Encoder<BE: Backend> {
    module: Module<BE>,
    slot_map: Vec<usize>,
}

impl<BE> Encoder<BE>
where
    BE: Backend,
    BE::ScalarPrep: Float + FloatConst + Debug,
    Module<BE>: ModuleNew<BE> + FFTModuleHandle<BE::ScalarPrep>,
{
    pub fn new(m: usize) -> Result<Self> {
        anyhow::ensure!(m.is_power_of_two(), "m must be a power of two, got {m}");
        anyhow::ensure!(m > 0, "m must be > 0, got {m}");
        // Returns the FFT-internal position for each user slot.
        //
        // User slot `k` has Galois exponent `5^k mod 2N`.  The FFT position
        // is `bit_reverse((5^k mod 2N − 1) / 2, log₂N)`.  No conjugate
        // folding is needed because `5 ≡ 1 mod 4` guarantees the result
        // stays in `[0, m)`.
        let two_n = 4 * m;
        let log_n = (2 * m).trailing_zeros();
        let mut slot_map = Vec::with_capacity(m);
        let mut exp = 1usize;
        for _ in 0..m {
            slot_map.push(((exp - 1) / 2).reverse_bits() >> (usize::BITS - log_n));
            exp = (exp * GALOISGENERATOR as usize) & (two_n - 1);
        }
        Ok(Self {
            module: Module::<BE>::new((m << 1) as u64),
            slot_map,
        })
    }

    pub fn encode_reim(
        &self,
        pt: &mut CKKSPlaintextRnx<BE::ScalarPrep>,
        re: &[BE::ScalarPrep],
        im: &[BE::ScalarPrep],
    ) -> Result<()> {
        let n = pt.n();
        let m = n / 2;

        let table = self.module.get_ifft_table();

        anyhow::ensure!(table.m() == m);
        anyhow::ensure!(re.len() == m);
        anyhow::ensure!(im.len() == m);

        pt.0.fill(BE::ScalarPrep::zero());
        for k in 0..m {
            let idx = self.slot_map[k];
            pt.0[idx] = re[k];
            pt.0[m + idx] = im[k];
        }

        table.execute(&mut pt.0);

        // Normalize by 1/m, matching vec_znx_idft_apply's divisor=m convention.
        let inv_m = <BE::ScalarPrep as NumCast>::from(m).unwrap().recip();
        pt.0.iter_mut().for_each(|x| *x = *x * inv_m);

        Ok(())
    }

    pub fn decode_reim(
        &self,
        pt: &CKKSPlaintextRnx<BE::ScalarPrep>,
        re: &mut [BE::ScalarPrep],
        im: &mut [BE::ScalarPrep],
    ) -> Result<()> {
        let n = pt.n();
        let m = n / 2;

        let table = self.module.get_fft_table();

        anyhow::ensure!(table.m() == m);
        anyhow::ensure!(re.len() == m);
        anyhow::ensure!(im.len() == m);

        let mut reim_tmp = vec![BE::ScalarPrep::zero(); n];
        reim_tmp.copy_from_slice(&pt.0);

        table.execute(&mut reim_tmp);

        for k in 0..m {
            let idx = self.slot_map[k];
            re[k] = reim_tmp[idx];
            im[k] = reim_tmp[m + idx];
        }

        Ok(())
    }
}

impl<F: Zero + Clone> CKKSPlaintextRnx<F> {
    pub fn alloc(n: usize) -> Result<Self> {
        anyhow::ensure!(n.is_power_of_two(), "n must be a power of two, got {n}");
        Ok(Self(vec![F::zero(); n]))
    }
}

impl<F> CKKSPlaintextRnx<F> {
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
impl CKKSPlaintextConversion for CKKSPlaintextRnx<f64> {
    const MAX_LOG_DECIMAL_PREC: usize = 53;

    /// TODO: use buffers internally instead of allocating.
    fn decode_from_znx<BE>(&mut self, other: &CKKSPlaintextZnx<impl DataRef>) -> Result<()>
    where
        BE: Backend,
    {
        let log_decimal = other.meta.log_decimal;
        let log_hom_rem = other.meta.log_hom_rem;
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
    fn to_znx<BE>(&self, other: &mut CKKSPlaintextZnx<impl DataMut>) -> Result<()>
    where
        BE: Backend,
    {
        let log_decimal = other.meta.log_decimal;
        let log_hom_rem = other.meta.log_hom_rem;

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

    fn set_log_decimal(&mut self, log_decimal: usize) -> Result<()> {
        ensure_log_decimal_fits(self.max_k().as_usize(), self.log_hom_rem(), log_decimal)?;
        self.meta.log_decimal = log_decimal;
        Ok(())
    }

    fn set_log_hom_rem(&mut self, log_hom_rem: usize) -> Result<()> {
        ensure_log_hom_rem_fits(self.max_k().as_usize(), self.log_decimal(), log_hom_rem)?;
        self.meta.log_hom_rem = log_hom_rem;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leveled::operations::pt_znx::CKKSPlaintextZnxOps;
    use poulpy_cpu_ref::{FFT64Ref, NTT120Ref};
    use poulpy_hal::{
        api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeTmpBytes},
        layouts::{Module, ScratchOwned},
    };

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    }

    fn roundtrip_f64(base2k: usize, prec: CKKS) {
        let n = 16usize;
        // Values spread across [-1, 1] to exercise both signs and fractional precision.
        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        let mut rnx = CKKSPlaintextRnx::<f64>::alloc(n).unwrap();
        rnx.0.copy_from_slice(&values);

        let mut znx = alloc_pt_znx(n.into(), base2k.into(), prec);
        rnx.to_znx::<NTT120Ref>(&mut znx).unwrap();

        let mut rnx_out = CKKSPlaintextRnx::<f64>::alloc(n).unwrap();
        rnx_out.decode_from_znx::<NTT120Ref>(&znx).unwrap();

        let err = max_err(&values, &rnx_out.0);
        // Rounding at scale 2^log_decimal_prec gives max error = 0.5 * 2^-log_decimal_prec.
        let bound = (prec.log_decimal as f64).exp2().recip();
        assert!(err < bound, "max_err={err:.2e} exceeds bound={bound:.2e}");
    }

    #[test]
    fn rnx_to_znx_roundtrip_i64_path() {
        // log_decimal_prec + log_hom_rem = 50 <= 63: uses encode_vec_i64.
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
        // log_decimal_prec + log_hom_rem = 70 > 63: uses encode_vec_i128.
        roundtrip_f64(
            16,
            CKKS {
                log_hom_rem: 30,
                log_decimal: 40,
            },
        );
    }

    // Encode values → CKKSPlaintextZnx → add_to a zero VecZnx →
    // extract_from → decode, compare.
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

        // Encode to ZNX.
        let mut rnx = CKKSPlaintextRnx::<f64>::alloc(n).unwrap();
        rnx.0.copy_from_slice(&values);
        let mut pt = alloc_pt_znx(n.into(), base2k.into(), prec);
        rnx.to_znx::<NTT120Ref>(&mut pt).unwrap();

        let mut dst = alloc_pt_znx(n.into(), base2k.into(), prec);

        // Add the plaintext at the correct offset via module helper.
        module.ckks_add_pt_znx(&mut dst, &pt, scratch.borrow()).unwrap();

        // Extract the bottom-aligned plaintext and decode.
        let mut pt_out = alloc_pt_znx(n.into(), base2k.into(), prec);
        module.ckks_extract_pt_znx(&mut pt_out, &dst, scratch.borrow()).unwrap();

        assert_eq!(pt.data(), pt_out.data())
    }

    #[test]
    fn encode_decode_reim_roundtrip() {
        let n = 16usize;
        let m = n / 2;

        let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64)).collect();
        let im_in: Vec<f64> = (0..m).map(|i| -((i as f64) / (m as f64))).collect();

        let encoder = Encoder::<FFT64Ref>::new(m).unwrap();

        let mut rnx = CKKSPlaintextRnx::<f64>::alloc(n).unwrap();
        encoder.encode_reim(&mut rnx, &re_in, &im_in).unwrap();

        let mut re_out = vec![0.0f64; m];
        let mut im_out = vec![0.0f64; m];
        encoder.decode_reim(&rnx, &mut re_out, &mut im_out).unwrap();

        let err_re = max_err(&re_in, &re_out);
        let err_im = max_err(&im_in, &im_out);
        let bound = 1e-10;
        assert!(err_re < bound, "re max_err={err_re:.2e} exceeds bound={bound:.2e}");
        assert!(err_im < bound, "im max_err={err_im:.2e} exceeds bound={bound:.2e}");
    }
}
