use std::fmt::Debug;

use anyhow::Result;
use poulpy_core::layouts::{Base2K, Degree, GLWEPlaintext, LWEInfos};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxRshAdd},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VecZnx},
    reference::fft64::reim::FFTModuleHandle,
};
use rand_distr::num_traits::{Float, FloatConst};

use crate::encoding::classical::{decode_reim, encode_reim};

pub struct CKKSPlaintextRnx<F>(Vec<F>)
where
    F: Float + FloatConst + Debug;
pub struct CKKSPlaintextZnx<D: Data> {
    data: GLWEPlaintext<D>,
    log_decimal_prec: usize,
    log_integer_prec: usize,
}

pub trait CKKSPlaintextConversion {
    const MAX_LOG_DECIMAL_PREC: usize;
    fn to_znx<BE>(&self, other: &mut CKKSPlaintextZnx<impl DataMut>) -> Result<()>
    where
        BE: Backend;
    fn from_znx<BE>(&mut self, other: &CKKSPlaintextZnx<impl DataRef>) -> Result<()>
    where
        BE: Backend;
}

impl<F: Float + FloatConst + Debug> CKKSPlaintextRnx<F> {
    pub fn alloc(n: usize) -> Self {
        assert!(n.is_power_of_two());
        Self(vec![F::zero(); n])
    }

    pub fn encode_reim<BE: Backend>(&mut self, module: &Module<BE>, re: &[F], im: &[F])
    where
        Module<BE>: FFTModuleHandle<F>,
    {
        encode_reim(module, &mut self.0, re, im);
    }

    pub fn decode_reim<BE: Backend>(&self, module: &Module<BE>, re: &mut [F], im: &mut [F])
    where
        Module<BE>: FFTModuleHandle<F>,
    {
        decode_reim(module, &self.0, re, im);
    }
}

impl CKKSPlaintextConversion for CKKSPlaintextRnx<f64> {
    const MAX_LOG_DECIMAL_PREC: usize = 53;

    /// TODO: use buffers internally instead of allocating.
    fn from_znx<BE>(&mut self, other: &CKKSPlaintextZnx<impl DataRef>) -> Result<()>
    where
        BE: Backend,
    {
        let log_decimal_prec = other.log_decimal_prec;
        let log_integer_prec = other.log_integer_prec;
        let n = other.data.n().as_usize();

        anyhow::ensure!(log_decimal_prec <= Self::MAX_LOG_DECIMAL_PREC);
        anyhow::ensure!(self.0.len() == other.data.n().as_usize());

        let scale = (-(log_decimal_prec as f64)).exp2();
        let k = other.data.max_k();
        if log_decimal_prec + log_integer_prec <= 63 {
            let mut data = vec![0i64; n];
            other.data.decode_vec_i64(&mut data, k);
            self.0.iter_mut().zip(data.iter()).for_each(|(f, i)| *f = (*i as f64) * scale);
        } else {
            let mut data = vec![0i128; n];
            other.data.decode_vec_i128(&mut data, k);
            self.0.iter_mut().zip(data.iter()).for_each(|(f, i)| *f = (*i as f64) * scale);
        }

        Ok(())
    }

    /// TODO: use buffers internally instead of allocating.
    fn to_znx<BE>(&self, other: &mut CKKSPlaintextZnx<impl DataMut>) -> Result<()>
    where
        BE: Backend,
    {
        let log_decimal_prec = other.log_decimal_prec;
        let log_integer_prec = other.log_integer_prec;

        anyhow::ensure!(log_decimal_prec <= Self::MAX_LOG_DECIMAL_PREC);
        anyhow::ensure!(self.0.len() == other.data.n().as_usize());

        let scale = (log_decimal_prec as f64).exp2();
        let k = other.data.max_k();
        if log_decimal_prec + log_integer_prec <= 63 {
            let data: Vec<i64> = self.0.iter().map(|&x| (x * scale).round() as i64).collect();
            other.data.encode_vec_i64(&data, k);
        } else {
            let data: Vec<i128> = self.0.iter().map(|&x| (x * scale).round() as i128).collect();
            other.data.encode_vec_i128(&data, k);
        }

        Ok(())
    }
}

impl<D: Data> CKKSPlaintextZnx<D> {
    pub fn log_decimal_prec(&self) -> usize {
        self.log_decimal_prec
    }

    pub fn log_integer_prec(&self) -> usize {
        self.log_integer_prec
    }

    pub fn plaintext(&self) -> &GLWEPlaintext<D> {
        &self.data
    }
}

impl<D: DataRef> CKKSPlaintextZnx<D> {
    /// Adds this bottom-aligned plaintext into `dst` at bit position
    /// `log_delta + log_decimal_prec` from the bottom.
    ///
    /// When the total offset is not a multiple of `base2k`, each plaintext
    /// limb is split across two adjacent destination limbs via a coefficient-
    /// level bit shift.
    pub(crate) fn add_to<BE: Backend>(
        &self,
        module: &Module<BE>,
        dst: &mut VecZnx<impl DataMut>,
        log_delta: usize,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: VecZnxRshAdd<BE>,
    {
        module.vec_znx_rsh_add(self.data.base2k().as_usize(), log_delta, dst, 0, self.data.data(), 0, scratch);
    }
}

impl<D: DataMut> CKKSPlaintextZnx<D> {
    /// Extracts a bottom-aligned plaintext from a `VecZnx` at bit position
    /// `log_delta + log_decimal_prec` from the bottom.
    ///
    /// When the total offset is not a multiple of `base2k`, adjacent source
    /// limbs are combined via a coefficient-level right shift.
    pub(crate) fn extract_from<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        src: &VecZnx<impl DataRef>,
        log_delta: usize,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: VecZnxLsh<BE>,
    {
        module.vec_znx_lsh(
            self.data.base2k().as_usize(),
            log_delta,
            self.data.data_mut(),
            0,
            src,
            0,
            scratch,
        );
    }
}

impl CKKSPlaintextZnx<Vec<u8>> {
    pub fn alloc(n: Degree, base2k: Base2K, log_decimal_prec: usize, log_integer_prec: usize) -> Self {
        Self {
            data: GLWEPlaintext::alloc(
                n,
                base2k,
                ((log_decimal_prec + log_integer_prec).next_multiple_of(base2k.as_usize())).into(),
            ),
            log_decimal_prec,
            log_integer_prec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_core::layouts::{Base2K, Degree};
    use poulpy_cpu_ref::NTT120Ref;
    use poulpy_hal::{
        api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeTmpBytes},
        layouts::{ScratchOwned, ZnxInfos},
    };

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    }

    fn roundtrip_f64(log_decimal_prec: usize, log_integer_prec: usize, base2k: u32) {
        let n = 16usize;
        // Values spread across [-1, 1] to exercise both signs and fractional precision.
        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        let mut rnx = CKKSPlaintextRnx::<f64>::alloc(n);
        rnx.0.copy_from_slice(&values);

        let mut znx = CKKSPlaintextZnx::alloc(Degree(n as u32), Base2K(base2k), log_decimal_prec, log_integer_prec);
        rnx.to_znx::<NTT120Ref>(&mut znx).unwrap();

        let mut rnx_out = CKKSPlaintextRnx::<f64>::alloc(n);
        rnx_out.from_znx::<NTT120Ref>(&znx).unwrap();

        let err = max_err(&values, &rnx_out.0);
        // Rounding at scale 2^log_decimal_prec gives max error = 0.5 * 2^-log_decimal_prec.
        let bound = (log_decimal_prec as f64).exp2().recip();
        assert!(err < bound, "max_err={err:.2e} exceeds bound={bound:.2e}");
    }

    #[test]
    fn rnx_to_znx_roundtrip_i64_path() {
        // log_decimal_prec + log_integer_prec = 50 <= 63: uses encode_vec_i64.
        roundtrip_f64(40, 10, 16);
    }

    #[test]
    fn rnx_to_znx_roundtrip_i128_path() {
        // log_decimal_prec + log_integer_prec = 70 > 63: uses encode_vec_i128.
        roundtrip_f64(40, 30, 16);
    }

    // Encode values → CKKSPlaintextZnx → add_to a zero VecZnx →
    // extract_from → decode, compare.
    fn add_extract_roundtrip(log_decimal_prec: usize, log_integer_prec: usize, base2k: u32, log_delta: usize) {
        println!("log_decimal_prec: {log_decimal_prec}");
        println!("log_integer_prec: {log_integer_prec}");
        println!("base2k: {base2k}");
        println!("log_delta: {log_delta}");

        use poulpy_hal::layouts::VecZnx;
        let n = 16usize;

        let module = Module::<NTT120Ref>::new(n as u64);
        let mut scratch = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        println!("values: {values:?}");

        // Encode to ZNX.
        let mut rnx = CKKSPlaintextRnx::<f64>::alloc(n);
        rnx.0.copy_from_slice(&values);
        let mut pt = CKKSPlaintextZnx::alloc(Degree(n as u32), Base2K(base2k), log_decimal_prec, log_integer_prec);
        rnx.to_znx::<NTT120Ref>(&mut pt).unwrap();

        println!("pt_in: {}", pt.data);

        // Build a zero VecZnx wide enough to hold the offset plaintext.
        let total_offset = log_delta + log_decimal_prec;
        let limb_offset = total_offset / base2k as usize;
        let pt_size = pt.plaintext().data.size();
        let full_limbs = limb_offset + pt_size + 1; // +1 for potential bit-shift carry
        let mut buf = VecZnx::alloc(n, 1, full_limbs);

        // Add the plaintext at the correct offset.
        pt.add_to(&module, &mut buf, log_delta, scratch.borrow());

        println!("buf: {}", buf);

        // Extract the bottom-aligned plaintext and decode.
        let mut pt_out = CKKSPlaintextZnx::alloc(Degree(n as u32), Base2K(base2k), log_decimal_prec, log_integer_prec);
        pt_out.extract_from(&module, &buf, log_delta, scratch.borrow());

        assert_eq!(pt.data.data(), pt_out.data.data())
    }

    #[test]
    fn add_extract_roundtrip_no_bit_shift() {
        // total_offset = 8 + 40 = 48 = 3 * 16 → bit_shift = 0.
        add_extract_roundtrip(40, 10, 16, 8);
    }

    #[test]
    fn add_extract_roundtrip_with_bit_shift() {
        // total_offset = 6 + 40 = 46 → bit_shift = 46 % 16 = 14.
        add_extract_roundtrip(40, 10, 16, 6);
    }
}
