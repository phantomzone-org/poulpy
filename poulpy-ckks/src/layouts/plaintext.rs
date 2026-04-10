use std::fmt::Debug;

use anyhow::Result;
use poulpy_core::layouts::{Base2K, Degree, GLWEPlaintext, LWEInfos, TorusPrecision};
use poulpy_hal::{
    GALOISGENERATOR,
    api::{VecZnxLsh, VecZnxRshAdd, VecZnxRshSub},
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, VecZnx},
    reference::fft64::reim::{ReimFFTTable, ReimIFFTTable},
};
use rand_distr::num_traits::{Float, FloatConst};

#[derive(Debug, Clone)]
pub struct CKKSPlaintextRnx<F>(Vec<F>)
where
    F: Float + FloatConst + Debug;
pub struct CKKSPlaintextZnx<D: Data> {
    pub data: GLWEPlaintext<D>,
    pub prec: PrecisionLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct PrecisionLayout {
    pub log_decimal: usize,
    pub log_integer: usize,
}

impl PrecisionLayout {
    pub fn k(&self, base2k: Base2K) -> TorusPrecision {
        ((self.log_decimal + self.log_integer).next_multiple_of(base2k.as_usize())).into()
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

/// Returns the FFT-internal position for each user slot.
///
/// User slot `k` has Galois exponent `5^k mod 2N`.  The FFT position
/// is `bit_reverse((5^k mod 2N − 1) / 2, log₂N)`.  No conjugate
/// folding is needed because `5 ≡ 1 mod 4` guarantees the result
/// stays in `[0, m)`.
fn slot_map(m: usize) -> Vec<usize> {
    let two_n = 4 * m;
    let log_n = (2 * m).trailing_zeros();
    let mut map = Vec::with_capacity(m);
    let mut exp = 1usize;
    for _ in 0..m {
        map.push(((exp - 1) / 2).reverse_bits() >> (usize::BITS - log_n));
        exp = (exp * GALOISGENERATOR as usize) % two_n;
    }
    map
}

impl<F: Float + FloatConst + Debug> CKKSPlaintextRnx<F> {
    pub fn alloc(n: usize) -> Self {
        assert!(n.is_power_of_two());
        Self(vec![F::zero(); n])
    }

    pub fn n(&self) -> usize {
        self.0.len()
    }

    pub fn data(&self) -> &[F] {
        &self.0
    }

    pub fn data_mut(&mut self) -> &mut [F] {
        &mut self.0
    }

    pub fn encode_reim(&mut self, table: &ReimIFFTTable<F>, re: &[F], im: &[F]) {
        let n = self.n();
        let m = n / 2;

        assert_eq!(table.m(), m);
        assert_eq!(re.len(), m);
        assert_eq!(im.len(), m);

        let sm = slot_map(m);

        self.0.fill(F::zero());
        for k in 0..m {
            self.0[sm[k]] = re[k];
            self.0[m + sm[k]] = im[k];
        }

        table.execute(&mut self.0);

        // Normalize by 1/m, matching vec_znx_idft_apply's divisor=m convention.
        let inv_m = F::from(m).unwrap().recip();
        self.0.iter_mut().for_each(|x| *x = *x * inv_m);
    }

    pub fn decode_reim(&self, table: &ReimFFTTable<F>, re: &mut [F], im: &mut [F]) {
        let n = self.n();
        let m = n / 2;

        assert_eq!(table.m(), m);
        assert!(re.len() <= m);
        assert!(im.len() <= m);

        let mut reim_tmp = vec![F::zero(); n];
        reim_tmp.copy_from_slice(&self.0);

        table.execute(&mut reim_tmp);

        let sm = slot_map(m);
        for (k, out) in re.iter_mut().enumerate() {
            *out = reim_tmp[sm[k]];
        }
        for (k, out) in im.iter_mut().enumerate() {
            *out = reim_tmp[m + sm[k]];
        }
    }
}

impl CKKSPlaintextConversion for CKKSPlaintextRnx<f64> {
    const MAX_LOG_DECIMAL_PREC: usize = 53;

    /// TODO: use buffers internally instead of allocating.
    fn decode_from_znx<BE>(&mut self, other: &CKKSPlaintextZnx<impl DataRef>) -> Result<()>
    where
        BE: Backend,
    {
        let log_decimal = other.prec.log_decimal;
        let log_integer = other.prec.log_integer;
        let n = other.data.n().as_usize();

        anyhow::ensure!(log_decimal <= Self::MAX_LOG_DECIMAL_PREC);
        anyhow::ensure!(self.0.len() == other.data.n().as_usize());
        anyhow::ensure!(log_decimal + log_integer <= 127);

        let scale = (-(log_decimal as f64)).exp2();
        let k = other.data.max_k();
        if log_decimal + log_integer <= 63 {
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
        let log_decimal = other.prec.log_decimal;
        let log_integer = other.prec.log_integer;

        anyhow::ensure!(log_decimal <= Self::MAX_LOG_DECIMAL_PREC);
        anyhow::ensure!(self.0.len() == other.data.n().as_usize());

        let scale = (log_decimal as f64).exp2();
        let k = other.data.max_k();
        if log_decimal + log_integer <= 63 {
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
        self.prec.log_decimal
    }

    pub fn log_integer_size(&self) -> usize {
        self.prec.log_integer
    }

    pub fn plaintext(&self) -> &GLWEPlaintext<D> {
        &self.data
    }

    pub(crate) fn offset(&self, log_scale: usize) -> usize {
        log_scale + self.log_decimal_prec() - self.data.max_k().as_usize()
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
        module.vec_znx_rsh_add(
            self.data.base2k().as_usize(),
            self.offset(log_delta),
            dst,
            0,
            self.data.data(),
            0,
            scratch,
        );
    }

    pub(crate) fn sub_to<BE: Backend>(
        &self,
        module: &Module<BE>,
        dst: &mut VecZnx<impl DataMut>,
        log_delta: usize,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: VecZnxRshSub<BE>,
    {
        module.vec_znx_rsh_sub(
            self.data.base2k().as_usize(),
            self.offset(log_delta),
            dst,
            0,
            self.data.data(),
            0,
            scratch,
        );
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
            self.offset(log_delta),
            self.data.data_mut(),
            0,
            src,
            0,
            scratch,
        );
    }
}

impl CKKSPlaintextZnx<Vec<u8>> {
    pub fn alloc(n: Degree, base2k: Base2K, prec: PrecisionLayout) -> Self {
        Self {
            data: GLWEPlaintext::alloc(n, base2k, prec.k(base2k)),
            prec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_cpu_ref::{FFT64Ref, NTT120Ref, fft64::FFT64ModuleHandle};
    use poulpy_hal::{
        api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeTmpBytes},
        layouts::ScratchOwned,
    };

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    }

    fn roundtrip_f64(base2k: usize, prec: PrecisionLayout) {
        let n = 16usize;
        // Values spread across [-1, 1] to exercise both signs and fractional precision.
        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        let mut rnx = CKKSPlaintextRnx::<f64>::alloc(n);
        rnx.0.copy_from_slice(&values);

        let mut znx = CKKSPlaintextZnx::alloc(n.into(), base2k.into(), prec);
        rnx.to_znx::<NTT120Ref>(&mut znx).unwrap();

        let mut rnx_out = CKKSPlaintextRnx::<f64>::alloc(n);
        rnx_out.decode_from_znx::<NTT120Ref>(&znx).unwrap();

        let err = max_err(&values, &rnx_out.0);
        // Rounding at scale 2^log_decimal_prec gives max error = 0.5 * 2^-log_decimal_prec.
        let bound = (prec.log_decimal as f64).exp2().recip();
        assert!(err < bound, "max_err={err:.2e} exceeds bound={bound:.2e}");
    }

    #[test]
    fn rnx_to_znx_roundtrip_i64_path() {
        // log_decimal_prec + log_integer_size = 50 <= 63: uses encode_vec_i64.
        roundtrip_f64(
            16,
            PrecisionLayout {
                log_integer: 10,
                log_decimal: 40,
            },
        );
    }

    #[test]
    fn rnx_to_znx_roundtrip_i128_path() {
        // log_decimal_prec + log_integer_size = 70 > 63: uses encode_vec_i128.
        roundtrip_f64(
            16,
            PrecisionLayout {
                log_integer: 30,
                log_decimal: 40,
            },
        );
    }

    // Encode values → CKKSPlaintextZnx → add_to a zero VecZnx →
    // extract_from → decode, compare.
    #[test]
    fn add_extract_roundtrip() {
        use poulpy_hal::layouts::VecZnx;
        let n = 16usize;

        let prec = PrecisionLayout {
            log_integer: 10,
            log_decimal: 40,
        };
        let base2k: usize = 52;

        let module = Module::<NTT120Ref>::new(n as u64);
        let mut scratch = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        let values: Vec<f64> = (0..n).map(|i| 2.0 * (i as f64) / (n as f64) - 1.0).collect();

        // Encode to ZNX.
        let mut rnx = CKKSPlaintextRnx::<f64>::alloc(n);
        rnx.0.copy_from_slice(&values);
        let mut pt = CKKSPlaintextZnx::alloc(n.into(), base2k.into(), prec);
        rnx.to_znx::<NTT120Ref>(&mut pt).unwrap();

        let log_delta: usize = 188;

        // Build a zero VecZnx wide enough to hold the offset plaintext.
        let mut buf = VecZnx::alloc(n, 1, log_delta.div_ceil(base2k));

        // Add the plaintext at the correct offset.
        pt.add_to(&module, &mut buf, log_delta, scratch.borrow());

        // Extract the bottom-aligned plaintext and decode.
        let mut pt_out = CKKSPlaintextZnx::alloc(n.into(), base2k.into(), prec);
        pt_out.extract_from(&module, &buf, log_delta, scratch.borrow());

        assert_eq!(pt.data.data(), pt_out.data.data())
    }

    #[test]
    fn encode_decode_reim_roundtrip() {
        let n = 16usize;
        let m = n / 2;
        let module = Module::<FFT64Ref>::new(n as u64);

        let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64)).collect();
        let im_in: Vec<f64> = (0..m).map(|i| -((i as f64) / (m as f64))).collect();

        let mut rnx = CKKSPlaintextRnx::<f64>::alloc(n);
        rnx.encode_reim(module.get_ifft_table(), &re_in, &im_in);

        let mut re_out = vec![0.0f64; m];
        let mut im_out = vec![0.0f64; m];
        rnx.decode_reim(module.get_fft_table(), &mut re_out, &mut im_out);

        let err_re = max_err(&re_in, &re_out);
        let err_im = max_err(&im_in, &im_out);
        let bound = 1e-10;
        assert!(err_re < bound, "re max_err={err_re:.2e} exceeds bound={bound:.2e}");
        assert!(err_im < bound, "im max_err={err_im:.2e} exceeds bound={bound:.2e}");
    }
}
