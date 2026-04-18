use std::fmt::Debug;

use anyhow::Result;
use poulpy_cpu_ref::reference::fft64::reim::{ReimFFTTable, ReimIFFTTable};
use poulpy_hal::GALOISGENERATOR;
use rand_distr::num_traits::{Float, FloatConst, NumCast};

use crate::layouts::plaintext::CKKSPlaintextVecRnx;

/// Slot encoder/decoder for CKKS real and imaginary vectors.
///
/// The encoder maps `m` complex slots onto an RNX plaintext of size `2m`
/// through the canonical FFT/IFFT packing used by the rest of the crate.
pub struct Encoder<F: Float + FloatConst + Debug> {
    fft_table: ReimFFTTable<F>,
    ifft_table: ReimIFFTTable<F>,
    slot_map: Vec<usize>,
}

impl<F> Encoder<F>
where
    F: Float + FloatConst + Debug,
{
    /// Creates an encoder for `m` complex CKKS slots.
    ///
    /// Inputs:
    /// - `m`: number of complex slots
    ///
    /// Output:
    /// - an encoder configured for plaintext polynomials of size `2m`
    ///
    /// Errors:
    /// - returns an error if `m == 0` or if `m` is not a power of two
    pub fn new(m: usize) -> Result<Self> {
        anyhow::ensure!(m.is_power_of_two(), "m must be a power of two, got {m}");
        anyhow::ensure!(m > 0, "m must be > 0, got {m}");
        let two_n = 4 * m;
        let log_n = (2 * m).trailing_zeros();
        let mut slot_map = Vec::with_capacity(m);
        let mut exp = 1usize;
        for _ in 0..m {
            slot_map.push(((exp - 1) / 2).reverse_bits() >> (usize::BITS - log_n));
            exp = (exp * GALOISGENERATOR as usize) & (two_n - 1);
        }
        Ok(Self {
            fft_table: ReimFFTTable::new(m),
            ifft_table: ReimIFFTTable::new(m),
            slot_map,
        })
    }

    /// Encodes complex slot values into an RNX plaintext buffer.
    ///
    /// Inputs:
    /// - `pt`: destination plaintext polynomial of size `2m`
    /// - `re`, `im`: real and imaginary slot vectors, each of length `m`
    ///
    /// Output:
    /// - fills `pt` with the packed coefficient representation
    ///
    /// Behavior:
    /// - writes slots according to the internal CKKS slot permutation
    /// - runs the inverse FFT and normalizes by `1/m`
    ///
    /// Errors:
    /// - returns an error if `pt`, `re`, and `im` do not match the encoder's
    ///   configured slot count
    pub fn encode_reim(&self, pt: &mut CKKSPlaintextVecRnx<F>, re: &[F], im: &[F]) -> Result<()> {
        let n = pt.n();
        let m = n / 2;

        let table = &self.ifft_table;

        anyhow::ensure!(table.m() == m);
        anyhow::ensure!(re.len() == m);
        anyhow::ensure!(im.len() == m);

        pt.data_mut().fill(F::zero());
        for k in 0..m {
            let idx = self.slot_map[k];
            pt.data_mut()[idx] = re[k];
            pt.data_mut()[m + idx] = im[k];
        }

        table.execute(pt.data_mut());

        let inv_m = <F as NumCast>::from(m).unwrap().recip();
        pt.data_mut().iter_mut().for_each(|x| *x = *x * inv_m);

        Ok(())
    }

    /// Decodes an RNX plaintext buffer back into complex slot vectors.
    ///
    /// Inputs:
    /// - `pt`: source RNX plaintext polynomial of size `2m`
    /// - `re`, `im`: output slot buffers of length `m`
    ///
    /// Output:
    /// - fills `re` and `im` with the decoded slot values
    ///
    /// Behavior:
    /// - runs the forward FFT and applies the inverse of the encoder slot map
    ///
    /// Errors:
    /// - returns an error if the provided buffers do not match the encoder's
    ///   configured slot count
    pub fn decode_reim(&self, pt: &CKKSPlaintextVecRnx<F>, re: &mut [F], im: &mut [F]) -> Result<()> {
        let n = pt.n();
        let m = n / 2;

        let table = &self.fft_table;

        anyhow::ensure!(table.m() == m);
        anyhow::ensure!(re.len() == m);
        anyhow::ensure!(im.len() == m);

        let mut reim_tmp = vec![F::zero(); n];
        reim_tmp.copy_from_slice(pt.data());

        table.execute(&mut reim_tmp);

        for k in 0..m {
            let idx = self.slot_map[k];
            re[k] = reim_tmp[idx];
            im[k] = reim_tmp[m + idx];
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn encode_decode_reim_roundtrip() {
        let n = 16usize;
        let m = n / 2;

        let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64)).collect();
        let im_in: Vec<f64> = (0..m).map(|i| -((i as f64) / (m as f64))).collect();

        let encoder = Encoder::<f64>::new(m).unwrap();

        let mut rnx = CKKSPlaintextVecRnx::<f64>::alloc(n).unwrap();
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
