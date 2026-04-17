use std::fmt::Debug;

use anyhow::Result;
use poulpy_cpu_ref::fft64::FFTModuleHandle;
use poulpy_hal::{
    GALOISGENERATOR,
    api::ModuleNew,
    layouts::{Backend, Module},
};
use rand_distr::num_traits::{Float, FloatConst, NumCast, Zero};

use crate::layouts::plaintext::CKKSPlaintextRnx;

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

        pt.data_mut().fill(BE::ScalarPrep::zero());
        for k in 0..m {
            let idx = self.slot_map[k];
            pt.data_mut()[idx] = re[k];
            pt.data_mut()[m + idx] = im[k];
        }

        table.execute(pt.data_mut());

        let inv_m = <BE::ScalarPrep as NumCast>::from(m).unwrap().recip();
        pt.data_mut().iter_mut().for_each(|x| *x = *x * inv_m);

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
    use poulpy_cpu_ref::FFT64Ref;

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
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
