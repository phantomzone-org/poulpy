use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDft,
    ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnxAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxOps,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::Infos,
    elem_rlwe::{RLWECt, RLWECtDft, RLWEPt},
    keys::SecretKeyDft,
    utils::derive_size,
};

pub struct GRLWECt<C, B: Backend> {
    pub data: MatZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> GRLWECt<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize, rows: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, 1, 2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<C> GRLWECt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
{
    pub fn get_row(&self, module: &Module<FFT64>, i: usize, res: &mut RLWECtDft<C, FFT64>)
    where
        VecZnxDft<C, FFT64>: VecZnxDftToMut<FFT64>,
    {
        module.vmp_extract_row(res, self, i, 0);
    }
}

impl<T, B: Backend> Infos for GRLWECt<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_k(&self) -> usize {
        self.log_k
    }
}

impl<C, B: Backend> MatZnxDftToMut<B> for GRLWECt<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> MatZnxDftToRef<B> for GRLWECt<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl GRLWECt<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_bytes(module: &Module<FFT64>, size: usize) -> usize {
        RLWECt::encrypt_sk_scratch_bytes(module, size)
            + module.bytes_of_vec_znx(2, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(2, size)
    }

    // pub fn encrypt_pk_scratch_bytes(module: &Module<FFT64>, pk_size: usize) -> usize {
    // RLWECt::encrypt_pk_scratch_bytes(module, pk_size)
    // }
}

pub fn encrypt_grlwe_sk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut GRLWECt<C, FFT64>,
    pt: &ScalarZnx<P>,
    sk: &SecretKeyDft<S, FFT64>,
    source_xa: &mut Source,
    source_xe: &mut Source,
    sigma: f64,
    bound: f64,
    scratch: &mut Scratch,
) where
    MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
    ScalarZnx<P>: ScalarZnxToRef,
    ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
{
    let rows: usize = ct.rows();
    let size: usize = ct.size();
    let log_base2k: usize = ct.log_base2k();

    let (tmp_znx_pt, scrach_1) = scratch.tmp_vec_znx(module, 1, size);
    let (tmp_znx_ct, scrach_2) = scrach_1.tmp_vec_znx(module, 2, size);
    let (mut vec_znx_dft_ct, scratch_3) = scrach_2.tmp_vec_znx_dft(module, 2, size);

    let mut vec_znx_pt: RLWEPt<&mut [u8]> = RLWEPt {
        data: tmp_znx_pt,
        log_base2k: log_base2k,
        log_k: ct.log_k(),
    };

    let mut vec_znx_ct: RLWECt<&mut [u8]> = RLWECt {
        data: tmp_znx_ct,
        log_base2k: log_base2k,
        log_k: ct.log_k(),
    };

    (0..rows).for_each(|row_i| {
        // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
        module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_i, pt, 0);
        module.vec_znx_normalize_inplace(log_base2k, &mut vec_znx_pt, 0, scratch_3);

        // rlwe encrypt of vec_znx_pt into vec_znx_ct
        vec_znx_ct.encrypt_sk(
            module,
            Some(&vec_znx_pt),
            sk,
            source_xa,
            source_xe,
            sigma,
            bound,
            scratch_3,
        );

        vec_znx_pt.data.zero(); // zeroes for next iteration

        // Switch vec_znx_ct into DFT domain
        module.vec_znx_dft(&mut vec_znx_dft_ct, 0, &vec_znx_ct, 0);
        module.vec_znx_dft(&mut vec_znx_dft_ct, 1, &vec_znx_ct, 1);

        // Stores vec_znx_dft_ct into thw i-th row of the MatZnxDft
        module.vmp_prepare_row(ct, row_i, 0, &vec_znx_dft_ct);
    });
}

impl<C> GRLWECt<C, FFT64> {
    pub fn encrypt_sk<P, S>(
        &mut self,
        module: &Module<FFT64>,
        pt: &ScalarZnx<P>,
        sk_dft: &SecretKeyDft<S, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        bound: f64,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<C, FFT64>: MatZnxDftToMut<FFT64>,
        ScalarZnx<P>: ScalarZnxToRef,
        ScalarZnxDft<S, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        encrypt_grlwe_sk(
            module, self, pt, sk_dft, source_xa, source_xe, sigma, bound, scratch,
        )
    }
}

#[cfg(test)]
mod tests {
    use base2k::{FFT64, Module, ScalarZnx, ScalarZnxAlloc, ScratchOwned, Stats, VecZnxOps};
    use sampling::source::Source;

    use crate::{
        elem::Infos,
        elem_rlwe::{RLWECtDft, RLWEPt},
        keys::{SecretKey, SecretKeyDft},
    };

    use super::GRLWECt;

    #[test]
    fn encrypt_sk_vec_znx_fft64() {
        let module: Module<FFT64> = Module::<FFT64>::new(2048);
        let log_base2k: usize = 8;
        let log_k_ct: usize = 54;
        let rows: usize = 4;

        let sigma: f64 = 3.2;
        let bound: f64 = sigma * 6.0;

        let mut ct: GRLWECt<Vec<u8>, FFT64> = GRLWECt::new(&module, log_base2k, log_k_ct, rows);
        let mut pt: RLWEPt<Vec<u8>> = RLWEPt::new(&module, log_base2k, log_k_ct);
        let mut pt_scalar: ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        pt_scalar.fill_ternary_hw(0, module.n(), &mut source_xs);

        let mut scratch: ScratchOwned = ScratchOwned::new(
            GRLWECt::encrypt_sk_scratch_bytes(&module, ct.size()) | RLWECtDft::decrypt_scratch_bytes(&module, ct.size()),
        );

        let mut sk: SecretKey<Vec<u8>> = SecretKey::new(&module);
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft: SecretKeyDft<Vec<u8>, FFT64> = SecretKeyDft::new(&module);
        sk_dft.dft(&module, &sk);

        ct.encrypt_sk(
            &module,
            &pt_scalar,
            &sk_dft,
            &mut source_xa,
            &mut source_xe,
            sigma,
            bound,
            scratch.borrow(),
        );

        let mut ct_rlwe_dft: RLWECtDft<Vec<u8>, FFT64> = RLWECtDft::new(&module, log_base2k, log_k_ct, 2);

        (0..ct.rows()).for_each(|row_i| {
            ct.get_row(&module, row_i, &mut ct_rlwe_dft);
            ct_rlwe_dft.decrypt(&module, &mut pt, &sk_dft, scratch.borrow());
            module.vec_znx_sub_scalar_inplace(&mut pt, 0, row_i, &pt_scalar, 0);
            let std_pt: f64 = pt.data.std(0, log_base2k) * (log_k_ct as f64).exp2();
            assert!((sigma - std_pt) <= 0.2, "{} {}", sigma, std_pt);
        });

        module.free();
    }
}
