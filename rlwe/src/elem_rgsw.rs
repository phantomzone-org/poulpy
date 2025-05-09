use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDft,
    ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnxAlloc, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxOps,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    elem::Infos,
    elem_rlwe::{RLWECt, RLWECtDft, RLWEPt, encrypt_rlwe_sk},
    keys::SecretKeyDft,
    utils::derive_size,
};

pub struct RGSWCt<C, B: Backend> {
    pub data: MatZnxDft<C, B>,
    pub log_base2k: usize,
    pub log_k: usize,
}

impl<B: Backend> RGSWCt<Vec<u8>, B> {
    pub fn new(module: &Module<B>, log_base2k: usize, log_k: usize, rows: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, 2, 2, derive_size(log_base2k, log_k)),
            log_base2k: log_base2k,
            log_k: log_k,
        }
    }
}

impl<T, B: Backend> Infos for RGSWCt<T, B> {
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

impl<C, B: Backend> MatZnxDftToMut<B> for RGSWCt<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> MatZnxDftToRef<B> for RGSWCt<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl RGSWCt<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, size: usize) -> usize {
        RLWECt::encrypt_sk_scratch_space(module, size)
            + module.bytes_of_vec_znx(2, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(2, size)
    }
}

impl<C> RGSWCt<C, FFT64>
where
    MatZnxDft<C, FFT64>: MatZnxDftToRef<FFT64>,
{
    pub fn get_row(&self, module: &Module<FFT64>, row_i: usize, col_j: usize, res: &mut RLWECtDft<C, FFT64>)
    where
        VecZnxDft<C, FFT64>: VecZnxDftToMut<FFT64>,
    {
        module.vmp_extract_row(res, self, row_i, col_j);
    }
}

pub fn encrypt_rgsw_sk<C, P, S>(
    module: &Module<FFT64>,
    ct: &mut RGSWCt<C, FFT64>,
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
    let size: usize = ct.size();
    let log_base2k: usize = ct.log_base2k();

    let (tmp_znx_pt, scratch_1) = scratch.tmp_vec_znx(module, 1, size);
    let (tmp_znx_ct, scrach_2) = scratch_1.tmp_vec_znx(module, 2, size);

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

    (0..ct.rows()).for_each(|row_j| {
        // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
        module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_j, pt, 0);
        module.vec_znx_normalize_inplace(log_base2k, &mut vec_znx_pt, 0, scrach_2);

        (0..ct.cols()).for_each(|col_i| {
            // rlwe encrypt of vec_znx_pt into vec_znx_ct
            encrypt_rlwe_sk(
                module,
                &mut vec_znx_ct,
                Some((&vec_znx_pt, col_i)),
                sk_dft,
                source_xa,
                source_xe,
                sigma,
                bound,
                scrach_2,
            );

            // Switch vec_znx_ct into DFT domain
            {
                let (mut vec_znx_dft_ct, _) = scrach_2.tmp_vec_znx_dft(module, 2, size);
                module.vec_znx_dft(&mut vec_znx_dft_ct, 0, &vec_znx_ct, 0);
                module.vec_znx_dft(&mut vec_znx_dft_ct, 1, &vec_znx_ct, 1);
                module.vmp_prepare_row(ct, row_j, col_i, &vec_znx_dft_ct);
            }
        });

        vec_znx_pt.data.zero(); // zeroes for next iteration
    });
}

impl<C> RGSWCt<C, FFT64> {
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
        encrypt_rgsw_sk(
            module, self, pt, sk_dft, source_xa, source_xe, sigma, bound, scratch,
        )
    }
}
