use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDft,
    ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnxAlloc, VecZnxDftAlloc, VecZnxDftOps, ZnxView, ZnxViewMut,
};
use sampling::source::Source;

use crate::{
    elem::Infos,
    elem_grlwe::GRLWECt,
    elem_rlwe::{RLWECt, RLWECtDft, RLWEPt},
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

impl GRLWECt<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_bytes(module: &Module<FFT64>, size: usize) -> usize {
        RLWECt::encrypt_sk_scratch_bytes(module, size)
            + module.bytes_of_vec_znx(2, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(2, size)
    }

    pub fn encrypt_pk_scratch_bytes(module: &Module<FFT64>, pk_size: usize) -> usize {
        RLWECt::encrypt_pk_scratch_bytes(module, pk_size)
    }

    pub fn decrypt_scratch_bytes(module: &Module<FFT64>, size: usize) -> usize {
        RLWECtDft::decrypt_scratch_bytes(module, size)
    }
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

    let (tmp_znx_pt, scrach_1) = scratch.tmp_vec_znx(module, 1, size);
    let (tmp_znx_ct, scrach_2) = scrach_1.tmp_vec_znx(module, 2, size);
    let (mut tmp_dft, scratch_3) = scrach_2.tmp_vec_znx_dft(module, 2, size);

    let mut tmp_pt: RLWEPt<&mut [u8]> = RLWEPt {
        data: tmp_znx_pt,
        log_base2k: ct.log_base2k(),
        log_k: ct.log_k(),
    };

    let mut tmp_ct: RLWECt<&mut [u8]> = RLWECt {
        data: tmp_znx_ct,
        log_base2k: ct.log_base2k(),
        log_k: ct.log_k(),
    };

    (0..rows).for_each(|row_i| {
        tmp_pt
            .data
            .at_mut(0, row_i)
            .copy_from_slice(&pt.to_ref().raw());

        tmp_ct.encrypt_sk(
            module,
            Some(&tmp_pt),
            sk,
            source_xa,
            source_xe,
            sigma,
            bound,
            scratch_3,
        );

        tmp_pt.data.at_mut(0, row_i).fill(0);

        module.vec_znx_dft(&mut tmp_dft, 0, &tmp_ct, 0);
        module.vec_znx_dft(&mut tmp_dft, 1, &tmp_ct, 1);

        module.vmp_prepare_row(ct, row_i, 0, &tmp_dft);
    });
}
