//! Backend extension points for vector-matrix product (VMP) operations
//! on [`VmpPMat`](poulpy_hal::layouts::VmpPMat).

use std::mem::size_of;

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        reim4::Reim4BlkMatVec,
        vmp::{
            vmp_apply_dft_to_dft as fft64_vmp_apply_dft_to_dft,
            vmp_apply_dft_to_dft_tmp_bytes as fft64_vmp_apply_dft_to_dft_tmp_bytes, vmp_prepare as fft64_vmp_prepare,
            vmp_prepare_tmp_bytes as fft64_vmp_prepare_tmp_bytes, vmp_zero as fft64_vmp_zero,
        },
    },
    ntt120::{
        NttCFromB, NttDFTExecute, NttExtract1BlkContiguous, NttFromZnx64, NttMulBbc1ColX2, NttMulBbc2ColsX2,
        ntt::NttTable,
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
        vmp::{
            ntt120_vmp_apply_dft_to_dft, ntt120_vmp_apply_dft_to_dft_tmp_bytes, ntt120_vmp_prepare, ntt120_vmp_prepare_tmp_bytes,
            ntt120_vmp_zero,
        },
    },
};
use poulpy_hal::{
    api::ScratchArenaTakeHost,
    layouts::{
        Backend, HostDataMut, HostDataRef, MatZnxToRef, Module, ScratchArena, VecZnxDft, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftToMut, VecZnxDftToRef, VmpPMat, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatToRef,
        ZnxInfos,
    },
};

#[doc(hidden)]
pub trait FFT64VmpDefaults<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::DataMut,
{
    fn vmp_prepare_tmp_bytes_default(module: &Module<BE>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        fft64_vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare_default<'s, A>(
        module: &Module<BE>,
        res: &mut VmpPMatBackendMut<'_, BE>,
        a: &A,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: 's,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        ScratchArena<'s, BE>: ScratchArenaTakeHost<'s, BE>,
        A: MatZnxToRef,
    {
        let bytes = fft64_vmp_prepare_tmp_bytes(module.n());
        let (tmp, _) = scratch.borrow().take_f64(bytes / size_of::<f64>());
        fft64_vmp_prepare(module.get_fft_table(), res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        fft64_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_default<'s, 'b, R, A>(
        _module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &VmpPMatBackendRef<'b, BE>,
        limb_offset: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: 'b,
        for<'a> BE: Backend<ScalarPrep = f64, BufRef<'a> = &'a [u8]> + ReimArith + Reim4BlkMatVec,
        ScratchArena<'s, BE>: ScratchArenaTakeHost<'s, BE>,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], BE> = a.to_ref();
        let b_ref: VmpPMat<&[u8], BE> = b.to_ref();

        let bytes = fft64_vmp_apply_dft_to_dft_tmp_bytes(a_ref.size(), b_ref.rows(), b_ref.cols_in());
        let (tmp, _) = scratch.borrow().take_f64(bytes / size_of::<f64>());
        fft64_vmp_apply_dft_to_dft(&mut res_ref, &a_ref, &b_ref, limb_offset, tmp);
    }

    fn vmp_apply_dft_to_dft_backend_ref_default<'s, 'r, 'a, 'b>(
        _module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        b: &VmpPMatBackendRef<'b, BE>,
        limb_offset: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: 'b,
        BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        ScratchArena<'s, BE>: ScratchArenaTakeHost<'s, BE>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], BE> = a.to_ref();
        let b_ref: VmpPMat<&[u8], BE> = b.to_ref();

        let bytes = fft64_vmp_apply_dft_to_dft_tmp_bytes(a_ref.size(), b_ref.rows(), b_ref.cols_in());
        let (tmp, _) = scratch.borrow().take_f64(bytes / size_of::<f64>());
        fft64_vmp_apply_dft_to_dft(&mut res_ref, &a_ref, &b_ref, limb_offset, tmp);
    }

    fn vmp_zero_default(_module: &Module<BE>, res: &mut VmpPMatBackendMut<'_, BE>)
    where
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        fft64_vmp_zero(res);
    }
}

impl<BE: Backend> FFT64VmpDefaults<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::DataMut {}

#[doc(hidden)]
pub trait NTT120VmpDefaults<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::DataMut,
{
    fn vmp_prepare_tmp_bytes_default(module: &Module<BE>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_vmp_prepare_tmp_bytes(module.n())
    }

    fn vmp_prepare_default<'s, A>(
        module: &Module<BE>,
        res: &mut VmpPMatBackendMut<'_, BE>,
        a: &A,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: 's,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        ScratchArena<'s, BE>: ScratchArenaTakeHost<'s, BE>,
        A: MatZnxToRef,
    {
        let bytes = ntt120_vmp_prepare_tmp_bytes(module.n());
        let (tmp, _) = scratch.borrow().take_u64(bytes / size_of::<u64>());
        ntt120_vmp_prepare::<VmpPMatBackendMut<'_, BE>, A, BE>(module, res, a, tmp);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes_default(
        _module: &Module<BE>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_default<'s, 'b, R, A>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        b: &VmpPMatBackendRef<'b, BE>,
        limb_offset: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: 's,
        BE: 'b,
        for<'a> BE: Backend<ScalarPrep = Q120bScalar, BufRef<'a> = &'a [u8]>
            + NttExtract1BlkContiguous
            + NttMulBbc1ColX2
            + NttMulBbc2ColsX2,
        ScratchArena<'s, BE>: ScratchArenaTakeHost<'s, BE>,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], BE> = a.to_ref();
        let b_ref: VmpPMat<&[u8], BE> = b.to_ref();

        let bytes = ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_ref.size(), b_ref.rows(), b_ref.cols_in());
        let (tmp, _) = scratch.borrow().take_u64(bytes / size_of::<u64>());
        ntt120_vmp_apply_dft_to_dft::<_, _, _, BE>(module, &mut res_ref, &a_ref, &b_ref, limb_offset, tmp);
    }

    fn vmp_apply_dft_to_dft_backend_ref_default<'s, 'r, 'a, 'b>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        a: &VecZnxDftBackendRef<'a, BE>,
        b: &VmpPMatBackendRef<'b, BE>,
        limb_offset: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: 's,
        BE: 'b,
        BE: Backend<ScalarPrep = Q120bScalar> + NttExtract1BlkContiguous + NttMulBbc1ColX2 + NttMulBbc2ColsX2,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        ScratchArena<'s, BE>: ScratchArenaTakeHost<'s, BE>,
    {
        let mut res_ref: VecZnxDft<&mut [u8], BE> = res.to_mut();
        let a_ref: VecZnxDft<&[u8], BE> = a.to_ref();
        let b_ref: VmpPMat<&[u8], BE> = b.to_ref();

        let bytes = ntt120_vmp_apply_dft_to_dft_tmp_bytes(a_ref.size(), b_ref.rows(), b_ref.cols_in());
        let (tmp, _) = scratch.borrow().take_u64(bytes / size_of::<u64>());
        ntt120_vmp_apply_dft_to_dft::<_, _, _, BE>(module, &mut res_ref, &a_ref, &b_ref, limb_offset, tmp);
    }

    fn vmp_zero_default(_module: &Module<BE>, res: &mut VmpPMatBackendMut<'_, BE>)
    where
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    {
        ntt120_vmp_zero::<VmpPMatBackendMut<'_, BE>, BE>(res);
    }
}

impl<BE: Backend> NTT120VmpDefaults<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::DataMut {}
