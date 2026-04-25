//! Backend extension points for scalar-vector product (SVP) operations
//! on [`SvpPPol`](poulpy_hal::layouts::SvpPPol).

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        svp::{
            svp_apply_dft as fft64_svp_apply_dft, svp_apply_dft_to_dft as fft64_svp_apply_dft_to_dft,
            svp_apply_dft_to_dft_assign as fft64_svp_apply_dft_to_dft_assign, svp_prepare as fft64_svp_prepare,
        },
    },
    ntt120::{
        NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc, NttZero,
        ntt::NttTable,
        primes::Primes30,
        svp::{ntt120_svp_apply_dft_to_dft, ntt120_svp_apply_dft_to_dft_assign, ntt120_svp_prepare},
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
    },
};
use poulpy_hal::{
    api::VecZnxDftApply,
    layouts::{
        Backend, HostDataRef, Module, ScalarZnxBackendRef, SvpPPolToMut, SvpPPolToRef, VecZnxBackendRef, VecZnxDft,
        VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendRef, VecZnxDftToMut, ZnxInfos,
    },
};

#[doc(hidden)]
pub trait FFT64SvpDefaults<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn svp_prepare_default<R>(module: &Module<BE>, res: &mut R, res_col: usize, a: &ScalarZnxBackendRef<'_, BE>, a_col: usize)
    where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        R: SvpPPolToMut<BE>,
    {
        fft64_svp_prepare::<R, BE>(module.get_fft_table(), res, res_col, a, a_col);
    }

    fn svp_apply_dft_default<'r, 'b, A>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'b, BE>,
        b_col: usize,
    ) where
        BE: 'r + 'b,
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        A: SvpPPolToRef<BE>,
    {
        fft64_svp_apply_dft(module.get_fft_table(), res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_dft_to_dft_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
    {
        let b_ref: VecZnxDft<&[u8], BE> = VecZnxDft::from_data(b.data.as_ref(), b.n, b.cols, b.size);
        fft64_svp_apply_dft_to_dft::<R, A, BE>(res, res_col, a, a_col, &b_ref, b_col);
    }

    fn svp_apply_dft_to_dft_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
    {
        fft64_svp_apply_dft_to_dft_assign::<R, A, BE>(res, res_col, a, a_col);
    }
}

impl<BE: Backend> FFT64SvpDefaults<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}

#[doc(hidden)]
pub trait NTT120SvpDefaults<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn svp_prepare_default<R>(module: &Module<BE>, res: &mut R, res_col: usize, a: &ScalarZnxBackendRef<'_, BE>, a_col: usize)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttCFromB,
        for<'a> BE::BufRef<'a>: HostDataRef,
        R: SvpPPolToMut<BE>,
    {
        ntt120_svp_prepare::<R, BE>(module, res, res_col, a, a_col);
    }

    fn svp_apply_dft_default<'r, 'b, A>(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'r, BE>,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxBackendRef<'b, BE>,
        b_col: usize,
    ) where
        BE: 'r + 'b,
        Module<BE>: NttModuleHandle + VecZnxDftApply<BE>,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttMulBbc + NttZero,
        for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
        A: SvpPPolToRef<BE>,
    {
        let b_size = b.size();
        let mut b_dft = poulpy_hal::layouts::VecZnxDftOwned::<BE>::alloc(module.n(), 1, b_size);
        let mut b_dft_ref = b_dft.to_mut();

        <Module<BE> as VecZnxDftApply<BE>>::vec_znx_dft_apply(module, 1, 0, &mut b_dft_ref, 0, b, b_col);
        let b_dft_backend = b_dft_ref.reborrow_backend_ref();
        let b_dft_host: VecZnxDft<&[u8], BE> = VecZnxDft::from_data(b_dft_backend.data, b_dft_backend.n, b_dft_backend.cols, b_dft_backend.size);
        ntt120_svp_apply_dft_to_dft(module, res, res_col, a, a_col, &b_dft_host, 0);
    }

    fn svp_apply_dft_to_dft_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttMulBbc + NttZero,
        for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
    {
        let b_ref: VecZnxDft<&[u8], BE> = VecZnxDft::from_data(b.data.as_ref(), b.n, b.cols, b.size);
        ntt120_svp_apply_dft_to_dft::<R, A, BE>(module, res, res_col, a, a_col, &b_ref, b_col);
    }

    fn svp_apply_dft_to_dft_assign_default<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttMulBbc,
        R: VecZnxDftToMut<BE>,
        A: SvpPPolToRef<BE>,
    {
        ntt120_svp_apply_dft_to_dft_assign::<R, A, BE>(module, res, res_col, a, a_col);
    }
}

impl<BE: Backend> NTT120SvpDefaults<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
