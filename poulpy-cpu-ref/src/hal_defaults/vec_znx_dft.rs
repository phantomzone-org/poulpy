//! Backend extension points for DFT-domain [`VecZnxDft`](poulpy_hal::layouts::VecZnxDft) operations.

use std::mem::size_of;

use crate::reference::{
    fft64::{
        module::FFTModuleHandle,
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
        vec_znx_dft::{
            vec_znx_dft_add_assign as fft64_vec_znx_dft_add_assign, vec_znx_dft_add_into as fft64_vec_znx_dft_add_into,
            vec_znx_dft_add_scaled_assign as fft64_vec_znx_dft_add_scaled_assign, vec_znx_dft_apply as fft64_vec_znx_dft_apply,
            vec_znx_dft_copy as fft64_vec_znx_dft_copy, vec_znx_dft_sub as fft64_vec_znx_dft_sub,
            vec_znx_dft_sub_inplace as fft64_vec_znx_dft_sub_inplace,
            vec_znx_dft_sub_negate_inplace as fft64_vec_znx_dft_sub_negate_inplace, vec_znx_dft_zero as fft64_vec_znx_dft_zero,
            vec_znx_idft_apply as fft64_vec_znx_idft_apply, vec_znx_idft_apply_consume as fft64_vec_znx_idft_apply_consume,
            vec_znx_idft_apply_tmpa as fft64_vec_znx_idft_apply_tmpa,
        },
    },
    ntt_ifma::{
        NttIfmaAdd, NttIfmaAddInplace, NttIfmaCopy, NttIfmaDFTExecute, NttIfmaFromZnx64, NttIfmaNegate, NttIfmaNegateInplace,
        NttIfmaSub, NttIfmaSubInplace, NttIfmaSubNegateInplace, NttIfmaToZnx128, NttIfmaZero,
        ntt::{NttIfmaTable, NttIfmaTableInv},
        primes::Primes40,
        vec_znx_dft::{
            NttIfmaModuleHandle, ntt_ifma_vec_znx_dft_add_assign as ntt_ifma_default_vec_znx_dft_add_assign,
            ntt_ifma_vec_znx_dft_add_into as ntt_ifma_default_vec_znx_dft_add_into,
            ntt_ifma_vec_znx_dft_add_scaled_assign as ntt_ifma_default_vec_znx_dft_add_scaled_assign,
            ntt_ifma_vec_znx_dft_apply as ntt_ifma_default_vec_znx_dft_apply,
            ntt_ifma_vec_znx_dft_copy as ntt_ifma_default_vec_znx_dft_copy,
            ntt_ifma_vec_znx_dft_sub as ntt_ifma_default_vec_znx_dft_sub,
            ntt_ifma_vec_znx_dft_sub_inplace as ntt_ifma_default_vec_znx_dft_sub_inplace,
            ntt_ifma_vec_znx_dft_sub_negate_inplace as ntt_ifma_default_vec_znx_dft_sub_negate_inplace,
            ntt_ifma_vec_znx_dft_zero as ntt_ifma_default_vec_znx_dft_zero,
            ntt_ifma_vec_znx_idft_apply as ntt_ifma_default_vec_znx_idft_apply,
            ntt_ifma_vec_znx_idft_apply_tmp_bytes as ntt_ifma_default_vec_znx_idft_apply_tmp_bytes,
            ntt_ifma_vec_znx_idft_apply_tmpa as ntt_ifma_default_vec_znx_idft_apply_tmpa,
        },
    },
    ntt120::{
        NttAdd, NttAddInplace, NttCopy, NttDFTExecute, NttFromZnx64, NttNegate, NttNegateInplace, NttSub, NttSubInplace,
        NttSubNegateInplace, NttToZnx128, NttZero,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        types::Q120bScalar,
        vec_znx_dft::{
            NttModuleHandle, ntt120_vec_znx_dft_add_assign as ntt120_default_vec_znx_dft_add_assign,
            ntt120_vec_znx_dft_add_into as ntt120_default_vec_znx_dft_add_into,
            ntt120_vec_znx_dft_add_scaled_assign as ntt120_default_vec_znx_dft_add_scaled_assign,
            ntt120_vec_znx_dft_apply as ntt120_default_vec_znx_dft_apply,
            ntt120_vec_znx_dft_copy as ntt120_default_vec_znx_dft_copy, ntt120_vec_znx_dft_sub as ntt120_default_vec_znx_dft_sub,
            ntt120_vec_znx_dft_sub_inplace as ntt120_default_vec_znx_dft_sub_inplace,
            ntt120_vec_znx_dft_sub_negate_inplace as ntt120_default_vec_znx_dft_sub_negate_inplace,
            ntt120_vec_znx_dft_zero as ntt120_default_vec_znx_dft_zero,
            ntt120_vec_znx_idft_apply as ntt120_default_vec_znx_idft_apply,
            ntt120_vec_znx_idft_apply_consume as ntt120_default_vec_znx_idft_apply_consume,
            ntt120_vec_znx_idft_apply_tmp_bytes as ntt120_default_vec_znx_idft_apply_tmp_bytes,
            ntt120_vec_znx_idft_apply_tmpa as ntt120_default_vec_znx_idft_apply_tmpa,
        },
    },
    znx::ZnxZero,
};
use poulpy_hal::{
    api::TakeSlice,
    layouts::{
        Backend, Data, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
    },
};

#[doc(hidden)]
pub trait FFT64VecZnxDftDefaults<BE: Backend>: Backend {
    fn vec_znx_dft_apply_default<R, A>(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64> + ReimArith + ReimFFTExecute<ReimFFTTable<f64>, f64>,
        R: VecZnxDftToMut<BE>,
        A: VecZnxToRef,
    {
        fft64_vec_znx_dft_apply(module.get_fft_table(), step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(_module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarPrep = f64>,
    {
        0
    }

    fn vec_znx_idft_apply_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        _scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
        R: VecZnxBigToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        fft64_vec_znx_idft_apply(module.get_ifft_table(), res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmpa_default<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64> + ZnxZero,
        R: VecZnxBigToMut<BE>,
        A: VecZnxDftToMut<BE>,
    {
        fft64_vec_znx_idft_apply_tmpa(module.get_ifft_table(), res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_consume_default<D: Data>(module: &Module<BE>, a: VecZnxDft<D, BE>) -> VecZnxBig<D, BE>
    where
        Module<BE>: FFTModuleHandle<f64>,
        BE: Backend<ScalarPrep = f64, ScalarBig = i64> + ReimArith + ReimFFTExecute<ReimIFFTTable<f64>, f64>,
        VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
    {
        fft64_vec_znx_idft_apply_consume(module.get_ifft_table(), a)
    }

    fn vec_znx_dft_add_into_default<R, A, D>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        D: VecZnxDftToRef<BE>,
    {
        fft64_vec_znx_dft_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_scaled_assign_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        a_scale: i64,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        fft64_vec_znx_dft_add_scaled_assign(res, res_col, a, a_col, a_scale);
    }

    fn vec_znx_dft_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        fft64_vec_znx_dft_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default<R, A, D>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        D: VecZnxDftToRef<BE>,
    {
        fft64_vec_znx_dft_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_inplace_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        fft64_vec_znx_dft_sub_inplace(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_inplace_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        fft64_vec_znx_dft_sub_negate_inplace(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default<R, A>(
        _module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        fft64_vec_znx_dft_copy(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<ScalarPrep = f64> + ReimArith,
        R: VecZnxDftToMut<BE>,
    {
        fft64_vec_znx_dft_zero(res, res_col);
    }
}

impl<BE: Backend> FFT64VecZnxDftDefaults<BE> for BE {}

#[doc(hidden)]
pub trait NTT120VecZnxDftDefaults<BE: Backend>: Backend {
    fn vec_znx_dft_apply_default<R, A>(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttZero,
        R: VecZnxDftToMut<BE>,
        A: VecZnxToRef,
    {
        ntt120_default_vec_znx_dft_apply(module, step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt120_default_vec_znx_idft_apply_tmp_bytes(module.n())
    }

    fn vec_znx_idft_apply_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128 + NttCopy,
        Scratch<BE>: TakeSlice,
        R: VecZnxBigToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        let (tmp, _) = scratch.take_slice(ntt120_default_vec_znx_idft_apply_tmp_bytes(module.n()) / size_of::<u64>());
        ntt120_default_vec_znx_idft_apply(module, res, res_col, a, a_col, tmp);
    }

    fn vec_znx_idft_apply_tmpa_default<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128,
        R: VecZnxBigToMut<BE>,
        A: VecZnxDftToMut<BE>,
    {
        ntt120_default_vec_znx_idft_apply_tmpa(module, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_consume_default<D: Data>(module: &Module<BE>, a: VecZnxDft<D, BE>) -> VecZnxBig<D, BE>
    where
        Module<BE>: NttModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>,
        VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
    {
        ntt120_default_vec_znx_idft_apply_consume(module, a)
    }

    fn vec_znx_dft_add_into_default<R, A, D>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttAdd + NttCopy + NttZero,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        D: VecZnxDftToRef<BE>,
    {
        ntt120_default_vec_znx_dft_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_scaled_assign_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        a_scale: i64,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttAddInplace,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt120_default_vec_znx_dft_add_scaled_assign(res, res_col, a, a_col, a_scale);
    }

    fn vec_znx_dft_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttAddInplace,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt120_default_vec_znx_dft_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default<R, A, D>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttSub + NttNegate + NttCopy + NttZero,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        D: VecZnxDftToRef<BE>,
    {
        ntt120_default_vec_znx_dft_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_inplace_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttSubInplace,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt120_default_vec_znx_dft_sub_inplace(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_inplace_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttSubNegateInplace + NttNegateInplace,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt120_default_vec_znx_dft_sub_negate_inplace(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default<R, A>(
        _module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttCopy + NttZero,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt120_default_vec_znx_dft_copy(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttZero,
        R: VecZnxDftToMut<BE>,
    {
        ntt120_default_vec_znx_dft_zero(res, res_col);
    }
}

impl<BE: Backend> NTT120VecZnxDftDefaults<BE> for BE {}

#[doc(hidden)]
pub trait NTTIfmaVecZnxDftDefaults<BE: Backend>: Backend {
    fn vec_znx_dft_apply_default<R, A>(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        Module<BE>: NttIfmaModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaDFTExecute<NttIfmaTable<Primes40>> + NttIfmaFromZnx64 + NttIfmaZero,
        R: VecZnxDftToMut<BE>,
        A: VecZnxToRef,
    {
        ntt_ifma_default_vec_znx_dft_apply(module, step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_tmp_bytes_default(module: &Module<BE>) -> usize
    where
        BE: Backend<ScalarPrep = Q120bScalar>,
    {
        ntt_ifma_default_vec_znx_idft_apply_tmp_bytes(module.n())
    }

    fn vec_znx_idft_apply_default<R, A>(
        module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: NttIfmaModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>
            + NttIfmaDFTExecute<NttIfmaTableInv<Primes40>>
            + NttIfmaToZnx128
            + NttIfmaCopy,
        Scratch<BE>: TakeSlice,
        R: VecZnxBigToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        let (tmp, _) = scratch.take_slice(ntt_ifma_default_vec_znx_idft_apply_tmp_bytes(module.n()) / size_of::<u64>());
        ntt_ifma_default_vec_znx_idft_apply(module, res, res_col, a, a_col, tmp);
    }

    fn vec_znx_idft_apply_tmpa_default<R, A>(module: &Module<BE>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        Module<BE>: NttIfmaModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttIfmaDFTExecute<NttIfmaTableInv<Primes40>> + NttIfmaToZnx128,
        R: VecZnxBigToMut<BE>,
        A: VecZnxDftToMut<BE>,
    {
        ntt_ifma_default_vec_znx_idft_apply_tmpa(module, res, res_col, a, a_col);
    }

    fn vec_znx_idft_apply_consume_default<D: Data>(_module: &Module<BE>, _a: VecZnxDft<D, BE>) -> VecZnxBig<D, BE>
    where
        Module<BE>: NttIfmaModuleHandle,
        BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>,
        VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
    {
        // No IFMA-specific consume variant exists yet (requires Primes40 CRT
        // compaction).  Backends that need it should override this method.
        todo!("ntt_ifma_vec_znx_idft_apply_consume: not yet implemented for Primes40")
    }

    fn vec_znx_dft_add_into_default<R, A, D>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaAdd + NttIfmaCopy + NttIfmaZero,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        D: VecZnxDftToRef<BE>,
    {
        ntt_ifma_default_vec_znx_dft_add_into(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_add_scaled_assign_default<R, A>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        a_scale: i64,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaAddInplace,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt_ifma_default_vec_znx_dft_add_scaled_assign(res, res_col, a, a_col, a_scale);
    }

    fn vec_znx_dft_add_assign_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaAddInplace,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt_ifma_default_vec_znx_dft_add_assign(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_default<R, A, D>(
        _module: &Module<BE>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &D,
        b_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaSub + NttIfmaNegate + NttIfmaCopy + NttIfmaZero,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
        D: VecZnxDftToRef<BE>,
    {
        ntt_ifma_default_vec_znx_dft_sub(res, res_col, a, a_col, b, b_col);
    }

    fn vec_znx_dft_sub_inplace_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaSubInplace,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt_ifma_default_vec_znx_dft_sub_inplace(res, res_col, a, a_col);
    }

    fn vec_znx_dft_sub_negate_inplace_default<R, A>(_module: &Module<BE>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaSubNegateInplace + NttIfmaNegateInplace,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt_ifma_default_vec_znx_dft_sub_negate_inplace(res, res_col, a, a_col);
    }

    fn vec_znx_dft_copy_default<R, A>(
        _module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaCopy + NttIfmaZero,
        R: VecZnxDftToMut<BE>,
        A: VecZnxDftToRef<BE>,
    {
        ntt_ifma_default_vec_znx_dft_copy(step, offset, res, res_col, a, a_col);
    }

    fn vec_znx_dft_zero_default<R>(_module: &Module<BE>, res: &mut R, res_col: usize)
    where
        BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaZero,
        R: VecZnxDftToMut<BE>,
    {
        ntt_ifma_default_vec_znx_dft_zero(res, res_col);
    }
}

impl<BE: Backend> NTTIfmaVecZnxDftDefaults<BE> for BE {}
