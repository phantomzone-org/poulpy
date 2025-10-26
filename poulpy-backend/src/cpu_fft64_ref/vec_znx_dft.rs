use poulpy_hal::{
    layouts::{
        Backend, Data, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef,
        VecZnxToRef,
    },
    oep::{
        VecZnxDftAddImpl, VecZnxDftAddInplaceImpl, VecZnxDftAddScaledInplaceImpl, VecZnxDftAllocBytesImpl, VecZnxDftAllocImpl,
        VecZnxDftApplyImpl, VecZnxDftCopyImpl, VecZnxDftFromBytesImpl, VecZnxDftSubImpl, VecZnxDftSubInplaceImpl,
        VecZnxDftSubNegateInplaceImpl, VecZnxDftZeroImpl, VecZnxIdftApplyConsumeImpl, VecZnxIdftApplyImpl,
        VecZnxIdftApplyTmpAImpl, VecZnxIdftApplyTmpBytesImpl,
    },
    reference::fft64::vec_znx_dft::{
        vec_znx_dft_add, vec_znx_dft_add_inplace, vec_znx_dft_add_scaled_inplace, vec_znx_dft_apply, vec_znx_dft_copy,
        vec_znx_dft_sub, vec_znx_dft_sub_inplace, vec_znx_dft_sub_negate_inplace, vec_znx_dft_zero, vec_znx_idft_apply,
        vec_znx_idft_apply_consume, vec_znx_idft_apply_tmpa,
    },
};

use crate::cpu_fft64_ref::{FFT64Ref, module::FFT64ModuleHandle};

unsafe impl VecZnxDftFromBytesImpl<Self> for FFT64Ref {
    fn vec_znx_dft_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<Self> {
        VecZnxDft::<Vec<u8>, Self>::from_bytes(n, cols, size, bytes)
    }
}

unsafe impl VecZnxDftAllocBytesImpl<Self> for FFT64Ref {
    fn vec_znx_dft_bytes_of_impl(n: usize, cols: usize, size: usize) -> usize {
        Self::layout_prep_word_count() * n * cols * size * size_of::<<FFT64Ref as Backend>::ScalarPrep>()
    }
}

unsafe impl VecZnxDftAllocImpl<Self> for FFT64Ref {
    fn vec_znx_dft_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxDftOwned<Self> {
        VecZnxDftOwned::alloc(n, cols, size)
    }
}

unsafe impl VecZnxIdftApplyTmpBytesImpl<Self> for FFT64Ref {
    fn vec_znx_idft_apply_tmp_bytes_impl(_module: &Module<Self>) -> usize {
        0
    }
}

unsafe impl VecZnxIdftApplyImpl<Self> for FFT64Ref {
    fn vec_znx_idft_apply_impl<R, A>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        _scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxBigToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        vec_znx_idft_apply(module.get_ifft_table(), res, res_col, a, a_col);
    }
}

unsafe impl VecZnxIdftApplyTmpAImpl<Self> for FFT64Ref {
    fn vec_znx_idft_apply_tmpa_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<Self>,
        A: VecZnxDftToMut<Self>,
    {
        vec_znx_idft_apply_tmpa(module.get_ifft_table(), res, res_col, a, a_col);
    }
}

unsafe impl VecZnxIdftApplyConsumeImpl<Self> for FFT64Ref {
    fn vec_znx_idft_apply_consume_impl<D: Data>(module: &Module<Self>, res: VecZnxDft<D, FFT64Ref>) -> VecZnxBig<D, FFT64Ref>
    where
        VecZnxDft<D, FFT64Ref>: VecZnxDftToMut<Self>,
    {
        vec_znx_idft_apply_consume(module.get_ifft_table(), res)
    }
}

unsafe impl VecZnxDftApplyImpl<Self> for FFT64Ref {
    fn vec_znx_dft_apply_impl<R, A>(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxToRef,
    {
        vec_znx_dft_apply(module.get_fft_table(), step, offset, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftAddImpl<Self> for FFT64Ref {
    fn vec_znx_dft_add_impl<R, A, B>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        B: VecZnxDftToRef<Self>,
    {
        vec_znx_dft_add(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxDftAddScaledInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_dft_add_scaled_inplace_impl<R, A>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        a_scale: i64,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        vec_znx_dft_add_scaled_inplace(res, res_col, a, a_col, a_scale);
    }
}

unsafe impl VecZnxDftAddInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_dft_add_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        vec_znx_dft_add_inplace(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftSubImpl<Self> for FFT64Ref {
    fn vec_znx_dft_sub_impl<R, A, B>(
        _module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
        B: VecZnxDftToRef<Self>,
    {
        vec_znx_dft_sub(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxDftSubInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_dft_sub_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        vec_znx_dft_sub_inplace(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftSubNegateInplaceImpl<Self> for FFT64Ref {
    fn vec_znx_dft_sub_negate_inplace_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        vec_znx_dft_sub_negate_inplace(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftCopyImpl<Self> for FFT64Ref {
    fn vec_znx_dft_copy_impl<R, A>(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxDftToMut<Self>,
        A: VecZnxDftToRef<Self>,
    {
        vec_znx_dft_copy(step, offset, res, res_col, a, a_col);
    }
}

unsafe impl VecZnxDftZeroImpl<Self> for FFT64Ref {
    fn vec_znx_dft_zero_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxDftToMut<Self>,
    {
        vec_znx_dft_zero(res, res_col);
    }
}
