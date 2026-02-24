//! Fourier-domain ring element vector operations for [`FFT64Ref`](crate::FFT64Ref).
//!
//! Implements the `VecZnxDft*` and `VecZnxIdft*` OEP traits. `VecZnxDft` stores
//! ring element vectors in the frequency domain (`ScalarPrep = f64`), where
//! polynomial multiplication reduces to coefficient-wise `f64` multiplication.
//!
//! Operations include:
//!
//! - **Allocation**: byte-size calculation, heap allocation, construction from raw bytes.
//! - **Forward DFT**: integer-domain `VecZnx` → frequency-domain `VecZnxDft`, with
//!   configurable step/offset for partial transforms.
//! - **Inverse DFT**: `VecZnxDft` → `VecZnxBig` (large-coefficient), with variants that
//!   consume, borrow, or use the input as temporary storage.
//! - **Frequency-domain arithmetic**: add, sub, negate, scaled-add, copy, zero.
//!
//! The IDFT does not require scratch space for this backend (`idft_apply_tmp_bytes = 0`).

use poulpy_hal::{
    layouts::{Data, Module, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef},
    oep::{
        VecZnxDftAddImpl, VecZnxDftAddInplaceImpl, VecZnxDftAddScaledInplaceImpl, VecZnxDftApplyImpl, VecZnxDftCopyImpl,
        VecZnxDftSubImpl, VecZnxDftSubInplaceImpl, VecZnxDftSubNegateInplaceImpl, VecZnxDftZeroImpl, VecZnxIdftApplyConsumeImpl,
        VecZnxIdftApplyImpl, VecZnxIdftApplyTmpAImpl, VecZnxIdftApplyTmpBytesImpl,
    },
    reference::fft64::vec_znx_dft::{
        vec_znx_dft_add, vec_znx_dft_add_inplace, vec_znx_dft_add_scaled_inplace, vec_znx_dft_apply, vec_znx_dft_copy,
        vec_znx_dft_sub, vec_znx_dft_sub_inplace, vec_znx_dft_sub_negate_inplace, vec_znx_dft_zero, vec_znx_idft_apply,
        vec_znx_idft_apply_consume, vec_znx_idft_apply_tmpa,
    },
};

use super::{FFT64Ref, module::FFT64ModuleHandle};

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
