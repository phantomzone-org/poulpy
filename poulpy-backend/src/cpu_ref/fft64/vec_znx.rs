use poulpy_hal::{
    api::{TakeSlice, VecZnxNormalizeTmpBytes},
    layouts::{Module, Scratch, VecZnxToMut, VecZnxToRef},
    oep::{
        TakeSliceImpl, VecZnxAddImpl, VecZnxAutomorphismImpl, VecZnxNormalizeImpl, VecZnxNormalizeInplaceImpl,
        VecZnxNormalizeTmpBytesImpl,
    },
    reference::{
        vec_znx::{vec_znx_add, vec_znx_automorphism, vec_znx_normalize, vec_znx_normalize_inplace, vec_znx_normalize_tmp_bytes},
        znx::{ZnxArithmeticRef, ZnxNormalizeRef},
    },
};

use crate::cpu_ref::fft64::FFT64;

unsafe impl VecZnxAddImpl<Self> for FFT64 {
    fn vec_znx_add_impl<R, A, C>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        vec_znx_add::<_, _, _, ZnxArithmeticRef>(res, res_col, a, a_col, b, b_col);
    }
}

unsafe impl VecZnxNormalizeTmpBytesImpl<Self> for FFT64 {
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_normalize_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxNormalizeInplaceImpl<Self> for FFT64
where
    Self: TakeSliceImpl<Self> + VecZnxNormalizeTmpBytesImpl<Self>,
{
    fn vec_znx_normalize_inplace_impl<R>(
        module: &Module<Self>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
    {
        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes() / size_of::<i64>());
        vec_znx_normalize_inplace::<_, ZnxNormalizeRef>(basek, res, res_col, tmp_bytes);
    }
}

unsafe impl VecZnxNormalizeImpl<Self> for FFT64
where
    Self: TakeSliceImpl<Self> + VecZnxNormalizeTmpBytesImpl<Self>,
{
    fn vec_znx_normalize_impl<R, A>(
        module: &Module<Self>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes() / size_of::<i64>());
        vec_znx_normalize::<_, _, ZnxArithmeticRef, ZnxNormalizeRef>(basek, res, res_col, a, a_col, carry);
    }
}

unsafe impl VecZnxAutomorphismImpl<Self> for FFT64 {
    fn vec_znx_automorphism_impl<R, A>(_module: &Module<Self>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_automorphism::<_, _, ZnxArithmeticRef>(p, res, res_col, a, a_col);
    }
}
