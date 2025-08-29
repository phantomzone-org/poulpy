use poulpy_hal::{
    api::{TakeSlice, VecZnxNormalizeTmpBytes},
    layouts::{Module, Scratch, VecZnxToMut, VecZnxToRef},
    oep::{
        TakeSliceImpl, VecZnxAddImpl, VecZnxAutomorphismImpl, VecZnxNormalizeImpl, VecZnxNormalizeInplaceImpl,
        VecZnxNormalizeTmpBytesImpl,
    },
    reference::vec_znx::{
        vec_znx_add_avx, vec_znx_add_ref, vec_znx_automorphism_avx, vec_znx_automorphism_ref, vec_znx_normalize_avx,
        vec_znx_normalize_inplace_avx, vec_znx_normalize_inplace_ref, vec_znx_normalize_ref,
    },
};

use crate::cpu_ref::{
    ffi::{module::module_info_t, vec_znx},
    fft64::FFT64,
};

unsafe impl VecZnxAddImpl<Self> for FFT64 {
    fn vec_znx_add_impl<R, A, C>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        C: VecZnxToRef,
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                vec_znx_add_avx(res, res_col, a, a_col, b, b_col);
            }
        } else {
            vec_znx_add_ref(res, res_col, a, a_col, b, b_col);
        }
    }
}

unsafe impl VecZnxNormalizeTmpBytesImpl<Self> for FFT64 {
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<Self>) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(module.ptr() as *const module_info_t) as usize }
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

        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                vec_znx_normalize_inplace_avx(basek, res, res_col, tmp_bytes);
            }
        } else {
            vec_znx_normalize_inplace_ref(basek, res, res_col, tmp_bytes);
        }
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
        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes() / size_of::<i64>());

        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                vec_znx_normalize_avx(basek, res, res_col, a, a_col, tmp_bytes);
            }
        } else {
            vec_znx_normalize_ref(basek, res, res_col, a, a_col, tmp_bytes);
        }
    }
}

unsafe impl VecZnxAutomorphismImpl<Self> for FFT64 {
    fn vec_znx_automorphism_impl<R, A>(_module: &Module<Self>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                vec_znx_automorphism_avx(p, res, res_col, a, a_col);
            }
        } else {
            vec_znx_automorphism_ref(p, res, res_col, a, a_col);
        }
    }
}
