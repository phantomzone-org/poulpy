use poulpy_hal::{
    api::TakeSlice,
    layouts::{Scratch, ZnToMut},
    oep::{TakeSliceImpl, ZnAddNormalImpl, ZnFillNormalImpl, ZnFillUniformImpl, ZnNormalizeInplaceImpl, ZnNormalizeTmpBytesImpl},
    reference::zn::{zn_add_normal, zn_fill_normal, zn_fill_uniform, zn_normalize_inplace, zn_normalize_tmp_bytes},
    source::Source,
};

use crate::cpu_fft64_ref::FFT64Ref;

unsafe impl ZnNormalizeTmpBytesImpl<Self> for FFT64Ref {
    fn zn_normalize_tmp_bytes_impl(n: usize) -> usize {
        zn_normalize_tmp_bytes(n)
    }
}

unsafe impl ZnNormalizeInplaceImpl<Self> for FFT64Ref
where
    Self: TakeSliceImpl<Self>,
{
    fn zn_normalize_inplace_impl<R>(n: usize, basek: usize, res: &mut R, res_col: usize, scratch: &mut Scratch<Self>)
    where
        R: ZnToMut,
    {
        let (carry, _) = scratch.take_slice(n);
        zn_normalize_inplace::<R, FFT64Ref>(n, basek, res, res_col, carry);
    }
}

unsafe impl ZnFillUniformImpl<Self> for FFT64Ref {
    fn zn_fill_uniform_impl<R>(n: usize, basek: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: ZnToMut,
    {
        zn_fill_uniform(n, basek, res, res_col, source);
    }
}

unsafe impl ZnFillNormalImpl<Self> for FFT64Ref {
    #[allow(clippy::too_many_arguments)]
    fn zn_fill_normal_impl<R>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut,
    {
        zn_fill_normal(n, basek, res, res_col, k, source, sigma, bound);
    }
}

unsafe impl ZnAddNormalImpl<Self> for FFT64Ref {
    #[allow(clippy::too_many_arguments)]
    fn zn_add_normal_impl<R>(
        n: usize,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: ZnToMut,
    {
        zn_add_normal(n, basek, res, res_col, k, source, sigma, bound);
    }
}
