use poulpy_hal::{
    api::TakeSlice,
    layouts::{Scratch, Zn, ZnToMut, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut},
    oep::{TakeSliceImpl, ZnAddNormalImpl, ZnFillNormalImpl, ZnFillUniformImpl, ZnNormalizeInplaceImpl},
    reference::zn::{zn_add_normal, zn_fill_normal, zn_fill_uniform},
    source::Source,
};

use crate::cpu_spqlios::{FFT64, ffi::zn64};

unsafe impl ZnNormalizeInplaceImpl<Self> for FFT64
where
    Self: TakeSliceImpl<Self>,
{
    fn zn_normalize_inplace_impl<A>(n: usize, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<Self>)
    where
        A: ZnToMut,
    {
        let mut a: Zn<&mut [u8]> = a.to_mut();

        let (tmp_bytes, _) = scratch.take_slice(n * size_of::<i64>());

        unsafe {
            zn64::zn64_normalize_base2k_ref(
                n as u64,
                basek as u64,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }
}

unsafe impl ZnFillUniformImpl<Self> for FFT64 {
    fn zn_fill_uniform_impl<R>(n: usize, basek: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: ZnToMut,
    {
        zn_fill_uniform(n, basek, res, res_col, source);
    }
}

unsafe impl ZnFillNormalImpl<Self> for FFT64 {
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

unsafe impl ZnAddNormalImpl<Self> for FFT64 {
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
