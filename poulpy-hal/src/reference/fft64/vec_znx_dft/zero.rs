use crate::{
    layouts::{Backend, VecZnxDftToMut, ZnxViewMut},
    reference::fft64::reim::ReimZero,
};

pub fn vec_znx_dft_zero<R, BE>(res: &mut R)
where
    R: VecZnxDftToMut<BE>,
    BE: Backend<ScalarPrep = f64> + ReimZero,
{
    BE::reim_zero(res.to_mut().raw_mut());
}
