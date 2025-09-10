use crate::{
    layouts::{Backend, VecZnxDftToMut, ZnxViewMut},
    reference::reim::ReimArithmetic,
};

pub fn vec_znx_dft_zero<R, BE, REIMARI>(res: &mut R)
where
    R: VecZnxDftToMut<BE>,
    BE: Backend<ScalarPrep = f64>,
    REIMARI: ReimArithmetic,
{
    REIMARI::reim_zero(res.to_mut().raw_mut());
}
