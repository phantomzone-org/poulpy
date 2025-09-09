use crate::{
    layouts::{Backend, VecZnxDftToMut, ZnxViewMut},
    reference::reim::reim_zero_ref,
};

pub fn vec_znx_dft_zero_ref<R, BE>(res: &mut R)
where
    R: VecZnxDftToMut<BE>,
    BE: Backend<ScalarPrep = f64>,
{
    reim_zero_ref(res.to_mut().raw_mut());
}
