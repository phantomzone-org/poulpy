use crate::{
    layouts::{Backend, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::fft64::reim::{ReimCopy, ReimZero},
};

pub fn vec_znx_dft_copy<R, A, BE>(step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = f64> + ReimCopy + ReimZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let steps: usize = a.size().div_ceil(step);
    let min_steps: usize = res.size().min(steps);

    (0..min_steps).for_each(|j| {
        let limb: usize = offset + j * step;
        if limb < a.size() {
            BE::reim_copy(res.at_mut(res_col, j), a.at(a_col, limb));
        }
    });
    (min_steps..res.size()).for_each(|j| {
        BE::reim_zero(res.at_mut(res_col, j));
    })
}
