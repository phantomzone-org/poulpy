use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut};

pub(crate) fn vec_znx_copy<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    assert_eq!(res.n(), a.n());

    let min_size = res.size().min(a.size());
    for j in 0..min_size {
        res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
    }
    for j in min_size..res.size() {
        res.at_mut(res_col, j).fill(0);
    }
}

pub(crate) fn vec_znx_sub_inplace<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    assert_eq!(res.n(), a.n());

    for j in 0..a.size().min(res.size()) {
        let res_j: &mut [i64] = cast_slice_mut(res.at_mut(res_col, j));
        let a_j: &[i64] = cast_slice(a.at(a_col, j));
        for (res_i, a_i) in res_j.iter_mut().zip(a_j.iter()) {
            *res_i -= *a_i;
        }
    }
}
