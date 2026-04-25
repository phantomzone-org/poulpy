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

pub(crate) fn vec_znx_sub<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    B: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n());
        assert_eq!(res.n(), b.n());
    }

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    for j in 0..a_size.min(b_size).min(res_size) {
        let res_j: &mut [i64] = cast_slice_mut(res.at_mut(res_col, j));
        let a_j: &[i64] = cast_slice(a.at(a_col, j));
        let b_j: &[i64] = cast_slice(b.at(b_col, j));
        for ((res_i, a_i), b_i) in res_j.iter_mut().zip(a_j.iter()).zip(b_j.iter()) {
            *res_i = *a_i - *b_i;
        }
    }

    if a_size > b_size {
        for j in b_size..a_size.min(res_size) {
            res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
        }
    } else {
        for j in a_size..b_size.min(res_size) {
            let res_j: &mut [i64] = cast_slice_mut(res.at_mut(res_col, j));
            let b_j: &[i64] = cast_slice(b.at(b_col, j));
            for (res_i, b_i) in res_j.iter_mut().zip(b_j.iter()) {
                *res_i = -*b_i;
            }
        }
    }

    for j in a_size.max(b_size)..res_size {
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

pub(crate) fn vec_znx_switch_ring<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    if res.n() == a.n() {
        vec_znx_copy(&mut res, res_col, &a, a_col);
        return;
    }

    #[cfg(debug_assertions)]
    {
        assert!(res.n().is_power_of_two());
        assert!(a.n().is_power_of_two());
        assert!(res.n().max(a.n()).is_multiple_of(res.n().min(a.n())));
    }

    let (gap_in, gap_out) = if a.n() > res.n() {
        (a.n() / res.n(), 1)
    } else {
        for j in 0..res.size() {
            res.at_mut(res_col, j).fill(0);
        }
        (1, res.n() / a.n())
    };

    for j in 0..a.size().min(res.size()) {
        let res_j: &mut [i64] = cast_slice_mut(res.at_mut(res_col, j));
        let a_j: &[i64] = cast_slice(a.at(a_col, j));
        res_j
            .iter_mut()
            .step_by(gap_out)
            .zip(a_j.iter().step_by(gap_in))
            .for_each(|(out, input)| *out = *input);
    }

    if a.n() > res.n() {
        for j in a.size().min(res.size())..res.size() {
            res.at_mut(res_col, j).fill(0);
        }
    }
}
