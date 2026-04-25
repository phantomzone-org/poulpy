use crate::{
    layouts::{ScalarZnx, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxAdd, ZnxAddInplace, ZnxCopy, ZnxZero},
};

pub fn vec_znx_add_scalar_into<R, B, ZNXARI>(
    res: &mut R,
    res_col: usize,
    a: &ScalarZnx<&[u8]>,
    a_col: usize,
    b: &B,
    b_col: usize,
    b_limb: usize,
) where
    R: VecZnxToMut,
    B: VecZnxToRef,
    ZNXARI: ZnxAdd + ZnxCopy + ZnxZero,
{
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let min_size: usize = b.size().min(res.size());

    #[cfg(debug_assertions)]
    {
        assert!(b_limb < min_size, "b_limb: {b_limb} > min_size: {min_size}");
    }

    for j in 0..min_size {
        if j == b_limb {
            ZNXARI::znx_add(res.at_mut(res_col, j), a.at(a_col, 0), b.at(b_col, j));
        } else {
            ZNXARI::znx_copy(res.at_mut(res_col, j), b.at(b_col, j));
        }
    }

    for j in min_size..res.size() {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_add_scalar_assign<R, ZNXARI>(res: &mut R, res_col: usize, res_limb: usize, a: &ScalarZnx<&[u8]>, a_col: usize)
where
    R: VecZnxToMut,
    ZNXARI: ZnxAddInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(res_limb < res.size());
    }

    ZNXARI::znx_add_assign(res.at_mut(res_col, res_limb), a.at(a_col, 0));
}
