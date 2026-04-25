use crate::layouts::{VecZnxToMut, VecZnxToRef};
use crate::{
    layouts::{ScalarZnx, VecZnx, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxSub, ZnxSubAssign, ZnxZero},
};

pub fn vec_znx_sub_scalar<R, B, ZNXARI>(
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
    ZNXARI: ZnxSub + ZnxZero,
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
            ZNXARI::znx_sub(res.at_mut(res_col, j), b.at(b_col, j), a.at(a_col, 0));
        } else {
            res.at_mut(res_col, j).copy_from_slice(b.at(b_col, j));
        }
    }

    for j in min_size..res.size() {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_sub_scalar_inplace<R, ZNXARI>(res: &mut R, res_col: usize, res_limb: usize, a: &ScalarZnx<&[u8]>, a_col: usize)
where
    R: VecZnxToMut,
    ZNXARI: ZnxSubInplace,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert!(res_limb < res.size());
    }

    ZNXARI::znx_sub_assign(res.at_mut(res_col, res_limb), a.at(a_col, 0));
}
