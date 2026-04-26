use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxCopy, ZnxZero},
};

pub fn vec_znx_copy<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxCopy + ZnxZero,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size = res.size();
    let a_size = a.size();

    let min_size = res_size.min(a_size);

    for j in 0..min_size {
        ZNXARI::znx_copy(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_copy_range<R, A, ZNXARI>(
    res: &mut R,
    res_col: usize,
    res_limb: usize,
    res_offset: usize,
    a: &A,
    a_col: usize,
    a_limb: usize,
    a_offset: usize,
    len: usize,
) where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxCopy,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    assert!(res_offset + len <= res.n());
    assert!(a_offset + len <= a.n());

    ZNXARI::znx_copy(
        &mut res.at_mut(res_col, res_limb)[res_offset..res_offset + len],
        &a.at(a_col, a_limb)[a_offset..a_offset + len],
    );
}
