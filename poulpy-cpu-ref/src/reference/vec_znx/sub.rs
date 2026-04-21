use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxCopy, ZnxNegate, ZnxNegateInplace, ZnxSub, ZnxSubInplace, ZnxSubNegateInplace, ZnxZero},
};

pub fn vec_znx_sub<R, A, B, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    B: VecZnxToRef,
    ZNXARI: ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            ZNXARI::znx_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            ZNXARI::znx_negate(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            ZNXARI::znx_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            ZNXARI::znx_copy(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            ZNXARI::znx_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_sub_inplace<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxSubInplace,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        ZNXARI::znx_sub_inplace(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn vec_znx_sub_negate_inplace<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxSubNegateInplace + ZnxNegateInplace,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        ZNXARI::znx_sub_negate_inplace(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in sum_size..res_size {
        ZNXARI::znx_negate_inplace(res.at_mut(res_col, j));
    }
}
