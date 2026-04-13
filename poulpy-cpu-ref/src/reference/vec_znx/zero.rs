use crate::{
    layouts::{VecZnx, VecZnxToMut, ZnxInfos, ZnxViewMut},
    reference::znx::ZnxZero,
};

pub fn vec_znx_zero<R, ZNXARI>(res: &mut R, res_col: usize)
where
    R: VecZnxToMut,
    ZNXARI: ZnxZero,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let res_size = res.size();
    for j in 0..res_size {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}
