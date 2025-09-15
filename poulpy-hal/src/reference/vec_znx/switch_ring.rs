use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::{
        vec_znx::vec_znx_copy,
        znx::{ZnxCopy, ZnxSwitchRing, ZnxZero},
    },
};

/// Maps between negacyclic rings by changing the polynomial degree.
/// Up:  Z[X]/(X^N+1) -> Z[X]/(X^{2^d N}+1) via X ↦ X^{2^d}
/// Down: Z[X]/(X^N+1) -> Z[X]/(X^{N/2^d}+1) by folding indices.
pub fn vec_znx_switch_ring<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxCopy + ZnxSwitchRing + ZnxZero,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let (n_in, n_out) = (a.n(), res.n());

    if n_in == n_out {
        vec_znx_copy::<_, _, ZNXARI>(&mut res, res_col, &a, a_col);
        return;
    }

    let min_size: usize = a.size().min(res.size());

    for j in 0..min_size {
        ZNXARI::znx_switch_ring(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        ZNXARI::znx_zero(res.at_mut(res_col, j));
    }
}
