use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero},
    reference::znx::znx_rotate_i64,
};

pub fn vec_znx_rotate_ref<R, A>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let min_size: usize = res_size.min(a_size);

    for j in 0..min_size {
        znx_rotate_i64(p, res.at_mut(res_col, j), a.at(a_col, j))
    }

    for j in min_size..res_size {
        res.zero_at(res_col, j);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub fn vec_znx_rotate_avx<R, A>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    use crate::reference::znx::znx_rotate_i64_avx;

    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let min_size: usize = res_size.min(a_size);

    for j in 0..min_size {
        znx_rotate_i64_avx(p, res.at_mut(res_col, j), a.at(a_col, j))
    }

    for j in min_size..res_size {
        res.zero_at(res_col, j);
    }
}
