use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero},
    reference::znx::{znx_automorphism_avx, znx_automorphism_ref},
};

pub fn vec_znx_automorphism_ref<R, A>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        use crate::layouts::ZnxInfos;

        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        znx_automorphism_ref(p, res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        res.zero_at(res_col, j);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_automorphism_avx<R, A>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        use crate::layouts::ZnxInfos;

        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        znx_automorphism_avx(p, res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        res.zero_at(res_col, j);
    }
}
