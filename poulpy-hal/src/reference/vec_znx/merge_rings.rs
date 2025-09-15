use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos},
    reference::{
        vec_znx::{vec_znx_rotate_inplace, vec_znx_switch_ring},
        znx::{ZnxCopy, ZnxRotate, ZnxSwitchRing, ZnxZero},
    },
};

pub fn vec_znx_merge_rings_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_merge_rings<R, A, ZNXARI>(res: &mut R, res_col: usize, a: &[A], a_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxCopy + ZnxSwitchRing + ZnxRotate + ZnxZero,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let (n_out, n_in) = (res.n(), a[0].to_ref().n());

    #[cfg(debug_assertions)]
    {
        assert_eq!(tmp.len(), res.n());

        debug_assert!(
            n_out > n_in,
            "invalid a: output ring degree should be greater"
        );
        a[1..].iter().for_each(|ai| {
            debug_assert_eq!(
                ai.to_ref().n(),
                n_in,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        assert!(n_out.is_multiple_of(n_in));
        assert_eq!(a.len(), n_out / n_in);
    }

    a.iter().for_each(|ai| {
        vec_znx_switch_ring::<_, _, ZNXARI>(&mut res, res_col, ai, a_col);
        vec_znx_rotate_inplace::<_, ZNXARI>(-1, &mut res, res_col, tmp);
    });

    vec_znx_rotate_inplace::<_, ZNXARI>(a.len() as i64, &mut res, res_col, tmp);
}
