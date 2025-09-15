use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxRotate, ZnxSwitchRing, ZnxZero},
};

pub fn vec_znx_split_ring_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_split_ring<R, A, ZNXARI>(res: &mut [R], res_col: usize, a: &A, a_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxSwitchRing + ZnxRotate + ZnxZero,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let a_size = a.size();

    let (n_in, n_out) = (a.n(), res[0].to_mut().n());

    #[cfg(debug_assertions)]
    {
        assert_eq!(tmp.len(), a.n());

        assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );

        res[1..].iter_mut().for_each(|bi| {
            assert_eq!(
                bi.to_mut().n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        assert!(n_in.is_multiple_of(n_out));
        assert_eq!(res.len(), n_in / n_out);
    }

    res.iter_mut().enumerate().for_each(|(i, bi)| {
        let mut bi: VecZnx<&mut [u8]> = bi.to_mut();

        let min_size = bi.size().min(a_size);

        if i == 0 {
            for j in 0..min_size {
                ZNXARI::znx_switch_ring(bi.at_mut(res_col, j), a.at(a_col, j));
            }
        } else {
            for j in 0..min_size {
                ZNXARI::znx_rotate(-(i as i64), tmp, a.at(a_col, j));
                ZNXARI::znx_switch_ring(bi.at_mut(res_col, j), tmp);
            }
        }

        for j in min_size..bi.size() {
            ZNXARI::znx_zero(bi.at_mut(res_col, j));
        }
    })
}
