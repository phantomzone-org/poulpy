use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero},
    reference::znx::{
        znx_normalize_beg_ref, znx_normalize_carry_only_ref, znx_normalize_end_ref, znx_normalize_inplace_beg_ref,
        znx_normalize_inplace_end_ref, znx_normalize_inplace_mid_ref, znx_normalize_mid_ref,
    },
};

pub fn vec_znx_normalize_ref<R, A>(basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(carry.len(), res.n());
    }

    let res_size: usize = res.size();
    let a_size = a.size();

    if a_size > res_size {
        for j in (res_size..a_size).rev() {
            znx_normalize_carry_only_ref(basek, a.at(a_col, j), carry);
        }

        for j in (1..res_size).rev() {
            znx_normalize_mid_ref(basek, res.at_mut(res_col, j), a.at(a_col, j), carry);
        }

        znx_normalize_end_ref(basek, res.at_mut(res_col, 0), a.at(a_col, 0), carry);
    } else {
        for j in (0..a_size).rev() {
            if j == a_size - 1 {
                znx_normalize_beg_ref(basek, res.at_mut(res_col, j), a.at(a_col, j), carry);
            } else if j == 0 {
                znx_normalize_end_ref(basek, res.at_mut(res_col, 0), a.at(a_col, 0), carry);
            } else {
                znx_normalize_mid_ref(basek, res.at_mut(res_col, j), a.at(a_col, j), carry);
            }
        }

        for j in a_size..res_size {
            res.zero_at(res_col, j);
        }
    }
}

pub fn vec_znx_normalize_inplace_ref<R: VecZnxToMut>(basek: usize, res: &mut R, res_col: usize, carry: &mut [i64]) {
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(carry.len(), res.n());
    }

    let res_size: usize = res.size();

    for j in (0..res_size).rev() {
        if j == res_size - 1 {
            znx_normalize_inplace_beg_ref(basek, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            znx_normalize_inplace_end_ref(basek, res.at_mut(res_col, j), carry);
        } else {
            znx_normalize_inplace_mid_ref(basek, res.at_mut(res_col, j), carry);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_normalize_avx<R, A>(basek: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, carry: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(carry.len(), res.n());
    }

    let res_size: usize = res.size();
    let a_size = a.size();

    use crate::reference::znx::{
        znx_normalize_beg_avx, znx_normalize_carry_only_avx, znx_normalize_end_avx, znx_normalize_mid_avx,
    };

    if a_size > res_size {
        for j in (res_size..a_size).rev() {
            znx_normalize_carry_only_avx(basek, a.at(a_col, j), carry);
        }

        for j in (1..res_size).rev() {
            znx_normalize_mid_avx(basek, res.at_mut(res_col, j), a.at(a_col, j), carry);
        }

        znx_normalize_end_avx(basek, res.at_mut(res_col, 0), a.at(a_col, 0), carry);
    } else {
        for j in (0..a_size).rev() {
            if j == a_size - 1 {
                znx_normalize_beg_avx(basek, res.at_mut(res_col, j), a.at(a_col, j), carry);
            } else if j == 0 {
                znx_normalize_end_avx(basek, res.at_mut(res_col, 0), a.at(a_col, 0), carry);
            } else {
                znx_normalize_mid_avx(basek, res.at_mut(res_col, j), a.at(a_col, j), carry);
            }
        }

        for j in a_size..res_size {
            res.zero_at(res_col, j);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_normalize_inplace_avx<R: VecZnxToMut>(basek: usize, res: &mut R, res_col: usize, carry: &mut [i64]) {
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(carry.len(), res.n());
    }

    let res_size: usize = res.size();

    use crate::reference::znx::{znx_normalize_inplace_beg_avx, znx_normalize_inplace_end_avx, znx_normalize_inplace_mid_avx};

    for j in (0..res_size).rev() {
        if j == res_size - 1 {
            znx_normalize_inplace_beg_avx(basek, res.at_mut(res_col, j), carry);
        } else if j == 0 {
            znx_normalize_inplace_end_avx(basek, res.at_mut(res_col, j), carry);
        } else {
            znx_normalize_inplace_mid_avx(basek, res.at_mut(res_col, j), carry);
        }
    }
}
