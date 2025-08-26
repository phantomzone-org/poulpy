use crate::{
    layouts::{VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero},
    reference::znx::znx_add_i64_ref,
};

pub fn vec_znx_add_ref<R, A, C>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    C: VecZnxToRef,
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
            znx_add_i64_ref(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            res.zero_at(res_col, j);
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            znx_add_i64_ref(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            res.at_mut(res_col, j).copy_from_slice(b.at(a_col, j));
        }

        for j in cpy_size..res_size {
            res.zero_at(res_col, j);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub fn vec_znx_add_avx<R, A, C>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    C: VecZnxToRef,
{
    use crate::reference::znx::znx_add_i64_avx;

    let a: VecZnx<&[u8]> = a.to_ref();
    let b: VecZnx<&[u8]> = b.to_ref();
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
        assert_ne!(a.as_ptr(), b.as_ptr());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            znx_add_i64_avx(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            res.zero_at(res_col, j);
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            znx_add_i64_avx(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            res.at_mut(res_col, j).copy_from_slice(b.at(a_col, j));
        }

        for j in cpy_size..res_size {
            res.zero_at(res_col, j);
        }
    }
}
