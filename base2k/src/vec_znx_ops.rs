use crate::ffi::vec_znx;
use crate::{
    Backend, Module, Scratch, VecZnx, VecZnxOwned, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero,
};
use itertools::izip;
use std::cmp::min;

pub trait VecZnxAlloc {
    /// Allocates a new [VecZnx].
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials.
    /// * `size`: the number small polynomials per column.
    fn new_vec_znx(&self, cols: usize, size: usize) -> VecZnxOwned;

    /// Instantiates a new [VecZnx] from a slice of bytes.
    /// The returned [VecZnx] takes ownership of the slice of bytes.
    ///
    /// # Arguments
    ///
    /// * `cols`: the number of polynomials.
    /// * `size`: the number small polynomials per column.
    ///
    /// # Panic
    /// Requires the slice of bytes to be equal to [VecZnxOps::bytes_of_vec_znx].
    fn new_vec_znx_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxOwned;

    /// Returns the number of bytes necessary to allocate
    /// a new [VecZnx] through [VecZnxOps::new_vec_znx_from_bytes]
    /// or [VecZnxOps::new_vec_znx_from_bytes_borrow].
    fn bytes_of_vec_znx(&self, cols: usize, size: usize) -> usize;
}

pub trait VecZnxOps {
    /// Normalizes the selected column of `a` and stores the result into the selected column of `res`.
    fn vec_znx_normalize<R, A>(&self, log_base2k: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    /// Normalizes the selected column of `a`.
    fn vec_znx_normalize_inplace<A>(&self, log_base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch)
    where
        A: VecZnxToMut;

    /// Adds the selected column of `a` to the selected column of `b` and writes the result on the selected column of `res`.
    fn vec_znx_add<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef;

    /// Adds the selected column of `a` to the selected column of `b` and writes the result on the selected column of `res`.
    fn vec_znx_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    /// Subtracts the selected column of `b` from the selected column of `a` and writes the result on the selected column of `res`.
    fn vec_znx_sub<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef;

    /// Subtracts the selected column of `a` from the selected column of `res` inplace.
    ///
    /// res[res_col] -= a[a_col]
    fn vec_znx_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    /// Subtracts the selected column of `res` from the selected column of `a` and inplace mutates `res`
    ///
    /// res[res_col] = a[a_col] - res[res_col]
    fn vec_znx_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    // Negates the selected column of `a` and stores the result in `res_col` of `res`.
    fn vec_znx_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    /// Negates the selected column of `a`.
    fn vec_znx_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;

    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;

    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut;

    /// Splits the selected columns of `b` into subrings and copies them them into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split<R, A>(&self, res: &mut Vec<R>, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    /// Merges the subrings of the selected column of `a` into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge<R, A>(&self, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;

    fn switch_degree<R, A>(&self, r: &mut R, col_b: usize, a: &A, col_a: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef;
}

pub trait VecZnxScratch {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

impl<B: Backend> VecZnxAlloc for Module<B> {
    fn new_vec_znx(&self, cols: usize, size: usize) -> VecZnxOwned {
        VecZnxOwned::new::<i64>(self.n(), cols, size)
    }

    fn bytes_of_vec_znx(&self, cols: usize, size: usize) -> usize {
        VecZnxOwned::bytes_of::<i64>(self.n(), cols, size)
    }

    fn new_vec_znx_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxOwned {
        VecZnxOwned::new_from_bytes::<i64>(self.n(), cols, size, bytes)
    }
}

impl<BACKEND: Backend> VecZnxOps for Module<BACKEND> {
    fn vec_znx_normalize<R, A>(&self, log_base2k: usize, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }

        let (tmp_bytes, _) = scratch.tmp_scalar_slice(self.vec_znx_normalize_tmp_bytes());

        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                self.ptr,
                log_base2k as u64,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vec_znx_normalize_inplace<A>(&self, log_base2k: usize, a: &mut A, a_col: usize, scratch: &mut Scratch)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
        }

        let (tmp_bytes, _) = scratch.tmp_scalar_slice(self.vec_znx_normalize_tmp_bytes());

        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                self.ptr,
                log_base2k as u64,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                tmp_bytes.as_mut_ptr(),
            );
        }
    }

    fn vec_znx_add<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_add_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_add(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }

    fn vec_znx_sub<R, A, B>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
        B: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(b.n(), self.n());
            assert_eq!(res.n(), self.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, 0),
                b.size() as u64,
                b.sl() as u64,
            )
        }
    }

    fn vec_znx_sub_ab_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_sub_ba_inplace<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }

    fn vec_znx_negate<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_negate(
                self.ptr,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_negate_inplace<A>(&self, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_negate(
                self.ptr,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_rotate<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_rotate(
                self.ptr,
                k,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_rotate_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_rotate(
                self.ptr,
                k,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_automorphism<R, A>(&self, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
            assert_eq!(res.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_automorphism_inplace<A>(&self, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), self.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                self.ptr,
                k,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }

    fn vec_znx_split<R, A>(&self, res: &mut Vec<R>, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();

        let (n_in, n_out) = (a.n(), res[0].to_mut().n());

        let (mut buf, _) = scratch.tmp_vec_znx(self, 1, a.size());

        debug_assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        res[1..].iter_mut().for_each(|bi| {
            debug_assert_eq!(
                bi.to_mut().n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        res.iter_mut().enumerate().for_each(|(i, bi)| {
            if i == 0 {
                self.switch_degree(bi, res_col, &a, a_col);
                self.vec_znx_rotate(-1, &mut buf, 0, &a, a_col);
            } else {
                self.switch_degree(bi, res_col, &mut buf, a_col);
                self.vec_znx_rotate_inplace(-1, &mut buf, a_col);
            }
        })
    }

    fn vec_znx_merge<R, A>(&self, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        let (n_in, n_out) = (res.n(), a[0].to_ref().n());

        debug_assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );
        a[1..].iter().for_each(|ai| {
            debug_assert_eq!(
                ai.to_ref().n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        a.iter().enumerate().for_each(|(_, ai)| {
            self.switch_degree(&mut res, res_col, ai, a_col);
            self.vec_znx_rotate_inplace(-1, &mut res, res_col);
        });

        self.vec_znx_rotate_inplace(a.len() as i64, &mut res, res_col);
    }

    fn switch_degree<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        let (n_in, n_out) = (a.n(), res.n());
        let (gap_in, gap_out): (usize, usize);

        if n_in > n_out {
            (gap_in, gap_out) = (n_in / n_out, 1)
        } else {
            (gap_in, gap_out) = (1, n_out / n_in);
            res.zero();
        }

        let size: usize = min(a.size(), res.size());

        (0..size).for_each(|i| {
            izip!(
                a.at(a_col, i).iter().step_by(gap_in),
                res.at_mut(res_col, i).iter_mut().step_by(gap_out)
            )
            .for_each(|(x_in, x_out)| *x_out = *x_in);
        });
    }
}

impl<B: Backend> VecZnxScratch for Module<B> {
    fn vec_znx_normalize_tmp_bytes(&self) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(self.ptr) as usize }
    }
}
