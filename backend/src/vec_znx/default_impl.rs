use crate::{
    Backend, Module, ScalarZnxToRef, Scratch, VecZnx, VecZnxAddImpl, VecZnxAddInplaceImpl, VecZnxAddScalarInplaceImpl,
    VecZnxAllocBytesImpl, VecZnxAllocImpl, VecZnxAutomorphismImpl, VecZnxAutomorphismInplaceImpl, VecZnxCopyImpl,
    VecZnxFromBytesImpl, VecZnxMergeImpl, VecZnxNegateImpl, VecZnxNegateInplaceImpl, VecZnxNormalizeImpl,
    VecZnxNormalizeInplaceImpl, VecZnxNormalizeTmpBytesImpl, VecZnxOwned, VecZnxRotateImpl, VecZnxRotateInplaceImpl,
    VecZnxShiftInplaceImpl, VecZnxSplitImpl, VecZnxStdImpl, VecZnxSubABInplaceImpl, VecZnxSubBAInplaceImpl, VecZnxSubImpl,
    VecZnxSubScalarInplaceImpl, VecZnxSwithcDegreeImpl, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero,
    ffi::{module::module_info_t, vec_znx},
    vec_znx::{
        layout::{VecZnxToMut, VecZnxToRef},
        traits::{VecZnxCopy, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxSwithcDegree},
    },
};
use crate::{
    Decoding, VecZnxAddDistF64, VecZnxAddDistF64Impl, VecZnxAddNormalImpl, VecZnxFillDistF64, VecZnxFillDistF64Impl,
    VecZnxFillNormalImpl, VecZnxFillUniformImpl,
};
use itertools::izip;
use rand_distr::Normal;
use rug::Float;
use rug::float::Round;
use rug::ops::{AddAssignRound, DivAssignRound, SubAssignRound};
use sampling::source::Source;

unsafe impl<B: Backend> VecZnxAllocImpl<B> for B {
    fn vec_znx_alloc_impl(module: &Module<B>, cols: usize, size: usize) -> VecZnxOwned {
        VecZnxOwned::new::<i64>(module.n(), cols, size)
    }
}

unsafe impl<B: Backend> VecZnxFromBytesImpl<B> for B {
    fn vec_znx_from_bytes_impl(module: &Module<B>, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxOwned {
        VecZnxOwned::new_from_bytes::<i64>(module.n(), cols, size, bytes)
    }
}

unsafe impl<B: Backend> VecZnxAllocBytesImpl<B> for B {
    fn vec_znx_alloc_bytes_impl(module: &Module<B>, cols: usize, size: usize) -> usize {
        VecZnxOwned::bytes_of::<i64>(module.n(), cols, size)
    }
}

unsafe impl<B: Backend> VecZnxNormalizeTmpBytesImpl<B> for B {
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<B>) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(module.ptr() as *const module_info_t) as usize }
    }
}

unsafe impl<B: Backend> VecZnxNormalizeImpl<B> for B {
    fn vec_znx_normalize_impl<R, A>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }

        let (tmp_bytes, _) = scratch.tmp_slice(module.vec_znx_normalize_tmp_bytes());

        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                module.ptr() as *const module_info_t,
                basek as u64,
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
}

unsafe impl<B: Backend> VecZnxNormalizeInplaceImpl<B> for B {
    fn vec_znx_normalize_inplace_impl<A>(module: &Module<B>, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
        }

        let (tmp_bytes, _) = scratch.tmp_slice(module.vec_znx_normalize_tmp_bytes());

        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                module.ptr() as *const module_info_t,
                basek as u64,
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
}

unsafe impl<B: Backend> VecZnxAddImpl<B> for B {
    fn vec_znx_add_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
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
            assert_eq!(a.n(), module.n());
            assert_eq!(b.n(), module.n());
            assert_eq!(res.n(), module.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_add(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxAddInplaceImpl<B> for B {
    /// Adds the selected column of `a` to the selected column of `res` and writes the result on the selected column of `res`.
    fn vec_znx_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_add(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxAddScalarInplaceImpl<B> for B {
    /// Adds the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_add_scalar_inplace_impl<R, A>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        res_limb: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        let a: crate::ScalarZnx<&[u8]> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }

        unsafe {
            vec_znx::vec_znx_add(
                module.ptr() as *const module_info_t,
                res.at_mut_ptr(res_col, res_limb),
                1 as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, res_limb),
                1 as u64,
                res.sl() as u64,
            )
        }
    }
}

unsafe impl<B: Backend> VecZnxSubImpl<B> for B {
    /// Subtracts the selected column of `b` from the selected column of `a` and writes the result on the selected column of `res`.
    fn vec_znx_sub_impl<R, A, C>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
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
            assert_eq!(a.n(), module.n());
            assert_eq!(b.n(), module.n());
            assert_eq!(res.n(), module.n());
            assert_ne!(a.as_ptr(), b.as_ptr());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxSubABInplaceImpl<B> for B {
    /// Subtracts the selected column of `a` from the selected column of `res` inplace.
    ///
    /// res[res_col] -= a[a_col]
    fn vec_znx_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxSubBAInplaceImpl<B> for B {
    /// Subtracts the selected column of `res` from the selected column of `a` and inplace mutates `res`
    ///
    /// res[res_col] = a[a_col] - res[res_col]
    fn vec_znx_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxSubScalarInplaceImpl<B> for B {
    /// Subtracts the selected column of `a` on the selected column and limb of `res`.
    fn vec_znx_sub_scalar_inplace_impl<R, A>(
        module: &Module<B>,
        res: &mut R,
        res_col: usize,
        res_limb: usize,
        a: &A,
        a_col: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
    {
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        let a: crate::ScalarZnx<&[u8]> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }

        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr() as *const module_info_t,
                res.at_mut_ptr(res_col, res_limb),
                1 as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, res_limb),
                1 as u64,
                res.sl() as u64,
            )
        }
    }
}

unsafe impl<B: Backend> VecZnxNegateImpl<B> for B {
    // Negates the selected column of `a` and stores the result in `res_col` of `res`.
    fn vec_znx_negate_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_negate(
                module.ptr() as *const module_info_t,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl<B: Backend> VecZnxNegateInplaceImpl<B> for B {
    /// Negates the selected column of `a`.
    fn vec_znx_negate_inplace_impl<A>(module: &Module<B>, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_negate(
                module.ptr() as *const module_info_t,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl<B: Backend> VecZnxShiftInplaceImpl<B> for B {
    /// Shifts by k bits all columns of `a`.
    /// A positive k applies a left shift, while a negative k applies a right shift.
    fn vec_znx_shift_inplace_impl<A>(_module: &Module<B>, basek: usize, k: i64, a: &mut A, scratch: &mut Scratch)
    where
        A: VecZnxToMut,
    {
        if k > 0 {
            a.to_mut().lsh(basek, k as usize, scratch);
        } else {
            a.to_mut().rsh(basek, k.abs() as usize, scratch);
        }
    }
}

unsafe impl<B: Backend> VecZnxRotateImpl<B> for B {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_rotate(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxRotateInplaceImpl<B> for B {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_rotate(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxAutomorphismImpl<B> for B {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxAutomorphismInplaceImpl<B> for B {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert!(
                k & 1 != 0,
                "invalid galois element: must be odd but is {}",
                k
            );
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                module.ptr() as *const module_info_t,
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
}

unsafe impl<B: Backend> VecZnxSplitImpl<B> for B {
    /// Splits the selected columns of `b` into subrings and copies them them into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split_impl<R, A>(module: &Module<B>, res: &mut Vec<R>, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();

        let (n_in, n_out) = (a.n(), res[0].to_mut().n());

        let (mut buf, _) = scratch.tmp_vec_znx(module, 1, a.size());

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
                module.vec_znx_switch_degree(bi, res_col, &a, a_col);
                module.vec_znx_rotate(-1, &mut buf, 0, &a, a_col);
            } else {
                module.vec_znx_switch_degree(bi, res_col, &mut buf, a_col);
                module.vec_znx_rotate_inplace(-1, &mut buf, a_col);
            }
        })
    }
}

unsafe impl<B: Backend> VecZnxMergeImpl<B> for B {
    /// Merges the subrings of the selected column of `a` into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
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
            module.vec_znx_switch_degree(&mut res, res_col, ai, a_col);
            module.vec_znx_rotate_inplace(-1, &mut res, res_col);
        });

        module.vec_znx_rotate_inplace(a.len() as i64, &mut res, res_col);
    }
}

unsafe impl<B: Backend> VecZnxSwithcDegreeImpl<B> for B {
    fn vec_znx_switch_degree_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        let (n_in, n_out) = (a.n(), res.n());

        if n_in == n_out {
            module.vec_znx_copy(&mut res, res_col, &a, a_col);
            return;
        }

        let (gap_in, gap_out): (usize, usize);
        if n_in > n_out {
            (gap_in, gap_out) = (n_in / n_out, 1)
        } else {
            (gap_in, gap_out) = (1, n_out / n_in);
            res.zero();
        }

        let size: usize = a.size().min(res.size());

        (0..size).for_each(|i| {
            izip!(
                a.at(a_col, i).iter().step_by(gap_in),
                res.at_mut(res_col, i).iter_mut().step_by(gap_out)
            )
            .for_each(|(x_in, x_out)| *x_out = *x_in);
        });
    }
}

unsafe impl<B: Backend> VecZnxCopyImpl<B> for B {
    fn vec_znx_copy_impl<R, A>(_module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let mut res_mut: VecZnx<&mut [u8]> = res.to_mut();
        let a_ref: VecZnx<&[u8]> = a.to_ref();

        let min_size: usize = res_mut.size().min(a_ref.size());

        (0..min_size).for_each(|j| {
            res_mut
                .at_mut(res_col, j)
                .copy_from_slice(a_ref.at(a_col, j));
        });
        (min_size..res_mut.size()).for_each(|j| {
            res_mut.zero_at(res_col, j);
        })
    }
}

unsafe impl<B: Backend> VecZnxStdImpl<B> for B {
    fn vec_znx_std_impl<A>(_module: &Module<B>, basek: usize, a: &A, a_col: usize) -> f64
    where
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let prec: u32 = (a.size() * basek) as u32;
        let mut data: Vec<Float> = (0..a.n()).map(|_| Float::with_val(prec, 0)).collect();
        a.decode_vec_float(a_col, basek, &mut data);
        // std = sqrt(sum((xi - avg)^2) / n)
        let mut avg: Float = Float::with_val(prec, 0);
        data.iter().for_each(|x| {
            avg.add_assign_round(x, Round::Nearest);
        });
        avg.div_assign_round(Float::with_val(prec, data.len()), Round::Nearest);
        data.iter_mut().for_each(|x| {
            x.sub_assign_round(&avg, Round::Nearest);
        });
        let mut std: Float = Float::with_val(prec, 0);
        data.iter().for_each(|x| std += x * x);
        std.div_assign_round(Float::with_val(prec, data.len()), Round::Nearest);
        std = std.sqrt();
        std.to_f64()
    }
}

unsafe impl<B: Backend> VecZnxFillUniformImpl<B> for B {
    fn vec_znx_fill_uniform_impl<R>(_module: &Module<B>, basek: usize, res: &mut R, res_col: usize, k: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = res.to_mut();
        let base2k: u64 = 1 << basek;
        let mask: u64 = base2k - 1;
        let base2k_half: i64 = (base2k >> 1) as i64;
        (0..k.div_ceil(basek)).for_each(|j| {
            a.at_mut(res_col, j)
                .iter_mut()
                .for_each(|x| *x = (source.next_u64n(base2k, mask) as i64) - base2k_half);
        })
    }
}

unsafe impl<B: Backend> VecZnxFillDistF64Impl<B> for B {
    fn vec_znx_fill_dist_f64_impl<R, D: rand::prelude::Distribution<f64>>(
        _module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = res.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = (k + basek - 1) / basek - 1;
        let basek_rem: usize = (limb + 1) * basek - k;

        if basek_rem != 0 {
            a.at_mut(res_col, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a = (dist_f64.round() as i64) << basek_rem;
            });
        } else {
            a.at_mut(res_col, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a = dist_f64.round() as i64
            });
        }
    }
}

unsafe impl<B: Backend> VecZnxAddDistF64Impl<B> for B {
    fn vec_znx_add_dist_f64_impl<R, D: rand::prelude::Distribution<f64>>(
        _module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        dist: D,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = res.to_mut();
        assert!(
            (bound.log2().ceil() as i64) < 64,
            "invalid bound: ceil(log2(bound))={} > 63",
            (bound.log2().ceil() as i64)
        );

        let limb: usize = (k + basek - 1) / basek - 1;
        let basek_rem: usize = (limb + 1) * basek - k;

        if basek_rem != 0 {
            a.at_mut(res_col, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += (dist_f64.round() as i64) << basek_rem;
            });
        } else {
            a.at_mut(res_col, limb).iter_mut().for_each(|a| {
                let mut dist_f64: f64 = dist.sample(source);
                while dist_f64.abs() > bound {
                    dist_f64 = dist.sample(source)
                }
                *a += dist_f64.round() as i64
            });
        }
    }
}

unsafe impl<B: Backend> VecZnxFillNormalImpl<B> for B {
    fn vec_znx_fill_normal_impl<R>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        module.vec_znx_fill_dist_f64(
            basek,
            res,
            res_col,
            k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}

unsafe impl<B: Backend> VecZnxAddNormalImpl<B> for B {
    fn vec_znx_add_normal_impl<R>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        module.vec_znx_add_dist_f64(
            basek,
            res,
            res_col,
            k,
            source,
            Normal::new(0.0, sigma).unwrap(),
            bound,
        );
    }
}
