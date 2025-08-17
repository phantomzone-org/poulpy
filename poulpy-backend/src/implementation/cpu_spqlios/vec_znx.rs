use itertools::izip;
use rand_distr::Normal;

use crate::{
    hal::{
        api::{
            TakeSlice, TakeVecZnx, VecZnxAddDistF64, VecZnxCopy, VecZnxFillDistF64, VecZnxNormalizeTmpBytes, VecZnxRotate,
            VecZnxRotateInplace, VecZnxSwithcDegree, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero,
        },
        layouts::{Backend, Module, ScalarZnx, ScalarZnxToRef, Scratch, VecZnx, VecZnxToMut, VecZnxToRef},
        oep::{
            VecZnxAddDistF64Impl, VecZnxAddImpl, VecZnxAddInplaceImpl, VecZnxAddNormalImpl, VecZnxAddScalarInplaceImpl,
            VecZnxAutomorphismImpl, VecZnxAutomorphismInplaceImpl, VecZnxCopyImpl, VecZnxFillDistF64Impl, VecZnxFillNormalImpl,
            VecZnxFillUniformImpl, VecZnxLshInplaceImpl, VecZnxMergeImpl, VecZnxMulXpMinusOneImpl,
            VecZnxMulXpMinusOneInplaceImpl, VecZnxNegateImpl, VecZnxNegateInplaceImpl, VecZnxNormalizeImpl,
            VecZnxNormalizeInplaceImpl, VecZnxNormalizeTmpBytesImpl, VecZnxRotateImpl, VecZnxRotateInplaceImpl,
            VecZnxRshInplaceImpl, VecZnxSplitImpl, VecZnxSubABInplaceImpl, VecZnxSubBAInplaceImpl, VecZnxSubImpl,
            VecZnxSubScalarInplaceImpl, VecZnxSwithcDegreeImpl,
        },
        source::Source,
    },
    implementation::cpu_spqlios::{
        CPUAVX,
        ffi::{module::module_info_t, vec_znx, znx},
    },
};

unsafe impl<B: Backend> VecZnxNormalizeTmpBytesImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<B>, n: usize) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(module.ptr() as *const module_info_t, n as u64) as usize }
    }
}

unsafe impl<B: Backend + CPUAVX> VecZnxNormalizeImpl<B> for B {
    fn vec_znx_normalize_impl<R, A>(
        module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), a.n());
        }

        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes(a.n()));

        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                module.ptr() as *const module_info_t,
                a.n() as u64,
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

unsafe impl<B: Backend + CPUAVX> VecZnxNormalizeInplaceImpl<B> for B {
    fn vec_znx_normalize_inplace_impl<A>(module: &Module<B>, basek: usize, a: &mut A, a_col: usize, scratch: &mut Scratch<B>)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();

        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes(a.n()));

        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                module.ptr() as *const module_info_t,
                a.n() as u64,
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

unsafe impl<B: Backend + CPUAVX> VecZnxAddImpl<B> for B {
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
            assert_eq!(a.n(), res.n());
            assert_eq!(b.n(), res.n());
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

unsafe impl<B: Backend + CPUAVX> VecZnxAddInplaceImpl<B> for B {
    fn vec_znx_add_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
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

unsafe impl<B: Backend + CPUAVX> VecZnxAddScalarInplaceImpl<B> for B {
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
        let a: ScalarZnx<&[u8]> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }

        unsafe {
            vec_znx::vec_znx_add(
                module.ptr() as *const module_info_t,
                res.at_mut_ptr(res_col, res_limb),
                1_u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, res_limb),
                1_u64,
                res.sl() as u64,
            )
        }
    }
}

unsafe impl<B: Backend + CPUAVX> VecZnxSubImpl<B> for B {
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
            assert_eq!(a.n(), res.n());
            assert_eq!(b.n(), res.n());
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

unsafe impl<B: Backend + CPUAVX> VecZnxSubABInplaceImpl<B> for B {
    fn vec_znx_sub_ab_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
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

unsafe impl<B: Backend + CPUAVX> VecZnxSubBAInplaceImpl<B> for B {
    fn vec_znx_sub_ba_inplace_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
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

unsafe impl<B: Backend + CPUAVX> VecZnxSubScalarInplaceImpl<B> for B {
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
        let a: ScalarZnx<&[u8]> = a.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }

        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr() as *const module_info_t,
                res.at_mut_ptr(res_col, res_limb),
                1_u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                res.at_ptr(res_col, res_limb),
                1_u64,
                res.sl() as u64,
            )
        }
    }
}

unsafe impl<B: Backend + CPUAVX> VecZnxNegateImpl<B> for B {
    fn vec_znx_negate_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
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

unsafe impl<B: Backend + CPUAVX> VecZnxNegateInplaceImpl<B> for B {
    fn vec_znx_negate_inplace_impl<A>(module: &Module<B>, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
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

unsafe impl<B: Backend + CPUAVX> VecZnxLshInplaceImpl<B> for B {
    fn vec_znx_lsh_inplace_impl<A>(_module: &Module<B>, basek: usize, k: usize, a: &mut A)
    where
        A: VecZnxToMut,
    {
        vec_znx_lsh_inplace_ref(basek, k, a)
    }
}

pub fn vec_znx_lsh_inplace_ref<A>(basek: usize, k: usize, a: &mut A)
where
    A: VecZnxToMut,
{
    let mut a: VecZnx<&mut [u8]> = a.to_mut();

    let n: usize = a.n();
    let cols: usize = a.cols();
    let size: usize = a.size();
    let steps: usize = k / basek;

    a.raw_mut().rotate_left(n * steps * cols);
    (0..cols).for_each(|i| {
        (size - steps..size).for_each(|j| {
            a.zero_at(i, j);
        })
    });

    let k_rem: usize = k % basek;

    if k_rem != 0 {
        let shift: usize = i64::BITS as usize - k_rem;
        (0..cols).for_each(|i| {
            (0..steps).for_each(|j| {
                a.at_mut(i, j).iter_mut().for_each(|xi| {
                    *xi <<= shift;
                });
            });
        });
    }
}

unsafe impl<B: Backend + CPUAVX> VecZnxRshInplaceImpl<B> for B {
    fn vec_znx_rsh_inplace_impl<A>(_module: &Module<B>, basek: usize, k: usize, a: &mut A)
    where
        A: VecZnxToMut,
    {
        vec_znx_rsh_inplace_ref(basek, k, a)
    }
}

pub fn vec_znx_rsh_inplace_ref<A>(basek: usize, k: usize, a: &mut A)
where
    A: VecZnxToMut,
{
    let mut a: VecZnx<&mut [u8]> = a.to_mut();
    let n: usize = a.n();
    let cols: usize = a.cols();
    let size: usize = a.size();
    let steps: usize = k / basek;

    a.raw_mut().rotate_right(n * steps * cols);
    (0..cols).for_each(|i| {
        (0..steps).for_each(|j| {
            a.zero_at(i, j);
        })
    });

    let k_rem: usize = k % basek;

    if k_rem != 0 {
        let mut carry: Vec<i64> = vec![0i64; n]; // ALLOC (but small so OK)
        let shift: usize = i64::BITS as usize - k_rem;
        (0..cols).for_each(|i| {
            carry.fill(0);
            (steps..size).for_each(|j| {
                izip!(carry.iter_mut(), a.at_mut(i, j).iter_mut()).for_each(|(ci, xi)| {
                    *xi += *ci << basek;
                    *ci = (*xi << shift) >> shift;
                    *xi = (*xi - *ci) >> k_rem;
                });
            });
        })
    }
}

unsafe impl<B: Backend + CPUAVX> VecZnxRotateImpl<B> for B {
    fn vec_znx_rotate_impl<R, A>(_module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), a.n());
        }
        unsafe {
            (0..a.size()).for_each(|j| {
                znx::znx_rotate_i64(
                    a.n() as u64,
                    k,
                    res.at_mut_ptr(res_col, j),
                    a.at_ptr(a_col, j),
                );
            });
        }
    }
}

unsafe impl<B: Backend + CPUAVX> VecZnxRotateInplaceImpl<B> for B {
    fn vec_znx_rotate_inplace_impl<A>(_module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        unsafe {
            (0..a.size()).for_each(|j| {
                znx::znx_rotate_inplace_i64(a.n() as u64, k, a.at_mut_ptr(a_col, j));
            });
        }
    }
}

unsafe impl<B: Backend + CPUAVX> VecZnxAutomorphismImpl<B> for B {
    fn vec_znx_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
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

unsafe impl<B: Backend + CPUAVX> VecZnxAutomorphismInplaceImpl<B> for B {
    fn vec_znx_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
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

unsafe impl<B: Backend + CPUAVX> VecZnxMulXpMinusOneImpl<B> for B {
    fn vec_znx_mul_xp_minus_one_impl<R, A>(module: &Module<B>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
            assert_eq!(res.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_mul_xp_minus_one(
                module.ptr() as *const module_info_t,
                p,
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

unsafe impl<B: Backend + CPUAVX> VecZnxMulXpMinusOneInplaceImpl<B> for B {
    fn vec_znx_mul_xp_minus_one_inplace_impl<R>(module: &Module<B>, p: i64, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), res.n());
        }
        unsafe {
            vec_znx::vec_znx_mul_xp_minus_one(
                module.ptr() as *const module_info_t,
                p,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                res.at_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
            )
        }
    }
}

unsafe impl<B: Backend + CPUAVX> VecZnxSplitImpl<B> for B {
    fn vec_znx_split_impl<R, A>(module: &Module<B>, res: &mut [R], res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_split_ref(module, res, res_col, a, a_col, scratch)
    }
}

pub fn vec_znx_split_ref<R, A, B>(
    module: &Module<B>,
    res: &mut [R],
    res_col: usize,
    a: &A,
    a_col: usize,
    scratch: &mut Scratch<B>,
) where
    B: Backend + CPUAVX,
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let a: VecZnx<&[u8]> = a.to_ref();

    let (n_in, n_out) = (a.n(), res[0].to_mut().n());

    let (mut buf, _) = scratch.take_vec_znx(n_in.max(n_out), 1, a.size());

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
            module.vec_znx_switch_degree(bi, res_col, &buf, a_col);
            module.vec_znx_rotate_inplace(-1, &mut buf, a_col);
        }
    })
}

unsafe impl<B: Backend + CPUAVX> VecZnxMergeImpl<B> for B {
    fn vec_znx_merge_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &[A], a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_merge_ref(module, res, res_col, a, a_col)
    }
}

pub fn vec_znx_merge_ref<R, A, B>(module: &Module<B>, res: &mut R, res_col: usize, a: &[A], a_col: usize)
where
    B: Backend + CPUAVX,
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

    a.iter().for_each(|ai| {
        module.vec_znx_switch_degree(&mut res, res_col, ai, a_col);
        module.vec_znx_rotate_inplace(-1, &mut res, res_col);
    });

    module.vec_znx_rotate_inplace(a.len() as i64, &mut res, res_col);
}

unsafe impl<B: Backend + CPUAVX> VecZnxSwithcDegreeImpl<B> for B {
    fn vec_znx_switch_degree_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_switch_degree_ref(module, res, res_col, a, a_col)
    }
}

pub fn vec_znx_switch_degree_ref<R, A, B>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    B: Backend + CPUAVX,
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

unsafe impl<B: Backend + CPUAVX> VecZnxCopyImpl<B> for B {
    fn vec_znx_copy_impl<R, A>(_module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_copy_ref(res, res_col, a, a_col)
    }
}

pub fn vec_znx_copy_ref<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
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

unsafe impl<B: Backend + CPUAVX> VecZnxFillUniformImpl<B> for B {
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

unsafe impl<B: Backend + CPUAVX> VecZnxFillDistF64Impl<B> for B {
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

        let limb: usize = k.div_ceil(basek) - 1;
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

unsafe impl<B: Backend + CPUAVX> VecZnxAddDistF64Impl<B> for B {
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

        let limb: usize = k.div_ceil(basek) - 1;
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

unsafe impl<B: Backend + CPUAVX> VecZnxFillNormalImpl<B> for B {
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

unsafe impl<B: Backend + CPUAVX> VecZnxAddNormalImpl<B> for B {
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
