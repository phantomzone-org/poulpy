use itertools::izip;
use rand_distr::Normal;
use rug::{
    Assign, Float,
    float::Round,
    ops::{AddAssignRound, DivAssignRound, SubAssignRound},
};
use sampling::source::Source;

use crate::{
    hal::{
        api::{
            TakeSlice, TakeVecZnx, VecZnxAddDistF64, VecZnxCopy, VecZnxDecodeVecFloat, VecZnxFillDistF64,
            VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxSwithcDegree, ZnxInfos, ZnxSliceSize, ZnxView,
            ZnxViewMut, ZnxZero,
        },
        layouts::{Backend, Module, ScalarZnx, ScalarZnxToRef, Scratch, VecZnx, VecZnxOwned, VecZnxToMut, VecZnxToRef},
        oep::{
            VecZnxAddDistF64Impl, VecZnxAddImpl, VecZnxAddInplaceImpl, VecZnxAddNormalImpl, VecZnxAddScalarInplaceImpl,
            VecZnxAllocBytesImpl, VecZnxAllocImpl, VecZnxAutomorphismImpl, VecZnxAutomorphismInplaceImpl, VecZnxCopyImpl,
            VecZnxDecodeCoeffsi64Impl, VecZnxDecodeVecFloatImpl, VecZnxDecodeVeci64Impl, VecZnxEncodeCoeffsi64Impl,
            VecZnxEncodeVeci64Impl, VecZnxFillDistF64Impl, VecZnxFillNormalImpl, VecZnxFillUniformImpl, VecZnxFromBytesImpl,
            VecZnxLshInplaceImpl, VecZnxMergeImpl, VecZnxMulXpMinusOneImpl, VecZnxMulXpMinusOneInplaceImpl, VecZnxNegateImpl,
            VecZnxNegateInplaceImpl, VecZnxNormalizeImpl, VecZnxNormalizeInplaceImpl, VecZnxNormalizeTmpBytesImpl,
            VecZnxRotateImpl, VecZnxRotateInplaceImpl, VecZnxRshInplaceImpl, VecZnxSplitImpl, VecZnxStdImpl,
            VecZnxSubABInplaceImpl, VecZnxSubBAInplaceImpl, VecZnxSubImpl, VecZnxSubScalarInplaceImpl, VecZnxSwithcDegreeImpl,
        },
    },
    implementation::cpu_spqlios::{
        CPUAVX,
        ffi::{module::module_info_t, vec_znx, znx},
    },
};

unsafe impl<B: Backend> VecZnxAllocImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_alloc_impl(n: usize, cols: usize, size: usize) -> VecZnxOwned {
        VecZnxOwned::new::<i64>(n, cols, size)
    }
}

unsafe impl<B: Backend> VecZnxFromBytesImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_from_bytes_impl(n: usize, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxOwned {
        VecZnxOwned::from_bytes::<i64>(n, cols, size, bytes)
    }
}

unsafe impl<B: Backend> VecZnxAllocBytesImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_alloc_bytes_impl(n: usize, cols: usize, size: usize) -> usize {
        VecZnxOwned::alloc_bytes::<i64>(n, cols, size)
    }
}

unsafe impl<B: Backend> VecZnxNormalizeTmpBytesImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<B>, n: usize) -> usize {
        unsafe { vec_znx::vec_znx_normalize_base2k_tmp_bytes(module.ptr() as *const module_info_t, n as u64) as usize }
    }
}

unsafe impl<B: Backend> VecZnxNormalizeImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxNormalizeInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxAddImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxAddInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxAddScalarInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxSubImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxSubABInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxSubBAInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxSubScalarInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxNegateImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxNegateInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxLshInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxRshInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxRotateImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxRotateInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxAutomorphismImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxAutomorphismInplaceImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxMulXpMinusOneImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_mul_xp_minus_one_impl<R, A>(module: &Module<B>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
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

unsafe impl<B: Backend> VecZnxMulXpMinusOneInplaceImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_mul_xp_minus_one_inplace_impl<R>(module: &Module<B>, p: i64, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), module.n());
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

unsafe impl<B: Backend> VecZnxSplitImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_split_impl<R, A>(
        module: &Module<B>,
        res: &mut Vec<R>,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_split_ref(module, res, res_col, a, a_col, scratch)
    }
}

pub fn vec_znx_split_ref<R, A, B: Backend>(
    module: &Module<B>,
    res: &mut Vec<R>,
    res_col: usize,
    a: &A,
    a_col: usize,
    scratch: &mut Scratch<B>,
) where
    B: CPUAVX,
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let a: VecZnx<&[u8]> = a.to_ref();

    let (n_in, n_out) = (a.n(), res[0].to_mut().n());

    let (mut buf, _) = scratch.take_vec_znx(module, 1, a.size());

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

unsafe impl<B: Backend> VecZnxMergeImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_merge_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_merge_ref(module, res, res_col, a, a_col)
    }
}

pub fn vec_znx_merge_ref<R, A, B: Backend>(module: &Module<B>, res: &mut R, res_col: usize, a: Vec<A>, a_col: usize)
where
    B: CPUAVX,
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

unsafe impl<B: Backend> VecZnxSwithcDegreeImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_switch_degree_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_switch_degree_ref(module, res, res_col, a, a_col)
    }
}

pub fn vec_znx_switch_degree_ref<R, A, B: Backend>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    B: CPUAVX,
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

unsafe impl<B: Backend> VecZnxCopyImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxStdImpl<B> for B
where
    B: CPUAVX,
{
    fn vec_znx_std_impl<A>(module: &Module<B>, basek: usize, a: &A, a_col: usize) -> f64
    where
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let prec: u32 = (a.size() * basek) as u32;
        let mut data: Vec<Float> = (0..a.n()).map(|_| Float::with_val(prec, 0)).collect();
        module.decode_vec_float(basek, &a, a_col, &mut data);
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

unsafe impl<B: Backend> VecZnxFillUniformImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxFillDistF64Impl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxAddDistF64Impl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxFillNormalImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxAddNormalImpl<B> for B
where
    B: CPUAVX,
{
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

unsafe impl<B: Backend> VecZnxEncodeVeci64Impl<B> for B
where
    B: CPUAVX,
{
    fn encode_vec_i64_impl<R>(
        _module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        data: &[i64],
        log_max: usize,
    ) where
        R: VecZnxToMut,
    {
        let size: usize = k.div_ceil(basek);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = res.to_mut();
            assert!(
                size <= a.size(),
                "invalid argument k: k.div_ceil(basek)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(res_col < a.cols());
            assert!(data.len() <= a.n())
        }

        let data_len: usize = data.len();
        let mut a: VecZnx<&mut [u8]> = res.to_mut();
        let k_rem: usize = basek - (k % basek);

        // Zeroes coefficients of the i-th column
        (0..a.size()).for_each(|i| {
            a.zero_at(res_col, i);
        });

        // If 2^{basek} * 2^{k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base2k.
        if log_max + k_rem < 63 || k_rem == basek {
            a.at_mut(res_col, size - 1)[..data_len].copy_from_slice(&data[..data_len]);
        } else {
            let mask: i64 = (1 << basek) - 1;
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size)
                .rev()
                .enumerate()
                .for_each(|(i, i_rev)| {
                    let shift: usize = i * basek;
                    izip!(a.at_mut(res_col, i_rev).iter_mut(), data.iter()).for_each(|(y, x)| *y = (x >> shift) & mask);
                })
        }

        // Case where self.prec % self.k != 0.
        if k_rem != basek {
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size).rev().for_each(|i| {
                a.at_mut(res_col, i)[..data_len]
                    .iter_mut()
                    .for_each(|x| *x <<= k_rem);
            })
        }
    }
}

unsafe impl<B: Backend> VecZnxEncodeCoeffsi64Impl<B> for B
where
    B: CPUAVX,
{
    fn encode_coeff_i64_impl<R>(
        _module: &Module<B>,
        basek: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        i: usize,
        data: i64,
        log_max: usize,
    ) where
        R: VecZnxToMut,
    {
        let size: usize = k.div_ceil(basek);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = res.to_mut();
            assert!(i < a.n());
            assert!(
                size <= a.size(),
                "invalid argument k: k.div_ceil(basek)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(res_col < a.cols());
        }

        let k_rem: usize = basek - (k % basek);
        let mut a: VecZnx<&mut [u8]> = res.to_mut();
        (0..a.size()).for_each(|j| a.at_mut(res_col, j)[i] = 0);

        // If 2^{basek} * 2^{k_rem} < 2^{63}-1, then we can simply copy
        // values on the last limb.
        // Else we decompose values base2k.
        if log_max + k_rem < 63 || k_rem == basek {
            a.at_mut(res_col, size - 1)[i] = data;
        } else {
            let mask: i64 = (1 << basek) - 1;
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size)
                .rev()
                .enumerate()
                .for_each(|(j, j_rev)| {
                    a.at_mut(res_col, j_rev)[i] = (data >> (j * basek)) & mask;
                })
        }

        // Case where prec % k != 0.
        if k_rem != basek {
            let steps: usize = size.min(log_max.div_ceil(basek));
            (size - steps..size).rev().for_each(|j| {
                a.at_mut(res_col, j)[i] <<= k_rem;
            })
        }
    }
}

unsafe impl<B: Backend> VecZnxDecodeVeci64Impl<B> for B
where
    B: CPUAVX,
{
    fn decode_vec_i64_impl<R>(_module: &Module<B>, basek: usize, res: &R, res_col: usize, k: usize, data: &mut [i64])
    where
        R: VecZnxToRef,
    {
        let size: usize = k.div_ceil(basek);
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = res.to_ref();
            assert!(
                data.len() >= a.n(),
                "invalid data: data.len()={} < a.n()={}",
                data.len(),
                a.n()
            );
            assert!(res_col < a.cols());
        }

        let a: VecZnx<&[u8]> = res.to_ref();
        data.copy_from_slice(a.at(res_col, 0));
        let rem: usize = basek - (k % basek);
        if k < basek {
            data.iter_mut().for_each(|x| *x >>= rem);
        } else {
            (1..size).for_each(|i| {
                if i == size - 1 && rem != basek {
                    let k_rem: usize = basek - rem;
                    izip!(a.at(res_col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << k_rem) + (x >> rem);
                    });
                } else {
                    izip!(a.at(res_col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << basek) + x;
                    });
                }
            })
        }
    }
}

unsafe impl<B: Backend> VecZnxDecodeCoeffsi64Impl<B> for B
where
    B: CPUAVX,
{
    fn decode_coeff_i64_impl<R>(_module: &Module<B>, basek: usize, res: &R, res_col: usize, k: usize, i: usize) -> i64
    where
        R: VecZnxToRef,
    {
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = res.to_ref();
            assert!(i < a.n());
            assert!(res_col < a.cols())
        }

        let a: VecZnx<&[u8]> = res.to_ref();
        let size: usize = k.div_ceil(basek);
        let mut res: i64 = 0;
        let rem: usize = basek - (k % basek);
        (0..size).for_each(|j| {
            let x: i64 = a.at(res_col, j)[i];
            if j == size - 1 && rem != basek {
                let k_rem: usize = basek - rem;
                res = (res << k_rem) + (x >> rem);
            } else {
                res = (res << basek) + x;
            }
        });
        res
    }
}

unsafe impl<B: Backend> VecZnxDecodeVecFloatImpl<B> for B
where
    B: CPUAVX,
{
    fn decode_vec_float_impl<R>(_module: &Module<B>, basek: usize, res: &R, res_col: usize, data: &mut [Float])
    where
        R: VecZnxToRef,
    {
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = res.to_ref();
            assert!(
                data.len() >= a.n(),
                "invalid data: data.len()={} < a.n()={}",
                data.len(),
                a.n()
            );
            assert!(res_col < a.cols());
        }

        let a: VecZnx<&[u8]> = res.to_ref();
        let size: usize = a.size();
        let prec: u32 = (basek * size) as u32;

        // 2^{basek}
        let base = Float::with_val(prec, (1 << basek) as f64);

        // y[i] = sum x[j][i] * 2^{-basek*j}
        (0..size).for_each(|i| {
            if i == 0 {
                izip!(a.at(res_col, size - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                    y.assign(*x);
                    *y /= &base;
                });
            } else {
                izip!(a.at(res_col, size - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y += Float::with_val(prec, *x);
                    *y /= &base;
                });
            }
        });
    }
}
