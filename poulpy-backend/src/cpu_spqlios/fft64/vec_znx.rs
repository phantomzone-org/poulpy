use poulpy_hal::{
    api::{
        TakeSlice, VecZnxLshTmpBytes, VecZnxMergeRingsTmpBytes, VecZnxNormalizeTmpBytes, VecZnxRshTmpBytes,
        VecZnxSplitRingTmpBytes,
    },
    layouts::{
        Module, ScalarZnx, ScalarZnxToRef, Scratch, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut,
    },
    oep::{
        TakeSliceImpl, VecZnxAddImpl, VecZnxAddInplaceImpl, VecZnxAddNormalImpl, VecZnxAddScalarImpl, VecZnxAddScalarInplaceImpl,
        VecZnxAutomorphismImpl, VecZnxAutomorphismInplaceImpl, VecZnxAutomorphismInplaceTmpBytesImpl, VecZnxCopyImpl,
        VecZnxFillNormalImpl, VecZnxFillUniformImpl, VecZnxLshImpl, VecZnxLshInplaceImpl, VecZnxLshTmpBytesImpl,
        VecZnxMergeRingsImpl, VecZnxMergeRingsTmpBytesImpl, VecZnxMulXpMinusOneImpl, VecZnxMulXpMinusOneInplaceImpl,
        VecZnxMulXpMinusOneInplaceTmpBytesImpl, VecZnxNegateImpl, VecZnxNegateInplaceImpl, VecZnxNormalizeImpl,
        VecZnxNormalizeInplaceImpl, VecZnxNormalizeTmpBytesImpl, VecZnxRotateImpl, VecZnxRotateInplaceImpl,
        VecZnxRotateInplaceTmpBytesImpl, VecZnxRshImpl, VecZnxRshInplaceImpl, VecZnxRshTmpBytesImpl, VecZnxSplitRingImpl,
        VecZnxSplitRingTmpBytesImpl, VecZnxSubImpl, VecZnxSubInplaceImpl, VecZnxSubNegateInplaceImpl, VecZnxSubScalarImpl,
        VecZnxSubScalarInplaceImpl, VecZnxSwitchRingImpl, VecZnxZeroImpl,
    },
    reference::{
        vec_znx::{
            vec_znx_add_normal_ref, vec_znx_automorphism_inplace_tmp_bytes, vec_znx_copy, vec_znx_fill_normal_ref,
            vec_znx_fill_uniform_ref, vec_znx_lsh, vec_znx_lsh_inplace, vec_znx_lsh_tmp_bytes, vec_znx_merge_rings,
            vec_znx_merge_rings_tmp_bytes, vec_znx_mul_xp_minus_one_inplace_tmp_bytes, vec_znx_normalize_tmp_bytes,
            vec_znx_rotate_inplace_tmp_bytes, vec_znx_rsh, vec_znx_rsh_inplace, vec_znx_rsh_tmp_bytes, vec_znx_split_ring,
            vec_znx_split_ring_tmp_bytes, vec_znx_switch_ring, vec_znx_zero,
        },
        znx::{znx_copy_ref, znx_zero_ref},
    },
    source::Source,
};

use crate::cpu_spqlios::{
    FFT64Spqlios,
    ffi::{module::module_info_t, vec_znx, znx},
};

unsafe impl VecZnxZeroImpl<Self> for FFT64Spqlios {
    fn vec_znx_zero_impl<R>(_module: &Module<Self>, res: &mut R, res_col: usize)
    where
        R: VecZnxToMut,
    {
        vec_znx_zero::<_, FFT64Spqlios>(res, res_col);
    }
}

unsafe impl VecZnxNormalizeTmpBytesImpl<Self> for FFT64Spqlios {
    fn vec_znx_normalize_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_normalize_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxNormalizeImpl<Self> for FFT64Spqlios
where
    Self: TakeSliceImpl<Self> + VecZnxNormalizeTmpBytesImpl<Self>,
{
    fn vec_znx_normalize_impl<R, A>(
        module: &Module<Self>,
        res_basek: usize,
        res: &mut R,
        res_col: usize,
        a_basek: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let a: VecZnx<&[u8]> = a.to_ref();
        let mut res: VecZnx<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.n(), a.n());
            assert_eq!(
                res_basek, a_basek,
                "res_basek != a_basek -> base2k conversion is not supported"
            )
        }

        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes());

        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                module.ptr() as *const module_info_t,
                res_basek as u64,
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

unsafe impl VecZnxNormalizeInplaceImpl<Self> for FFT64Spqlios
where
    Self: TakeSliceImpl<Self> + VecZnxNormalizeTmpBytesImpl<Self>,
{
    fn vec_znx_normalize_inplace_impl<A>(
        module: &Module<Self>,
        base2k: usize,
        a: &mut A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();

        let (tmp_bytes, _) = scratch.take_slice(module.vec_znx_normalize_tmp_bytes());

        unsafe {
            vec_znx::vec_znx_normalize_base2k(
                module.ptr() as *const module_info_t,
                base2k as u64,
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

unsafe impl VecZnxAddImpl<Self> for FFT64Spqlios {
    fn vec_znx_add_impl<R, A, C>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
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

unsafe impl VecZnxAddInplaceImpl<Self> for FFT64Spqlios {
    fn vec_znx_add_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
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

unsafe impl VecZnxAddScalarInplaceImpl<Self> for FFT64Spqlios {
    fn vec_znx_add_scalar_inplace_impl<R, A>(
        module: &Module<Self>,
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

unsafe impl VecZnxAddScalarImpl<Self> for FFT64Spqlios {
    fn vec_znx_add_scalar_impl<R, A, B>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef,
    {
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        let a: ScalarZnx<&[u8]> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }

        let min_size: usize = b.size().min(res.size());

        unsafe {
            vec_znx::vec_znx_add(
                module.ptr() as *const module_info_t,
                res.at_mut_ptr(res_col, b_limb),
                1_u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                b.at_ptr(b_col, b_limb),
                1_u64,
                b.sl() as u64,
            );

            for j in 0..min_size {
                if j != b_limb {
                    znx_copy_ref(res.at_mut(res_col, j), b.at(b_col, j));
                }
            }

            for j in min_size..res.size() {
                znx_zero_ref(res.at_mut(res_col, j));
            }
        }
    }
}

unsafe impl VecZnxSubImpl<Self> for FFT64Spqlios {
    fn vec_znx_sub_impl<R, A, C>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &C, b_col: usize)
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

unsafe impl VecZnxSubInplaceImpl<Self> for FFT64Spqlios {
    fn vec_znx_sub_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
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

unsafe impl VecZnxSubNegateInplaceImpl<Self> for FFT64Spqlios {
    fn vec_znx_sub_negate_inplace_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
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

unsafe impl VecZnxSubScalarImpl<Self> for FFT64Spqlios {
    fn vec_znx_sub_scalar_impl<R, A, B>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        b: &B,
        b_col: usize,
        b_limb: usize,
    ) where
        R: VecZnxToMut,
        A: ScalarZnxToRef,
        B: VecZnxToRef,
    {
        let mut res: VecZnx<&mut [u8]> = res.to_mut();
        let a: ScalarZnx<&[u8]> = a.to_ref();
        let b: VecZnx<&[u8]> = b.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), res.n());
        }

        let min_size: usize = b.size().min(res.size());

        unsafe {
            vec_znx::vec_znx_sub(
                module.ptr() as *const module_info_t,
                res.at_mut_ptr(res_col, b_limb),
                1_u64,
                res.sl() as u64,
                b.at_ptr(b_col, b_limb),
                1_u64,
                b.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            );

            for j in 0..min_size {
                if j != b_limb {
                    res.at_mut(res_col, j).copy_from_slice(b.at(b_col, j))
                }
            }

            for j in min_size..res.size() {
                znx_zero_ref(res.at_mut(res_col, j));
            }
        }
    }
}

unsafe impl VecZnxSubScalarInplaceImpl<Self> for FFT64Spqlios {
    fn vec_znx_sub_scalar_inplace_impl<R, A>(
        module: &Module<Self>,
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
                res.at_ptr(res_col, res_limb),
                1_u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl VecZnxNegateImpl<Self> for FFT64Spqlios {
    fn vec_znx_negate_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
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

unsafe impl VecZnxNegateInplaceImpl<Self> for FFT64Spqlios {
    fn vec_znx_negate_inplace_impl<A>(module: &Module<Self>, a: &mut A, a_col: usize)
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

unsafe impl VecZnxLshTmpBytesImpl<Self> for FFT64Spqlios {
    fn vec_znx_lsh_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_lsh_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxRshTmpBytesImpl<Self> for FFT64Spqlios {
    fn vec_znx_rsh_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_rsh_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxLshImpl<Self> for FFT64Spqlios
where
    Module<Self>: VecZnxNormalizeTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_lsh_impl<R, A>(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_lsh_tmp_bytes() / size_of::<i64>());
        vec_znx_lsh::<_, _, FFT64Spqlios>(base2k, k, res, res_col, a, a_col, carry)
    }
}

unsafe impl VecZnxLshInplaceImpl<Self> for FFT64Spqlios
where
    Module<Self>: VecZnxNormalizeTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_lsh_inplace_impl<A>(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        a: &mut A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        A: VecZnxToMut,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_lsh_tmp_bytes() / size_of::<i64>());
        vec_znx_lsh_inplace::<_, FFT64Spqlios>(base2k, k, a, a_col, carry)
    }
}

unsafe impl VecZnxRshImpl<Self> for FFT64Spqlios
where
    Module<Self>: VecZnxNormalizeTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_rsh_impl<R, A>(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut R,
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_rsh_tmp_bytes() / size_of::<i64>());
        vec_znx_rsh::<_, _, FFT64Spqlios>(base2k, k, res, res_col, a, a_col, carry)
    }
}

unsafe impl VecZnxRshInplaceImpl<Self> for FFT64Spqlios
where
    Module<Self>: VecZnxNormalizeTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_rsh_inplace_impl<A>(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        a: &mut A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        A: VecZnxToMut,
    {
        let (carry, _) = scratch.take_slice(module.vec_znx_rsh_tmp_bytes() / size_of::<i64>());
        vec_znx_rsh_inplace::<_, FFT64Spqlios>(base2k, k, a, a_col, carry)
    }
}

unsafe impl VecZnxRotateImpl<Self> for FFT64Spqlios {
    fn vec_znx_rotate_impl<R, A>(_module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
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
            let min_size = res.size().min(a.size());
            (0..min_size).for_each(|j| {
                znx::znx_rotate_i64(
                    a.n() as u64,
                    k,
                    res.at_mut_ptr(res_col, j),
                    a.at_ptr(a_col, j),
                );
            });

            (min_size..res.size()).for_each(|j| {
                znx_zero_ref(res.at_mut(res_col, j));
            })
        }
    }
}

unsafe impl VecZnxRotateInplaceTmpBytesImpl<Self> for FFT64Spqlios
where
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_rotate_inplace_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_rotate_inplace_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxRotateInplaceImpl<Self> for FFT64Spqlios
where
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_rotate_inplace_impl<A>(_module: &Module<Self>, k: i64, a: &mut A, a_col: usize, _scratch: &mut Scratch<Self>)
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

unsafe impl VecZnxAutomorphismImpl<Self> for FFT64Spqlios {
    fn vec_znx_automorphism_impl<R, A>(module: &Module<Self>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
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

unsafe impl VecZnxAutomorphismInplaceTmpBytesImpl<Self> for FFT64Spqlios {
    fn vec_znx_automorphism_inplace_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_automorphism_inplace_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxAutomorphismInplaceImpl<Self> for FFT64Spqlios {
    fn vec_znx_automorphism_inplace_impl<A>(module: &Module<Self>, k: i64, a: &mut A, a_col: usize, _scratch: &mut Scratch<Self>)
    where
        A: VecZnxToMut,
    {
        let mut a: VecZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert!(k & 1 != 0, "invalid galois element: must be odd but is {k}");
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

unsafe impl VecZnxMulXpMinusOneImpl<Self> for FFT64Spqlios {
    fn vec_znx_mul_xp_minus_one_impl<R, A>(module: &Module<Self>, p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
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

unsafe impl VecZnxMulXpMinusOneInplaceTmpBytesImpl<Self> for FFT64Spqlios {
    fn vec_znx_mul_xp_minus_one_inplace_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_mul_xp_minus_one_inplace_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxMulXpMinusOneInplaceImpl<Self> for FFT64Spqlios {
    fn vec_znx_mul_xp_minus_one_inplace_impl<R>(
        module: &Module<Self>,
        p: i64,
        res: &mut R,
        res_col: usize,
        _scratch: &mut Scratch<Self>,
    ) where
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

unsafe impl VecZnxSplitRingTmpBytesImpl<Self> for FFT64Spqlios {
    fn vec_znx_split_ring_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_split_ring_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxSplitRingImpl<Self> for FFT64Spqlios
where
    Module<Self>: VecZnxSplitRingTmpBytes,
    Scratch<Self>: TakeSlice,
{
    fn vec_znx_split_ring_impl<R, A>(
        module: &Module<Self>,
        res: &mut [R],
        res_col: usize,
        a: &A,
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_split_ring_tmp_bytes() / size_of::<i64>());
        vec_znx_split_ring::<_, _, FFT64Spqlios>(res, res_col, a, a_col, tmp);
    }
}

unsafe impl VecZnxMergeRingsTmpBytesImpl<Self> for FFT64Spqlios {
    fn vec_znx_merge_rings_tmp_bytes_impl(module: &Module<Self>) -> usize {
        vec_znx_merge_rings_tmp_bytes(module.n())
    }
}

unsafe impl VecZnxMergeRingsImpl<Self> for FFT64Spqlios
where
    Module<Self>: VecZnxMergeRingsTmpBytes,
{
    fn vec_znx_merge_rings_impl<R, A>(
        module: &Module<Self>,
        res: &mut R,
        res_col: usize,
        a: &[A],
        a_col: usize,
        scratch: &mut Scratch<Self>,
    ) where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        let (tmp, _) = scratch.take_slice(module.vec_znx_merge_rings_tmp_bytes() / size_of::<i64>());
        vec_znx_merge_rings::<_, _, FFT64Spqlios>(res, res_col, a, a_col, tmp);
    }
}

unsafe impl VecZnxSwitchRingImpl<Self> for FFT64Spqlios
where
    Self: VecZnxCopyImpl<Self>,
{
    fn vec_znx_switch_ring_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_switch_ring::<_, _, FFT64Spqlios>(res, res_col, a, a_col);
    }
}

unsafe impl VecZnxCopyImpl<Self> for FFT64Spqlios {
    fn vec_znx_copy_impl<R, A>(_module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxToMut,
        A: VecZnxToRef,
    {
        vec_znx_copy::<_, _, FFT64Spqlios>(res, res_col, a, a_col)
    }
}

unsafe impl VecZnxFillUniformImpl<Self> for FFT64Spqlios {
    fn vec_znx_fill_uniform_impl<R>(_module: &Module<Self>, base2k: usize, res: &mut R, res_col: usize, source: &mut Source)
    where
        R: VecZnxToMut,
    {
        vec_znx_fill_uniform_ref(base2k, res, res_col, source)
    }
}

unsafe impl VecZnxFillNormalImpl<Self> for FFT64Spqlios {
    fn vec_znx_fill_normal_impl<R>(
        _module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        vec_znx_fill_normal_ref(base2k, res, res_col, k, sigma, bound, source);
    }
}

unsafe impl VecZnxAddNormalImpl<Self> for FFT64Spqlios {
    fn vec_znx_add_normal_impl<R>(
        _module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        k: usize,
        source: &mut Source,
        sigma: f64,
        bound: f64,
    ) where
        R: VecZnxToMut,
    {
        vec_znx_add_normal_ref(base2k, res, res_col, k, sigma, bound, source);
    }
}
