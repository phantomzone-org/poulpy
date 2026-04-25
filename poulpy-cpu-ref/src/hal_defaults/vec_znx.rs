//! Backend extension points for coefficient-domain [`VecZnx`](poulpy_hal::layouts::VecZnx) operations.

use std::mem::size_of;

use crate::reference::vec_znx::{
    vec_znx_add_into, vec_znx_add_normal_ref, vec_znx_add_scalar_assign, vec_znx_add_scalar_into, vec_znx_automorphism,
    vec_znx_automorphism_inplace, vec_znx_automorphism_inplace_tmp_bytes, vec_znx_copy, vec_znx_fill_normal_ref,
    vec_znx_fill_uniform_ref, vec_znx_lsh, vec_znx_lsh_inplace, vec_znx_lsh_sub, vec_znx_lsh_tmp_bytes, vec_znx_merge_rings,
    vec_znx_merge_rings_tmp_bytes, vec_znx_mul_xp_minus_one, vec_znx_mul_xp_minus_one_inplace,
    vec_znx_mul_xp_minus_one_inplace_tmp_bytes, vec_znx_negate, vec_znx_negate_inplace, vec_znx_normalize,
    vec_znx_normalize_inplace, vec_znx_normalize_tmp_bytes, vec_znx_rotate, vec_znx_rotate_inplace,
    vec_znx_rotate_inplace_tmp_bytes, vec_znx_rsh, vec_znx_rsh_inplace, vec_znx_rsh_sub, vec_znx_rsh_tmp_bytes,
    vec_znx_split_ring, vec_znx_split_ring_tmp_bytes, vec_znx_sub, vec_znx_sub_inplace, vec_znx_sub_negate_inplace,
    vec_znx_sub_scalar, vec_znx_sub_scalar_inplace, vec_znx_switch_ring, vec_znx_zero,
};
use crate::reference::znx::{
    ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign,
    ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep,
    ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign,
    ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepSub, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign,
    ZnxSwitchRing, ZnxZero,
};
use poulpy_hal::{
    api::HostBufMut,
    layouts::{
        Backend, HostDataMut, Module, NoiseInfos, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
        ZnxView, ZnxViewMut,
    },
    source::Source,
};

#[inline]
fn take_host_typed<'a, BE, T>(arena: ScratchArena<'a, BE>, len: usize) -> (&'a mut [T], ScratchArena<'a, BE>)
where
    BE: Backend + 'a,
    BE::BufMut<'a>: HostBufMut<'a>,
    T: Copy,
{
    debug_assert!(
        BE::SCRATCH_ALIGN.is_multiple_of(std::mem::align_of::<T>()),
        "B::SCRATCH_ALIGN ({}) must be a multiple of align_of::<T>() ({})",
        BE::SCRATCH_ALIGN,
        std::mem::align_of::<T>()
    );
    let (buf, arena) = arena.take_region(len * std::mem::size_of::<T>());
    let bytes: &'a mut [u8] = buf.into_bytes();
    let slice = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) };
    (slice, arena)
}
#[doc(hidden)]
pub trait HalVecZnxDefaults<BE: Backend>: Backend
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
{
    fn vec_znx_zero_backend_default<'r>(_module: &Module<BE>, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize)
    where
        BE: ZnxZero,
        BE::BufMut<'r>: HostDataMut,
    {
        vec_znx_zero::<VecZnxBackendMut<'r, BE>, BE>(res, res_col);
    }

    fn vec_znx_normalize_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_normalize_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_default<'s, 'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_base2k: usize,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxCopy
            + ZnxAddAssign
            + ZnxMulPowerOfTwoAssign
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStep
            + ZnxExtractDigitAddMul
            + ZnxNormalizeMiddleStepAssign
            + ZnxNormalizeFinalStepAssign
            + ZnxNormalizeDigit,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let byte_count = vec_znx_normalize_tmp_bytes(module.n());
        assert!(
            byte_count.is_multiple_of(size_of::<i64>()),
            "Scratch buffer size {} must be divisible by {}",
            byte_count,
            size_of::<i64>()
        );
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), byte_count / size_of::<i64>());
        vec_znx_normalize::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(
            res, res_base2k, res_offset, res_col, a, a_base2k, a_col, carry,
        );
    }

    fn vec_znx_normalize_inplace_backend_default<'s, 'r>(
        module: &Module<BE>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxNormalizeFirstStepInplace + ZnxNormalizeMiddleStepInplace + ZnxNormalizeFinalStepInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let byte_count = vec_znx_normalize_tmp_bytes(module.n());
        assert!(
            byte_count.is_multiple_of(size_of::<i64>()),
            "Scratch buffer size {} must be divisible by {}",
            byte_count,
            size_of::<i64>()
        );
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), byte_count / size_of::<i64>());
        vec_znx_normalize_inplace::<VecZnxBackendMut<'r, BE>, BE>(base2k, res, res_col, carry);
    }

    fn vec_znx_add_into_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
    ) where
        BE: ZnxAdd + ZnxCopy + ZnxZero,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: PartialEq + Eq + Sized + Default + AsRef<[u8]> + Sync,
    {
        vec_znx_add_into::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, VecZnxBackendRef<'a, BE>, BE>(
            res, res_col, a, a_col, b, b_col,
        );
    }

    fn vec_znx_add_assign_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxAddInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n, res.n);
        }

        let sum_size: usize = a.size.min(res.size);

        for j in 0..sum_size {
            BE::znx_add_inplace(res.at_mut(res_col, j), a.at(a_col, j));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
        b_limb: usize,
    ) where
        BE: ZnxAdd + ZnxCopy + ZnxZero,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        let a_ref = a.to_ref();
        vec_znx_add_scalar_into::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(
            res, res_col, &a_ref, a_col, b, b_col, b_limb,
        );
    }

    fn vec_znx_add_scalar_assign_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxAddInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        let a_ref = a.to_ref();
        vec_znx_add_scalar_assign::<VecZnxBackendMut<'r, BE>, BE>(res, res_col, res_limb, &a_ref, a_col);
    }

    fn vec_znx_sub_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
    ) where
        BE: ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, VecZnxBackendRef<'a, BE>, BE>(
            res, res_col, a, a_col, b, b_col,
        );
    }

    fn vec_znx_sub_inplace_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxSubInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_inplace::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_sub_negate_inplace_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxSubNegateInplace + ZnxNegateInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_sub_negate_inplace::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(res, res_col, a, a_col);
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
        b_limb: usize,
    ) where
        BE: ZnxSub + ZnxZero,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        let a_ref = a.to_ref();
        vec_znx_sub_scalar::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(
            res, res_col, &a_ref, a_col, b, b_col, b_limb,
        );
    }

    fn vec_znx_sub_scalar_inplace_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxSubInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        let a_ref = a.to_ref();
        vec_znx_sub_scalar_inplace::<VecZnxBackendMut<'r, BE>, BE>(res, res_col, res_limb, &a_ref, a_col);
    }

    fn vec_znx_negate_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxNegate + ZnxZero,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_negate::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_negate_inplace_backend_default<'r>(_module: &Module<BE>, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize)
    where
        BE: ZnxNegateInplace,
        BE::BufMut<'r>: HostDataMut,
    {
        vec_znx_negate_inplace::<VecZnxBackendMut<'r, BE>, BE>(res, res_col);
    }

    fn vec_znx_rsh_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_rsh_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_backend_default<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepInplace
            + ZnxNormalizeFirstStepInplace
            + ZnxNormalizeFinalStepInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_rsh::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE, true>(
            base2k, k, res, res_col, a, a_col, carry,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_into_backend_default<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepInplace
            + ZnxNormalizeFirstStepInplace
            + ZnxNormalizeFinalStepInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_rsh::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE, false>(
            base2k, k, res, res_col, a, a_col, carry,
        );
    }

    fn vec_znx_lsh_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_lsh_tmp_bytes(module.n())
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_backend_default<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxCopy
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_lsh::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE, true>(
            base2k, k, res, res_col, a, a_col, carry,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_into_backend_default<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxNormalizeFirstStep
            + ZnxNormalizeMiddleStep
            + ZnxCopy
            + ZnxNormalizeFinalStep
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_lsh::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE, false>(
            base2k, k, res, res_col, a, a_col, carry,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub_backend_default<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepSub
            + ZnxNormalizeFinalStepSub
            + ZnxNormalizeMiddleStepCarryOnly,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_lsh_sub::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(
            base2k, k, res, res_col, a, a_col, carry,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub_backend_default<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStepSub
            + ZnxNormalizeMiddleStepInplace
            + ZnxNormalizeFirstStepInplace
            + ZnxNormalizeFinalStepInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_rsh_sub::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(
            base2k, k, res, res_col, a, a_col, carry,
        );
    }

    fn vec_znx_rsh_inplace_backend_default<'s, 'r>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero
            + ZnxCopy
            + ZnxNormalizeFirstStepCarryOnly
            + ZnxNormalizeMiddleStepCarryOnly
            + ZnxNormalizeMiddleStep
            + ZnxNormalizeMiddleStepInplace
            + ZnxNormalizeFirstStepInplace
            + ZnxNormalizeFinalStepInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_rsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_rsh_inplace::<VecZnxBackendMut<'r, BE>, BE>(base2k, k, res, res_col, carry);
    }

    fn vec_znx_lsh_inplace_backend_default<'s, 'r>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxZero + ZnxCopy + ZnxNormalizeFirstStepInplace + ZnxNormalizeMiddleStepInplace + ZnxNormalizeFinalStepInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (carry, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_lsh_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_lsh_inplace::<VecZnxBackendMut<'r, BE>, BE>(base2k, k, res, res_col, carry);
    }

    fn vec_znx_rotate_backend_default<'r, 'a>(
        _module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxRotate + ZnxZero,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_rotate::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_rotate_assign_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_rotate_assign_tmp_bytes(module.n())
    }

    fn vec_znx_rotate_inplace_backend_default<'s, 'r>(
        module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxRotate + ZnxCopy,
        BE::BufMut<'r>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            vec_znx_rotate_inplace_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_rotate_inplace::<VecZnxBackendMut<'r, BE>, BE>(p, res, res_col, tmp);
    }

    fn vec_znx_automorphism_backend_default<'r, 'a>(
        _module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxAutomorphism + ZnxZero,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_automorphism::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_automorphism_assign_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_automorphism_assign_tmp_bytes(module.n())
    }

    fn vec_znx_automorphism_inplace_default<'s, 'r>(
        module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxAutomorphism + ZnxCopy,
        BE::BufMut<'r>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            vec_znx_automorphism_inplace_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_automorphism_inplace::<VecZnxBackendMut<'r, BE>, BE>(p, res, res_col, tmp);
    }

    fn vec_znx_mul_xp_minus_one_backend_default<'r, 'a>(
        _module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    )
    where
        BE: ZnxRotate + ZnxZero + ZnxSubInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_mul_xp_minus_one::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(p, res, res_col, a, a_col);
    }

    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_mul_xp_minus_one_assign_tmp_bytes(module.n())
    }

    fn vec_znx_mul_xp_minus_one_inplace_backend_default<'s, 'r>(
        module: &Module<BE>,
        p: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxRotate + ZnxNegate + ZnxSubNegateInplace,
        BE::BufMut<'r>: HostDataMut,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(
            scratch.borrow(),
            vec_znx_mul_xp_minus_one_inplace_tmp_bytes(module.n()) / size_of::<i64>(),
        );
        vec_znx_mul_xp_minus_one_inplace::<VecZnxBackendMut<'r, BE>, BE>(p, res, res_col, tmp);
    }

    fn vec_znx_split_ring_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_split_ring_tmp_bytes(module.n())
    }

    fn vec_znx_split_ring_backend_default<'s>(
        module: &Module<BE>,
        res: &mut [VecZnxBackendMut<'_, BE>],
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxSwitchRing + ZnxRotate + ZnxZero,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_split_ring_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_split_ring::<VecZnxBackendMut<'_, BE>, VecZnxBackendRef<'_, BE>, BE>(res, res_col, a, a_col, tmp);
    }

    fn vec_znx_merge_rings_tmp_bytes_default(module: &Module<BE>) -> usize {
        vec_znx_merge_rings_tmp_bytes(module.n())
    }

    fn vec_znx_merge_rings_backend_default<'s>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &[VecZnxBackendRef<'_, BE>],
        a_col: usize,
        scratch: &'s mut ScratchArena<'s, BE>,
    ) where
        BE: 's,
        BE: ZnxCopy + ZnxSwitchRing + ZnxRotate + ZnxZero,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
        BE::BufMut<'s>: HostBufMut<'s>,
    {
        let (tmp, _) = take_host_typed::<BE, i64>(scratch.borrow(), vec_znx_merge_rings_tmp_bytes(module.n()) / size_of::<i64>());
        vec_znx_merge_rings::<VecZnxBackendMut<'_, BE>, VecZnxBackendRef<'_, BE>, BE>(res, res_col, a, a_col, tmp);
    }

    fn vec_znx_switch_ring_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxCopy + ZnxSwitchRing + ZnxZero,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_switch_ring::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_copy_backend_default<'r, 'a>(
        _module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    ) where
        BE: ZnxCopy + ZnxZero,
        BE::BufMut<'r>: HostDataMut,
        BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    {
        vec_znx_copy::<VecZnxBackendMut<'r, BE>, VecZnxBackendRef<'a, BE>, BE>(res, res_col, a, a_col);
    }

    fn vec_znx_fill_uniform_backend_default(
        _module: &Module<BE>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        seed: [u8; 32],
    ) where
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let mut source = Source::new(seed);
        vec_znx_fill_uniform_ref(base2k, res, res_col, &mut source);
    }

    fn vec_znx_fill_normal_backend_default(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) where
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let mut source = Source::new(seed);
        vec_znx_fill_normal_ref(res_base2k, res, res_col, noise_infos, &mut source);
    }

    fn vec_znx_add_normal_backend_default(
        _module: &Module<BE>,
        res_base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    ) where
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        let mut source = Source::new(seed);
        vec_znx_add_normal_ref(res_base2k, res, res_col, noise_infos, &mut source);
    }
}

impl<BE: Backend> HalVecZnxDefaults<BE> for BE where BE::OwnedBuf: poulpy_hal::layouts::HostDataMut {}
