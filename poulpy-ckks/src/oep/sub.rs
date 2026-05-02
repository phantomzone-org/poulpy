#[macro_export]
macro_rules! impl_ckks_sub_default_methods {
    ($backend:ty) => {
        fn ckks_sub_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_vec_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_tmp_bytes_default(module)
        }

        fn ckks_sub_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            a: &$crate::CKKSCiphertextRef<'_, $backend>,
            b: &$crate::CKKSCiphertextRef<'_, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWESub<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_into_default(module, &mut dst, &a, &b, scratch)
        }

        fn ckks_sub_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            a: &$crate::CKKSCiphertextRef<'_, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWESub<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_assign_default(module, &mut dst, &a, scratch)
        }

        fn ckks_sub_pt_vec_znx_into<P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            a: &$crate::CKKSCiphertextRef<'_, $backend>,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            P: $crate::CKKSPlaintexToBackendRef<$backend> + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSubBackend<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_into_default(module, &mut dst, &a, pt_znx, scratch)
        }

        fn ckks_sub_pt_vec_znx_assign<P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            P: $crate::CKKSPlaintexToBackendRef<$backend> + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshSubBackend<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_vec_znx_assign_default(module, &mut dst, pt_znx, scratch)
        }

        fn ckks_sub_pt_const_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_tmp_bytes_default(module)
        }

        fn ckks_sub_pt_const_znx_into<P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            a: &$crate::CKKSCiphertextRef<'_, $backend>,
            dst_coeff: usize,
            pt_znx: &P,
            pt_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            P: $crate::CKKSPlaintexToBackendRef<$backend> + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshCoeffBackend<$backend> + poulpy_hal::api::VecZnxSubAssignBackend<$backend> + poulpy_hal::api::VecZnxAddConstAssignBackend<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_znx_into_default(module, &mut dst, &a, dst_coeff, pt_znx, pt_coeff, scratch)
        }

        fn ckks_sub_pt_const_znx_assign<P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            dst_coeff: usize,
            pt_znx: &P,
            pt_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            P: $crate::CKKSPlaintexToBackendRef<$backend> + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshCoeffBackend<$backend> + poulpy_hal::api::VecZnxSubAssignBackend<$backend> + poulpy_hal::api::VecZnxAddConstAssignBackend<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::sub::CKKSSubDefault<$backend>>::ckks_sub_pt_const_znx_assign_default(module, &mut dst, dst_coeff, pt_znx, pt_coeff, scratch)
        }

    };
}

pub use crate::impl_ckks_sub_default_methods;
