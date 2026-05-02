#[macro_export]
macro_rules! impl_ckks_add_default_methods {
    ($backend:ty) => {
        fn ckks_add_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_tmp_bytes_default(module)
        }

        fn ckks_add_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            a: &$crate::CKKSCiphertextRef<'_, $backend>,
            b: &$crate::CKKSCiphertextRef<'_, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_into_default(module, &mut dst, &a, &b, scratch)
        }

        fn ckks_add_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            a: &$crate::CKKSCiphertextRef<'_, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAdd<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_assign_default(module, &mut dst, &a, scratch)
        }

        fn ckks_add_pt_vec_znx_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_znx_tmp_bytes_default(module)
        }

        fn ckks_add_pt_vec_znx_into<P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            a: &$crate::CKKSCiphertextRef<'_, $backend>,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            P: $crate::CKKSPlaintexToBackendRef<$backend> + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshAddIntoBackend<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_znx_into_default(module, &mut dst, &a, pt_znx, scratch)
        }

        fn ckks_add_pt_vec_znx_assign<P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            pt_znx: &P,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            P: $crate::CKKSPlaintexToBackendRef<$backend> + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshAddIntoBackend<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_pt_vec_znx_assign_default(module, &mut dst, pt_znx, scratch)
        }

        fn ckks_add_pt_const_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_pt_const_tmp_bytes_default(module)
        }

        fn ckks_add_pt_const_znx_into<P>(
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
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend> + poulpy_hal::api::VecZnxRshAddCoeffIntoBackend<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_pt_const_znx_into_default(module, &mut dst, &a, dst_coeff, pt_znx, pt_coeff, scratch)
        }

        fn ckks_add_pt_const_znx_assign<P>(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            dst_coeff: usize,
            pt_znx: &P,
            pt_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            P: $crate::CKKSPlaintexToBackendRef<$backend> + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_hal::api::VecZnxRshAddCoeffIntoBackend<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::add::CKKSAddDefault<$backend>>::ckks_add_pt_const_znx_assign_default(module, &mut dst, dst_coeff, pt_znx, pt_coeff, scratch)
        }

    };
}

pub use crate::impl_ckks_add_default_methods;
