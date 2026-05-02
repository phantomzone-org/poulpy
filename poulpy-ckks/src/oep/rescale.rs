#[macro_export]
macro_rules! impl_ckks_rescale_default_methods {
    ($backend:ty) => {
        fn ckks_rescale_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_rescale_tmp_bytes_default(module)
        }

        fn ckks_rescale_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut ct: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            k: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_rescale_assign_default(
                module, &mut ct, k, scratch,
            )
        }

        fn ckks_rescale_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            k: usize,
            src: &$crate::CKKSCiphertextRef<'_, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_rescale_into_default(
                module, &mut dst, k, &src, scratch,
            )
        }

        fn ckks_align_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut a: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            mut b: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_align_assign_default(
                module, &mut a, &mut b, scratch,
            )
        }

        fn ckks_align_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_align_tmp_bytes_default(module)
        }
    };
}

pub use crate::impl_ckks_rescale_default_methods;
