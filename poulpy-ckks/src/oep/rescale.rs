#[macro_export]
macro_rules! impl_ckks_rescale_default_methods {
    ($backend:ty) => {
        fn ckks_rescale_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_rescale_tmp_bytes_default(module)
        }

        fn ckks_rescale_assign<Dst>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct: &mut Dst,
            k: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_rescale_assign_default(module, ct, k, scratch)
        }

        fn ckks_rescale_into<Dst, Src>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            k: usize,
            src: &Src,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            Src: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_rescale_into_default(module, dst, k, src, scratch)
        }

        fn ckks_align_assign<A, B>(
            module: &poulpy_hal::layouts::Module<$backend>,
            a: &mut A,
            b: &mut B,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            A: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            B: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rescale::CKKSRescaleOpsDefault<$backend>>::ckks_align_assign_default(module, a, b, scratch)
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
