#[macro_export]
macro_rules! impl_ckks_neg_default_methods {
    ($backend:ty) => {
        fn ckks_neg_tmp_bytes(module: &poulpy_hal::layouts::Module<$backend>) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEShift<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_tmp_bytes_default(module)
        }

        fn ckks_neg_into<Dst, Src>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            src: &Src,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            Src: $crate::GLWEToBackendRef<$backend>
                + poulpy_core::layouts::GLWEInfos
                + poulpy_core::layouts::LWEInfos
                + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWENegate<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_into_default(module, dst, src, scratch)
        }

        fn ckks_neg_assign<Dst>(module: &poulpy_hal::layouts::Module<$backend>, dst: &mut Dst) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWENegate<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::neg::CKKSNegDefault<$backend>>::ckks_neg_assign_default(module, dst)
        }
    };
}

pub use crate::impl_ckks_neg_default_methods;
