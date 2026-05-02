#[macro_export]
macro_rules! impl_ckks_rotate_default_methods {
    ($backend:ty) => {
        fn ckks_rotate_tmp_bytes<C: poulpy_core::layouts::GLWEInfos, K: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct_infos: &C,
            key_infos: &K,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_tmp_bytes_default(module, ct_infos, key_infos)
        }

        fn ckks_rotate_into<Dst, Src, K>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            src: &Src,
            key: &K,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            Src: $crate::GLWEToBackendRef<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos,
            K: poulpy_core::layouts::GetGaloisElement
                + poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<$backend>
                + poulpy_core::layouts::GGLWEInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_into_default(module, dst, src, key, scratch)
        }

        fn ckks_rotate_assign<Dst, K>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut Dst,
            key: &K,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: $crate::GLWEToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            K: poulpy_core::layouts::GetGaloisElement
                + poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<$backend>
                + poulpy_core::layouts::GGLWEInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::rotate::CKKSRotateDefault<$backend>>::ckks_rotate_assign_default(module, dst, key, scratch)
        }
    };
}

pub use crate::impl_ckks_rotate_default_methods;
