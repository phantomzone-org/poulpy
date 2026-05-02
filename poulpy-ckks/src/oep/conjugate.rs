#[macro_export]
macro_rules! impl_ckks_conjugate_default_methods {
    ($backend:ty) => {
        fn ckks_conjugate_tmp_bytes<C: poulpy_core::layouts::GLWEInfos, K: poulpy_core::layouts::GGLWEInfos>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct_infos: &C,
            key_infos: &K,
        ) -> usize
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_tmp_bytes_default(module, ct_infos, key_infos)
        }

        fn ckks_conjugate_into(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            src: &$crate::CKKSCiphertextRef<'_, $backend>,
            key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend> + poulpy_core::GLWEShift<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_into_default(
                module, &mut dst, &src, &key, scratch,
            )
        }

        fn ckks_conjugate_assign(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut dst: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_assign_default(
                module, &mut dst, &key, scratch,
            )
        }
    };
}

pub use crate::impl_ckks_conjugate_default_methods;
