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

        fn ckks_conjugate_into<Dst, Src, K>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            src: &$crate::layouts::CKKSCiphertext<Src>,
            key: &poulpy_core::layouts::GLWEAutomorphismKeyPrepared<K, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: poulpy_hal::layouts::Data,
            Src: poulpy_hal::layouts::Data,
            K: poulpy_hal::layouts::Data,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend> + poulpy_core::GLWEShift<$backend>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            $crate::layouts::CKKSCiphertext<Src>: poulpy_core::layouts::GLWEToBackendRef<$backend>,
            poulpy_core::layouts::GLWEAutomorphismKeyPrepared<K, $backend>:
                poulpy_core::layouts::GGLWEPreparedToBackendRef<$backend> + poulpy_core::layouts::GGLWEInfos,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_into_default(module, dst, src, key, scratch)
        }

        fn ckks_conjugate_assign<Dst, K>(
            module: &poulpy_hal::layouts::Module<$backend>,
            dst: &mut $crate::layouts::CKKSCiphertext<Dst>,
            key: &poulpy_core::layouts::GLWEAutomorphismKeyPrepared<K, $backend>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Dst: poulpy_hal::layouts::Data,
            K: poulpy_hal::layouts::Data,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEAutomorphism<$backend>,
            $crate::layouts::CKKSCiphertext<Dst>: poulpy_core::layouts::GLWEToBackendMut<$backend>,
            poulpy_core::layouts::GLWEAutomorphismKeyPrepared<K, $backend>:
                poulpy_core::layouts::GGLWEPreparedToBackendRef<$backend> + poulpy_core::layouts::GGLWEInfos,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::conjugate::CKKSConjugateDefault<$backend>>::ckks_conjugate_assign_default(module, dst, key, scratch)
        }
    };
}

pub use crate::impl_ckks_conjugate_default_methods;
