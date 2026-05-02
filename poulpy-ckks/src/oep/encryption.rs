#[macro_export]
macro_rules! impl_ckks_encryption_default_methods {
    ($backend:ty) => {
        fn ckks_encrypt_sk_tmp_bytes<A>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct_infos: &A,
        ) -> usize
        where
            A: poulpy_core::layouts::GLWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEEncryptSk<$backend>
                + poulpy_hal::api::VecZnxRshAddIntoBackend<$backend>
                + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::encryption::CKKSEncryptionDefault<$backend>>::ckks_encrypt_sk_tmp_bytes_default(module, ct_infos)
        }

        fn ckks_encrypt_sk<'s, S, E, Pt>(
            module: &poulpy_hal::layouts::Module<$backend>,
            mut ct: &mut $crate::CKKSCiphertextMut<'_, $backend>,
            pt: &Pt,
            sk: &S,
            enc_infos: &E,
            source_xa: &mut poulpy_hal::source::Source,
            source_xe: &mut poulpy_hal::source::Source,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'s, $backend>,
        ) -> anyhow::Result<()>
        where
            E: poulpy_core::EncryptionInfos,
            Pt: $crate::CKKSPlaintexToBackendRef<$backend> + $crate::CKKSInfos,
            S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<$backend>,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEEncryptSk<$backend>
                + poulpy_hal::api::VecZnxRshAddIntoBackend<$backend>
                + $crate::leveled::default::pt_znx::CKKSPlaintextZnxDefault<$backend>,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>:
                poulpy_hal::api::ScratchAvailable + poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::encryption::CKKSEncryptionDefault<$backend>>::ckks_encrypt_sk_default(
                module, &mut ct, pt, sk, enc_infos, source_xa, source_xe, scratch,
            )
        }

        fn ckks_decrypt_tmp_bytes<A>(
            module: &poulpy_hal::layouts::Module<$backend>,
            ct_infos: &A,
        ) -> usize
        where
            A: poulpy_core::layouts::GLWEInfos + $crate::CKKSInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEDecrypt<$backend>
                + poulpy_hal::api::VecZnxLshBackend<$backend>
                + poulpy_hal::api::VecZnxLshTmpBytes
                + poulpy_hal::api::VecZnxRshBackend<$backend>
                + poulpy_hal::api::VecZnxRshTmpBytes,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::encryption::CKKSEncryptionDefault<$backend>>::ckks_decrypt_tmp_bytes_default(module, ct_infos)
        }

        fn ckks_decrypt<S, Pt>(
            module: &poulpy_hal::layouts::Module<$backend>,
            pt: &mut Pt,
            ct: &$crate::CKKSCiphertextRef<'_, $backend>,
            sk: &S,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, $backend>,
        ) -> anyhow::Result<()>
        where
            Pt: $crate::CKKSPlaintextVecZnxToBackendMut<$backend> + poulpy_core::layouts::LWEInfos + $crate::CKKSInfos + $crate::SetCKKSInfos,
            S: poulpy_core::layouts::GLWESecretPreparedToBackendRef<$backend> + poulpy_core::layouts::GLWEInfos,
            poulpy_hal::layouts::Module<$backend>: poulpy_core::GLWEDecrypt<$backend>
                + poulpy_hal::api::VecZnxLshBackend<$backend>
                + poulpy_hal::api::VecZnxLshTmpBytes
                + poulpy_hal::api::VecZnxRshBackend<$backend>
                + poulpy_hal::api::VecZnxRshTmpBytes
                + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = <$backend as poulpy_hal::layouts::Backend>::OwnedBuf>
                + $crate::leveled::default::pt_znx::CKKSPlaintextZnxDefault<$backend>,
            <$backend as poulpy_hal::layouts::Backend>::OwnedBuf: poulpy_hal::layouts::HostDataMut,
            for<'a> poulpy_hal::layouts::ScratchArena<'a, $backend>: poulpy_core::ScratchArenaTakeCore<'a, $backend>,
        {
            <poulpy_hal::layouts::Module<$backend> as $crate::leveled::default::encryption::CKKSEncryptionDefault<$backend>>::ckks_decrypt_default(
                module, pt, &ct, sk, scratch,
            )
        }
    };
}

pub use crate::impl_ckks_encryption_default_methods;
