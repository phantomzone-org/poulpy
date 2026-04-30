use anyhow::Result;
use poulpy_core::{EncryptionInfos, ScratchArenaTakeCore, layouts::GLWEInfos};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, ScratchArena},
};

use crate::{
    layouts::{CKKSCiphertext, plaintext::CKKSPlaintextVecZnx},
    oep::CKKSImpl,
};
use poulpy_core::layouts::GLWESecretPreparedToBackendRef;
use poulpy_hal::source::Source;

pub trait CKKSEncrypt<BE: Backend + CKKSImpl<BE>> {
    /// Returns the scratch size, in bytes, required by [`Self::ckks_encrypt_sk`].
    ///
    /// The returned size depends on the ciphertext layout and backend.
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Encrypts a CKKS plaintext vector under a secret key.
    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<'s, Dct: Data, Dpt: Data, S, E: EncryptionInfos>(
        &self,
        ct: &mut CKKSCiphertext<Dct>,
        pt: &CKKSPlaintextVecZnx<Dpt>,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        CKKSCiphertext<Dct>: poulpy_core::layouts::GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<Dpt>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;
}

pub trait CKKSDecrypt<BE: Backend + CKKSImpl<BE>> {
    /// Returns the scratch size, in bytes, required by [`Self::ckks_decrypt`].
    ///
    /// The returned size includes raw GLWE decryption plus plaintext extraction.
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Decrypts a ciphertext into a caller-provided CKKS plaintext layout.
    fn ckks_decrypt<Dpt: Data, Dct: Data, S>(
        &self,
        pt: &mut CKKSPlaintextVecZnx<Dpt>,
        ct: &CKKSCiphertext<Dct>,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        CKKSPlaintextVecZnx<Dpt>: poulpy_core::layouts::GLWEPlaintextToBackendMut<BE>,
        CKKSCiphertext<Dct>: poulpy_core::layouts::GLWEToBackendRef<BE>,
        BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

// Suppress unused import warnings for trait bounds used only in delegates
#[allow(unused_imports)]
use poulpy_hal::api::{VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshTmpBytes};
#[allow(unused_imports)]
use poulpy_hal::layouts::Module;
