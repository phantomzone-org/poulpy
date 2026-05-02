use anyhow::Result;
use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::{CKKSCiphertextToBackendMut, CKKSCiphertextToBackendRef, CKKSPlaintexToBackendRef, CKKSPlaintextVecZnxToBackendMut};

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSImpl};

pub trait CKKSEncrypt<BE: Backend + CKKSImpl<BE>> {
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos;

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<'s, Dct, Dpt, S, E: EncryptionInfos>(
        &self,
        ct: &mut Dct,
        pt: &Dpt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE>,
        Dct: CKKSCiphertextToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Dpt: CKKSPlaintexToBackendRef<BE> + CKKSInfos,
        BE: 's;
}

pub trait CKKSDecrypt<BE: Backend + CKKSImpl<BE>> {
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos;

    fn ckks_decrypt<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: CKKSPlaintextVecZnxToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
        Dct: CKKSCiphertextToBackendRef<BE> + CKKSInfos;
}
