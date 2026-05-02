use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, ModuleCoreAlloc};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos, oep::CKKSImpl};

pub(crate) trait CKKSEncryptionOep<BE: Backend + CKKSImpl<BE>> {
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Self: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes;

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<'s, Dct, Pt, S, E>(
        &self,
        ct: &mut Dct,
        pt: &Pt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dct: GLWEToBackendMut<BE> + SetCKKSInfos + LWEInfos,
        E: EncryptionInfos,
        Pt: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        Self: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>;

    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Self: GLWEDecrypt<BE> + VecZnxLshBackend<BE> + VecZnxLshTmpBytes + VecZnxRshBackend<BE> + VecZnxRshTmpBytes;

    fn ckks_decrypt<Pt, Dct, S>(&self, pt: &mut Pt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Pt: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Self: GLWEDecrypt<BE>
            + VecZnxLshBackend<BE>
            + VecZnxLshTmpBytes
            + VecZnxRshBackend<BE>
            + VecZnxRshTmpBytes
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSEncryptionOep<BE> for Module<BE> {
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Self: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_encrypt_sk_tmp_bytes(self, ct_infos)
    }

    fn ckks_encrypt_sk<'s, Dct, Pt, S, E>(
        &self,
        ct: &mut Dct,
        pt: &Pt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dct: GLWEToBackendMut<BE> + CKKSInfos + LWEInfos + SetCKKSInfos,
        E: EncryptionInfos,
        Pt: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        Self: GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_encrypt_sk(self, ct, pt, sk, enc_infos, source_xa, source_xe, scratch)
    }

    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
        Self: GLWEDecrypt<BE> + VecZnxLshBackend<BE> + VecZnxLshTmpBytes + VecZnxRshBackend<BE> + VecZnxRshTmpBytes,
    {
        BE::ckks_decrypt_tmp_bytes(self, ct_infos)
    }

    fn ckks_decrypt<Pt, Dct, S>(&self, pt: &mut Pt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Pt: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Self: GLWEDecrypt<BE>
            + VecZnxLshBackend<BE>
            + VecZnxLshTmpBytes
            + VecZnxRshBackend<BE>
            + VecZnxRshTmpBytes
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        BE::ckks_decrypt(self, pt, ct, sk, scratch)
    }
}
