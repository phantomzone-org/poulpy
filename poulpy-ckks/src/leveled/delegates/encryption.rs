use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, HostDataMut, Module, ScratchArena},
    source::Source,
};

use crate::{
    CKKSInfos, SetCKKSInfos,
    leveled::{
        api::{CKKSDecrypt, CKKSEncrypt},
        oep::CKKSEncryptionOep,
    },
    oep::CKKSImpl,
};

impl<BE: Backend + CKKSImpl<BE>> CKKSEncrypt<BE> for Module<BE>
where
    Self: CKKSEncryptionOep<BE> + GLWEEncryptSk<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
    {
        CKKSEncryptionOep::ckks_encrypt_sk_tmp_bytes(self, ct_infos)
    }

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
        Dct: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Dpt: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        BE: 's,
    {
        CKKSEncryptionOep::ckks_encrypt_sk(self, ct, pt, sk, enc_infos, source_xa, source_xe, scratch)
    }
}

impl<BE: Backend + CKKSImpl<BE>> CKKSDecrypt<BE> for Module<BE>
where
    Self: CKKSEncryptionOep<BE>
        + GLWEDecrypt<BE>
        + VecZnxLshBackend<BE>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<BE>
        + VecZnxRshTmpBytes
        + poulpy_core::layouts::ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    BE::OwnedBuf: HostDataMut,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
    {
        CKKSEncryptionOep::ckks_decrypt_tmp_bytes(self, ct_infos)
    }

    fn ckks_decrypt<Dpt, Dct, S>(&self, pt: &mut Dpt, ct: &Dct, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        Dpt: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        CKKSEncryptionOep::ckks_decrypt(self, pt, ct, sk, scratch)
    }
}
