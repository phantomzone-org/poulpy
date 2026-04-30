use anyhow::Result;
use poulpy_core::layouts::{
    GLWEInfos, GLWEPlaintext, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc,
};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, GLWEShift, ScratchArenaTakeCore};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Data, HostDataMut, Module, ScratchArena},
    source::Source,
};

use crate::{
    CKKSInfos, checked_log_budget_sub,
    layouts::{CKKSCiphertext, plaintext::CKKSPlaintextVecZnx},
    leveled::api::{CKKSAddOps, CKKSDecrypt, CKKSEncrypt, CKKSPlaintextZnxOps},
    oep::CKKSImpl,
};

impl<BE: Backend + CKKSImpl<BE>> CKKSEncrypt<BE> for Module<BE>
where
    Self: GLWEEncryptSk<BE> + GLWEShift<BE> + VecZnxRshAddIntoBackend<BE> + VecZnxRshTmpBytes,
{
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_encrypt_sk_tmp_bytes(ct_infos)
            .max(self.ckks_add_pt_vec_znx_tmp_bytes())
    }

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
        CKKSCiphertext<Dct>: GLWEToBackendMut<BE>,
        CKKSPlaintextVecZnx<Dpt>: poulpy_core::layouts::GLWEPlaintextToBackendRef<BE>,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
    {
        self.glwe_encrypt_zero_sk(ct, sk, enc_infos, source_xe, source_xa, scratch);
        let log_budget = checked_log_budget_sub("ckks_encrypt_sk", enc_infos.noise_infos().k, pt.log_delta())?;
        ct.meta.log_budget = log_budget;
        ct.meta.log_delta = pt.log_delta();
        self.ckks_add_pt_vec_znx_assign(ct, pt, scratch)?;
        Ok(())
    }
}

impl<BE: Backend + CKKSImpl<BE>> CKKSDecrypt<BE> for Module<BE>
where
    Self: GLWEDecrypt<BE>
        + VecZnxLshBackend<BE>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<BE>
        + VecZnxRshTmpBytes
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
{
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(ct_infos)
            + self
                .glwe_decrypt_tmp_bytes(ct_infos)
                .max(self.ckks_extract_pt_znx_tmp_bytes())
    }

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
        CKKSCiphertext<Dct>: GLWEToBackendRef<BE>,
        BE::OwnedBuf: HostDataMut,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let mut full_pt = self.glwe_plaintext_alloc_from_infos(ct);
        self.glwe_decrypt(ct, &mut full_pt, sk, scratch);
        self.ckks_extract_pt_znx(pt, &full_pt, ct, scratch)?;
        Ok(())
    }
}
