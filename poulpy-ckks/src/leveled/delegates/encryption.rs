use anyhow::Result;
use poulpy_core::layouts::{GLWEPlaintext, GLWESecretPreparedToRef};
use poulpy_core::{EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore, layouts::GLWEInfos};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use crate::{
    CKKSInfos, checked_log_hom_rem_sub,
    layouts::{CKKSCiphertext, plaintext::CKKSPlaintextVecZnx},
    leveled::api::{CKKSDecrypt, CKKSEncrypt, CKKSPlaintextZnxOps},
    oep::CKKSImpl,
};

impl<BE: Backend + CKKSImpl<BE>> CKKSEncrypt<BE> for Module<BE>
where
    Self: GLWEEncryptSk<BE> + VecZnxRshAddInto<BE> + VecZnxRshTmpBytes,
{
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_encrypt_sk_tmp_bytes(ct_infos).max(self.ckks_add_pt_znx_tmp_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<S, E: EncryptionInfos>(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        pt: &CKKSPlaintextVecZnx<impl DataRef>,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_encrypt_zero_sk(ct, sk, enc_infos, source_xe, source_xa, scratch);
        let log_hom_rem = checked_log_hom_rem_sub("ckks_encrypt_sk", enc_infos.noise_infos().k, pt.log_decimal())?;
        ct.meta.log_hom_rem = log_hom_rem;
        ct.meta.log_decimal = pt.log_decimal();
        self.ckks_add_pt_vec_znx(ct, pt, scratch)?;
        Ok(())
    }
}

impl<BE: Backend + CKKSImpl<BE>> CKKSDecrypt<BE> for Module<BE>
where
    Self: GLWEDecrypt<BE> + VecZnxLsh<BE> + VecZnxLshTmpBytes + VecZnxRsh<BE> + VecZnxRshTmpBytes,
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

    fn ckks_decrypt<S>(
        &self,
        pt: &mut CKKSPlaintextVecZnx<impl DataMut>,
        ct: &CKKSCiphertext<impl DataRef>,
        sk: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let (mut full_pt, scratch_rest) = scratch.take_glwe_plaintext(ct);
        self.glwe_decrypt(ct, &mut full_pt, sk, scratch_rest);
        self.ckks_extract_pt_znx(pt, &full_pt, ct, scratch_rest)?;
        Ok(())
    }
}
