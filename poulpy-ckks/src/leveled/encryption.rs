//! CKKS secret-key encryption and decryption.

use crate::{
    CKKSInfos,
    layouts::plaintext::{CKKSPlaintextZnx, attach_meta},
    leveled::operations::pt_znx::CKKSPlaintextZnxOps,
};
use poulpy_core::{
    EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{GLWEInfos, GLWEPlaintext, GLWESecretPreparedToRef, GLWEToMut, GLWEToRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxNormalizeTmpBytes, VecZnxRshAddInto},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use anyhow::{Ok, Result};

pub trait CKKSEncrypt<BE: Backend> {
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<C, S, E: EncryptionInfos>(
        &self,
        ct: &mut C,
        pt: &CKKSPlaintextZnx<impl DataRef>,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        C: GLWEToMut + LWEInfos + CKKSInfos,
        S: GLWESecretPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> CKKSEncrypt<BE> for Module<BE>
where
    Self: GLWEEncryptSk<BE> + VecZnxNormalizeTmpBytes + VecZnxRshAddInto<BE>,
{
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        GLWEPlaintext::<Vec<u8>, ()>::bytes_of_from_infos(ct_infos)
            + self
                .vec_znx_normalize_tmp_bytes()
                .max(self.glwe_encrypt_sk_tmp_bytes(ct_infos))
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<C, S, E: EncryptionInfos>(
        &self,
        ct: &mut C,
        pt: &CKKSPlaintextZnx<impl DataRef>,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        C: GLWEToMut + LWEInfos + CKKSInfos,
        S: GLWESecretPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_encrypt_zero_sk(ct, sk, enc_infos, source_xe, source_xa, scratch);
        let log_hom_rem = enc_infos.noise_infos().k - pt.meta.log_decimal;
        ct.set_log_hom_rem(log_hom_rem).unwrap();
        ct.set_log_decimal(pt.log_decimal()).unwrap();
        self.ckks_add_pt_znx(ct, pt, scratch)?;
        Ok(())
    }
}

pub trait CKKSDecrypt<BE: Backend> {
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;
    fn ckks_decrypt<C, S>(
        &self,
        pt: &mut CKKSPlaintextZnx<impl DataMut>,
        ct: &C,
        sk: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        C: GLWEToRef + GLWEInfos + CKKSInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> CKKSDecrypt<BE> for Module<BE>
where
    Self: GLWEDecrypt<BE> + VecZnxLsh<BE>,
{
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        GLWEPlaintext::<Vec<u8>, ()>::bytes_of_from_infos(ct_infos) + self.glwe_decrypt_tmp_bytes(ct_infos)
    }

    fn ckks_decrypt<C, S>(&self, pt: &mut CKKSPlaintextZnx<impl DataMut>, ct: &C, sk: &S, scratch: &mut Scratch<BE>) -> Result<()>
    where
        C: GLWEToRef + GLWEInfos + CKKSInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let (full_pt, scratch_rest) = scratch.take_glwe_plaintext(ct);
        let mut full_pt = attach_meta(full_pt, ct.meta());
        self.glwe_decrypt(ct, &mut full_pt, sk, scratch_rest);
        self.ckks_extract_pt_znx(pt, &full_pt, scratch_rest)?;
        Ok(())
    }
}
