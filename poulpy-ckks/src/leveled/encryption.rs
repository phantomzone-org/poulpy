//! CKKS secret-key encryption and decryption.

use crate::{
    CKKSInfos, checked_log_hom_rem_sub,
    layouts::{CKKSCiphertext, plaintext::CKKSPlaintextVecZnx},
    leveled::operations::pt_znx::CKKSPlaintextZnxOps,
};
use poulpy_core::{
    EncryptionInfos, GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{GLWEInfos, GLWEPlaintext, GLWESecretPreparedToRef},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
    source::Source,
};

use anyhow::Result;

pub trait CKKSEncrypt<BE: Backend> {
    /// Returns the scratch size, in bytes, required by [`Self::ckks_encrypt_sk`].
    ///
    /// The returned size depends on the ciphertext layout and backend.
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Encrypts a CKKS plaintext vector under a secret key.
    ///
    /// Inputs:
    /// - `ct`: destination ciphertext buffer
    /// - `pt`: plaintext vector to encrypt
    /// - `sk`: prepared secret key
    /// - `enc_infos`: encryption/noise parameters
    /// - `source_xa`, `source_xe`: randomness sources used by the backend
    /// - `scratch`: temporary workspace
    ///
    /// Output:
    /// - Returns `Ok(())` and writes the encrypted ciphertext into `ct`
    ///
    /// Behavior:
    /// - encrypts zero first, then injects `pt` into the ciphertext
    /// - sets `ct.log_decimal = pt.log_decimal()`
    /// - sets `ct.log_hom_rem = enc_infos.noise_infos().k - pt.log_decimal()`
    ///
    /// Errors:
    /// - `InsufficientHomomorphicCapacity` if the requested plaintext
    ///   `log_decimal` exceeds the available encryption precision `k`
    /// - any alignment/base mismatch error returned while adding the plaintext
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
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> CKKSEncrypt<BE> for Module<BE>
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

pub trait CKKSDecrypt<BE: Backend> {
    /// Returns the scratch size, in bytes, required by [`Self::ckks_decrypt`].
    ///
    /// The returned size includes raw GLWE decryption plus plaintext extraction.
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Decrypts a ciphertext into a caller-provided CKKS plaintext layout.
    ///
    /// Inputs:
    /// - `pt`: destination plaintext buffer and metadata
    /// - `ct`: source ciphertext
    /// - `sk`: prepared secret key
    /// - `scratch`: temporary workspace
    ///
    /// Output:
    /// - Returns `Ok(())` and fills `pt` on success
    ///
    /// Behavior:
    /// - first decrypts `ct` into a raw GLWE plaintext buffer
    /// - then calls [`CKKSPlaintextZnxOps::ckks_extract_pt_znx`] to align that
    ///   buffer into the destination CKKS plaintext format
    /// - the destination metadata is respected rather than rewritten
    ///
    /// Errors:
    /// - any raw decryption backend error
    /// - `PlaintextBase2KMismatch` if `pt` and `ct` use different `base2k`
    /// - `PlaintextAlignmentImpossible` if `pt` requests more semantic
    ///   precision than `ct` can provide
    fn ckks_decrypt<S>(
        &self,
        pt: &mut CKKSPlaintextVecZnx<impl DataMut>,
        ct: &CKKSCiphertext<impl DataRef>,
        sk: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
}

impl<BE: Backend> CKKSDecrypt<BE> for Module<BE>
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
