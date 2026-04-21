//! Scratch-size helpers for CKKS workflows.
//!
//! These helpers return a single maximum scratch requirement covering a broad
//! set of CKKS operations, so callers do not need to manually chain
//! `max(...)` over every individual `*_tmp_bytes` method.

use crate::{
    CKKSMeta,
    leveled::{
        encryption::{CKKSDecrypt, CKKSEncrypt},
        operations::{
            add::CKKSAddOps, conjugate::CKKSConjugateOps, mul::CKKSMulOps, neg::CKKSNegOps, pow2::CKKSPow2Ops,
            rotate::CKKSRotateOps, sub::CKKSSubOps,
        },
        rescale::CKKSRescaleOps,
    },
    oep::CKKSImpl,
};
use poulpy_core::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWEShift, GLWETensorKeyEncryptSk,
    GLWETensoring,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWETensorKeyPreparedFactory},
};
use poulpy_hal::{
    api::{ModuleN, VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, Module},
};

/// Helpers that return the maximum scratch size needed across broad CKKS
/// operation sets.
pub trait CKKSAllOpsTmpBytes<BE: Backend> {
    /// Returns a scratch size large enough for the common CKKS workflow using
    /// ciphertext ops, plaintext ops, encryption/decryption, multiplication,
    /// and tensor-key setup.
    ///
    /// Inputs:
    /// - `ct_infos`: ciphertext layout used by runtime CKKS operations
    /// - `tsk_infos`: tensor-key layout used by multiplication and tensor-key
    ///   setup
    /// - `pt_prec`: representative plaintext precision used for RNX/constant
    ///   conversion scratch estimates
    ///
    /// Output:
    /// - the maximum `tmp_bytes` across the supported operation set
    ///
    /// Behavior:
    /// - includes secret-key encryption/decryption scratch
    /// - includes ciphertext add/sub/neg/pow2/rescale scratch
    /// - includes ZNX/RNX plaintext add/sub/mul scratch
    /// - includes ciphertext-ciphertext multiply/square and constant multiply
    /// - includes tensor-key encryption and preparation scratch
    fn ckks_all_ops_tmp_bytes<C, T>(&self, ct_infos: &C, tsk_infos: &T, pt_prec: &CKKSMeta) -> usize
    where
        C: GLWEInfos,
        T: GGLWEInfos;

    /// Returns a scratch size large enough for [`Self::ckks_all_ops_tmp_bytes`]
    /// plus automorphism-key setup, rotation, and conjugation.
    ///
    /// Inputs:
    /// - `ct_infos`: ciphertext layout used by runtime CKKS operations
    /// - `tsk_infos`: tensor-key layout used by multiplication and tensor-key
    ///   setup
    /// - `atk_infos`: automorphism-key layout used by rotation/conjugation and
    ///   automorphism-key setup
    /// - `pt_prec`: representative plaintext precision used for RNX/constant
    ///   conversion scratch estimates
    ///
    /// Output:
    /// - the maximum `tmp_bytes` across the supported operation set including
    ///   automorphisms
    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A>(&self, ct_infos: &C, tsk_infos: &T, atk_infos: &A, pt_prec: &CKKSMeta) -> usize
    where
        C: GLWEInfos,
        T: GGLWEInfos,
        A: GGLWEInfos;
}

impl<BE: Backend + CKKSImpl<BE>> CKKSAllOpsTmpBytes<BE> for Module<BE>
where
    Self: CKKSEncrypt<BE>
        + CKKSDecrypt<BE>
        + CKKSAddOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSSubOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRescaleOps<BE>
        + CKKSRotateOps<BE>
        + CKKSMulOps<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + ModuleN
        + GLWEShift<BE>
        + GLWEMulPlain<BE>
        + GLWEMulConst<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + VecZnxLsh<BE>
        + VecZnxLshTmpBytes
        + VecZnxRsh<BE>
        + VecZnxRshAddInto<BE>
        + VecZnxRshSub<BE>
        + VecZnxRshTmpBytes,
{
    fn ckks_all_ops_tmp_bytes<C, T>(&self, ct_infos: &C, tsk_infos: &T, pt_prec: &CKKSMeta) -> usize
    where
        C: GLWEInfos,
        T: GGLWEInfos,
    {
        self.ckks_encrypt_sk_tmp_bytes(ct_infos)
            .max(self.ckks_decrypt_tmp_bytes(ct_infos))
            .max(self.ckks_add_tmp_bytes())
            .max(self.ckks_add_pt_vec_znx_tmp_bytes())
            .max(self.ckks_add_pt_vec_rnx_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_add_const_tmp_bytes())
            .max(self.ckks_sub_tmp_bytes())
            .max(self.ckks_sub_pt_vec_znx_tmp_bytes())
            .max(self.ckks_sub_pt_vec_rnx_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_sub_const_tmp_bytes())
            .max(self.ckks_neg_tmp_bytes())
            .max(self.ckks_mul_pow2_tmp_bytes())
            .max(self.ckks_div_pow2_tmp_bytes())
            .max(self.ckks_rescale_tmp_bytes())
            .max(self.ckks_align_tmp_bytes())
            .max(self.ckks_mul_tmp_bytes(ct_infos, tsk_infos))
            .max(self.ckks_square_tmp_bytes(ct_infos, tsk_infos))
            .max(self.ckks_mul_pt_vec_znx_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_mul_pt_vec_rnx_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_mul_const_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.prepare_tensor_key_tmp_bytes(tsk_infos))
            .max(self.glwe_tensor_key_encrypt_sk_tmp_bytes(tsk_infos))
    }

    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A>(&self, ct_infos: &C, tsk_infos: &T, atk_infos: &A, pt_prec: &CKKSMeta) -> usize
    where
        C: GLWEInfos,
        T: GGLWEInfos,
        A: GGLWEInfos,
    {
        self.ckks_all_ops_tmp_bytes(ct_infos, tsk_infos, pt_prec)
            .max(self.ckks_rotate_tmp_bytes(ct_infos, atk_infos))
            .max(self.ckks_conjugate_tmp_bytes(ct_infos, atk_infos))
            .max(self.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk_infos))
            .max(self.glwe_automorphism_key_prepare_tmp_bytes(atk_infos))
    }
}
