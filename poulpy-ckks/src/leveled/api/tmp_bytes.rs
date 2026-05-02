use crate::CKKSInfos;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos};
use poulpy_hal::layouts::Backend;

/// Helpers that return the maximum scratch size needed across broad CKKS
/// operation sets.
pub trait CKKSAllOpsTmpBytes<BE: Backend> {
    /// Returns a scratch size large enough for the common CKKS workflow using
    /// ciphertext ops, plaintext ops, encryption/decryption, multiplication,
    /// and tensor-key setup.
    fn ckks_all_ops_tmp_bytes<C, T, P>(&self, ct_infos: &C, tsk_infos: &T, pt_prec: &P) -> usize
    where
        C: GLWEInfos + CKKSInfos,
        T: GGLWEInfos,
        P: CKKSInfos;

    /// Returns a scratch size large enough for [`Self::ckks_all_ops_tmp_bytes`]
    /// plus automorphism-key setup, rotation, and conjugation.
    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A, P>(&self, ct_infos: &C, tsk_infos: &T, atk_infos: &A, pt_prec: &P) -> usize
    where
        C: GLWEInfos + CKKSInfos,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos;
}

// Re-export the CKKSImpl bound so downstream doesn't need to import it separately
#[allow(unused_imports)]
use crate::oep::CKKSImpl as _CKKSImpl;
