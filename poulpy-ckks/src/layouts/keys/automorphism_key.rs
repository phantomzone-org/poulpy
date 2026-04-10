//! CKKS automorphism key for slot rotations and conjugation.

use poulpy_core::{
    EncryptionInfos, GLWEAutomorphismKeyEncryptSk,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWESecretToRef, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    layouts::{Backend, Data, Module, Scratch},
    source::Source,
};

/// Automorphism key for CKKS slot rotations and complex conjugation.
///
/// Wraps a [`GLWEAutomorphismKey`] with rank-1 and CKKS-specific layout.
/// Each key is bound to a single Galois element `p`:
///
/// - Slot rotation by `r` positions uses `galois_element(r)`.
/// - Complex conjugation uses Galois element `-1`.
///
/// The key precision is `k + base2k` (same rationale as the tensor key).
///
/// ## Lifecycle
///
/// 1. Allocate with [`CKKSAutomorphismKey::alloc`].
/// 2. Encrypt for Galois element `p` with [`CKKSAutomorphismKey::encrypt_sk`].
/// 3. Prepare into a [`CKKSAutomorphismKeyPrepared`](super::automorphism_key_prepared::CKKSAutomorphismKeyPrepared)
///    before evaluation.
pub struct CKKSAutomorphismKey<D: Data> {
    pub inner: GLWEAutomorphismKey<D>,
}

impl CKKSAutomorphismKey<Vec<u8>> {
    /// Allocates a new automorphism key from CKKS parameters.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        CKKSAutomorphismKey {
            inner: GLWEAutomorphismKey::alloc_from_infos(&Self::layout(n, base2k, k)),
        }
    }

    /// Computes the automorphism key layout for CKKS (rank=1, dsize=1).
    ///
    /// The key precision is `k + base2k`: one extra digit for the
    /// decomposition products during automorphism evaluation.
    pub fn layout(n: Degree, base2k: Base2K, k: TorusPrecision) -> GLWEAutomorphismKeyLayout {
        let atk_k = TorusPrecision(k.0 + base2k.0);
        GLWEAutomorphismKeyLayout {
            n,
            base2k,
            k: atk_k,
            rank: Rank(1),
            dnum: Dnum(k.0.div_ceil(base2k.0)),
            dsize: Dsize(1),
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Encrypts the automorphism key for Galois element `p` under a secret key.
    pub fn encrypt_sk<BE: Backend, S, E: EncryptionInfos>(
        &mut self,
        module: &Module<BE>,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>,
        S: GLWESecretToRef,
    {
        self.inner.encrypt_sk(module, p, sk, enc_infos, source_xe, source_xa, scratch);
    }
}
