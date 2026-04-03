//! CKKS relinearization key (tensor switching key).

use poulpy_core::{
    GLWETensorKeyEncryptSk, GetDistribution, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWEInfos, GLWESecretToRef, GLWETensorKey, GLWETensorKeyLayout, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    layouts::{Backend, Data, Module, Scratch},
    source::Source,
};

/// Relinearization key for CKKS ciphertext × ciphertext multiplication.
///
/// Wraps a [`GLWETensorKey`] with rank-1 and CKKS-specific layout parameters.
/// The key precision is `k + base2k` to accommodate the extra digit needed
/// by the tensor switching decomposition.
///
/// ## Lifecycle
///
/// 1. Allocate with [`CKKSTensorKey::alloc`].
/// 2. Encrypt with [`CKKSTensorKey::encrypt_sk`].
/// 3. Prepare into a [`CKKSTensorKeyPrepared`](super::tensor_key_prepared::CKKSTensorKeyPrepared)
///    before evaluation.
pub struct CKKSTensorKey<D: Data> {
    pub inner: GLWETensorKey<D>,
}

impl CKKSTensorKey<Vec<u8>> {
    /// Allocates a new tensor key from CKKS parameters.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        CKKSTensorKey {
            inner: GLWETensorKey::alloc_from_infos(&Self::layout(n, base2k, k)),
        }
    }

    /// Computes the tensor key layout for CKKS (rank=1, dsize=1).
    ///
    /// The key precision is `k + base2k`: one extra digit to hold the
    /// decomposition products during relinearization.
    pub fn layout(n: Degree, base2k: Base2K, k: TorusPrecision) -> GLWETensorKeyLayout {
        let tsk_k = TorusPrecision(k.0 + base2k.0);
        GLWETensorKeyLayout {
            n,
            base2k,
            k: tsk_k,
            rank: Rank(1),
            dnum: Dnum(k.0.div_ceil(base2k.0)),
            dsize: Dsize(1),
        }
    }

    /// Encrypts the tensor key under a secret key.
    pub fn encrypt_sk<BE: Backend, S>(
        &mut self,
        module: &Module<BE>,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GLWETensorKeyEncryptSk<BE>,
        S: GLWESecretToRef + GetDistribution + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.inner.encrypt_sk(module, sk, source_xa, source_xe, scratch);
    }
}
