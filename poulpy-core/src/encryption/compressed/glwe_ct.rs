use poulpy_hal::{
    layouts::{Backend, DataMut, Module, Scratch},
    source::Source,
};

use crate::{
    encryption::{GLWEEncryptSk, GLWEEncryptSkInternal, SIGMA},
    layouts::{
        GLWECompressedSeedMut, GLWEInfos, GLWEPlaintextToRef, LWEInfos,
        compressed::{GLWECompressed, GLWECompressedToMut},
        prepared::GLWESecretPreparedToRef,
    },
};

impl GLWECompressed<Vec<u8>> {
    /// Returns the scratch buffer size in bytes required by [`GLWECompressed::encrypt_sk`].
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GLWEInfos,
        M: GLWECompressedEncryptSk<BE>,
    {
        module.glwe_compressed_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> GLWECompressed<D> {
    /// Encrypts a plaintext under a secret key, producing a compressed GLWE ciphertext.
    ///
    /// The mask is deterministically generated from `seed_xa` rather than stored explicitly.
    /// Only the ciphertext body and the seed are retained in the output.
    ///
    /// - `pt`: the plaintext to encrypt.
    /// - `sk`: the GLWE secret key in prepared form.
    /// - `seed_xa`: seed for deterministic mask generation.
    /// - `source_xe`: PRNG source for sampling encryption noise.
    /// - `scratch`: scratch buffer (see [`GLWECompressed::encrypt_sk_tmp_bytes`] for sizing).
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, P, S, BE: Backend>(
        &mut self,
        module: &M,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        M: GLWECompressedEncryptSk<BE>,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<BE>,
    {
        module.glwe_compressed_encrypt_sk(self, pt, sk, seed_xa, source_xe, scratch);
    }
}

/// Compressed secret-key encryption of a GLWE ciphertext.
///
/// Produces a [`GLWECompressed`] where the mask is derived from a seed
/// instead of being stored explicitly.
pub trait GLWECompressedEncryptSk<BE: Backend> {
    /// Returns the scratch buffer size in bytes required by
    /// [`glwe_compressed_encrypt_sk`](Self::glwe_compressed_encrypt_sk).
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Encrypts a plaintext under a GLWE secret key into a compressed GLWE ciphertext.
    ///
    /// - `res`: output compressed GLWE ciphertext.
    /// - `pt`: the plaintext to encrypt.
    /// - `sk`: the GLWE secret key in prepared form.
    /// - `seed_xa`: seed for deterministic mask generation.
    /// - `source_xe`: PRNG source for sampling encryption noise.
    /// - `scratch`: scratch buffer.
    fn glwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWECompressedToMut + GLWECompressedSeedMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GLWECompressedEncryptSk<BE> for Module<BE>
where
    Self: GLWEEncryptSkInternal<BE> + GLWEEncryptSk<BE>,
{
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_encrypt_sk_tmp_bytes(infos)
    }

    fn glwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWECompressedToMut + GLWECompressedSeedMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<BE>,
    {
        {
            let res: &mut GLWECompressed<&mut [u8]> = &mut res.to_mut();
            let mut source_xa: Source = Source::new(seed_xa);
            let cols: usize = (res.rank() + 1).into();

            self.glwe_encrypt_sk_internal(
                res.base2k().into(),
                res.k().into(),
                &mut res.data,
                cols,
                true,
                Some((pt, 0)),
                sk,
                &mut source_xa,
                source_xe,
                SIGMA,
                scratch,
            );
        }

        res.seed_mut().copy_from_slice(&seed_xa);
    }
}
