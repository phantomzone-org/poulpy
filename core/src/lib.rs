pub mod automorphism;
pub mod elem;
pub mod gglwe_ciphertext;
pub mod ggsw_ciphertext;
pub mod glwe_ciphertext;
pub mod glwe_ciphertext_fourier;
pub mod glwe_ops;
pub mod glwe_packing;
pub mod glwe_plaintext;
pub mod keys;
pub mod keyswitch_key;
pub mod tensor_key;
#[cfg(test)]
mod test_fft64;
pub mod trace;
mod utils;

pub use automorphism::*;
use backend::Backend;
use backend::FFT64;
use backend::Module;
pub use elem::*;
pub use gglwe_ciphertext::*;
pub use ggsw_ciphertext::*;
pub use glwe_ciphertext::*;
pub use glwe_ciphertext_fourier::*;
pub use glwe_ops::*;
pub use glwe_packing::*;
pub use glwe_plaintext::*;
pub use keys::*;
pub use keyswitch_key::*;
pub use tensor_key::*;

pub use backend::Scratch;
pub use backend::ScratchOwned;
use utils::derive_size;

pub(crate) const SIX_SIGMA: f64 = 6.0;

pub trait ScratchCore<B: Backend> {
    fn tmp_glwe(&mut self, module: &Module<B>, basek: usize, k: usize, rank: usize) -> (GLWECiphertext<&mut [u8]>, &mut Self);
    fn tmp_gglwe(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertext<&mut [u8], B>, &mut Self);
    fn tmp_ggsw(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
    ) -> (GGSWCiphertext<&mut [u8], B>, &mut Self);
    fn tmp_glwe_fourier(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWECiphertextFourier<&mut [u8], B>, &mut Self);
    fn tmp_sk(&mut self, module: &Module<B>, rank: usize) -> (SecretKey<&mut [u8]>, &mut Self);
    fn tmp_sk_fourier(&mut self, module: &Module<B>, rank: usize) -> (SecretKeyFourier<&mut [u8], B>, &mut Self);
    fn tmp_glwe_pk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWEPublicKey<&mut [u8], B>, &mut Self);
    fn tmp_glwe_ksk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GLWESwitchingKey<&mut [u8], B>, &mut Self);
    fn tmp_tsk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
    ) -> (TensorKey<&mut [u8], B>, &mut Self);
    fn tmp_autokey(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
    ) -> (AutomorphismKey<&mut [u8], B>, &mut Self);
}

impl ScratchCore<FFT64> for Scratch {
    fn tmp_glwe(
        &mut self,
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWECiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_vec_znx(module, rank + 1, derive_size(basek, k));
        (GLWECiphertext { data, basek, k }, scratch)
    }

    fn tmp_gglwe(
        &mut self,
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertext<&mut [u8], FFT64>, &mut Self) {
        let (data, scratch) = self.tmp_mat_znx_dft(module, rows, rank_in, rank_out + 1, derive_size(basek, k));
        (
            GGLWECiphertext {
                data: data,
                basek: basek,
                k,
            },
            scratch,
        )
    }

    fn tmp_ggsw(
        &mut self,
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
    ) -> (GGSWCiphertext<&mut [u8], FFT64>, &mut Self) {
        let (data, scratch) = self.tmp_mat_znx_dft(module, rows, rank + 1, rank + 1, derive_size(basek, k));
        (
            GGSWCiphertext {
                data: data,
                basek: basek,
                k,
            },
            scratch,
        )
    }

    fn tmp_glwe_fourier(
        &mut self,
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWECiphertextFourier<&mut [u8], FFT64>, &mut Self) {
        let (data, scratch) = self.tmp_vec_znx_dft(module, rank + 1, derive_size(basek, k));
        (GLWECiphertextFourier { data, basek, k }, scratch)
    }

    fn tmp_glwe_pk(
        &mut self,
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWEPublicKey<&mut [u8], FFT64>, &mut Self) {
        let (data, scratch) = self.tmp_glwe_fourier(module, basek, k, rank);
        (
            GLWEPublicKey {
                data,
                dist: SecretDistribution::NONE,
            },
            scratch,
        )
    }

    fn tmp_sk(&mut self, module: &Module<FFT64>, rank: usize) -> (SecretKey<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_scalar_znx(module, rank + 1);
        (
            SecretKey {
                data,
                dist: SecretDistribution::NONE,
            },
            scratch,
        )
    }

    fn tmp_sk_fourier(&mut self, module: &Module<FFT64>, rank: usize) -> (SecretKeyFourier<&mut [u8], FFT64>, &mut Self) {
        let (data, scratch) = self.tmp_scalar_znx_dft(module, rank + 1);
        (
            SecretKeyFourier {
                data,
                dist: SecretDistribution::NONE,
            },
            scratch,
        )
    }

    fn tmp_glwe_ksk(
        &mut self,
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GLWESwitchingKey<&mut [u8], FFT64>, &mut Self) {
        let (data, scratch) = self.tmp_gglwe(module, basek, k, rows, rank_in, rank_out);
        (GLWESwitchingKey(data), scratch)
    }

    fn tmp_autokey(
        &mut self,
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
    ) -> (AutomorphismKey<&mut [u8], FFT64>, &mut Self) {
        let (data, scratch) = self.tmp_glwe_ksk(module, basek, k, rows, rank, rank);
        (AutomorphismKey { key: data, p: 0 }, scratch)
    }

    fn tmp_tsk(
        &mut self,
        module: &Module<FFT64>,
        basek: usize,
        k: usize,
        rows: usize,
        rank: usize,
    ) -> (TensorKey<&mut [u8], FFT64>, &mut Self) {
        let mut keys: Vec<GLWESwitchingKey<&mut [u8], FFT64>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);

        let mut scratch: &mut Scratch = self;

        if pairs != 0 {
            let (gglwe, s) = scratch.tmp_glwe_ksk(module, basek, k, rows, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.tmp_glwe_ksk(module, basek, k, rows, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        (TensorKey { keys }, scratch)
    }
}
