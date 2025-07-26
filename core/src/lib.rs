#![feature(trait_alias)]
mod blind_rotation;
mod dist;
mod elem;
mod gglwe;
mod ggsw;
mod glwe;
mod lwe;
mod noise;

use backend::Backend;
use backend::Module;
use backend::SvpPPolAllocBytes;
use backend::VecZnxDftAllocBytes;
pub use blind_rotation::{
    BlindRotationKeyCGGI, BlindRotationKeyCGGIExec, BlindRotationKeyCGGIExecLayoutFamily, CCGIBlindRotationFamily, LookUpTable,
    cggi_blind_rotate, cggi_blind_rotate_scratch_space,
};
pub use elem::{Infos, SetMetaData};
pub use gglwe::{
    AutomorphismKey, AutomorphismKeyEncryptSkFamily, AutomorphismKeyExec, GGLWECiphertext, GGLWECiphertextExec,
    GGLWEEncryptSkFamily, GGLWEExecLayoutFamily, GLWESwitchingKey, GLWESwitchingKeyEncryptSkFamily, GLWESwitchingKeyExec,
    GLWETensorKey, GLWETensorKeyEncryptSkFamily, GLWETensorKeyExec,
};
pub use ggsw::{
    GGSWAssertNoiseFamily, GGSWCiphertext, GGSWCiphertextExec, GGSWEncryptSkFamily, GGSWKeySwitchFamily, GGSWLayoutFamily,
};
pub use glwe::{
    AutomorphismExecFamily, GLWECiphertext, GLWEDecryptFamily, GLWEEncryptPkFamily, GLWEEncryptSkFamily,
    GLWEExternalProductFamily, GLWEKeyswitchFamily, GLWEOps, GLWEPacker, GLWEPlaintext, GLWEPublicKey, GLWEPublicKeyFamily,
    GLWESecret, GLWESecretExec, GLWESecretFamily,
};
pub use lwe::{LWECiphertext, LWESecret};

pub use backend;
pub use backend::Scratch;
pub use backend::ScratchOwned;
pub(crate) use glwe::{GLWECiphertextToMut, GLWECiphertextToRef};

use crate::dist::Distribution;

pub(crate) const SIX_SIGMA: f64 = 6.0;

pub trait ScratchCore<B: Backend> {
    fn tmp_glwe_ct(&mut self, module: &Module<B>, basek: usize, k: usize, rank: usize) -> (GLWECiphertext<&mut [u8]>, &mut Self);
    fn tmp_vec_glwe_ct(
        &mut self,
        size: usize,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (Vec<GLWECiphertext<&mut [u8]>>, &mut Self);
    fn tmp_glwe_pt(&mut self, module: &Module<B>, basek: usize, k: usize) -> (GLWEPlaintext<&mut [u8]>, &mut Self);
    fn tmp_gglwe(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertext<&mut [u8]>, &mut Self);
    fn tmp_ggsw(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGSWCiphertext<&mut [u8]>, &mut Self);
    fn tmp_glwe_secret(&mut self, module: &Module<B>, rank: usize) -> (GLWESecret<&mut [u8]>, &mut Self);
    fn tmp_glwe_secret_exec(&mut self, module: &Module<B>, rank: usize) -> (GLWESecretExec<&mut [u8], B>, &mut Self)
    where
        Module<B>: SvpPPolAllocBytes;
    fn tmp_glwe_pk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWEPublicKey<&mut [u8], B>, &mut Self)
    where
        Module<B>: VecZnxDftAllocBytes;
    fn tmp_glwe_ksk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GLWESwitchingKey<&mut [u8]>, &mut Self);
    fn tmp_tsk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GLWETensorKey<&mut [u8]>, &mut Self);
    fn tmp_autokey(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (AutomorphismKey<&mut [u8]>, &mut Self);
}

impl<B: Backend> ScratchCore<B> for Scratch {
    fn tmp_glwe_ct(&mut self, module: &Module<B>, basek: usize, k: usize, rank: usize) -> (GLWECiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_vec_znx(module, rank + 1, k.div_ceil(basek));
        (GLWECiphertext { data, basek, k }, scratch)
    }

    fn tmp_vec_glwe_ct(
        &mut self,
        size: usize,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (Vec<GLWECiphertext<&mut [u8]>>, &mut Self) {
        let mut scratch: &mut Scratch = self;
        let mut cts: Vec<GLWECiphertext<&mut [u8]>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.tmp_glwe_ct(module, basek, k, rank);
            scratch = new_scratch;
            cts.push(ct);
        }
        (cts, scratch)
    }

    fn tmp_glwe_pt(&mut self, module: &Module<B>, basek: usize, k: usize) -> (GLWEPlaintext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_vec_znx(module, 1, k.div_ceil(basek));
        (GLWEPlaintext { data, basek, k }, scratch)
    }

    fn tmp_gglwe(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_mat_znx(
            module,
            rows.div_ceil(digits),
            rank_in,
            rank_out + 1,
            k.div_ceil(basek),
        );
        (
            GGLWECiphertext {
                data: data,
                basek: basek,
                k,
                digits,
            },
            scratch,
        )
    }

    fn tmp_ggsw(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGSWCiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_mat_znx(
            module,
            rows.div_ceil(digits),
            rank + 1,
            rank + 1,
            k.div_ceil(basek),
        );
        (
            GGSWCiphertext {
                data,
                basek,
                k,
                digits,
            },
            scratch,
        )
    }

    fn tmp_glwe_pk(&mut self, module: &Module<B>, basek: usize, k: usize, rank: usize) -> (GLWEPublicKey<&mut [u8], B>, &mut Self)
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        let (data, scratch) = self.tmp_vec_znx_dft(module, rank + 1, k.div_ceil(basek));
        (
            GLWEPublicKey {
                data,
                k,
                basek,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }

    fn tmp_glwe_secret(&mut self, module: &Module<B>, rank: usize) -> (GLWESecret<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_scalar_znx(module, rank);
        (
            GLWESecret {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }

    fn tmp_glwe_secret_exec(&mut self, module: &Module<B>, rank: usize) -> (GLWESecretExec<&mut [u8], B>, &mut Self)
    where
        Module<B>: SvpPPolAllocBytes,
    {
        let (data, scratch) = self.tmp_svp_ppol(module, rank);
        (
            GLWESecretExec {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }

    fn tmp_glwe_ksk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GLWESwitchingKey<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_gglwe(module, basek, k, rows, digits, rank_in, rank_out);
        (
            GLWESwitchingKey {
                key: data,
                sk_in_n: 0,
                sk_out_n: 0,
            },
            scratch,
        )
    }

    fn tmp_autokey(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (AutomorphismKey<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.tmp_glwe_ksk(module, basek, k, rows, digits, rank, rank);
        (AutomorphismKey { key: data, p: 0 }, scratch)
    }

    fn tmp_tsk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GLWETensorKey<&mut [u8]>, &mut Self) {
        let mut keys: Vec<GLWESwitchingKey<&mut [u8]>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);

        let mut scratch: &mut Scratch = self;

        if pairs != 0 {
            let (gglwe, s) = scratch.tmp_glwe_ksk(module, basek, k, rows, digits, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.tmp_glwe_ksk(module, basek, k, rows, digits, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        (GLWETensorKey { keys }, scratch)
    }
}
