#![feature(trait_alias)]
mod blind_rotation;
mod dist;
mod elem;
mod gglwe;
mod ggsw;
mod glwe;
mod lwe;
mod noise;

use backend::hal::{
    api::{ScratchTakeMatZnx, ScratchTakeScalarZnx, ScratchTakeSvpPPol, ScratchTakeVecZnx, ScratchTakeVecZnxDft},
    layouts::{Backend, Module, Scratch},
    oep::{ScratchTakeSvpPPolImpl, ScratchTakeVecZnxDftImpl},
};
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
pub(crate) use glwe::{GLWECiphertextToMut, GLWECiphertextToRef};
pub use lwe::{LWECiphertext, LWESecret};

use crate::dist::Distribution;

pub(crate) const SIX_SIGMA: f64 = 6.0;

pub trait TakeGLWECt<B: Backend> {
    fn take_glwe_ct(&mut self, module: &Module<B>, basek: usize, k: usize, rank: usize)
    -> (GLWECiphertext<&mut [u8]>, &mut Self);
}

pub trait TakeGLWECtSlice<B: Backend> {
    fn take_glwe_ct_slice(
        &mut self,
        size: usize,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (Vec<GLWECiphertext<&mut [u8]>>, &mut Self);
}

pub trait TakeGLWEPt<B: Backend> {
    fn take_glwe_pt(&mut self, module: &Module<B>, basek: usize, k: usize) -> (GLWEPlaintext<&mut [u8]>, &mut Self);
}

pub trait TakeGGLWE<B: Backend> {
    fn take_gglwe(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertext<&mut [u8]>, &mut Self);
}

pub trait TakeGGSW<B: Backend> {
    fn take_ggsw(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGSWCiphertext<&mut [u8]>, &mut Self);
}

pub trait TakeGLWESecret<B: Backend> {
    fn take_glwe_secret(&mut self, module: &Module<B>, rank: usize) -> (GLWESecret<&mut [u8]>, &mut Self);
}

pub trait TakeGLWESecretExec<B: Backend> {
    fn take_glwe_secret_exec(&mut self, module: &Module<B>, rank: usize) -> (GLWESecretExec<&mut [u8], B>, &mut Self);
}

pub trait TakeGLWEPk<B: Backend> {
    fn take_glwe_pk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWEPublicKey<&mut [u8], B>, &mut Self);
}

pub trait TakeGLWESwitchingKey<B: Backend> {
    fn take_glwe_switching_key(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GLWESwitchingKey<&mut [u8]>, &mut Self);
}

pub trait TakeTensorKey<B: Backend> {
    fn take_tensor_key(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GLWETensorKey<&mut [u8]>, &mut Self);
}

pub trait TakeAutomorphismKey<B: Backend> {
    fn take_automorphism_key(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (AutomorphismKey<&mut [u8]>, &mut Self);
}

impl<B: Backend> TakeGLWECt<B> for Scratch<B> {
    fn take_glwe_ct(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWECiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_vec_znx(module, rank + 1, k.div_ceil(basek));
        (GLWECiphertext { data, basek, k }, scratch)
    }
}

impl<B: Backend> TakeGLWECtSlice<B> for Scratch<B> {
    fn take_glwe_ct_slice(
        &mut self,
        size: usize,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (Vec<GLWECiphertext<&mut [u8]>>, &mut Self) {
        let mut scratch: &mut Scratch<B> = self;
        let mut cts: Vec<GLWECiphertext<&mut [u8]>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_glwe_ct(module, basek, k, rank);
            scratch = new_scratch;
            cts.push(ct);
        }
        (cts, scratch)
    }
}

impl<B: Backend> TakeGLWEPt<B> for Scratch<B> {
    fn take_glwe_pt(&mut self, module: &Module<B>, basek: usize, k: usize) -> (GLWEPlaintext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_vec_znx(module, 1, k.div_ceil(basek));
        (GLWEPlaintext { data, basek, k }, scratch)
    }
}

impl<B: Backend> TakeGGLWE<B> for Scratch<B> {
    fn take_gglwe(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_mat_znx(
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
}

impl<B: Backend> TakeGGSW<B> for Scratch<B> {
    fn take_ggsw(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGSWCiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_mat_znx(
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
}

impl<B: Backend> TakeGLWEPk<B> for Scratch<B>
where
    B: ScratchTakeVecZnxDftImpl<B>,
{
    fn take_glwe_pk(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWEPublicKey<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_vec_znx_dft(module, rank + 1, k.div_ceil(basek));
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
}

impl<B: Backend> TakeGLWESecret<B> for Scratch<B> {
    fn take_glwe_secret(&mut self, module: &Module<B>, rank: usize) -> (GLWESecret<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_scalar_znx(module, rank);
        (
            GLWESecret {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESecretExec<B> for Scratch<B>
where
    B: ScratchTakeSvpPPolImpl<B>,
{
    fn take_glwe_secret_exec(&mut self, module: &Module<B>, rank: usize) -> (GLWESecretExec<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_svp_ppol(module, rank);
        (
            GLWESecretExec {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESwitchingKey<B> for Scratch<B> {
    fn take_glwe_switching_key(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GLWESwitchingKey<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_gglwe(module, basek, k, rows, digits, rank_in, rank_out);
        (
            GLWESwitchingKey {
                key: data,
                sk_in_n: 0,
                sk_out_n: 0,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeAutomorphismKey<B> for Scratch<B> {
    fn take_automorphism_key(
        &mut self,
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (AutomorphismKey<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_glwe_switching_key(module, basek, k, rows, digits, rank, rank);
        (AutomorphismKey { key: data, p: 0 }, scratch)
    }
}

impl<B: Backend> TakeTensorKey<B> for Scratch<B> {
    fn take_tensor_key(
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

        let mut scratch: &mut Scratch<B> = self;

        if pairs != 0 {
            let (gglwe, s) = scratch.take_glwe_switching_key(module, basek, k, rows, digits, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_glwe_switching_key(module, basek, k, rows, digits, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        (GLWETensorKey { keys }, scratch)
    }
}
