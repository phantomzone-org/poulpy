use poulpy_hal::layouts::{Backend, Scratch};

use crate::layouts::{GLWECiphertextToMut, GLWECiphertextToRef, prepared::GGSWCiphertextPreparedToRef};

mod gglwe_atk;
mod gglwe_ksk;
mod ggsw_ct;
mod glwe_ct;

pub trait GLWEExternalProduct<BE: Backend> {
    fn external_product<R, A, D>(&self, res: &mut R, a: &A, ggsw: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWECiphertextToMut,
        A: GLWECiphertextToRef,
        D: GGSWCiphertextPreparedToRef<BE>;
}

pub trait GLWEExternalProductInplace<BE: Backend> {
    fn external_product_inplace<R, D>(&self, res: &mut R, ggsw: &D, scratch: &mut Scratch<BE>)
    where
        R: GLWECiphertextToMut,
        D: GGSWCiphertextPreparedToRef<BE>;
}
