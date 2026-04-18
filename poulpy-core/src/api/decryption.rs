use poulpy_hal::layouts::{Backend, DataMut, DataRef, Scratch};

use crate::layouts::{
    GLWEInfos, GLWEPlaintext, GLWEPlaintextToMut, GLWESecretPrepared, GLWESecretPreparedToRef, GLWESecretTensorPrepared,
    GLWETensor, GLWEToRef, LWEInfos, LWEPlaintextToMut, LWESecretToRef, LWEToRef, SetLWEInfos,
};

pub trait GLWEDecrypt<BE: Backend> {
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos;
}

pub trait LWEDecrypt<BE: Backend> {
    fn lwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        Scratch<BE>: crate::ScratchTakeCore<BE>;

    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;
}

pub trait GLWETensorDecrypt<BE: Backend> {
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_tensor_decrypt<R, P, S0, S1>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataRef,
        P: DataMut,
        S0: DataRef,
        S1: DataRef;
}
