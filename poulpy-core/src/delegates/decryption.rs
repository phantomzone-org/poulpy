use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    api::{GLWEDecrypt, GLWETensorDecrypt, LWEDecrypt},
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWEPlaintextToMut, GLWESecretPrepared, GLWESecretPreparedToRef, GLWESecretTensorPrepared,
        GLWETensor, LWEInfos, LWEPlaintextToMut, LWESecretToRef, LWEToRef, SetLWEInfos,
    },
    oep::CoreImpl,
};

impl<BE> GLWEDecrypt<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_decrypt_tmp_bytes(self, infos)
    }

    fn glwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: crate::layouts::GLWEToRef + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
    {
        BE::glwe_decrypt(self, res, pt, sk, scratch)
    }
}

impl<BE> LWEDecrypt<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn lwe_decrypt<R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::lwe_decrypt(self, res, pt, sk, scratch)
    }

    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        BE::lwe_decrypt_tmp_bytes(self, infos)
    }
}

impl<BE> GLWETensorDecrypt<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_tensor_decrypt_tmp_bytes(self, infos)
    }

    fn glwe_tensor_decrypt<R, P, PM, S0, S1>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P, PM>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataRef,
        P: DataMut,
        S0: DataRef,
        S1: DataRef,
    {
        BE::glwe_tensor_decrypt(self, res, pt, sk, sk_tensor, scratch)
    }
}
