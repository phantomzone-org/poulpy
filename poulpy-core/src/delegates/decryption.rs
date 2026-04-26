use poulpy_hal::layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena};

use crate::{
    api::{GLWEDecrypt, GLWETensorDecrypt, LWEDecrypt},
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWEPlaintextToBackendMut, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor,
        GLWEToBackendRef, LWEInfos, LWEPlaintextToBackendMut, LWEPlaintextToMut, LWESecretToRef, LWEToRef, SetLWEInfos,
        prepared::GLWESecretPreparedToBackendRef,
    },
    oep::DecryptionImpl,
};

macro_rules! impl_decryption_delegate {
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + HostBackend + DecryptionImpl<BE>,
        {
            $($body)+
        }
    };
}

impl_decryption_delegate!(
    GLWEDecrypt<BE>,
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_decrypt_tmp_bytes(self, infos)
    },
    fn glwe_decrypt<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    {
        BE::glwe_decrypt(self, res, pt, sk, scratch)
    }
);

impl_decryption_delegate!(
    LWEDecrypt<BE>,
    fn lwe_decrypt<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    {
        BE::lwe_decrypt(self, res, pt, sk, scratch)
    },
    fn lwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        BE::lwe_decrypt_tmp_bytes(self, infos)
    }
);

impl_decryption_delegate!(
    GLWETensorDecrypt<BE>,
    fn glwe_tensor_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_tensor_decrypt_tmp_bytes(self, infos)
    },
    fn glwe_tensor_decrypt<R, P, S0, S1>(
        &self,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: HostDataRef,
        GLWETensor<R>: GLWEToBackendRef<BE> + GLWEInfos,
        P: HostDataMut,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S0: HostDataRef,
        S1: HostDataRef,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    {
        BE::glwe_tensor_decrypt(self, res, pt, sk, sk_tensor, scratch)
    }
);
