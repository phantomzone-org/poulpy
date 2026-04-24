use poulpy_hal::layouts::{Backend, HostDataMut, HostDataRef, ScratchArena};

use crate::layouts::{
    GLWEInfos, GLWEPlaintext, GLWEPlaintextToBackendMut, GLWEPlaintextToMut, GLWESecretPrepared, GLWESecretTensorPrepared,
    GLWETensor, GLWEToBackendRef, GLWEToRef, LWEInfos, LWEPlaintextToBackendMut, LWEPlaintextToMut, LWESecretToRef, LWEToRef,
    SetLWEInfos, prepared::GLWESecretPreparedToBackendRef,
};

pub trait GLWEDecrypt<BE: Backend> {
    fn glwe_decrypt_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut;
}

pub trait LWEDecrypt<BE: Backend> {
    fn lwe_decrypt<'s, R, P, S>(&self, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut;

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
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: HostDataRef,
        GLWETensor<R>: GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: HostDataMut,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S0: HostDataRef,
        S1: HostDataRef,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut;
}
