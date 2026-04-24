use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    decryption::{GLWEDecryptDefault, GLWETensorDecryptDefault, LWEDecryptDefault},
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWEPlaintextToBackendMut, GLWEPlaintextToMut, GLWESecretPrepared, GLWESecretTensorPrepared,
        GLWETensor, GLWEToBackendRef, LWEInfos, LWEPlaintextToBackendMut, LWEPlaintextToMut, LWESecretToRef, LWEToRef,
        SetLWEInfos, prepared::GLWESecretPreparedToBackendRef,
    },
};

/// Backend-provided decryption operations.
///
/// # Safety
/// Implementations must interpret ciphertexts, plaintexts, and secrets according to their layout
/// metadata, avoid out-of-bounds or aliased writes, and only use scratch space within the
/// advertised temporary-size contracts.
pub unsafe trait DecryptionImpl<BE: Backend>: Backend {
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn glwe_tensor_decrypt<R, P, S0, S1>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: DataRef,
        GLWETensor<R>: crate::layouts::GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: DataMut,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S0: DataRef,
        S1: DataRef,
        for<'a> BE::BufMut<'a>: DataMut;

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;
}

#[doc(hidden)]
pub trait DecryptionDefaults<BE: Backend>: Backend {
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn glwe_tensor_decrypt<R, P, S0, S1>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: DataRef,
        GLWETensor<R>: crate::layouts::GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: DataMut,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S0: DataRef,
        S1: DataRef,
        for<'a> BE::BufMut<'a>: DataMut;

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;
}

impl<BE: Backend> DecryptionDefaults<BE> for BE
where
    Module<BE>: GLWEDecryptDefault<BE> + LWEDecryptDefault<BE> + GLWETensorDecryptDefault<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWEDecryptDefault<BE>>::glwe_decrypt_tmp_bytes_default(module, infos)
    }

    fn glwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        <Module<BE> as GLWEDecryptDefault<BE>>::glwe_decrypt_default(module, res, pt, sk, scratch)
    }

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        <Module<BE> as LWEDecryptDefault<BE>>::lwe_decrypt_tmp_bytes_default(module, infos)
    }

    fn lwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        <Module<BE> as LWEDecryptDefault<BE>>::lwe_decrypt_default(module, res, pt, sk, scratch)
    }

    fn glwe_tensor_decrypt<R, P, S0, S1>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: DataRef,
        GLWETensor<R>: crate::layouts::GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: DataMut,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S0: DataRef,
        S1: DataRef,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        <Module<BE> as GLWETensorDecryptDefault<BE>>::glwe_tensor_decrypt_default(module, res, pt, sk, sk_tensor, scratch)
    }

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWETensorDecryptDefault<BE>>::glwe_tensor_decrypt_tmp_bytes_default(module, infos)
    }
}
