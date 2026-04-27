use poulpy_hal::layouts::{Backend, Data, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    decryption::{GLWEDecryptDefault, GLWETensorDecryptDefault, LWEDecryptDefault},
    layouts::{
        GLWEInfos, GLWEPlaintext, GLWEPlaintextToBackendMut, GLWESecretPrepared, GLWESecretTensorPrepared, GLWETensor, LWEInfos,
        LWEPlaintextToBackendMut, LWESecretToBackendRef, LWEToBackendRef, SetLWEInfos,
        prepared::{GLWESecretPreparedToBackendRef, GLWESecretTensorPreparedToBackendRef},
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
        R: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        GLWETensor<R>: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        GLWESecretPrepared<S0, BE>: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos;

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
        R: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;

    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        GLWETensor<R>: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        GLWESecretPrepared<S0, BE>: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos;

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;
}

impl<BE: Backend + HostBackend> DecryptionDefaults<BE> for BE
where
    Module<BE>: GLWEDecryptDefault<BE> + LWEDecryptDefault<BE> + GLWETensorDecryptDefault<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        <Module<BE> as GLWEDecryptDefault<BE>>::glwe_decrypt_tmp_bytes_default(module, infos)
    }

    fn glwe_decrypt<'s, R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut ScratchArena<'s, BE>)
    where
        R: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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
        R: LWEToBackendRef<BE> + LWEInfos,
        P: LWEPlaintextToBackendMut<BE> + SetLWEInfos + LWEInfos,
        S: LWESecretToBackendRef<BE> + LWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        <Module<BE> as LWEDecryptDefault<BE>>::lwe_decrypt_default(module, res, pt, sk, scratch)
    }

    fn glwe_tensor_decrypt<R: Data, P: Data, S0: Data, S1: Data>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        GLWETensor<R>: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
        GLWEPlaintext<P>: GLWEPlaintextToBackendMut<BE> + GLWEInfos + SetLWEInfos,
        GLWESecretPrepared<S0, BE>: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        GLWESecretTensorPrepared<S1, BE>: GLWESecretTensorPreparedToBackendRef<BE> + GLWEInfos,
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
