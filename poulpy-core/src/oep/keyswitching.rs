use poulpy_hal::layouts::{Backend, HostDataMut, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    keyswitching::{GGLWEKeyswitchDefault, GGSWKeyswitchDefault, GLWEKeyswitchDefault, LWEKeySwitchDefault},
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGLWEToMut, GGLWEToRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef,
        GGSWToMut, GGSWToRef, GLWEBackendMut, GLWEBackendRef, GLWEInfos, LWEInfos, LWEToBackendMut, LWEToBackendRef,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

/// Backend-provided GLWE key-switching operations.
///
/// # Safety
/// Implementations must satisfy the documented key-switch semantics, honor layout metadata and
/// prepared-key interpretation, and keep all reads and writes within the described backend buffers.
pub unsafe trait GLWEKeyswitchImpl<BE: Backend>: Backend {
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch<'s, K>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'_, BE>,
        a: &GLWEBackendRef<'_, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_keyswitch_inplace<'s, K>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'_, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

/// Backend-provided GGLWE key-switching operations.
///
/// # Safety
/// Implementations must preserve ciphertext invariants, use scratch space according to the
/// advertised temporary-size contract, and uphold aliasing guarantees for backend-owned buffers.
pub unsafe trait GGLWEKeyswitchImpl<BE: Backend>: Backend {
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch<'s, R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn gglwe_keyswitch_inplace<'s, R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

/// Backend-provided GGSW key-switching operations.
///
/// # Safety
/// Implementations must correctly interpret prepared key material for the backend, respect all
/// layout-derived bounds, and avoid invalid aliasing or mutation through scratch-backed views.
pub unsafe trait GGSWKeyswitchImpl<BE: Backend>: Backend {
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<'s, R, A, K, T>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn ggsw_keyswitch_inplace<'s, R, K, T>(
        module: &Module<BE>,
        res: &mut R,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

/// Backend-provided LWE key-switching operations.
///
/// # Safety
/// Implementations must only access the ciphertext and key regions described by the layouts and
/// must produce results matching the logical key-switch operation for the backend.
pub unsafe trait LWEKeyswitchImpl<BE: Backend>: Backend {
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<'s, R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut;
}

#[doc(hidden)]
pub trait GLWEKeyswitchDefaults<BE: Backend>: Backend {
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch<'s, 'r, 'a, K>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn glwe_keyswitch_inplace<'s, 'r, K>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

#[doc(hidden)]
pub trait GGLWEKeyswitchDefaults<BE: Backend>: Backend {
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch<'s, R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn gglwe_keyswitch_inplace<'s, R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

#[doc(hidden)]
pub trait GGSWKeyswitchDefaults<BE: Backend>: Backend {
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<'s, R, A, K, T>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn ggsw_keyswitch_inplace<'s, R, K, T>(
        module: &Module<BE>,
        res: &mut R,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

#[doc(hidden)]
pub trait LWEKeyswitchDefaults<BE: Backend>: Backend {
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<'s, R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut;
}

impl<BE: Backend> GLWEKeyswitchDefaults<BE> for BE
where
    Module<BE>: GLWEKeyswitchDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWEKeyswitchDefault<BE>>::glwe_keyswitch_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn glwe_keyswitch<'s, 'r, 'a, K>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        a: &GLWEBackendRef<'a, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        <Module<BE> as GLWEKeyswitchDefault<BE>>::glwe_keyswitch_default(module, res, a, key, scratch)
    }

    fn glwe_keyswitch_inplace<'s, 'r, K>(
        module: &Module<BE>,
        res: &mut GLWEBackendMut<'r, BE>,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        <Module<BE> as GLWEKeyswitchDefault<BE>>::glwe_keyswitch_assign_default(module, res, key, scratch)
    }
}

impl<BE: Backend> GGLWEKeyswitchDefaults<BE> for BE
where
    Module<BE>: GGLWEKeyswitchDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GGLWEKeyswitchDefault<BE>>::gglwe_keyswitch_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn gglwe_keyswitch<'s, R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        A: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        <Module<BE> as GGLWEKeyswitchDefault<BE>>::gglwe_keyswitch_default(module, res, a, key, scratch)
    }

    fn gglwe_keyswitch_inplace<'s, R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGLWEToMut + GGLWEToBackendMut<BE> + GGLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        <Module<BE> as GGLWEKeyswitchDefault<BE>>::gglwe_keyswitch_assign_default(module, res, key, scratch)
    }
}

impl<BE: Backend> GGSWKeyswitchDefaults<BE> for BE
where
    Module<BE>: GGSWKeyswitchDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        <Module<BE> as GGSWKeyswitchDefault<BE>>::ggsw_keyswitch_tmp_bytes_default(
            module, res_infos, a_infos, key_infos, tsk_infos,
        )
    }

    fn ggsw_keyswitch<'s, R, A, K, T>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        key: &K,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        <Module<BE> as GGSWKeyswitchDefault<BE>>::ggsw_keyswitch_default(module, res, a, key, tsk, scratch)
    }

    fn ggsw_keyswitch_inplace<'s, R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        <Module<BE> as GGSWKeyswitchDefault<BE>>::ggsw_keyswitch_assign_default(module, res, key, tsk, scratch)
    }
}

impl<BE: Backend> LWEKeyswitchDefaults<BE> for BE
where
    Module<BE>: LWEKeySwitchDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as LWEKeySwitchDefault<BE>>::lwe_keyswitch_tmp_bytes_default(module, res_infos, a_infos, key_infos)
    }

    fn lwe_keyswitch<'s, R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: LWEToBackendMut<BE> + LWEInfos,
        A: LWEToBackendRef<BE> + LWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        <Module<BE> as LWEKeySwitchDefault<BE>>::lwe_keyswitch_default(module, res, a, ksk, scratch)
    }
}
