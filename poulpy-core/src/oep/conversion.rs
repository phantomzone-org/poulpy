use poulpy_hal::layouts::{Backend, DataMut, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    conversion::{GGSWExpandRowsDefault, GGSWFromGGLWEDefault, GLWEFromLWEDefault, LWEFromGLWEDefault},
    layouts::{
        GGLWEInfos, GGLWEToRef, GGSWBackendMut, GGSWInfos, GGSWToBackendMut, GGSWToMut, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef, GLWEToMut, GLWEToRef, LWEInfos, LWEToMut, LWEToRef,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
};

/// Backend-provided ciphertext conversion operations.
///
/// # Safety
/// Implementations must only read and write the regions described by the provided layouts, respect
/// scratch-space requirements, and produce results equivalent to the documented conversion
/// semantics for the backend.
pub unsafe trait ConversionImpl<BE: Backend>: Backend {
    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<'s, R, A, K>(module: &Module<BE>, res: &mut R, lwe: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToMut + GLWEToBackendMut<BE>,
        A: LWEToRef,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe<'s, R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToMut,
        A: GLWEToRef + GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<'s, R, A, T>(module: &Module<BE>, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToRef + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<'s, 'r, T>(
        module: &Module<BE>,
        res: &mut GGSWBackendMut<'r, BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>;
}

#[doc(hidden)]
pub trait ConversionDefaults<BE: Backend>: Backend {
    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<'s, R, A, K>(module: &Module<BE>, res: &mut R, lwe: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToMut + GLWEToBackendMut<BE>,
        A: LWEToRef,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe<'s, R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToMut,
        A: GLWEToRef + GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<'s, R, A, T>(module: &Module<BE>, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToRef + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<'s, 'r, T>(
        module: &Module<BE>,
        res: &mut GGSWBackendMut<'r, BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>;
}

impl<BE: Backend> ConversionDefaults<BE> for BE
where
    Module<BE>: GLWEFromLWEDefault<BE> + LWEFromGLWEDefault<BE> + GGSWFromGGLWEDefault<BE> + GGSWExpandRowsDefault<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
    for<'s> BE::BufMut<'s>: DataMut,
{
    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as GLWEFromLWEDefault<BE>>::glwe_from_lwe_tmp_bytes_default(module, glwe_infos, lwe_infos, key_infos)
    }

    fn glwe_from_lwe<'s, R, A, K>(module: &Module<BE>, res: &mut R, lwe: &A, ksk: &K, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToMut + GLWEToBackendMut<BE>,
        A: LWEToRef,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        <Module<BE> as GLWEFromLWEDefault<BE>>::glwe_from_lwe_default(module, res, lwe, ksk, scratch)
    }

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        <Module<BE> as LWEFromGLWEDefault<BE>>::lwe_from_glwe_tmp_bytes_default(module, lwe_infos, glwe_infos, key_infos)
    }

    fn lwe_from_glwe<'s, R, A, K>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: LWEToMut,
        A: GLWEToRef + GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        <Module<BE> as LWEFromGLWEDefault<BE>>::lwe_from_glwe_default(module, res, a, a_idx, key, scratch)
    }

    fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        <Module<BE> as GGSWFromGGLWEDefault<BE>>::ggsw_from_gglwe_tmp_bytes_default(module, res_infos, tsk_infos)
    }

    fn ggsw_from_gglwe<'s, R, A, T>(module: &Module<BE>, res: &mut R, a: &A, tsk: &T, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToRef + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        <Module<BE> as GGSWFromGGLWEDefault<BE>>::ggsw_from_gglwe_default(module, res, a, tsk, scratch)
    }

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        <Module<BE> as GGSWExpandRowsDefault<BE>>::ggsw_expand_rows_tmp_bytes_default(module, res_infos, tsk_infos)
    }

    fn ggsw_expand_row<'s, 'r, T>(
        module: &Module<BE>,
        res: &mut GGSWBackendMut<'r, BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
    {
        <Module<BE> as GGSWExpandRowsDefault<BE>>::ggsw_expand_row_default(module, res, tsk, scratch)
    }
}
