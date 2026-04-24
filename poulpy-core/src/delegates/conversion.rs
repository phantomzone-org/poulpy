use poulpy_hal::layouts::{Backend, DataMut, Module, ScratchArena};

use crate::{
    api::{GGSWExpandRows, GGSWFromGGLWE, GLWEFromLWE, LWEFromGLWE},
    conversion::LWEFromGLWEDefault,
    layouts::{
        GGLWEInfos, GGLWEToRef, GGSWBackendMut, GGSWInfos, GGSWToBackendMut, GGSWToMut, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef, GLWEToMut, GLWEToRef, LWEInfos, LWEToMut, LWEToRef,
        prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef},
    },
    oep::{ConversionImpl, GLWEKeyswitchImpl, GLWERotateImpl},
};

macro_rules! impl_conversion_delegate {
    ($trait:ty, [$($bounds:tt)+], $($body:item)+) => {
        impl<BE> $trait for Module<BE>
        where
            $($bounds)+
        {
            $($body)+
        }
    };
}

impl_conversion_delegate!(
    GLWEFromLWE<BE>,
    [BE: Backend + ConversionImpl<BE> + GLWEKeyswitchImpl<BE>],
    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_from_lwe_tmp_bytes(self, glwe_infos, lwe_infos, key_infos)
    }

    fn glwe_from_lwe<'s, R, A, K>(
        &self,
        res: &mut R,
        lwe: &A,
        ksk: &K,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GLWEToMut + GLWEToBackendMut<BE>,
        A: LWEToRef,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        BE::glwe_from_lwe(self, res, lwe, ksk, scratch)
    }
);

impl_conversion_delegate!(
    LWEFromGLWE<BE>,
    [BE: Backend + ConversionImpl<BE> + GLWEKeyswitchImpl<BE> + GLWERotateImpl<BE>, Module<BE>: LWEFromGLWEDefault<BE>],
    fn lwe_from_glwe_tmp_bytes<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::lwe_from_glwe_tmp_bytes(self, lwe_infos, glwe_infos, key_infos)
    }

    fn lwe_from_glwe<'s, R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        a_idx: usize,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: LWEToMut,
        A: GLWEToRef + GLWEToBackendRef<BE>,
        K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        BE::lwe_from_glwe(self, res, a, a_idx, key, scratch)
    }
);

impl_conversion_delegate!(
    GGSWFromGGLWE<BE>,
    [BE: Backend + ConversionImpl<BE>],
    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        BE::ggsw_from_gglwe_tmp_bytes(self, res_infos, tsk_infos)
    }

    fn ggsw_from_gglwe<'s, R, A, T>(
        &self,
        res: &mut R,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        R: GGSWToMut + GGSWToBackendMut<BE> + GGSWInfos,
        A: GGLWEToRef + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        BE::ggsw_from_gglwe(self, res, a, tsk, scratch)
    }
);

impl_conversion_delegate!(
    GGSWExpandRows<BE>,
    [BE: Backend + ConversionImpl<BE>],
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        BE::ggsw_expand_rows_tmp_bytes(self, res_infos, tsk_infos)
    }

    fn ggsw_expand_row<'s, 'r, T>(
        &self,
        res: &mut GGSWBackendMut<'r, BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'s, BE>,
    )
    where
        T: GGLWEToGGSWKeyPreparedToBackendRef<BE> + GGLWEInfos,
        ScratchArena<'s, BE>: crate::ScratchArenaTakeCore<'s, BE>,
    {
        BE::ggsw_expand_row(self, res, tsk, scratch)
    }
);
