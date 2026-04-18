use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::{
    api::{GGSWExpandRows, GGSWFromGGLWE, GLWEFromLWE, LWEFromGLWE},
    conversion::LWEFromGLWEDefault,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyPreparedToRef, GGLWEToRef, GGSWInfos, GGSWToMut, GLWEInfos, GLWEToMut,
        GLWEToRef, LWEInfos, LWEToMut, LWEToRef,
    },
    oep::CoreImpl,
};

impl<BE> GLWEFromLWE<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn glwe_from_lwe_tmp_bytes<R, A, K>(&self, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos,
    {
        BE::glwe_from_lwe_tmp_bytes(self, glwe_infos, lwe_infos, key_infos)
    }

    fn glwe_from_lwe<R, A, K>(&self, res: &mut R, lwe: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
    {
        BE::glwe_from_lwe(self, res, lwe, ksk, scratch)
    }
}

impl<BE> LWEFromGLWE<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
    Module<BE>: LWEFromGLWEDefault<BE>,
{
    fn lwe_from_glwe_tmp_bytes<R, A, K>(&self, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::lwe_from_glwe_tmp_bytes(self, lwe_infos, glwe_infos, key_infos)
    }

    fn lwe_from_glwe<R, A, K>(&self, res: &mut R, a: &A, a_idx: usize, key: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::lwe_from_glwe(self, res, a, a_idx, key, scratch)
    }
}

impl<BE> GGSWFromGGLWE<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn ggsw_from_gglwe_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        BE::ggsw_from_gglwe_tmp_bytes(self, res_infos, tsk_infos)
    }

    fn ggsw_from_gglwe<R, A, T>(&self, res: &mut R, a: &A, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGLWEToRef,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::ggsw_from_gglwe(self, res, a, tsk, scratch)
    }
}

impl<BE> GGSWExpandRows<BE> for Module<BE>
where
    BE: Backend + CoreImpl<BE>,
{
    fn ggsw_expand_rows_tmp_bytes<R, A>(&self, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos,
    {
        BE::ggsw_expand_rows_tmp_bytes(self, res_infos, tsk_infos)
    }

    fn ggsw_expand_row<R, T>(&self, res: &mut R, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: crate::ScratchTakeCore<BE>,
    {
        BE::ggsw_expand_row(self, res, tsk, scratch)
    }
}
