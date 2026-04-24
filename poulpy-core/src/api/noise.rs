use poulpy_hal::{
    layouts::DataMut,
    layouts::ScalarZnxToRef,
    layouts::{Backend, ScratchArena, Stats},
};

use crate::layouts::{
    GGLWEInfos, GGLWEToBackendRef, GGLWEToRef, GGSWInfos, GGSWToBackendRef, GGSWToRef, GLWEInfos, GLWESecretPreparedToBackendRef,
    GLWEToBackendRef, GLWEToRef,
};

pub trait GLWENoise<BE: Backend> {
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_noise<'s, R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut ScratchArena<'s, BE>) -> Stats
    where
        R: GLWEToRef + GLWEToBackendRef<BE> + GLWEInfos,
        P: GLWEToRef,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: poulpy_hal::layouts::DataMut;
}

pub trait GGLWENoise<BE: Backend> {
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_noise<'s, R, S, P>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &P,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        R: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        P: ScalarZnxToRef,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;
}

pub trait GGSWNoise<BE: Backend> {
    fn ggsw_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_noise<'s, R, S, P>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &P,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        R: GGSWToRef + GGSWToBackendRef<BE> + GGSWInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        P: ScalarZnxToRef,
        for<'a> ScratchArena<'a, BE>: crate::ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut;
}
