use poulpy_hal::{
    layouts::ScalarZnxToRef,
    layouts::{Backend, Scratch, Stats},
};

use crate::layouts::{GGLWEInfos, GGLWEToRef, GGSWInfos, GGSWToRef, GLWEInfos, GLWESecretPreparedToRef, GLWEToRef};

pub trait GLWENoise<BE: Backend> {
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_noise<R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut Scratch<BE>) -> Stats
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEToRef,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos;
}

pub trait GGLWENoise<BE: Backend> {
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_noise<R, S, P>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &P,
        sk_prepared: &S,
        scratch: &mut Scratch<BE>,
    ) -> Stats
    where
        R: GGLWEToRef + GGLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        P: ScalarZnxToRef;
}

pub trait GGSWNoise<BE: Backend> {
    fn ggsw_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_noise<R, S, P>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &P,
        sk_prepared: &S,
        scratch: &mut Scratch<BE>,
    ) -> Stats
    where
        R: GGSWToRef,
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef;
}
