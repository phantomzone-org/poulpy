use poulpy_hal::{
    api::VecZnxAddScalarInplace,
    layouts::{Backend, DataRef, Module, ScalarZnxToRef, Scratch, Stats, ZnxZero},
};

use crate::{
    GLWENoise,
    layouts::{GGLWE, GGLWEInfos, GGLWEToRef, GLWEInfos, prepared::GLWESecretPreparedToRef},
};
use crate::{ScratchTakeCore, layouts::GLWEPlaintext};

impl<D: DataRef> GGLWE<D> {
    pub fn noise<M, S, P, BE: Backend>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &P,
        sk_prepared: &S,
        scratch: &mut Scratch<BE>,
    ) -> Stats
    where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        P: ScalarZnxToRef,
        M: GGLWENoise<BE>,
    {
        module.gglwe_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
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
        R: GGLWEToRef,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        P: ScalarZnxToRef;
}

impl<BE: Backend> GGLWENoise<BE> for Module<BE>
where
    Module<BE>: VecZnxAddScalarInplace + GLWENoise<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        GLWEPlaintext::bytes_of_from_infos(infos) + self.glwe_noise_tmp_bytes(infos)
    }

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
        R: GGLWEToRef,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        P: ScalarZnxToRef,
    {
        let res: &GGLWE<&[u8]> = &res.to_ref();
        let dsize: usize = res.dsize().into();
        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(res);
        pt.data_mut().zero();
        self.vec_znx_add_scalar_inplace(&mut pt.data, 0, (dsize - 1) + res_row * dsize, pt_want, res_col);
        self.glwe_noise(&res.at(res_row, res_col), &pt, sk_prepared, scratch_1)
    }
}
