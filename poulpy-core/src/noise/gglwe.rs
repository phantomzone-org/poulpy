use poulpy_hal::{
    api::VecZnxAddScalarAssign,
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScratchArena, Stats, ZnxZero},
};

use crate::noise::glwe::glwe_noise_backend_inner;
use crate::{
    GLWENormalize,
    api::{GGLWENoise, GLWENoise},
    decryption::{GLWEDecrypt, GLWEDecryptDefault},
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToBackendRef, GGLWEToRef, GLWEInfos, gglwe_at_backend_ref_from_ref,
        prepared::GLWESecretPreparedToBackendRef,
    },
};
use crate::{ScratchArenaTakeCore, layouts::GLWEPlaintext};

impl<D: HostDataRef> GGLWE<D> {
    pub fn noise<'s, M, S, BE: Backend>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &ScalarZnx<&[u8]>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        GGLWE<D>: GGLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        M: GGLWENoise<BE>,
        BE: HostBackend,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        module.gglwe_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
}

impl<BE: Backend + HostBackend> GGLWENoise<BE> for Module<BE>
where
    Module<BE>: VecZnxAddScalarAssign + GLWENoise<BE> + GLWEDecrypt<BE> + GLWEDecryptDefault<BE> + GLWENormalize<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn gglwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_noise_tmp_bytes(infos);

        lvl_0 + lvl_1
    }

    fn gglwe_noise<'s, R, S>(
        &self,
        res: &R,
        res_row: usize,
        res_col: usize,
        pt_want: &ScalarZnx<&[u8]>,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        R: GGLWEToRef + GGLWEToBackendRef<BE> + GGLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert!(
            scratch.available() >= self.gglwe_noise_tmp_bytes(res),
            "scratch.available(): {} < GGLWENoise::gglwe_noise_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_noise_tmp_bytes(res)
        );

        let res_ref = res.to_ref();
        let res_backend = res.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();
        let dsize: usize = res_ref.dsize().into();
        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext(&res_ref);
        pt.data_mut().zero();
        self.vec_znx_add_scalar_assign(&mut pt.data, 0, (dsize - 1) + res_row * dsize, pt_want, res_col);
        let res_at_ref = res_ref.at(res_row, res_col);
        let res_at_backend = gglwe_at_backend_ref_from_ref::<BE>(&res_backend, res_row, res_col);
        glwe_noise_backend_inner(self, &res_at_ref, &res_at_backend, &pt, &sk_backend, &mut scratch_1)
    }
}
