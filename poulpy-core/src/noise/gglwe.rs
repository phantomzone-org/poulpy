use poulpy_hal::{
    api::VecZnxAddScalarAssignBackend,
    layouts::{
        Backend, DataView, HostBackend, HostDataMut, HostDataRef, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, Stats,
        VecZnx, VecZnxReborrowBackendMut, ZnxZero,
    },
};

use crate::noise::glwe::glwe_noise_backend_inner;
use crate::{
    GLWENormalize,
    api::{GGLWENoise, GLWENoise},
    decryption::{GLWEDecrypt, GLWEDecryptDefault},
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToBackendRef, GLWE, GLWEInfos, gglwe_at_backend_ref_from_ref,
        prepared::GLWESecretPreparedToBackendRef,
    },
};
use crate::{ScratchArenaTakeCore, layouts::GLWEPlaintext};

impl<D: HostDataRef> GGLWE<D> {
    pub fn noise<'s, M, S, BE>(
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
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        module.gglwe_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
}

impl<BE: Backend + HostBackend> GGLWENoise<BE> for Module<BE>
where
    Module<BE>: VecZnxAddScalarAssignBackend<BE> + GLWENoise<BE> + GLWEDecrypt<BE> + GLWEDecryptDefault<BE> + GLWENormalize<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
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
        R: GGLWEToBackendRef<BE> + GGLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        BE: HostBackend,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        assert!(
            scratch.available() >= self.gglwe_noise_tmp_bytes(res),
            "scratch.available(): {} < GGLWENoise::gglwe_noise_tmp_bytes: {}",
            scratch.available(),
            self.gglwe_noise_tmp_bytes(res)
        );

        let res_backend = res.to_backend_ref();
        let res_ref = GGLWE {
            base2k: res_backend.base2k,
            dsize: res_backend.dsize(),
            data: poulpy_hal::layouts::MatZnx::from_data(
                res_backend.data.data().as_ref(),
                res_backend.data.n(),
                res_backend.data.rows(),
                res_backend.data.cols_in(),
                res_backend.data.cols_out(),
                res_backend.data.size(),
            ),
        };
        let sk_backend = sk_prepared.to_backend_ref();
        let dsize: usize = res_ref.dsize().into();
        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext(&res_ref);
        pt.data_mut().zero();
        let pt_want_backend: ScalarZnx<BE::OwnedBuf> =
            ScalarZnx::from_data(BE::from_host_bytes(pt_want.data), pt_want.n(), pt_want.cols());
        {
            let mut pt_data =
                <poulpy_hal::layouts::VecZnx<BE::BufMut<'_>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(&mut pt.data);
            self.vec_znx_add_scalar_assign_backend(
                &mut pt_data,
                0,
                (dsize - 1) + res_row * dsize,
                &<ScalarZnx<BE::OwnedBuf> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&pt_want_backend),
                res_col,
            );
        }
        let res_at_ref = res_ref.at(res_row, res_col);
        let res_at_backend = gglwe_at_backend_ref_from_ref::<BE>(&res_backend, res_row, res_col);
        let pt_ref = GLWE {
            base2k: pt.base2k,
            data: VecZnx::from_data_with_max_size(
                pt.data.data().as_ref(),
                pt.data.n(),
                pt.data.cols(),
                pt.data.size(),
                pt.data.max_size(),
            ),
        };
        glwe_noise_backend_inner(self, &res_at_ref, &res_at_backend, &pt_ref, &sk_backend, &mut scratch_1)
    }
}
