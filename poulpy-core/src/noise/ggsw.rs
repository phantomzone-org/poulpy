use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, SvpApplyDftToDftInplace, VecZnxAddScalarAssign, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume,
    },
    layouts::{
        Backend, DataMut, DataRef, Module, ScalarZnxToRef, ScratchArena, Stats, ZnxZero, vec_znx_backend_ref_from_mut,
        vec_znx_big_backend_ref_from_mut,
    },
};

use crate::layouts::{GGSW, GGSWInfos, GGSWToBackendRef, GGSWToRef, LWEInfos};
use crate::noise::glwe::glwe_noise_backend_inner;
use crate::{
    GLWENormalize, GLWESub,
    api::{GGSWNoise, GLWENoise},
    decryption::{GLWEDecrypt, GLWEDecryptDefault},
    layouts::{ggsw_at_backend_ref_from_ref, prepared::GLWESecretPreparedToBackendRef},
};
use crate::{ScratchArenaTakeCore, layouts::GLWEPlaintext};

impl<D: DataRef> GGSW<D> {
    pub fn noise<'s, M, BE: Backend, P, S>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &P,
        sk_prepared: &S,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Stats
    where
        GGSW<D>: GGSWToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
        P: ScalarZnxToRef,
        M: GGSWNoise<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: DataMut,
    {
        module.ggsw_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
}

impl<BE: Backend> GGSWNoise<BE> for Module<BE>
where
    Module<BE>: VecZnxAddScalarAssign
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftAssign<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWENoise<BE>
        + GLWEDecrypt<BE>
        + GLWEDecryptDefault<BE>
        + GLWESub
        + GLWENormalize<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ggsw_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1_glwe_noise: usize = self.glwe_noise_tmp_bytes(infos);
        let lvl_1_mul: usize = self.bytes_of_vec_znx_dft(1, infos.size()) + self.vec_znx_big_normalize_tmp_bytes();
        let lvl_1: usize = lvl_1_glwe_noise.max(lvl_1_mul);

        lvl_0 + lvl_1
    }

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
        for<'a> BE::BufMut<'a>: DataMut,
    {
        let res_ref = res.to_ref();
        let res_backend = res.to_backend_ref();
        let sk_backend = sk_prepared.to_backend_ref();

        let base2k: usize = res_ref.base2k().into();
        let dsize: usize = res_ref.dsize().into();
        assert!(
            scratch.available() >= self.ggsw_noise_tmp_bytes(res),
            "scratch.available(): {} < GGSWNoise::ggsw_noise_tmp_bytes: {}",
            scratch.available(),
            self.ggsw_noise_tmp_bytes(res)
        );

        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext(&res_ref);
        pt.data_mut().zero();
        self.vec_znx_add_scalar_assign(&mut pt.data, 0, (dsize - 1) + res_row * dsize, pt_want, 0);

        // mul with sk[col_j-1]
        if res_col > 0 {
            let scratch_mul = scratch_1.borrow();
            let (mut pt_dft, mut scratch_2) = scratch_mul.take_vec_znx_dft(self, 1, res_ref.size());
            self.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &vec_znx_backend_ref_from_mut::<BE>(&pt.data), 0);
            self.svp_apply_dft_to_dft_inplace(&mut pt_dft, 0, &sk_backend.data, res_col - 1);
            let pt_big = self.vec_znx_idft_apply_consume(pt_dft);
            self.vec_znx_big_normalize(
                &mut pt.data,
                base2k,
                0,
                0,
                &vec_znx_big_backend_ref_from_mut::<BE>(&pt_big),
                base2k,
                0,
                &mut scratch_2,
            );
        }

        let res_at_ref = res_ref.at(res_row, res_col);
        let res_at_backend = ggsw_at_backend_ref_from_ref::<BE>(&res_backend, res_row, res_col);
        glwe_noise_backend_inner(self, &res_at_ref, &res_at_backend, &pt, &sk_backend, &mut scratch_1)
    }
}
