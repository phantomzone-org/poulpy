use poulpy_hal::{
    api::{
        ScratchTakeBasic, SvpApplyDftToDftInplace, VecZnxAddScalarInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApplyConsume,
    },
    layouts::{Backend, DataRef, Module, ScalarZnxToRef, Scratch, Stats, ZnxZero},
};

use crate::layouts::{GGSW, GGSWInfos, GGSWToRef, LWEInfos, prepared::GLWESecretPrepared};
use crate::{GLWENoise, layouts::prepared::GLWESecretPreparedToRef};
use crate::{ScratchTakeCore, layouts::GLWEPlaintext};

impl<D: DataRef> GGSW<D> {
    pub fn noise<M, BE: Backend, P, S>(
        &self,
        module: &M,
        row: usize,
        col: usize,
        pt_want: &P,
        sk_prepared: &S,
        scratch: &mut Scratch<BE>,
    ) -> Stats
    where
        S: GLWESecretPreparedToRef<BE>,
        P: ScalarZnxToRef,
        M: GGSWNoise<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.ggsw_noise(self, row, col, pt_want, sk_prepared, scratch)
    }
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

impl<BE: Backend> GGSWNoise<BE> for Module<BE>
where
    Module<BE>: VecZnxAddScalarInplace
        + VecZnxDftApply<BE>
        + SvpApplyDftToDftInplace<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWENoise<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn ggsw_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        GLWEPlaintext::bytes_of_from_infos(infos)
            + (self.bytes_of_vec_znx_dft(1, infos.size()) + self.vec_znx_big_normalize_tmp_bytes())
                .max(self.glwe_noise_tmp_bytes(infos))
    }

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
        P: ScalarZnxToRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &GGSW<&[u8]> = &res.to_ref();
        let sk_prepared: &GLWESecretPrepared<&[u8], BE> = &sk_prepared.to_ref();

        let base2k: usize = res.base2k().into();
        let dsize: usize = res.dsize().into();

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(res);
        pt.data_mut().zero();
        self.vec_znx_add_scalar_inplace(&mut pt.data, 0, (dsize - 1) + res_row * dsize, pt_want, 0);

        // mul with sk[col_j-1]
        if res_col > 0 {
            let (mut pt_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, 1, res.size());
            self.vec_znx_dft_apply(1, 0, &mut pt_dft, 0, &pt.data, 0);
            self.svp_apply_dft_to_dft_inplace(&mut pt_dft, 0, &sk_prepared.data, res_col - 1);
            let pt_big = self.vec_znx_idft_apply_consume(pt_dft);
            self.vec_znx_big_normalize(base2k, &mut pt.data, 0, base2k, &pt_big, 0, scratch_2);
        }

        self.glwe_noise(&res.at(res_row, res_col), &pt, sk_prepared, scratch_1)
    }
}
