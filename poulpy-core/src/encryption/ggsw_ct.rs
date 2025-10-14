use poulpy_hal::{
    api::{VecZnxAddScalarInplace, VecZnxDftAllocBytes, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, VecZnx, ZnxZero},
    source::Source,
};

use crate::{
    SIGMA, TakeGLWEPt,
    encryption::glwe_ct::GLWEEncryptSkInternal,
    layouts::{
        GGSW, GGSWInfos, GGSWToMut, GLWE, GLWEInfos, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl GGSW<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGSWInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes,
    {
        let size = infos.size();
        GLWE::encrypt_sk_scratch_space(module, &infos.glwe_layout())
            + VecZnx::bytes_of(module.n(), (infos.rank() + 1).into(), size)
            + VecZnx::bytes_of(module.n(), 1, size)
            + module.vec_znx_dft_bytes_of((infos.rank() + 1).into(), size)
    }
}

pub trait GGSWEncryptSk<B: Backend> {
    fn ggsw_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<B>;
}

impl<B: Backend> GGSWEncryptSk<B> for Module<B>
where
    Module<B>: GLWEEncryptSkInternal<B> + VecZnxAddScalarInplace + VecZnxNormalizeInplace<B>,
    Scratch<B>: TakeGLWEPt<B>,
{
    fn ggsw_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<B>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let pt: &ScalarZnx<&[u8]> = &pt.to_ref();
        let sk: &GLWESecretPrepared<&[u8], B> = &sk.to_ref();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::layouts::ZnxInfos;

            assert_eq!(res.rank(), sk.rank());
            assert_eq!(res.n(), self.n() as u32);
            assert_eq!(pt.n(), self.n());
            assert_eq!(sk.n(), self.n() as u32);
        }

        let k: usize = res.k().into();
        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let dsize: usize = res.dsize().into();
        let cols: usize = (rank + 1).into();

        let (mut tmp_pt, scratch_1) = scratch.take_glwe_pt(&res.glwe_layout());

        for row_i in 0..res.dnum().into() {
            tmp_pt.data.zero();
            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            self.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, 0);
            self.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scratch_1);
            for col_j in 0..rank + 1 {
                self.glwe_encrypt_sk_internal(
                    base2k,
                    k,
                    res.at_mut(row_i, col_j).data_mut(),
                    cols,
                    false,
                    Some((&tmp_pt, col_j)),
                    sk,
                    source_xa,
                    source_xe,
                    SIGMA,
                    scratch_1,
                );
            }
        }
    }
}

impl<DataSelf: DataMut> GGSW<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGSWEncryptSk<B>,
    {
        module.ggsw_encrypt_sk(self, pt, sk, source_xa, source_xe, scratch);
    }
}
