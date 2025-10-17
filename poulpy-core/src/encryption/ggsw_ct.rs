use poulpy_hal::{
    api::{ModuleN, VecZnxAddScalarInplace, VecZnxDftBytesOf, VecZnxNormalizeInplace},
    layouts::{Backend, DataMut, Module, ScalarZnx, ScalarZnxToRef, Scratch, VecZnx, ZnxInfos, ZnxZero},
    source::Source,
};

use crate::{
    SIGMA, ScratchTakeCore,
    encryption::glwe_ct::{GLWEEncryptSk, GLWEEncryptSkInternal},
    layouts::{
        GGSW, GGSWInfos, GGSWToMut, GLWEInfos, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl GGSW<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: GGSWEncryptSk<BE>,
    {
        module.ggsw_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> GGSW<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<P, S, M, BE: Backend>(
        &mut self,
        module: &M,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<BE>,
        M: GGSWEncryptSk<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.ggsw_encrypt_sk(self, pt, sk, source_xa, source_xe, scratch);
    }
}

pub trait GGSWEncryptSk<B: Backend> {
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

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
    Module<B>: ModuleN
        + GLWEEncryptSkInternal<B>
        + GLWEEncryptSk<B>
        + VecZnxDftBytesOf
        + VecZnxNormalizeInplace<B>
        + VecZnxAddScalarInplace,
    Scratch<B>: ScratchTakeCore<B>,
{
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        let size = infos.size();
        self.glwe_encrypt_sk_tmp_bytes(infos)
            + VecZnx::bytes_of(self.n(), (infos.rank() + 1).into(), size)
            + VecZnx::bytes_of(self.n(), 1, size)
            + self.bytes_of_vec_znx_dft((infos.rank() + 1).into(), size)
    }

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

        assert_eq!(res.rank(), sk.rank());
        assert_eq!(res.n(), self.n() as u32);
        assert_eq!(pt.n(), self.n());
        assert_eq!(sk.n(), self.n() as u32);

        let k: usize = res.k().into();
        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let dsize: usize = res.dsize().into();
        let cols: usize = (rank + 1).into();

        let (mut tmp_pt, scratch_1) = scratch.take_glwe_pt(self, &res.glwe_layout());

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
