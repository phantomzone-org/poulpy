use poulpy_hal::{
    api::{ModuleN, VecZnxAddScalarInplace, VecZnxNormalizeInplace},
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxZero},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::{SIGMA, ggsw_ct::GGSWEncryptSk, glwe_ct::GLWEEncryptSkInternal},
    layouts::{
        GGSWInfos, GLWEInfos, LWEInfos,
        compressed::{GGSWCompressed, GGSWCompressedToMut},
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl GGSWCompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: GGSWCompressedEncryptSk<BE>,
    {
        module.ggsw_compressed_encrypt_sk_tmp_bytes(infos)
    }
}

impl<DataSelf: DataMut> GGSWCompressed<DataSelf> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, BE>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: GGSWCompressedEncryptSk<BE>,
    {
        module.ggsw_compressed_encrypt_sk(self, pt, sk, seed_xa, source_xe, scratch);
    }
}


pub trait GGSWCompressedEncryptSk<BE: Backend> {
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWCompressedToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GGSWCompressedEncryptSk<BE> for Module<BE>
where
    Module<BE>: ModuleN + GLWEEncryptSkInternal<BE> + GGSWEncryptSk<BE> + VecZnxAddScalarInplace + VecZnxNormalizeInplace<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
        where A: GGSWInfos,
    {
        self.ggsw_encrypt_sk_tmp_bytes(infos)
    }

    fn ggsw_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWCompressedToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<BE>,
    {
        let res: &mut GGSWCompressed<&mut [u8]> = &mut res.to_mut();
        let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();
        let pt: &ScalarZnx<&[u8]> = &pt.to_ref();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::layouts::ZnxInfos;

            assert_eq!(res.rank(), sk.rank());
            assert_eq!(res.n(), sk.n());
            assert_eq!(pt.n() as u32, sk.n());
        }

        let base2k: usize = res.base2k().into();
        let rank: usize = res.rank().into();
        let cols: usize = rank + 1;
        let dsize: usize = res.dsize().into();

        let (mut tmp_pt, scratch_1) = scratch.take_glwe_pt(self, &res.glwe_layout());

        let mut source = Source::new(seed_xa);

        res.seed = vec![[0u8; 32]; res.dnum().0 as usize * cols];

        for row_i in 0..res.dnum().into() {
            tmp_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            self.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, 0);
            self.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scratch_1);

            for col_j in 0..rank + 1 {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                let (seed, mut source_xa_tmp) = source.branch();

                res.seed[row_i * cols + col_j] = seed;

                self.glwe_encrypt_sk_internal(
                    res.base2k().into(),
                    res.k().into(),
                    &mut res.at_mut(row_i, col_j).data,
                    cols,
                    true,
                    Some((&tmp_pt, col_j)),
                    sk,
                    &mut source_xa_tmp,
                    source_xe,
                    SIGMA,
                    scratch_1,
                );
            }
        }
    }
}