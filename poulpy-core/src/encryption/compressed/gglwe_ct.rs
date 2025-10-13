use poulpy_hal::{
    api::{
        ScratchAvailable, VecZnxAddScalarInplace, VecZnxDftAllocBytes, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
        ZnNormalizeInplace,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxZero},
    source::Source,
};

use crate::{
    TakeGLWEPt,
    encryption::{SIGMA, glwe_ct::GLWEEncryptSkInternal},
    layouts::{
        GGLWE, GGLWEInfos, LWEInfos,
        compressed::{GGLWECiphertextCompressed, GGLWECiphertextCompressedToMut},
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl<D: DataMut> GGLWECiphertextCompressed<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGLWECompressedEncryptSk<B>,
    {
        module.gglwe_compressed_encrypt_sk(self, pt, sk, seed, source_xe, scratch);
    }
}

impl GGLWECiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftAllocBytes + VecZnxNormalizeTmpBytes,
    {
        GGLWE::encrypt_sk_scratch_space(module, infos)
    }
}

pub trait GGLWECompressedEncryptSk<B: Backend> {
    fn gglwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GGLWECiphertextCompressedToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<B>;
}

impl<B: Backend> GGLWECompressedEncryptSk<B> for Module<B>
where
    Module<B>: GLWEEncryptSkInternal<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftAllocBytes
        + VecZnxAddScalarInplace
        + ZnNormalizeInplace<B>,
    Scratch<B>: TakeGLWEPt<B> + ScratchAvailable,
{
    fn gglwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GGLWECiphertextCompressedToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<B>,
    {
        let res: &mut GGLWECiphertextCompressed<&mut [u8]> = &mut res.to_mut();
        let pt: &ScalarZnx<&[u8]> = &pt.to_ref();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::layouts::ZnxInfos;
            let sk = &sk.to_ref();

            assert_eq!(
                res.rank_in(),
                pt.cols() as u32,
                "res.rank_in(): {} != pt.cols(): {}",
                res.rank_in(),
                pt.cols()
            );
            assert_eq!(
                res.rank_out(),
                sk.rank(),
                "res.rank_out(): {} != sk.rank(): {}",
                res.rank_out(),
                sk.rank()
            );
            assert_eq!(res.n(), sk.n());
            assert_eq!(pt.n() as u32, sk.n());
            assert!(
                scratch.available() >= GGLWECiphertextCompressed::encrypt_sk_scratch_space(self, res),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GGLWECiphertextCompressed::encrypt_sk_scratch_space(self, res)
            );
            assert!(
                res.dnum().0 * res.dsize().0 * res.base2k().0 <= res.k().0,
                "res.dnum() : {} * res.dsize() : {} * res.base2k() : {} = {} >= res.k() = {}",
                res.dnum(),
                res.dsize(),
                res.base2k(),
                res.dnum().0 * res.dsize().0 * res.base2k().0,
                res.k()
            );
        }

        let dnum: usize = res.dnum().into();
        let dsize: usize = res.dsize().into();
        let base2k: usize = res.base2k().into();
        let rank_in: usize = res.rank_in().into();
        let cols: usize = (res.rank_out() + 1).into();

        let mut source_xa = Source::new(seed);

        let (mut tmp_pt, scrach_1) = scratch.take_glwe_pt(res);
        (0..rank_in).for_each(|col_i| {
            (0..dnum).for_each(|d_i| {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                tmp_pt.data.zero(); // zeroes for next iteration
                self.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (dsize - 1) + d_i * dsize, pt, col_i);
                self.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scrach_1);

                let (seed, mut source_xa_tmp) = source_xa.branch();
                res.seed[col_i * dnum + d_i] = seed;

                self.glwe_encrypt_sk_internal(
                    res.base2k().into(),
                    res.k().into(),
                    &mut res.at_mut(d_i, col_i).data,
                    cols,
                    true,
                    Some((&tmp_pt, 0)),
                    sk,
                    &mut source_xa_tmp,
                    source_xe,
                    SIGMA,
                    scrach_1,
                );
            });
        });
    }
}
