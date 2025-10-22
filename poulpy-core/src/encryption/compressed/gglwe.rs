use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddScalarInplace, VecZnxDftBytesOf, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxInfos, ZnxZero},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::{GLWEEncryptSk, GLWEEncryptSkInternal, SIGMA},
    layouts::{
        GGLWECompressedSeedMut, GGLWEInfos, GLWEPlaintext, GLWESecretPrepared, LWEInfos,
        compressed::{GGLWECompressed, GGLWECompressedToMut},
        prepared::GLWESecretPreparedToRef,
    },
};

impl<D: DataMut> GGLWECompressed<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<M, P, S, BE: Backend>(
        &mut self,
        module: &M,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<BE>,
        M: GGLWECompressedEncryptSk<BE>,
    {
        module.gglwe_compressed_encrypt_sk(self, pt, sk, seed, source_xe, scratch);
    }
}

impl GGLWECompressed<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, BE: Backend, A>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWECompressedEncryptSk<BE>,
    {
        module.gglwe_compressed_encrypt_sk_tmp_bytes(infos)
    }
}

pub trait GGLWECompressedEncryptSk<BE: Backend> {
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GGLWECompressedEncryptSk<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSkInternal<BE>
        + GLWEEncryptSk<BE>
        + VecZnxDftBytesOf
        + VecZnxNormalizeInplace<BE>
        + VecZnxAddScalarInplace
        + VecZnxNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_encrypt_sk_tmp_bytes(infos)
            .max(self.vec_znx_normalize_tmp_bytes())
            + GLWEPlaintext::bytes_of_from_infos(infos)
    }

    fn gglwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<BE>,
    {
        let mut seeds: Vec<[u8; 32]> = vec![[0u8; 32]; res.seed_mut().len()];

        {
            let res: &mut GGLWECompressed<&mut [u8]> = &mut res.to_mut();
            let pt: &ScalarZnx<&[u8]> = &pt.to_ref();
            let sk: &GLWESecretPrepared<&[u8], BE> = &sk.to_ref();

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
                scratch.available() >= GGLWECompressed::encrypt_sk_tmp_bytes(self, res),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                GGLWECompressed::encrypt_sk_tmp_bytes(self, res)
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

            let dnum: usize = res.dnum().into();
            let dsize: usize = res.dsize().into();
            let base2k: usize = res.base2k().into();
            let rank_in: usize = res.rank_in().into();
            let cols: usize = (res.rank_out() + 1).into();

            let mut source_xa = Source::new(seed);

            let (mut tmp_pt, scrach_1) = scratch.take_glwe_plaintext(res);
            for col_i in 0..rank_in {
                for d_i in 0..dnum {
                    // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                    tmp_pt.data.zero(); // zeroes for next iteration
                    self.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (dsize - 1) + d_i * dsize, pt, col_i);
                    self.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scrach_1);

                    let (seed, mut source_xa_tmp) = source_xa.branch();
                    seeds[col_i * dnum + d_i] = seed;

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
                }
            }
        }

        res.seed_mut().copy_from_slice(&seeds);
    }
}
