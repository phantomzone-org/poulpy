use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddScalarInplace, VecZnxDftBytesOf, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxInfos, ZnxZero},
    source::Source,
};

use crate::{
    ScratchTakeCore,
    encryption::glwe_ct::GLWEEncryptSk,
    layouts::GLWEInfos,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToMut, GLWEPlaintext, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl GGLWE<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEEncryptSk<BE>,
    {
        module.gglwe_encrypt_sk_tmp_bytes(infos)
    }

    pub fn encrypt_pk_tmp_bytes<M, A, BE: Backend>(module: &M, infos: &A) -> usize
    where
        A: GGLWEInfos,
        M: GGLWEEncryptSk<BE>,
    {
        module.gglwe_encrypt_sk_tmp_bytes(infos)
    }
}

impl<D: DataMut> GGLWE<D> {
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
        M: GGLWEEncryptSk<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        module.gglwe_encrypt_sk(self, pt, sk, source_xa, source_xe, scratch);
    }
}

pub trait GGLWEEncryptSk<BE: Backend> {
    fn gglwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<BE>;
}

impl<BE: Backend> GGLWEEncryptSk<BE> for Module<BE>
where
    Self: ModuleN
        + GLWEEncryptSk<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxAddScalarInplace
        + VecZnxNormalizeInplace<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn gglwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_encrypt_sk_tmp_bytes(infos)
            + GLWEPlaintext::bytes_of_from_infos(self, infos).max(self.vec_znx_normalize_tmp_bytes())
    }

    fn gglwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<BE>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
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
            scratch.available() >= self.gglwe_encrypt_sk_tmp_bytes(res),
            "scratch.available: {} < GGLWE::encrypt_sk_tmp_bytes(self, res.rank()={}, res.size()={}): {}",
            scratch.available(),
            res.rank_out(),
            res.size(),
            self.gglwe_encrypt_sk_tmp_bytes(res)
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

        let (mut tmp_pt, scrach_1) = scratch.take_glwe_pt(self, &res.glwe_layout());
        // For each input column (i.e. rank) produces a GGLWE of rank_out+1 columns
        //
        // Example for ksk rank 2 to rank 3:
        //
        // (-(a0*s0 + a1*s1 + a2*s2) + s0', a0, a1, a2)
        // (-(b0*s0 + b1*s1 + b2*s2) + s0', b0, b1, b2)
        //
        // Example ksk rank 2 to rank 1
        //
        // (-(a*s) + s0, a)
        // (-(b*s) + s1, b)

        for col_i in 0..rank_in {
            for row_i in 0..dnum {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                tmp_pt.data.zero(); // zeroes for next iteration
                self.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (dsize - 1) + row_i * dsize, pt, col_i);
                self.vec_znx_normalize_inplace(base2k, &mut tmp_pt.data, 0, scrach_1);
                self.glwe_encrypt_sk(
                    &mut res.at_mut(row_i, col_i),
                    &tmp_pt,
                    sk,
                    source_xa,
                    source_xe,
                    scrach_1,
                );
            }
        }
    }
}
