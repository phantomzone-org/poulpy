use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAddScalarInplace, VecZnxDftBytesOf, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxZero},
    source::Source,
};

use crate::{
    encryption::glwe_ct::GLWEEncryptSk, layouts::GLWEInfos, ScratchTakeCore,
    layouts::{
        GGLWE, GGLWEInfos, GGLWEToMut, GLWE, GLWEPlaintext, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl GGLWE<Vec<u8>> {
    pub fn encrypt_sk_tmp_bytes<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxNormalizeTmpBytes,
    {
        GLWE::encrypt_sk_tmp_bytes(module, &infos.glwe_layout())
            + (GLWEPlaintext::bytes_of_from_infos(module, &infos.glwe_layout()) | module.vec_znx_normalize_tmp_bytes())
    }

    pub fn encrypt_pk_tmp_bytes<B: Backend, A>(_module: &Module<B>, _infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        unimplemented!()
    }
}

pub trait GGLWEEncryptSk<B: Backend> {
    fn gglwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<B>;
}

impl<B: Backend> GGLWEEncryptSk<B> for Module<B>
where
    Module<B>: ModuleN + GLWEEncryptSk<B> + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf + VecZnxAddScalarInplace + VecZnxNormalizeInplace<B>,
    Scratch<B>: ScratchAvailable + ScratchTakeCore<B>,
{
    fn gglwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToRef,
        S: GLWESecretPreparedToRef<B>,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let pt: &ScalarZnx<&[u8]> = &pt.to_ref();

        #[cfg(debug_assertions)]
        {
            use poulpy_hal::layouts::ZnxInfos;
            let sk: GLWESecretPrepared<&[u8], B> = sk.to_ref();

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
                scratch.available() >= GGLWE::encrypt_sk_tmp_bytes(self, res),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_tmp_bytes(self, res.rank()={}, res.size()={}): {}",
                scratch.available(),
                res.rank_out(),
                res.size(),
                GGLWE::encrypt_sk_tmp_bytes(self, res)
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

        let (mut tmp_pt, scrach_1) = scratch.take_glwe_pt(self, &res.glwe_layout());
        // For each input column (i.e. rank) produces a GGLWE ciphertext of rank_out+1 columns
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

impl<DataSelf: DataMut> GGLWE<DataSelf> {
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
        Module<B>: GGLWEEncryptSk<B>,
    {
        module.gglwe_encrypt_sk(self, pt, sk, source_xa, source_xe, scratch);
    }
}
