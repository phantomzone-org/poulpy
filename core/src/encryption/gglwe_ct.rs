use backend::hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
        ZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch},
};
use sampling::source::Source;

use crate::{
    TakeGLWEPt,
    layouts::{GGLWECiphertext, GLWECiphertext, GLWEPlaintext, Infos, prepared::GLWESecretPrepared},
};

use crate::trait_families::GGLWEEncryptSkFamily;

impl GGLWECiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWECiphertext::encrypt_sk_scratch_space(module, n, basek, k)
            + (GLWEPlaintext::byte_of(n, basek, k) | module.vec_znx_normalize_tmp_bytes(n))
    }

    pub fn encrypt_pk_scratch_space<B: Backend>(_module: &Module<B>, _n: usize, _basek: usize, _k: usize, _rank: usize) -> usize {
        unimplemented!()
    }
}

impl<DataSelf: DataMut> GGLWECiphertext<DataSelf> {
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGLWEEncryptSkFamily<B> + VecZnxAddScalarInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            use backend::hal::api::ZnxInfos;

            assert_eq!(
                self.rank_in(),
                pt.cols(),
                "self.rank_in(): {} != pt.cols(): {}",
                self.rank_in(),
                pt.cols()
            );
            assert_eq!(
                self.rank_out(),
                sk.rank(),
                "self.rank_out(): {} != sk.rank(): {}",
                self.rank_out(),
                sk.rank()
            );
            assert_eq!(self.n(), sk.n());
            assert_eq!(pt.n(), sk.n());
            assert!(
                scratch.available() >= GGLWECiphertext::encrypt_sk_scratch_space(module, sk.n(), self.basek(), self.k()),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GGLWECiphertext::encrypt_sk_scratch_space(module, sk.n(), self.basek(), self.k())
            );
            assert!(
                self.rows() * self.digits() * self.basek() <= self.k(),
                "self.rows() : {} * self.digits() : {} * self.basek() : {} = {} >= self.k() = {}",
                self.rows(),
                self.digits(),
                self.basek(),
                self.rows() * self.digits() * self.basek(),
                self.k()
            );
        }

        let rows: usize = self.rows();
        let digits: usize = self.digits();
        let basek: usize = self.basek();
        let k: usize = self.k();
        let rank_in: usize = self.rank_in();

        let (mut tmp_pt, scrach_1) = scratch.take_glwe_pt(sk.n(), basek, k);
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
        (0..rank_in).for_each(|col_i| {
            (0..rows).for_each(|row_i| {
                // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
                tmp_pt.data.zero(); // zeroes for next iteration
                module.vec_znx_add_scalar_inplace(
                    &mut tmp_pt.data,
                    0,
                    (digits - 1) + row_i * digits,
                    pt,
                    col_i,
                );
                module.vec_znx_normalize_inplace(basek, &mut tmp_pt.data, 0, scrach_1);

                // rlwe encrypt of vec_znx_pt into vec_znx_ct
                self.at_mut(row_i, col_i)
                    .encrypt_sk(module, &tmp_pt, sk, source_xa, source_xe, sigma, scrach_1);
            });
        });
    }
}
