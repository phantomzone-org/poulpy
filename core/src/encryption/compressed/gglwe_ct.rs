use backend::hal::{
    api::{ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxNormalizeInplace, ZnxZero},
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch},
};
use sampling::source::Source;

use crate::{
    TakeGLWEPt,
    encryption::glwe_encrypt_sk_internal,
    layouts::{GGLWECiphertext, Infos, compressed::GGLWECiphertextCompressed, prepared::GLWESecretPrepared},
};

use crate::trait_families::{GGLWEEncryptSkFamily, GGLWESwitchingKeyEncryptSkFamily};

impl GGLWECiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWESwitchingKeyEncryptSkFamily<B>,
    {
        GGLWECiphertext::encrypt_sk_scratch_space(module, n, basek, k)
    }
}

impl<D: DataMut> GGLWECiphertextCompressed<D> {
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        seed: [u8; 32],
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
                scratch.available()
                    >= GGLWECiphertextCompressed::encrypt_sk_scratch_space(module, sk.n(), self.basek(), self.k()),
                "scratch.available: {} < GGLWECiphertext::encrypt_sk_scratch_space(module, self.rank()={}, self.size()={}): {}",
                scratch.available(),
                self.rank(),
                self.size(),
                GGLWECiphertextCompressed::encrypt_sk_scratch_space(module, sk.n(), self.basek(), self.k())
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
        let cols: usize = self.rank_out() + 1;

        let mut source_xa = Source::new(seed);

        let (mut tmp_pt, scrach_1) = scratch.take_glwe_pt(sk.n(), basek, k);
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

                let (seed, mut source_xa_tmp) = source_xa.branch();
                self.seed[col_i * rows + row_i] = seed;

                glwe_encrypt_sk_internal(
                    module,
                    self.basek(),
                    self.k(),
                    &mut self.at_mut(row_i, col_i).data,
                    cols,
                    true,
                    Some((&tmp_pt, 0)),
                    sk,
                    &mut source_xa_tmp,
                    source_xe,
                    sigma,
                    scrach_1,
                );
            });
        });
    }
}
