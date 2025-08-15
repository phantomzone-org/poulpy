use backend::hal::{
    api::{ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxNormalizeInplace, ZnxZero},
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnx, Scratch},
};
use sampling::source::Source;

use crate::{
    TakeGLWEPt,
    encryption::glwe_encrypt_sk_internal,
    layouts::{GGSWCiphertext, Infos, compressed::GGSWCiphertextCompressed, prepared::GLWESecretPrepared},
};

use crate::trait_families::GGSWEncryptSkFamily;

impl GGSWCiphertextCompressed<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GGSWEncryptSkFamily<B>,
    {
        GGSWCiphertext::encrypt_sk_scratch_space(module, n, basek, k, rank)
    }
}

impl<DataSelf: DataMut> GGSWCiphertextCompressed<DataSelf> {
    pub fn encrypt_sk<DataPt: DataRef, DataSk: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretPrepared<DataSk, B>,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGSWEncryptSkFamily<B> + VecZnxAddScalarInplace,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            use backend::hal::api::ZnxInfos;

            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), sk.n());
            assert_eq!(pt.n(), sk.n());
        }

        let basek: usize = self.basek();
        let k: usize = self.k();
        let rank: usize = self.rank();
        let cols: usize = rank + 1;
        let digits: usize = self.digits();

        let (mut tmp_pt, scratch_1) = scratch.take_glwe_pt(self.n(), basek, k);

        let mut source = Source::new(seed_xa);

        self.seed = vec![[0u8; 32]; self.rows() * cols];

        (0..self.rows()).for_each(|row_i| {
            tmp_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (digits - 1) + row_i * digits, pt, 0);
            module.vec_znx_normalize_inplace(basek, &mut tmp_pt.data, 0, scratch_1);

            (0..rank + 1).for_each(|col_j| {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                let (seed, mut source_xa_tmp) = source.branch();

                self.seed[row_i * cols + col_j] = seed;

                glwe_encrypt_sk_internal(
                    module,
                    self.basek(),
                    self.k(),
                    &mut self.at_mut(row_i, col_j).data,
                    cols,
                    true,
                    Some((&tmp_pt, col_j)),
                    sk,
                    &mut source_xa_tmp,
                    source_xe,
                    sigma,
                    scratch_1,
                );
            });
        });
    }
}
