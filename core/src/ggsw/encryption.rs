use backend::hal::{
    api::{ScratchTakeVecZnxDft, VecZnxAddScalarInplace, VecZnxAllocBytes, VecZnxNormalizeInplace, ZnxZero},
    layouts::{Backend, Module, ScalarZnx, Scratch},
};
use sampling::source::Source;

use crate::{GGSWCiphertext, GLWECiphertext, GLWEEncryptSkFamily, GLWESecretExec, Infos, TakeGLWEPt};

pub trait GGSWEncryptSkFamily<B: Backend> = GLWEEncryptSkFamily<B>;

impl GGSWCiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: GGSWEncryptSkFamily<B>,
    {
        let size = k.div_ceil(basek);
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k)
            + module.vec_znx_alloc_bytes(rank + 1, size)
            + module.vec_znx_alloc_bytes(1, size)
            + module.vec_znx_dft_alloc_bytes(rank + 1, size)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGSWCiphertext<DataSelf> {
    pub fn encrypt_sk<DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecretExec<DataSk, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        Module<B>: GGSWEncryptSkFamily<B>,
        Scratch<B>: ScratchTakeVecZnxDft<B>,
    {
        #[cfg(debug_assertions)]
        {
            use backend::hal::api::ZnxInfos;

            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk.n(), module.n());
        }

        let basek: usize = self.basek();
        let k: usize = self.k();
        let rank: usize = self.rank();
        let digits: usize = self.digits();

        let (mut tmp_pt, scratch1) = scratch.take_glwe_pt(module, basek, k);

        (0..self.rows()).for_each(|row_i| {
            tmp_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, (digits - 1) + row_i * digits, pt, 0);
            module.vec_znx_normalize_inplace(basek, &mut tmp_pt.data, 0, scratch1);

            (0..rank + 1).for_each(|col_j| {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                self.at_mut(row_i, col_j).encrypt_sk_private(
                    module,
                    Some((&tmp_pt, col_j)),
                    sk,
                    source_xa,
                    source_xe,
                    sigma,
                    scratch1,
                );
            });
        });
    }
}
