//! CKKS ciphertext multiplication.

use poulpy_core::{
    GLWETensoring, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWELayout, GLWETensor, GLWETensorKeyPrepared, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::{PrecisionInfos, ciphertext::CKKSCiphertext};
use anyhow::Result;

impl CKKSCiphertext<Vec<u8>> {
    pub fn mul_relin_tmp_bytes<R, T, BE: Backend>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Module<BE>: GLWETensoring<BE>,
    {
        let glwe_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32() + 4 * res.base2k().as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = module
            .glwe_tensor_apply_tmp_bytes(&glwe_layout, 0, res, res)
            .max(module.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

        lvl_0 + lvl_1
    }
}

impl<D: DataMut> CKKSCiphertext<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn mul<BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &CKKSCiphertext<impl DataRef>,
        b: &CKKSCiphertext<impl DataRef>,
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        let log_decimal = a.log_decimal();

        let offset = a.log_hom_rem() + a.log_decimal();

        let tensor_layout = GLWELayout {
            n: self.inner.n(),
            base2k: self.base2k(),
            k: TorusPrecision(self.max_k().as_u32() + a.base2k().as_u32()),
            rank: self.rank(),
        };

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&tensor_layout);

        module.glwe_tensor_apply(&mut tmp, offset, &a.inner, &b.inner, scratch_1);

        // TODO: Chose correct optimal size based on noise
        module.glwe_tensor_relinearize(&mut self.inner, &tmp, tsk, tsk.size(), scratch_1);

        self.set_log_hom_rem(a.log_hom_rem() - log_decimal)?;
        self.set_log_decimal(a.log_decimal().max(b.log_decimal()))?;
        Ok(())
    }
}
