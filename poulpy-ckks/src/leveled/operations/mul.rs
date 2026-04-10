//! CKKS ciphertext multiplication.

use poulpy_core::{
    GLWETensoring, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWEInfos, GLWETensor, GLWETensorKeyPrepared, LWEInfos},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::layouts::ciphertext::CKKSCiphertext;
use anyhow::Result;

impl CKKSCiphertext<Vec<u8>> {
    pub fn mul_relin_tmp_bytes<R, T, BE: Backend>(module: &Module<BE>, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Module<BE>: GLWETensoring<BE>,
    {
        let lvl_0 = GLWETensor::bytes_of_from_infos(res);
        let lvl_1 = module
            .glwe_tensor_apply_tmp_bytes(res, 0, res, res)
            .max(module.glwe_tensor_relinearize_tmp_bytes(res, res, tsk));

        lvl_0 + lvl_1
    }
}

impl<D: DataMut> CKKSCiphertext<D> {
    pub fn mul_relin<BE: Backend>(
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
        let a_max_k = a.inner.max_k().as_usize();
        let a_decimal = a_max_k - a.log_delta;

        let offset = a_max_k;

        let (mut tmp, scratch_1) = scratch.take_glwe_tensor(&self.inner);

        module.glwe_tensor_apply(&mut tmp, offset, &a.inner, &b.inner, scratch_1);

        // TODO: Chose correct optimal size based on noise
        module.glwe_tensor_relinearize(&mut self.inner, &tmp, tsk, tsk.size(), scratch_1);

        self.log_delta = a.log_delta - a_decimal;
        Ok(())
    }
}
