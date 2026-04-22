//! Product of many CKKS ciphertexts via a balanced binary tree.

use anyhow::{Result, bail};
use poulpy_core::{
    GLWECopy, GLWETensoring, ScratchTakeCore,
    layouts::{GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWETensorKeyPrepared, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{CKKSInfos, CKKSMeta, layouts::CKKSCiphertext, leveled::operations::mul::CKKSMulOps, oep::CKKSImpl};

pub trait CKKSMulManyOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_many_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE> + CKKSMulOps<BE>;

    fn ckks_mul_many<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        inputs: &[&CKKSCiphertext<D>],
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWECopy + GLWETensoring<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

fn ceil_log2(n: usize) -> usize {
    debug_assert!(n >= 1);
    if n <= 1 { 0 } else { (n - 1).ilog2() as usize + 1 }
}

fn mul_many_rec<BE, D, M>(
    module: &M,
    dst: &mut CKKSCiphertext<impl DataMut>,
    inputs: &[&CKKSCiphertext<D>],
    tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
    scratch: &mut Scratch<BE>,
) -> Result<()>
where
    BE: Backend + CKKSImpl<BE>,
    D: DataRef,
    M: GLWECopy + GLWETensoring<BE> + CKKSMulOps<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    match inputs.len() {
        1 => {
            module.glwe_copy(dst, inputs[0]);
            dst.meta = inputs[0].meta();
            Ok(())
        }
        2 => module.ckks_mul(dst, inputs[0], inputs[1], tsk, scratch),
        _ => {
            let mid: usize = inputs.len() / 2;
            let (left_slice, right_slice) = inputs.split_at(mid);

            let log_decimal: usize = inputs[0].log_decimal();
            let left_min_eff_k: usize = left_slice.iter().map(|c| c.effective_k()).min().unwrap();
            let right_min_eff_k: usize = right_slice.iter().map(|c| c.effective_k()).min().unwrap();
            let left_max_k: usize = left_min_eff_k.saturating_sub(ceil_log2(left_slice.len()) * log_decimal);
            let right_max_k: usize = right_min_eff_k.saturating_sub(ceil_log2(right_slice.len()) * log_decimal);

            let left_layout = GLWELayout {
                n: dst.n(),
                base2k: dst.base2k(),
                k: TorusPrecision(left_max_k as u32),
                rank: dst.rank(),
            };
            let right_layout = GLWELayout {
                n: dst.n(),
                base2k: dst.base2k(),
                k: TorusPrecision(right_max_k as u32),
                rank: dst.rank(),
            };

            let (left_glwe, scratch_a) = scratch.take_glwe(&left_layout);
            let (right_glwe, scratch_b) = scratch_a.take_glwe(&right_layout);
            let mut left: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(left_glwe, CKKSMeta::default());
            let mut right: CKKSCiphertext<&mut [u8]> = CKKSCiphertext::from_inner(right_glwe, CKKSMeta::default());

            mul_many_rec(module, &mut left, left_slice, tsk, scratch_b)?;
            mul_many_rec(module, &mut right, right_slice, tsk, scratch_b)?;

            module.ckks_mul(dst, &left, &right, tsk, scratch_b)
        }
    }
}

impl<BE: Backend + CKKSImpl<BE>> CKKSMulManyOps<BE> for Module<BE> {
    fn ckks_mul_many_tmp_bytes<R, T>(&self, n: usize, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE> + CKKSMulOps<BE>,
    {
        let mul_scratch: usize = self.ckks_mul_tmp_bytes(res, tsk);
        if n <= 2 {
            return mul_scratch;
        }
        let depth: usize = ceil_log2(n);
        2 * depth * GLWE::<Vec<u8>>::bytes_of_from_infos(res) + mul_scratch
    }

    fn ckks_mul_many<D: DataRef>(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        inputs: &[&CKKSCiphertext<D>],
        tsk: &GLWETensorKeyPrepared<impl DataRef, BE>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWECopy + GLWETensoring<BE> + CKKSMulOps<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        if inputs.is_empty() {
            bail!("ckks_mul_many: inputs must contain at least one ciphertext");
        }
        mul_many_rec(self, dst, inputs, tsk, scratch)
    }
}
