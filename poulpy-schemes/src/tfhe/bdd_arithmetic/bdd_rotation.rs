use poulpy_core::{
    GLWECopy, GLWERotate, ScratchTakeCore,
    layouts::{GLWE, GLWEToMut},
};
use poulpy_hal::layouts::{Backend, Scratch};

use crate::tfhe::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};

pub trait BDDRotation<T: UnsignedInteger, BE: Backend>
where
    Self: GLWECopy + GLWERotate<BE> + Cmux<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    /// Homomorphic multiplication of res by X^{k[bit_start..bit_start + bit_size] * bit_step}.
    fn bdd_rotate<R, K, D>(
        &self,
        res: &mut R,
        k: K,
        bit_start: usize,
        bit_size: usize,
        bit_step: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        K: GetGGSWBit<T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        let (mut tmp_res, scratch_1) = scratch.take_glwe(res);

        self.glwe_copy(&mut tmp_res, res);

        for i in 1..bit_size {
            // res' = res * X^2^(i * bit_step)
            self.glwe_rotate(1 << (i + bit_step), &mut tmp_res, res);

            // res = (res - res') * GGSW(b[i]) + res'
            self.cmux_inplace(res, &tmp_res, &k.get_bit(i + bit_start), scratch_1);
        }
    }
}
