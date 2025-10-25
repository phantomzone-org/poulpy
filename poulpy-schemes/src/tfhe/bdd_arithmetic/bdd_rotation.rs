use poulpy_core::{
    GLWECopy, GLWERotate, ScratchTakeCore,
    layouts::{GGSW, GGSWInfos, GGSWToMut, GLWE, GLWEInfos, GLWEToMut},
};
use poulpy_hal::layouts::{Backend, Scratch};

use crate::tfhe::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};

pub trait GGSWBlindRotation<T: UnsignedInteger, BE: Backend>
where
    Self: GLWEBlindRotation<T, BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn ggsw_blind_rotation<R, K>(
        &self,
        res: &mut R,
        k: &K,
        bit_start: usize,
        bit_size: usize,
        bit_step: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        K: GetGGSWBit<T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();

        for row in 0..res.dnum().into() {
            for col in 0..(res.rank() + 1).into() {
                self.glwe_blind_rotation(
                    &mut res.at_mut(row, col),
                    k,
                    bit_start,
                    bit_size,
                    bit_step,
                    scratch,
                );
            }
        }
    }
}

pub trait GLWEBlindRotation<T: UnsignedInteger, BE: Backend>
where
    Self: GLWECopy + GLWERotate<BE> + Cmux<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_blind_rotation_tmp_bytes<R, A, B>(&self, res_infos: &R, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        B: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, b_infos) + GLWE::bytes_of_from_infos(res_infos)
    }

    /// Homomorphic multiplication of res by X^{k[bit_start..bit_start + bit_size] * bit_step}.
    fn glwe_blind_rotation<R, K>(
        &self,
        res: &mut R,
        k: &K,
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
