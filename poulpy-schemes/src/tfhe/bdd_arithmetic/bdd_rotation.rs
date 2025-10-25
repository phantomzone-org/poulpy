use poulpy_core::{
    GLWECopy, GLWERotate, ScratchTakeCore,
    layouts::{GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWE, GLWEInfos, GLWEToMut, GLWEToRef},
};
use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::tfhe::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};

impl<T: UnsignedInteger, BE: Backend> GGSWBlindRotation<T, BE> for Module<BE>
where
    Self: GLWEBlindRotation<T, BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}

pub trait GGSWBlindRotation<T: UnsignedInteger, BE: Backend>
where
    Self: GLWEBlindRotation<T, BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos)
    }

    fn ggsw_blind_rotation<R, G, K>(
        &self,
        res: &mut R,
        test_ggsw: &G,
        k: &K,
        bit_start: usize,
        bit_size: usize,
        bit_step: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        G: GGSWToRef,
        K: GetGGSWBit<T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let test_ggsw: &GGSW<&[u8]> = &test_ggsw.to_ref();

        for row in 0..res.dnum().into() {
            for col in 0..(res.rank() + 1).into() {
                self.glwe_blind_rotation(
                    &mut res.at_mut(row, col),
                    &test_ggsw.at(row, col),
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

impl<T: UnsignedInteger, BE: Backend> GLWEBlindRotation<T, BE> for Module<BE>
where
    Self: GLWECopy + GLWERotate<BE> + Cmux<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}

pub trait GLWEBlindRotation<T: UnsignedInteger, BE: Backend>
where
    Self: GLWECopy + GLWERotate<BE> + Cmux<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, k_infos) + GLWE::bytes_of_from_infos(res_infos)
    }

    /// Homomorphic multiplication of res by X^{k[bit_start..bit_start + bit_size] * bit_step}.
    fn glwe_blind_rotation<R, G, K>(
        &self,
        res: &mut R,
        test_glwe: &G,
        k: &K,
        bit_start: usize,
        bit_size: usize,
        bit_step: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        G: GLWEToRef,
        K: GetGGSWBit<T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(bit_start + bit_size <= T::WORD_SIZE);

        let mut res: GLWE<&mut [u8]> = res.to_mut();

        let (mut tmp_res, scratch_1) = scratch.take_glwe(&res);

        // res <- test_glwe
        self.glwe_copy(&mut res, test_glwe);

        // a_is_res = true  => (a, b) = (&mut res, &mut tmp_res)
        // a_is_res = false => (a, b) = (&mut tmp_res, &mut res)
        let mut a_is_res: bool = true;

        for i in 0..bit_size {
            let (a, b) = if a_is_res {
                (&mut res, &mut tmp_res)
            } else {
                (&mut tmp_res, &mut res)
            };

            // a <- a ; b <- a * X^{-2^{i + bit_step}}
            self.glwe_rotate(-1 << (i + bit_step), b, a);

            // b <- (b - a) * GGSW(b[i]) + a
            self.cmux_inplace(b, a, &k.get_bit(i + bit_start), scratch_1);

            // ping-pong roles for next iter
            a_is_res = !a_is_res;
        }

        // Ensure the final value ends up in `res`
        if !a_is_res {
            self.glwe_copy(&mut res, &tmp_res);
        }
    }
}
