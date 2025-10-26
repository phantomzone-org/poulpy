use poulpy_core::{
    GLWECopy, GLWERotate, ScratchTakeCore,
    layouts::{GGSW, GGSWInfos, GGSWToMut, GGSWToRef, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxAddScalarInplace, VecZnxNormalizeInplace},
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToRef, Scratch, ZnxZero},
};

use crate::tfhe::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};

impl<T: UnsignedInteger, BE: Backend> GGSWBlindRotation<T, BE> for Module<BE>
where
    Self: GLWEBlindRotation<T, BE> + VecZnxAddScalarInplace + VecZnxNormalizeInplace<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}

pub trait GGSWBlindRotation<T: UnsignedInteger, BE: Backend>
where
    Self: GLWEBlindRotation<T, BE> + VecZnxAddScalarInplace + VecZnxNormalizeInplace<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn ggsw_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_to_glwe_blind_rotation_tmp_bytes(res_infos, k_infos)
    }

    /// res <- a * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn ggsw_to_ggsw_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        k: &K,
        bit_start: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GetGGSWBit<T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                self.glwe_to_glwe_blind_rotation(
                    &mut res.at_mut(row, col),
                    &a.at(row, col),
                    k,
                    bit_start,
                    bit_mask,
                    bit_lsh,
                    scratch,
                );
            }
        }
    }

    fn scalar_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_to_glwe_blind_rotation_tmp_bytes(res_infos, k_infos) + GLWE::bytes_of_from_infos(res_infos)
    }

    fn scalar_to_ggsw_blind_rotation<R, S, K>(
        &self,
        res: &mut R,
        test_vector: &S,
        k: &K,
        bit_start: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        S: ScalarZnxToRef,
        K: GetGGSWBit<T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let test_vector: &ScalarZnx<&[u8]> = &test_vector.to_ref();

        let base2k: usize = res.base2k().into();
        let dsize: usize = res.dsize().into();

        let (mut tmp_glwe, scratch_1) = scratch.take_glwe(res);

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                tmp_glwe.data_mut().zero();
                self.vec_znx_add_scalar_inplace(
                    tmp_glwe.data_mut(),
                    col,
                    (dsize - 1) + row * dsize,
                    test_vector,
                    0,
                );
                self.vec_znx_normalize_inplace(base2k, tmp_glwe.data_mut(), col, scratch_1);

                self.glwe_to_glwe_blind_rotation(
                    &mut res.at_mut(row, col),
                    &tmp_glwe,
                    k,
                    bit_start,
                    bit_mask,
                    bit_lsh,
                    scratch_1,
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
    fn glwe_to_glwe_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, k_infos) + GLWE::bytes_of_from_infos(res_infos)
    }

    /// res <- a * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn glwe_to_glwe_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        k: &K,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetGGSWBit<T, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(bit_rsh + bit_mask <= T::WORD_SIZE);

        let mut res: GLWE<&mut [u8]> = res.to_mut();

        let (mut tmp_res, scratch_1) = scratch.take_glwe(&res);

        // a <- a ; b <- a * X^{-2^{i + bit_lsh}}
        self.glwe_rotate(-1 << bit_lsh, &mut res, a);

        // b <- (b - a) * GGSW(b[i]) + a
        self.cmux_inplace(&mut res, a, &k.get_bit(bit_rsh), scratch_1);

        // a_is_res = true  => (a, b) = (&mut res, &mut tmp_res)
        // a_is_res = false => (a, b) = (&mut tmp_res, &mut res)
        let mut a_is_res: bool = true;

        for i in 1..bit_mask {
            let (a, b) = if a_is_res {
                (&mut res, &mut tmp_res)
            } else {
                (&mut tmp_res, &mut res)
            };

            // a <- a ; b <- a * X^{-2^{i + bit_lsh}}
            self.glwe_rotate(-1 << (i + bit_lsh), b, a);

            // b <- (b - a) * GGSW(b[i]) + a
            self.cmux_inplace(b, a, &k.get_bit(i + bit_rsh), scratch_1);

            // ping-pong roles for next iter
            a_is_res = !a_is_res;
        }

        // Ensure the final value ends up in `res`
        if !a_is_res {
            self.glwe_copy(&mut res, &tmp_res);
        }
    }
}
