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
    Self: GLWEBlindRotation<BE> + VecZnxAddScalarInplace + VecZnxNormalizeInplace<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}

pub trait GGSWBlindRotation<T: UnsignedInteger, BE: Backend>
where
    Self: GLWEBlindRotation<BE> + VecZnxAddScalarInplace + VecZnxNormalizeInplace<BE>,
{
    fn ggsw_to_ggsw_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos)
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- res * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn ggsw_blind_rotation_inplace<R, K>(
        &self,
        res: &mut R,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                self.glwe_blind_rotation_inplace(
                    &mut res.at_mut(row, col),
                    fhe_uint,
                    sign,
                    bit_rsh,
                    bit_mask,
                    bit_lsh,
                    scratch,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- a * X^{((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn ggsw_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let a: &GGSW<&[u8]> = &a.to_ref();

        assert!(res.dnum() <= a.dnum());
        assert_eq!(res.dsize(), a.dsize());

        for col in 0..(res.rank() + 1).into() {
            for row in 0..res.dnum().into() {
                self.glwe_blind_rotation(
                    &mut res.at_mut(row, col),
                    &a.at(row, col),
                    fhe_uint,
                    sign,
                    bit_rsh,
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
        self.glwe_blind_rotation_tmp_bytes(res_infos, k_infos) + GLWE::bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_to_ggsw_blind_rotation<R, S, K>(
        &self,
        res: &mut R,
        test_vector: &S,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        S: ScalarZnxToRef,
        K: GetGGSWBit<BE>,
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

                self.glwe_blind_rotation(
                    &mut res.at_mut(row, col),
                    &tmp_glwe,
                    fhe_uint,
                    sign,
                    bit_rsh,
                    bit_mask,
                    bit_lsh,
                    scratch_1,
                );
            }
        }
    }
}

impl<BE: Backend> GLWEBlindRotation<BE> for Module<BE>
where
    Self: GLWECopy + GLWERotate<BE> + Cmux<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}

pub trait GLWEBlindRotation<BE: Backend>
where
    Self: GLWECopy + GLWERotate<BE> + Cmux<BE>,
{
    fn glwe_blind_rotation_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, k_infos) + GLWE::bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_rotation_inplace<R, K>(
        &self,
        res: &mut R,
        value: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let mut res: GLWE<&mut [u8]> = res.to_mut();

        let (mut tmp_res, scratch_1) = scratch.take_glwe(&res);

        // a_is_res = true  => (a, b) = (&mut res, &mut tmp_res)
        // a_is_res = false => (a, b) = (&mut tmp_res, &mut res)
        let mut a_is_res: bool = true;

        for i in 0..bit_mask {
            let (a, b) = if a_is_res {
                (&mut res, &mut tmp_res)
            } else {
                (&mut tmp_res, &mut res)
            };

            // a <- a ; b <- a * X^{-2^{i + bit_lsh}}
            match sign {
                true => self.glwe_rotate(1 << (i + bit_lsh), b, a),
                false => self.glwe_rotate(-1 << (i + bit_lsh), b, a),
            }

            // b <- (b - a) * GGSW(b[i]) + a
            self.cmux_inplace(b, a, &value.get_bit(i + bit_rsh), scratch_1);

            // ping-pong roles for next iter
            a_is_res = !a_is_res;
        }

        // Ensure the final value ends up in `res`
        if !a_is_res {
            self.glwe_copy(&mut res, &tmp_res);
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// res <- a * X^{sign * ((k>>bit_rsh) % 2^bit_mask) << bit_lsh}.
    fn glwe_blind_rotation<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        fhe_uint: &K,
        sign: bool,
        bit_rsh: usize,
        bit_mask: usize,
        bit_lsh: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        A: GLWEToRef,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.glwe_copy(res, a);
        self.glwe_blind_rotation_inplace(res, fhe_uint, sign, bit_rsh, bit_mask, bit_lsh, scratch);
    }
}
