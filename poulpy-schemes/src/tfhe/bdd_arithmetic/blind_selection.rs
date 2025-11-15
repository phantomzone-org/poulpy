use std::collections::HashMap;

use poulpy_core::{
    GLWECopy, GLWEDecrypt, ScratchTakeCore,
    layouts::{GGSWInfos, GGSWPrepared, GLWE, GLWEInfos, GLWEToMut},
};
use poulpy_hal::layouts::{Backend, Module, Scratch, ZnxZero};

use crate::tfhe::bdd_arithmetic::{Cmux, GetGGSWBit, UnsignedInteger};

impl<T: UnsignedInteger, BE: Backend> GLWEBlinSelection<T, BE> for Module<BE> where Self: GLWECopy + Cmux<BE> + GLWEDecrypt<BE> {}

pub trait GLWEBlinSelection<T: UnsignedInteger, BE: Backend>
where
    Self: GLWECopy + Cmux<BE> + GLWEDecrypt<BE>,
{
    fn glwe_blind_selection_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cmux_tmp_bytes(res_infos, res_infos, k_infos) + GLWE::bytes_of_from_infos(res_infos)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_blind_selection<R, A, K>(
        &self,
        res: &mut R,
        mut a: HashMap<usize, &mut A>,
        fhe_uint: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        A: GLWEToMut,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(bit_rsh + bit_mask <= T::BITS as usize);

        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        for i in 0..bit_mask {
            let t: usize = 1 << (bit_mask - i - 1);

            let bit: &GGSWPrepared<&[u8], BE> = &fhe_uint.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal

            for j in 0..t {
                let hi: Option<&mut A> = a.remove(&j);
                let lo: Option<&mut A> = a.remove(&(j + t));

                match (lo, hi) {
                    (Some(lo), Some(hi)) => {
                        self.cmux_inplace(lo, hi, bit, scratch);
                        a.insert(j, lo);
                    }

                    (Some(lo), None) => {
                        let (mut zero, scratch_1) = scratch.take_glwe(res);
                        zero.data_mut().zero();
                        self.cmux_inplace(lo, &zero, bit, scratch_1);
                        a.insert(j, lo);
                    }

                    (None, Some(hi)) => {
                        let (mut zero, scratch_1) = scratch.take_glwe(res);
                        zero.data_mut().zero();
                        self.cmux_inplace(&mut zero, hi, bit, scratch_1);
                        self.glwe_copy(hi, &zero);
                        a.insert(j, hi);
                    }

                    (None, None) => {
                        // No low or high branch — nothing to insert
                        // leave empty; future iterations will combine actual ciphertexts
                    }
                }
            }
        }

        let out: Option<&mut A> = a.remove(&0);

        if let Some(out) = out {
            self.glwe_copy(res, out);
        } else {
            res.data_mut().zero();
        }
    }
}
