use itertools::{Itertools, izip};
use poulpy_core::layouts::{GLWECiphertext, GLWEPlaintext, TorusPrecision, prepared::GGSWCiphertextPrepared};
use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxSub,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{DataMut, DataRef, Module, Scratch},
};

use crate::tfhe::arithmetic::{BitCircuitInfo, Circuit, Helper};

/// Binary operations 1 x 1 -> 1
pub(crate) enum BinOps2x1<C: BitCircuitInfo> {
    AND(Circuit<C, 1>),
    OR(Circuit<C, 1>),
    XOR(Circuit<C, 1>),
}

impl<C: BitCircuitInfo> BinOps2x1<C> {
    pub(crate) fn circuit(&self) -> &Circuit<C, 1> {
        match self {
            BinOps2x1::AND(circuit) => circuit,
            BinOps2x1::OR(circuit) => circuit,
            BinOps2x1::XOR(circuit) => circuit,
        }
    }

    pub(crate) fn tmp_glwe_bounds<const WORD_SIZE: usize>(&self) -> (usize, Vec<usize>) {
        match self {
            BinOps2x1::OR(_) | BinOps2x1::AND(_) => {
                // AND and OR circuit have same structure
                let glwe_bounds = (0..WORD_SIZE).map(|i| i + 1).collect_vec();
                (WORD_SIZE, glwe_bounds)
            }
            BinOps2x1::XOR(_) => {
                let glwe_bounds = (0..WORD_SIZE).map(|i| (i + 1) * 2).collect_vec();
                (WORD_SIZE * 2, glwe_bounds)
            }
            _ => {
                unimplemented!()
            }
        }
    }
}
