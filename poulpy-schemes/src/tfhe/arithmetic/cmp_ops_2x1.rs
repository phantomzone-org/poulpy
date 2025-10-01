use itertools::{Itertools, izip};
use poulpy_core::layouts::{
    GLWECiphertext, GLWEPlaintext, TorusPrecision,
    prepared::{GGSWCiphertextPrepared, GGSWCiphertextPreparedToRef},
};
use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxSub,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::tfhe::arithmetic::{BitCircuit, BitCircuitInfo, Circuit, Helper, circuits, execute};

/// Comparison operations Z x Z -> 1
pub(crate) enum CmpOps2x1<C: BitCircuitInfo + 'static, const WORD_SIZE: usize> {
    SLT(&'static Circuit<C, 1>),
    SLTU(&'static Circuit<C, 1>),
}

pub(crate) const SLT_OP32: CmpOps2x1<BitCircuit<97usize, 65usize>, 32> = CmpOps2x1::SLT(&circuits::slt_32::OUTPUT_CIRCUIT);
pub(crate) const SLTU_OP32: CmpOps2x1<BitCircuit<97usize, 65usize>, 32> = CmpOps2x1::SLTU(&circuits::sltu_32::OUTPUT_CIRCUIT);

impl<C: BitCircuitInfo, const WORD_SIZE: usize> CmpOps2x1<C, WORD_SIZE> {
    pub(crate) const fn word_size(&self) -> usize {
        WORD_SIZE
    }

    pub(crate) fn circuit(&self) -> &Circuit<C, 1> {
        match self {
            CmpOps2x1::SLT(circuit) => circuit,
            CmpOps2x1::SLTU(circuit) => circuit,
        }
    }

    pub(crate) fn tmp_glwe_bounds(&self) -> (usize, Vec<usize>) {
        let mut bounds = vec![];
        let mut curr_idx = 0;
        self.circuit().0.iter().for_each(|c| {
            let (nodes, _) = c.info();
            // skip 0,1 terminal nodes since they are fixed to plaintext constants 0,1
            // skip last node since its output is written to the actual output bit
            nodes
                .iter()
                .take(nodes.len() - 1)
                .skip(2)
                .for_each(|_| curr_idx += 1);
            bounds.push(curr_idx);
        });
        (curr_idx, bounds)
    }
}

impl<C: BitCircuitInfo, const WORD_SIZE: usize> CmpOps2x1<C, WORD_SIZE> {
    // Sets res <- a.op(b)
    pub(crate) fn execute<R: DataMut, A: DataRef, B: DataRef, BE: Backend>(
        &self,
        module: &Module<BE>,
        res: &mut [GLWECiphertext<R>],
        a: &[GGSWCiphertextPrepared<A, BE>],
        b: &[GGSWCiphertextPrepared<B, BE>],
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: VecZnxSub
            + VecZnxCopy
            + VecZnxNegateInplace
            + VecZnxDftAllocBytes
            + VecZnxAddInplace
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<BE>
            + VmpApplyDftToDft<BE>
            + VmpApplyDftToDftAdd<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxNormalize<BE>,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(res.len(), 1);
            assert_eq!(a.len(), WORD_SIZE);
            assert_eq!(b.len(), WORD_SIZE);
        }

        // Allocates terminal nodes: [GLWE(0), GLWE(1)]
        // TODO: provide them from the evaluator?
        let mut terminal_nodes: [GLWEPlaintext<Vec<u8>>; 2] = [GLWEPlaintext::alloc(&res[0]), GLWEPlaintext::alloc(&res[0])];
        terminal_nodes[1].encode_coeff_i64(1, TorusPrecision(1), 0);

        // Allocates the helper
        // TODO: provide it from the evaluator?
        let (total_tmp_glwes, tmp_glwes_bounds) = self.tmp_glwe_bounds();

        let tmp_glwes: Vec<GLWECiphertext<Vec<u8>>> = (0..total_tmp_glwes)
            .map(|_| GLWECiphertext::alloc(&res[0]))
            .collect_vec();

        let mut helper: Helper = Helper {
            tmp_glwes,
            tmp_glwes_bounds,
            single_bit_output: false,
        };

        let tmp: Vec<&mut [GLWECiphertext<Vec<u8>>]> = helper.split_tmp_glwes_mut();

        // Collects inputs into a single array
        let inputs: Vec<&dyn GGSWCiphertextPreparedToRef<BE>> = a
            .iter()
            .map(|x| x as &dyn GGSWCiphertextPreparedToRef<BE>)
            .chain(b.iter().map(|x| x as &dyn GGSWCiphertextPreparedToRef<BE>))
            .collect_vec();

        // Evaluates out[i] = circuit[i](a, b)
        for (circuit_i, out_i, tmp_outs) in izip!(self.circuit().0.iter(), res.iter_mut(), tmp) {
            execute(
                circuit_i,
                module,
                out_i,
                &inputs[..],
                &terminal_nodes,
                tmp_outs,
                scratch,
            );
        }
    }
}
