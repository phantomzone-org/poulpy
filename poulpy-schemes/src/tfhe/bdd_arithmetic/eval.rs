use itertools::{Itertools, izip};
use poulpy_core::{
    GLWEOperations, TakeGLWECtSlice,
    layouts::{
        GLWECiphertext, GLWECiphertextLayout, GLWECiphertextToMut, GLWEInfos, LWEInfos,
        prepared::{GGSWCiphertextPrepared, GGSWCiphertextPreparedToRef},
    },
};
use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxSub,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxZero},
};

pub trait BitCircuitInfo {
    fn info(&self) -> (&[Node], &[usize], usize);
}

pub(crate) struct BitCircuit<const N: usize, const K: usize> {
    pub(crate) nodes: [Node; N],
    pub(crate) levels: [usize; K],
    pub(crate) max_inter_state: usize,
}

pub struct Circuit<C: BitCircuitInfo, const N: usize>(pub [C; N]);

pub trait CircuitExecute<BE: Backend> {
    fn execute<R>(
        &self,
        module: &Module<BE>,
        out: &mut [GLWECiphertext<R>],
        inputs: &[&dyn GGSWCiphertextPreparedToRef<BE>],
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut;
}

impl<C: BitCircuitInfo, const N: usize, BE: Backend> CircuitExecute<BE> for Circuit<C, N>
where
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
    Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeGLWECtSlice,
{
    fn execute<R>(
        &self,
        module: &Module<BE>,
        out: &mut [GLWECiphertext<R>],
        inputs: &[&dyn GGSWCiphertextPreparedToRef<BE>],
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
    {
        let glwe_infos: GLWECiphertextLayout = out[0].glwe_layout();

        for (circuit_i, out_i) in izip!(self.0.iter(), out.iter_mut()) {
            let (nodes, levels, max_inter_state) = circuit_i.info();

            let (mut level, scratch_1) = scratch.take_glwe_ct_slice(max_inter_state * 2, &glwe_infos);

            level.iter_mut().for_each(|ct| ct.data_mut().zero());

            // TODO: implement API on GLWE
            level[1]
                .data_mut()
                .encode_coeff_i64(glwe_infos.base2k().into(), 0, 2, 0, 1);

            let mut level_ref = level.iter_mut().map(|c| c).collect_vec();
            let (mut prev_level, mut next_level) = level_ref.split_at_mut(max_inter_state);

            for i in 0..levels.len() - 1 {
                let start: usize = levels[i];
                let end: usize = levels[i + 1];

                let nodes_lvl: &[Node] = &nodes[start..end];

                for (j, node) in nodes_lvl.iter().enumerate() {
                    if node.low_index == node.high_index {
                        next_level[j].copy(module, prev_level[node.low_index]);
                    } else {
                        next_level[j].cmux(
                            module,
                            prev_level[node.high_index],
                            prev_level[node.low_index],
                            &inputs[node.input_index].to_ref(),
                            scratch_1,
                        );
                    }
                }

                (prev_level, next_level) = (next_level, prev_level);
            }

            // handle last output
            // there's always only 1 node at last level
            let node: &Node = nodes.last().unwrap();
            out_i.cmux(
                module,
                prev_level[node.high_index],
                prev_level[node.low_index],
                &inputs[node.input_index].to_ref(),
                scratch_1,
            );
        }
    }
}

impl<const N: usize, const K: usize> BitCircuit<N, K> {
    pub(crate) const fn new(nodes: [Node; N], levels: [usize; K], max_inter_state: usize) -> Self {
        Self {
            nodes,
            levels,
            max_inter_state,
        }
    }
}
impl<const N: usize, const K: usize> BitCircuitInfo for BitCircuit<N, K> {
    fn info(&self) -> (&[Node], &[usize], usize) {
        (
            self.nodes.as_ref(),
            self.levels.as_ref(),
            self.max_inter_state,
        )
    }
}

pub(crate) struct Node {
    input_index: usize,
    high_index: usize,
    low_index: usize,
}

impl Node {
    pub(crate) const fn new(input_index: usize, high_index: usize, low_index: usize) -> Self {
        Self {
            input_index,
            high_index,
            low_index,
        }
    }
}

pub trait Cmux<BE: Backend> {
    fn cmux<T, F, S>(
        &mut self,
        module: &Module<BE>,
        t: &GLWECiphertext<T>,
        f: &GLWECiphertext<F>,
        s: &GGSWCiphertextPrepared<S, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        T: DataRef,
        F: DataRef,
        S: DataRef;
}

impl<BE: Backend, A: GLWECiphertextToMut> Cmux<BE> for A
where
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
    fn cmux<T, F, S>(
        &mut self,
        module: &Module<BE>,
        t: &GLWECiphertext<T>,
        f: &GLWECiphertext<F>,
        s: &GGSWCiphertextPrepared<S, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        T: DataRef,
        F: DataRef,
        S: DataRef,
    {
        let mut self_mut: GLWECiphertext<&mut [u8]> = self.to_mut();
        self_mut.sub(module, t, f);
        self_mut.external_product_inplace(module, s, scratch);
        self_mut.add_inplace(module, f);
    }
}
