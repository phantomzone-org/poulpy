use poulpy_core::{
    GLWEOperations,
    layouts::{
        GLWECiphertext, GLWECiphertextToRef, GLWEInfos, GLWEPlaintext,
        prepared::{GGSWCiphertextPrepared, GGSWCiphertextPreparedToRef},
    },
};
use poulpy_hal::{
    api::{
        ScratchAvailable, TakeVecZnx, TakeVecZnxDft, VecZnxAddInplace, VecZnxBigNormalize, VecZnxCopy, VecZnxDftAllocBytes,
        VecZnxDftApply, VecZnxIdftApplyConsume, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxSub,
        VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

pub trait BitCircuitInfo {
    fn info(&self) -> (&[Node], &[usize]);
}

pub(crate) fn execute<OUT, C: BitCircuitInfo + ?Sized, B: Backend>(
    circuit: &C,
    module: &Module<B>,
    out: &mut GLWECiphertext<OUT>,
    inputs: &[&dyn GGSWCiphertextPreparedToRef<B>],
    terminal_nodes: &[GLWEPlaintext<Vec<u8>>; 2],
    tmp_outs: &mut [GLWECiphertext<Vec<u8>>],
    scratch: &mut Scratch<B>,
) where
    OUT: DataMut,
    Module<B>: VecZnxSub
        + VecZnxCopy
        + VecZnxNegateInplace
        + VecZnxDftAllocBytes
        + VecZnxAddInplace
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<B>
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalize<B>,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
{
    let (nodes, levels) = circuit.info();

    // Note: tmp_outs does not store the expected first two terminal nodes. Thus, j^th node's tmp output
    // is stored at tmp_outs[j-2]

    // start at level 1
    for i in 0..levels.len() - 2 {
        let start = levels[i];
        let end = levels[i + 1];
        let (tmp_outs_ref, tmp_outs_mut) = tmp_outs.split_at_mut(start - 2);
        for j in start..end {
            let node = &nodes[j];
            pointer_hurdle(
                module,
                node,
                &mut tmp_outs_mut[j - start],
                &tmp_outs_ref,
                inputs,
                terminal_nodes,
                scratch,
            );
        }
    }

    // handle last output
    // there's always only 1 node at last level
    let node = nodes.last().unwrap();
    pointer_hurdle(
        module,
        node,
        out,
        &tmp_outs,
        inputs,
        terminal_nodes,
        scratch,
    );
}

pub(crate) struct BitCircuit<const N: usize, const K: usize> {
    pub(crate) nodes: [Node; N],
    pub(crate) levels: [usize; K],
}

pub struct Circuit<C: BitCircuitInfo, const N: usize>(pub(crate) [C; N]);

impl<const N: usize, const K: usize> BitCircuit<N, K> {
    pub(crate) const fn new(nodes: [Node; N], levels: [usize; K]) -> Self {
        Self { nodes, levels }
    }
}
impl<const N: usize, const K: usize> BitCircuitInfo for BitCircuit<N, K> {
    fn info(&self) -> (&[Node], &[usize]) {
        (self.nodes.as_ref(), self.levels.as_ref())
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

fn cmux<G1, G2, G3, G4, B: Backend>(
    module: &Module<B>,
    o: &mut GLWECiphertext<G4>,
    t: &G1,
    f: &G2,
    s: &GGSWCiphertextPrepared<G3, B>,
    scratch: &mut Scratch<B>,
) where
    G1: GLWECiphertextToRef + GLWEInfos,
    G2: GLWECiphertextToRef + GLWEInfos,
    G3: DataRef,
    G4: DataMut,
    Module<B>: VecZnxSub
        + VecZnxCopy
        + VecZnxNegateInplace
        + VecZnxDftAllocBytes
        + VecZnxAddInplace
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<B>
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalize<B>,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
{
    o.sub(module, t, f);
    o.external_product_inplace(module, s, scratch);
    o.add_inplace(module, f);
}

fn pointer_hurdle<G1, G2, G4, B: Backend>(
    module: &Module<B>,
    node: &Node,
    out: &mut GLWECiphertext<G1>,
    tmp_outs: &[GLWECiphertext<G2>],
    inputs: &[&dyn GGSWCiphertextPreparedToRef<B>],
    terminal_nodes: &[GLWEPlaintext<G4>; 2],
    scratch: &mut Scratch<B>,
) where
    G1: DataMut,
    G2: DataRef,
    G4: DataRef,
    Module<B>: VecZnxSub
        + VecZnxCopy
        + VecZnxNegateInplace
        + VecZnxDftAllocBytes
        + VecZnxAddInplace
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxDftApply<B>
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalize<B>,
    Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx,
{
    let if_true: usize = node.high_index;
    let if_false: usize = node.low_index;

    match (if_true < 2, if_false < 2) {
        (true, true) => {
            // both children nodes are terminal nodes
            cmux(
                module,
                out,
                &terminal_nodes[if_true],
                &terminal_nodes[if_false],
                &inputs[node.input_index].to_ref(),
                scratch,
            );
        }
        (true, false) => {
            cmux(
                module,
                out,
                &terminal_nodes[if_true],
                &tmp_outs[if_false - 2],
                &inputs[node.input_index].to_ref(),
                scratch,
            );
        }
        (false, true) => {
            cmux(
                module,
                out,
                &tmp_outs[if_true - 2],
                &terminal_nodes[if_false],
                &inputs[node.input_index].to_ref(),
                scratch,
            );
        }
        _ => {
            cmux(
                module,
                out,
                &tmp_outs[if_true - 2],
                &tmp_outs[if_false - 2],
                &inputs[node.input_index].to_ref(),
                scratch,
            );
        }
    }
}

pub(crate) struct Helper {
    pub(crate) tmp_glwes: Vec<GLWECiphertext<Vec<u8>>>,
    pub(crate) tmp_glwes_bounds: Vec<usize>,
    pub(crate) single_bit_output: bool,
}

impl Helper {
    pub(crate) fn split_tmp_glwes_mut(&mut self) -> Vec<&mut [GLWECiphertext<Vec<u8>>]> {
        let mut chunks = Vec::with_capacity(self.tmp_glwes_bounds.len());
        let mut slice = self.tmp_glwes.as_mut_slice();
        let mut last_end = 0;

        for &end in &self.tmp_glwes_bounds {
            let (left, right) = slice.split_at_mut(end - last_end);
            chunks.push(left);
            slice = right;
            last_end = end;
        }

        chunks
    }
}
