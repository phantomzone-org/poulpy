use itertools::Itertools;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEExternalProduct, GLWESub, ScratchTakeCore,
    layouts::{
        GLWE, LWEInfos,
        prepared::{GGSWPrepared, GGSWPreparedToRef},
    },
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch, ZnxZero};

use crate::tfhe::bdd_arithmetic::UnsignedInteger;

pub trait BitCircuitInfo {
    fn info(&self) -> (&[Node], &[usize], usize);
}

pub trait GetBitCircuitInfo<T: UnsignedInteger> {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn get_circuit(&self, bit: usize) -> (&[Node], &[usize], usize);
}

pub(crate) struct BitCircuit<const N: usize, const K: usize> {
    pub(crate) nodes: [Node; N],
    pub(crate) levels: [usize; K],
    pub(crate) max_inter_state: usize,
}

pub struct Circuit<C: BitCircuitInfo, const N: usize>(pub [C; N]);

pub trait ExecuteBDDCircuit<T: UnsignedInteger, BE: Backend> {
    fn execute_bdd_circuit<C, O>(
        &self,
        out: &mut [GLWE<O>],
        inputs: &[&dyn GGSWPreparedToRef<BE>],
        circuit: &C,
        scratch: &mut Scratch<BE>,
    ) where
        C: GetBitCircuitInfo<T>,
        O: DataMut;
}

impl<T: UnsignedInteger, BE: Backend> ExecuteBDDCircuit<T, BE> for Module<BE>
where
    Self: Cmux<BE> + GLWECopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn execute_bdd_circuit<C, O>(
        &self,
        out: &mut [GLWE<O>],
        inputs: &[&dyn GGSWPreparedToRef<BE>],
        circuit: &C,
        scratch: &mut Scratch<BE>,
    ) where
        C: GetBitCircuitInfo<T>,
        O: DataMut,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(inputs.len(), circuit.input_size());
            assert!(out.len() >= circuit.output_size());
        }

        for (i, out_i) in out.iter_mut().enumerate().take(circuit.output_size()) {
            let (nodes, levels, max_inter_state) = circuit.get_circuit(i);

            let (mut level, scratch_1) = scratch.take_glwe_slice(max_inter_state * 2, out_i);

            level.iter_mut().for_each(|ct| ct.data_mut().zero());

            // TODO: implement API on GLWE
            level[1]
                .data_mut()
                .encode_coeff_i64(out_i.base2k().into(), 0, 2, 0, 1);

            let mut level_ref = level.iter_mut().collect_vec();
            let (mut prev_level, mut next_level) = level_ref.split_at_mut(max_inter_state);

            for i in 0..levels.len() - 1 {
                let start: usize = levels[i];
                let end: usize = levels[i + 1];

                let nodes_lvl: &[Node] = &nodes[start..end];

                for (j, node) in nodes_lvl.iter().enumerate() {
                    if node.low_index == node.high_index {
                        self.glwe_copy(next_level[j], prev_level[node.low_index]);
                    } else {
                        self.cmux(
                            next_level[j],
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
            self.cmux(
                out_i,
                prev_level[node.high_index],
                prev_level[node.low_index],
                &inputs[node.input_index].to_ref(),
                scratch_1,
            );
        }

        for out_i in out.iter_mut().skip(circuit.output_size()) {
            out_i.data_mut().zero();
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

#[derive(Debug)]
pub struct Node {
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
    fn cmux<O, T, F, S>(&self, out: &mut GLWE<O>, t: &GLWE<T>, f: &GLWE<F>, s: &GGSWPrepared<S, BE>, scratch: &mut Scratch<BE>)
    where
        O: DataMut,
        T: DataRef,
        F: DataRef,
        S: DataRef;
}

impl<BE: Backend> Cmux<BE> for Module<BE>
where
    Module<BE>: GLWEExternalProduct<BE> + GLWESub + GLWEAdd,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn cmux<O, T, F, S>(&self, out: &mut GLWE<O>, t: &GLWE<T>, f: &GLWE<F>, s: &GGSWPrepared<S, BE>, scratch: &mut Scratch<BE>)
    where
        O: DataMut,
        T: DataRef,
        F: DataRef,
        S: DataRef,
    {
        // let mut out: GLWECiphertext<&mut [u8]> = out.to_mut();
        self.glwe_sub(out, t, f);
        self.glwe_external_product_inplace(out, s, scratch);
        self.glwe_add_inplace(out, f);
    }
}
