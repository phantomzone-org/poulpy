use core::panic;

use itertools::Itertools;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEExternalProduct, GLWESub, ScratchTakeCore,
    layouts::{GGSWInfos, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, prepared::GGSWPreparedToRef},
};
use poulpy_hal::layouts::{Backend, DataMut, Module, Scratch, ZnxZero};

use crate::tfhe::bdd_arithmetic::UnsignedInteger;

pub trait BitCircuitInfo {
    fn info(&self) -> (&[Node], usize);
}

pub trait GetBitCircuitInfo<T: UnsignedInteger> {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn get_circuit(&self, bit: usize) -> (&[Node], usize);
}

pub(crate) struct BitCircuit<const N: usize> {
    pub(crate) nodes: [Node; N],
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
            let (nodes, max_inter_state) = circuit.get_circuit(i);

            assert!(nodes.len().is_multiple_of(max_inter_state));

            let (mut level, scratch_1) = scratch.take_glwe_slice(max_inter_state * 2, out_i);

            level.iter_mut().for_each(|ct| ct.data_mut().zero());

            // TODO: implement API on GLWE
            level[1]
                .data_mut()
                .encode_coeff_i64(out_i.base2k().into(), 0, 2, 0, 1);

            let mut level_ref = level.iter_mut().collect_vec();
            let (mut prev_level, mut next_level) = level_ref.split_at_mut(max_inter_state);

            let (all_but_last, last) = nodes.split_at(nodes.len() - max_inter_state);

            for nodes_lvl in all_but_last.chunks_exact(max_inter_state) {
                for (j, node) in nodes_lvl.iter().enumerate() {
                    match node {
                        Node::Cmux(in_idx, hi_idx, lo_idx) => {
                            self.cmux(
                                next_level[j],
                                prev_level[*hi_idx],
                                prev_level[*lo_idx],
                                &inputs[*in_idx].to_ref(),
                                scratch_1,
                            );
                        }
                        Node::Copy => self.glwe_copy(next_level[j], prev_level[j]), /* Update BDD circuits to order Cmux -> Copy -> None so that mem swap can be used */
                        Node::None => {}
                    }
                }

                (prev_level, next_level) = (next_level, prev_level);
            }

            // Last chunck of max_inter_state Nodes is always structured as
            // [CMUX, NONE, NONE, ..., NONE]
            match &last[0] {
                Node::Cmux(in_idx, hi_idx, lo_idx) => {
                    self.cmux(
                        out_i,
                        prev_level[*hi_idx],
                        prev_level[*lo_idx],
                        &inputs[*in_idx].to_ref(),
                        scratch_1,
                    );
                }
                _ => {
                    panic!("invalid last node, should be CMUX")
                }
            }
        }

        for out_i in out.iter_mut().skip(circuit.output_size()) {
            out_i.data_mut().zero();
        }
    }
}

impl<const N: usize> BitCircuit<N> {
    pub(crate) const fn new(nodes: [Node; N], max_inter_state: usize) -> Self {
        Self {
            nodes,
            max_inter_state,
        }
    }
}
impl<const N: usize> BitCircuitInfo for BitCircuit<N> {
    fn info(&self) -> (&[Node], usize) {
        (self.nodes.as_ref(), self.max_inter_state)
    }
}

#[derive(Debug)]
pub enum Node {
    Cmux(usize, usize, usize),
    Copy,
    None,
}

pub trait Cmux<BE: Backend>
where
    Self: GLWEExternalProduct<BE> + GLWESub + GLWEAdd,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn cmux_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        self.glwe_external_product_tmp_bytes(res_infos, a_infos, b_infos)
    }

    fn cmux<R, T, F, S>(&self, res: &mut R, t: &T, f: &F, s: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        T: GLWEToRef,
        F: GLWEToRef,
        S: GGSWPreparedToRef<BE>,
    {
        self.glwe_sub(res, t, f);
        self.glwe_external_product_inplace(res, s, scratch);
        self.glwe_add_inplace(res, f);
    }

    fn cmux_inplace<R, A, S>(&self, res: &mut R, a: &A, s: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        S: GGSWPreparedToRef<BE>,
    {
        self.glwe_sub_inplace(res, a);
        self.glwe_external_product_inplace(res, s, scratch);
        self.glwe_add_inplace(res, a);
    }
}

impl<BE: Backend> Cmux<BE> for Module<BE>
where
    Self: GLWEExternalProduct<BE> + GLWESub + GLWEAdd,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}
