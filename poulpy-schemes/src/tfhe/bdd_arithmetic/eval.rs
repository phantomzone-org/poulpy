use core::panic;

use itertools::Itertools;
use poulpy_core::{
    GLWECopy, GLWEExternalProductInternal, GLWENormalize, GLWESub, ScratchTakeCore,
    layouts::{GGSWInfos, GGSWPrepared, GLWE, GLWEInfos, GLWEToMut, GLWEToRef, LWEInfos, prepared::GGSWPreparedToRef},
};
use poulpy_hal::{
    api::{ScratchTakeBasic, VecZnxBigAddSmallInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftBytesOf},
    layouts::{Backend, DataMut, Module, Scratch, VecZnxBig, ZnxZero},
};

use crate::tfhe::bdd_arithmetic::{GetGGSWBit, UnsignedInteger};

pub trait BitCircuitInfo {
    fn info(&self) -> (&[Node], usize);
}

pub trait GetBitCircuitInfo<T: UnsignedInteger> {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn get_circuit(&self, bit: usize) -> (&[Node], usize);
}

pub struct BitCircuit<const N: usize> {
    pub nodes: [Node; N],
    pub max_inter_state: usize,
}

pub trait BitCircuitFamily {
    const INPUT_BITS: usize;
    const OUTPUT_BITS: usize;
}

pub struct Circuit<C: BitCircuitInfo, const N: usize>(pub [C; N]);

impl<C, T: UnsignedInteger, const N: usize> GetBitCircuitInfo<T> for Circuit<C, N>
where
    C: BitCircuitInfo + BitCircuitFamily,
{
    fn input_size(&self) -> usize {
        C::INPUT_BITS
    }
    fn output_size(&self) -> usize {
        C::OUTPUT_BITS
    }
    fn get_circuit(&self, bit: usize) -> (&[Node], usize) {
        self.0[bit].info()
    }
}

pub trait ExecuteBDDCircuit<T: UnsignedInteger, BE: Backend> {
    fn execute_bdd_circuit<C, G, O>(&self, out: &mut [GLWE<O>], inputs: &G, circuit: &C, scratch: &mut Scratch<BE>)
    where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo<T>,
        O: DataMut;
}

pub trait BitSize {
    fn bit_size(&self) -> usize;
}

impl<T: UnsignedInteger, BE: Backend> ExecuteBDDCircuit<T, BE> for Module<BE>
where
    Self: Cmux<BE> + GLWECopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn execute_bdd_circuit<C, G, O>(&self, out: &mut [GLWE<O>], inputs: &G, circuit: &C, scratch: &mut Scratch<BE>)
    where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo<T>,
        O: DataMut,
    {
        #[cfg(debug_assertions)]
        {
            assert!(
                inputs.bit_size() >= circuit.input_size(),
                "inputs.bit_size(): {} < circuit.input_size():{}",
                inputs.bit_size(),
                circuit.input_size()
            );
            assert!(
                out.len() >= circuit.output_size(),
                "out.len(): {} < circuit.output_size(): {}",
                out.len(),
                circuit.output_size()
            );
        }

        for (i, out_i) in out.iter_mut().enumerate().take(circuit.output_size()) {
            let (nodes, max_inter_state) = circuit.get_circuit(i);

            if max_inter_state == 0 {
                out_i.data_mut().zero();
            } else {
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
                                    &inputs.get_bit(*in_idx),
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
                            &inputs.get_bit(*in_idx),
                            scratch_1,
                        );
                    }
                    _ => {
                        panic!("invalid last node, should be CMUX")
                    }
                }
            }
        }

        for out_i in out.iter_mut().skip(circuit.output_size()) {
            out_i.data_mut().zero();
        }
    }
}

impl<const N: usize> BitCircuit<N> {
    pub const fn new(nodes: [Node; N], max_inter_state: usize) -> Self {
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
    Self: Sized
        + GLWEExternalProductInternal<BE>
        + GLWESub
        + VecZnxBigAddSmallInplace<BE>
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    fn cmux_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let res_dft: usize = self.bytes_of_vec_znx_dft((b_infos.rank() + 1).into(), b_infos.size());
        res_dft
            + self
                .glwe_external_product_internal_tmp_bytes(res_infos, a_infos, b_infos)
                .max(self.vec_znx_big_normalize_tmp_bytes())
    }

    fn cmux<R, T, F, S>(&self, res: &mut R, t: &T, f: &F, s: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        T: GLWEToRef,
        F: GLWEToRef,
        S: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let s: &GGSWPrepared<&[u8], BE> = &s.to_ref();
        let f: GLWE<&[u8]> = f.to_ref();

        let res_base2k: usize = res.base2k().into();
        let ggsw_base2k: usize = s.base2k().into();

        self.glwe_sub(res, t, &f);
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), s.size()); // Todo optimise
        let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_external_product_internal(res_dft, res, s, scratch_1);
        for j in 0..(res.rank() + 1).into() {
            self.vec_znx_big_add_small_inplace(&mut res_big, j, f.data(), j);
            self.vec_znx_big_normalize(
                res_base2k,
                res.data_mut(),
                j,
                ggsw_base2k,
                &res_big,
                j,
                scratch_1,
            );
        }
    }

    fn cmux_inplace<R, A, S>(&self, res: &mut R, a: &A, s: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        S: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let s: &GGSWPrepared<&[u8], BE> = &s.to_ref();
        let a: GLWE<&[u8]> = a.to_ref();
        let res_base2k: usize = res.base2k().into();
        let ggsw_base2k: usize = s.base2k().into();
        self.glwe_sub_inplace(res, &a);
        let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (res.rank() + 1).into(), s.size()); // Todo optimise
        let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_external_product_internal(res_dft, res, s, scratch_1);
        for j in 0..(res.rank() + 1).into() {
            self.vec_znx_big_add_small_inplace(&mut res_big, j, a.data(), j);
            self.vec_znx_big_normalize(
                res_base2k,
                res.data_mut(),
                j,
                ggsw_base2k,
                &res_big,
                j,
                scratch_1,
            );
        }
    }
}

impl<BE: Backend> Cmux<BE> for Module<BE>
where
    Self: Sized
        + GLWEExternalProductInternal<BE>
        + GLWESub
        + VecZnxBigAddSmallInplace<BE>
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
}
