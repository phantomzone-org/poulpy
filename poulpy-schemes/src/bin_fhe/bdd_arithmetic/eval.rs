use core::panic;
use std::thread;

use itertools::Itertools;
use poulpy_core::{
    GLWECopy, GLWEExternalProductInternal, GLWENormalize, GLWESub, ScratchTakeCore,
    layouts::{
        GGSWInfos, GGSWPrepared, GLWE, GLWEInfos, GLWELayout, GLWEToMut, GLWEToRef, LWEInfos, prepared::GGSWPreparedToRef,
    },
};
use poulpy_hal::{
    api::{
        ScratchAvailable, ScratchTakeBasic, VecZnxBigAddSmall, VecZnxBigAddSmallInplace, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallA, VecZnxDftBytesOf,
    },
    layouts::{Backend, DataMut, Module, Scratch, VecZnxBig, ZnxInfos, ZnxZero},
};

use crate::bin_fhe::bdd_arithmetic::GetGGSWBit;

pub trait BitCircuitInfo: Sync {
    fn info(&self) -> (&[Node], usize);
}

pub trait GetBitCircuitInfo: Sync {
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

impl<C, const N: usize> GetBitCircuitInfo for Circuit<C, N>
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

pub trait ExecuteBDDCircuit<BE: Backend> {
    fn execute_bdd_circuit_tmp_bytes<R, G>(&self, res_infos: &R, state_size: usize, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos;

    fn execute_bdd_circuit<C, G, O>(&self, out: &mut [GLWE<O>], inputs: &G, circuit: &C, scratch: &mut Scratch<BE>)
    where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: DataMut,
    {
        self.execute_bdd_circuit_multi_thread(1, out, inputs, circuit, scratch);
    }

    fn execute_bdd_circuit_multi_thread<C, G, O>(
        &self,
        threads: usize,
        out: &mut [GLWE<O>],
        inputs: &G,
        circuit: &C,
        scratch: &mut Scratch<BE>,
    ) where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: DataMut;
}

pub trait BitSize {
    fn bit_size(&self) -> usize;
}

impl<BE: Backend> ExecuteBDDCircuit<BE> for Module<BE>
where
    Self: Cmux<BE> + GLWECopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn execute_bdd_circuit_tmp_bytes<R, G>(&self, res_infos: &R, state_size: usize, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
    {
        2 * state_size * GLWE::bytes_of_from_infos(res_infos) + self.cmux_tmp_bytes(res_infos, res_infos, ggsw_infos)
    }

    fn execute_bdd_circuit_multi_thread<C, G, O>(
        &self,
        threads: usize,
        out: &mut [GLWE<O>],
        inputs: &G,
        circuit: &C,
        scratch: &mut Scratch<BE>,
    ) where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
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

        let mut max_state_size = 0;
        for i in 0..circuit.output_size() {
            let (_, state_size) = circuit.get_circuit(i);
            max_state_size = max_state_size.max(state_size)
        }

        let scratch_thread_size: usize = self.execute_bdd_circuit_tmp_bytes(&out[0], max_state_size, &inputs.get_bit(0));

        assert!(
            scratch.available() >= threads * scratch_thread_size,
            "scratch.available(): {} < threads:{threads} * scratch_thread_size: {scratch_thread_size}",
            scratch.available()
        );

        let (mut scratches, _) = scratch.split_mut(threads, scratch_thread_size);

        let chunk_size: usize = circuit.output_size().div_ceil(threads);

        thread::scope(|scope| {
            for (thread_idx, (scratch_thread, out_chunk)) in scratches
                .iter_mut()
                .zip(out[..circuit.output_size()].chunks_mut(chunk_size))
                .enumerate()
            {
                // Capture chunk + thread scratch by move
                scope.spawn(move || {
                    for (idx, out_i) in out_chunk.iter_mut().enumerate() {
                        let (nodes, state_size) = circuit.get_circuit(thread_idx * chunk_size + idx);

                        if state_size == 0 {
                            out_i.data_mut().zero();
                        } else {
                            eval_level(self, out_i, inputs, nodes, state_size, *scratch_thread);
                        }
                    }
                });
            }
        });

        for out_i in out.iter_mut().skip(circuit.output_size()) {
            out_i.data_mut().zero();
        }
    }
}

fn eval_level<M, R, G, BE: Backend>(
    module: &M,
    res: &mut R,
    inputs: &G,
    nodes: &[Node],
    state_size: usize,
    scratch: &mut Scratch<BE>,
) where
    M: Cmux<BE> + GLWECopy,
    R: GLWEToMut,
    G: GetGGSWBit<BE> + BitSize,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    assert!(nodes.len().is_multiple_of(state_size));
    let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

    let (mut level, scratch_1) = scratch.take_glwe_slice(state_size * 2, res);

    level.iter_mut().for_each(|ct| ct.data_mut().zero());

    // TODO: implement API on GLWE
    level[1].data_mut().encode_coeff_i64(res.base2k().into(), 0, 2, 0, 1);

    let mut level_ref: Vec<&mut GLWE<&mut [u8]>> = level.iter_mut().collect_vec();
    let (mut prev_level, mut next_level) = level_ref.split_at_mut(state_size);

    let (all_but_last, last) = nodes.split_at(nodes.len() - state_size);

    for nodes_lvl in all_but_last.chunks_exact(state_size) {
        for (j, node) in nodes_lvl.iter().enumerate() {
            match node {
                Node::Cmux(in_idx, hi_idx, lo_idx) => {
                    module.cmux(
                        next_level[j],
                        prev_level[*hi_idx],
                        prev_level[*lo_idx],
                        &inputs.get_bit(*in_idx),
                        scratch_1,
                    );
                }
                Node::Copy => module.glwe_copy(next_level[j], prev_level[j]), /* Update BDD circuits to order Cmux -> Copy -> None so that mem swap can be used */
                Node::None => {}
            }
        }

        (prev_level, next_level) = (next_level, prev_level);
    }

    // Last chunck of max_inter_state Nodes is always structured as
    // [CMUX, NONE, NONE, ..., NONE]
    match &last[0] {
        Node::Cmux(in_idx, hi_idx, lo_idx) => {
            module.cmux(
                res,
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

impl<const N: usize> BitCircuit<N> {
    pub const fn new(nodes: [Node; N], max_inter_state: usize) -> Self {
        Self { nodes, max_inter_state }
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

impl<BE: Backend> Cswap<BE> for Module<BE> where
    Self: Sized
        + GLWEExternalProductInternal<BE>
        + GLWESub
        + VecZnxBigAddSmallInplace<BE>
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWENormalize<BE>
        + VecZnxBigAddSmall<BE>
        + VecZnxBigSubSmallA<BE>
        + VecZnxBigBytesOf
{
}

pub trait Cswap<BE: Backend>
where
    Self: Sized
        + GLWEExternalProductInternal<BE>
        + GLWESub
        + GLWECopy
        + VecZnxBigAddSmallInplace<BE>
        + GLWENormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + GLWENormalize<BE>
        + VecZnxBigAddSmall<BE>
        + VecZnxBigSubSmallA<BE>
        + VecZnxBigBytesOf,
{
    fn cswap_tmp_bytes<R, A, S>(&self, res_a_infos: &R, res_b_infos: &A, s_infos: &S) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        S: GGSWInfos,
    {
        let res_dft: usize = self.bytes_of_vec_znx_dft((s_infos.rank() + 1).into(), s_infos.size());
        let mut tot = res_dft
            + (self.glwe_external_product_internal_tmp_bytes(res_a_infos, res_b_infos, s_infos)
                + GLWE::bytes_of_from_infos(&GLWELayout {
                    n: s_infos.n(),
                    base2k: s_infos.base2k(),
                    k: res_a_infos.k().max(res_b_infos.k()),
                    rank: s_infos.rank(),
                }))
            .max(self.vec_znx_big_normalize_tmp_bytes());

        if res_a_infos.base2k() != s_infos.base2k() {
            tot += GLWE::bytes_of_from_infos(&GLWELayout {
                n: res_a_infos.n(),
                base2k: s_infos.base2k(),
                k: res_a_infos.k(),
                rank: res_a_infos.rank(),
            });

            tot += GLWE::bytes_of_from_infos(&GLWELayout {
                n: res_b_infos.n(),
                base2k: s_infos.base2k(),
                k: res_b_infos.k(),
                rank: res_b_infos.rank(),
            });
        }

        tot += self.bytes_of_vec_znx_big(1, s_infos.size());

        tot
    }

    fn cswap<A, B, S>(&self, res_a: &mut A, res_b: &mut B, s: &S, scratch: &mut Scratch<BE>)
    where
        A: GLWEToMut,
        B: GLWEToMut,
        S: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res_a: &mut GLWE<&mut [u8]> = &mut res_a.to_mut();
        let res_b: &mut GLWE<&mut [u8]> = &mut res_b.to_mut();
        let s: &GGSWPrepared<&[u8], BE> = &s.to_ref();
        assert_eq!(res_a.base2k(), res_b.base2k());

        let res_base2k: usize = res_a.base2k().as_usize();
        let s_base2k: usize = s.base2k().as_usize();

        if res_base2k == s_base2k {
            let res_big: VecZnxBig<&mut [u8], BE>;
            let (res_dft, scratch_1) = scratch.take_vec_znx_dft(self, (s.rank() + 1).into(), s.size()); // Todo optimise
            {
                // Temporary value storing a - b
                let tmp_c_infos: GLWELayout = GLWELayout {
                    n: s.n(),
                    base2k: s.base2k(),
                    k: res_a.k().max(res_b.k()),
                    rank: s.rank(),
                };
                let (mut tmp_c, scratch_2) = scratch_1.take_glwe(&tmp_c_infos);
                self.glwe_sub(&mut tmp_c, res_b, res_a);
                res_big = self.glwe_external_product_internal(res_dft, &tmp_c, s, scratch_2);
            }

            // Single column res_big to store temporary value before normalization
            let (mut res_big_tmp, scratch_2) = scratch_1.take_vec_znx_big::<_, BE>(self, 1, res_big.size());

            // res_a = (b-a) * bit + a
            for j in 0..(res_a.rank() + 1).into() {
                self.vec_znx_big_add_small(&mut res_big_tmp, 0, &res_big, j, res_a.data(), j);
                self.vec_znx_big_normalize(res_a.data_mut(), res_base2k, 0, j, &res_big_tmp, s_base2k, 0, scratch_2);
            }

            // res_b = a - (a - b) * bit = (b - a) * bit + a
            for j in 0..(res_b.rank() + 1).into() {
                self.vec_znx_big_sub_small_a(&mut res_big_tmp, 0, res_b.data(), j, &res_big, j);
                self.vec_znx_big_normalize(res_b.data_mut(), res_base2k, 0, j, &res_big_tmp, s_base2k, 0, scratch_2);
            }
        } else {
            let (mut tmp_a, scratch_1) = scratch.take_glwe(&GLWELayout {
                n: res_a.n(),
                base2k: s.base2k(),
                k: res_a.k(),
                rank: res_a.rank(),
            });

            let (mut tmp_b, scratch_2) = scratch_1.take_glwe(&GLWELayout {
                n: res_b.n(),
                base2k: s.base2k(),
                k: res_b.k(),
                rank: res_b.rank(),
            });

            self.glwe_normalize(&mut tmp_a, res_a, scratch_2);
            self.glwe_normalize(&mut tmp_b, res_b, scratch_2);

            let res_big: VecZnxBig<&mut [u8], BE>;
            let (res_dft, scratch_3) = scratch_2.take_vec_znx_dft(self, (s.rank() + 1).into(), s.size()); // Todo optimise
            {
                // Temporary value storing a - b
                let tmp_c_infos: GLWELayout = GLWELayout {
                    n: s.n(),
                    base2k: s.base2k(),
                    k: res_a.k().max(res_b.k()),
                    rank: s.rank(),
                };
                let (mut tmp_c, scratch_4) = scratch_3.take_glwe(&tmp_c_infos);
                self.glwe_sub(&mut tmp_c, res_b, res_a);
                res_big = self.glwe_external_product_internal(res_dft, &tmp_c, s, scratch_4);
            }

            // Single column res_big to store temporary value before normalization
            let (mut res_big_tmp, scratch_4) = scratch_3.take_vec_znx_big::<_, BE>(self, 1, res_big.size());

            // res_a = (b-a) * bit + a
            for j in 0..(res_a.rank() + 1).into() {
                self.vec_znx_big_add_small(&mut res_big_tmp, 0, &res_big, j, tmp_a.data(), j);
                self.vec_znx_big_normalize(res_a.data_mut(), res_base2k, 0, j, &res_big_tmp, s_base2k, 0, scratch_4);
            }

            // res_b = a - (a - b) * bit = (b - a) * bit + a
            for j in 0..(res_b.rank() + 1).into() {
                self.vec_znx_big_sub_small_a(&mut res_big_tmp, 0, tmp_b.data(), j, &res_big, j);
                self.vec_znx_big_normalize(res_b.data_mut(), res_base2k, 0, j, &res_big_tmp, s_base2k, 0, scratch_4);
            }
        }
    }
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
    fn cmux_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, selector_infos: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let res_dft: usize = self.bytes_of_vec_znx_dft((selector_infos.rank() + 1).into(), selector_infos.size());
        res_dft
            + self
                .glwe_external_product_internal_tmp_bytes(res_infos, a_infos, selector_infos)
                .max(self.vec_znx_big_normalize_tmp_bytes())
    }

    // res = (t - f) * s + f
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
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, j, &res_big, ggsw_base2k, j, scratch_1);
        }
    }

    // res = (a - res) * s + res
    fn cmux_inplace_neg<R, A, S>(&self, res: &mut R, a: &A, s: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        S: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let s: &GGSWPrepared<&[u8], BE> = &s.to_ref();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.base2k(), a.base2k());

        let res_base2k: usize = res.base2k().into();
        let ggsw_base2k: usize = s.base2k().into();
        let (mut tmp, scratch_1) = scratch.take_glwe(&GLWELayout {
            n: s.n(),
            base2k: res.base2k(),
            k: res.k().max(a.k()),
            rank: res.rank(),
        });
        self.glwe_sub(&mut tmp, a, res);
        let (res_dft, scratch_2) = scratch_1.take_vec_znx_dft(self, (res.rank() + 1).into(), s.size()); // Todo optimise
        let mut res_big: VecZnxBig<&mut [u8], BE> = self.glwe_external_product_internal(res_dft, &tmp, s, scratch_2);
        for j in 0..(res.rank() + 1).into() {
            self.vec_znx_big_add_small_inplace(&mut res_big, j, res.data(), j);
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, j, &res_big, ggsw_base2k, j, scratch_2);
        }
    }

    // res = (res - a) * s + a
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
            self.vec_znx_big_normalize(res.data_mut(), res_base2k, 0, j, &res_big, ggsw_base2k, j, scratch_1);
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
