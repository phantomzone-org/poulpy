use itertools::Itertools;
use poulpy_core::{
    GLWECopy, ScratchTakeCore,
    layouts::{GGSWInfos, GGSWPrepared, GLWE, GLWEInfos, GLWEToMut, GLWEToRef},
};
use poulpy_hal::layouts::{Backend, Module, Scratch};

use crate::bin_fhe::bdd_arithmetic::{Cmux, Cswap, GetGGSWBit};

pub struct GLWEBlindRetriever {
    accumulators: Vec<Accumulator>,
    counter: usize,
}

impl GLWEBlindRetriever {
    pub fn alloc<A>(infos: &A, size: usize) -> Self
    where
        A: GLWEInfos,
    {
        let bit_size: usize = (u32::BITS - (size as u32 - 1).leading_zeros()) as usize;
        Self {
            accumulators: (0..bit_size)
                .map(|_| Accumulator::alloc(infos))
                .collect_vec(),
            counter: 0,
        }
    }

    pub fn retrieve_tmp_bytes<M, R, S, BE: Backend>(module: &M, res: &R, selector: &S) -> usize
    where
        M: Cmux<BE>,
        R: GLWEInfos,
        S: GGSWInfos,
    {
        module.cmux_tmp_bytes(res, res, selector)
    }

    pub fn retrieve<M, R, A, S, BE: Backend>(
        &mut self,
        module: &M,
        res: &mut R,
        data: &[A],
        selector: &S,
        offset: usize,
        scratch: &mut Scratch<BE>,
    ) where
        M: GLWECopy + Cmux<BE>,
        R: GLWEToMut,
        A: GLWEToRef,
        S: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.reset();
        for ct in data {
            self.add(module, ct, selector, offset, scratch);
        }
        self.flush(module, res, selector, offset, scratch);
    }

    pub fn add<A, S, M, BE: Backend>(&mut self, module: &M, a: &A, selector: &S, offset: usize, scratch: &mut Scratch<BE>)
    where
        A: GLWEToRef,
        S: GetGGSWBit<BE>,
        M: GLWECopy + Cmux<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        assert!(
            (self.counter as u32) < 1 << self.accumulators.len(),
            "Accumulating limit of {} reached",
            1 << self.accumulators.len()
        );

        add_core(
            module,
            a,
            &mut self.accumulators,
            0,
            selector,
            offset,
            scratch,
        );
        self.counter += 1;
    }

    pub fn flush<R, M, S, BE: Backend>(&mut self, module: &M, res: &mut R, selector: &S, offset: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        S: GetGGSWBit<BE>,
        M: GLWECopy + Cmux<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        for i in 0..self.accumulators.len() - 1 {
            let (acc_prev, acc_next) = self.accumulators.split_at_mut(i + 1);
            if acc_prev[i].num != 0 {
                add_core(
                    module,
                    &acc_prev[i].data,
                    acc_next,
                    i + 1,
                    selector,
                    offset,
                    scratch,
                );
                acc_prev[0].num = 0
            }
        }
        module.glwe_copy(res, &self.accumulators.last().unwrap().data);
        self.reset()
    }

    fn reset(&mut self) {
        for acc in self.accumulators.iter_mut() {
            acc.num = 0;
        }
        self.counter = 0;
    }
}

struct Accumulator {
    data: GLWE<Vec<u8>>,
    num: usize, // Number of accumulated values
}

impl Accumulator {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self {
            data: GLWE::alloc_from_infos(infos),
            num: 0,
        }
    }
}

fn add_core<A, S, M, BE: Backend>(
    module: &M,
    a: &A,
    accumulators: &mut [Accumulator],
    i: usize,
    selector: &S,
    offset: usize,
    scratch: &mut Scratch<BE>,
) where
    A: GLWEToRef,
    S: GetGGSWBit<BE>,
    M: GLWECopy + Cmux<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    // Isolate the first accumulator
    let (acc_prev, acc_next) = accumulators.split_at_mut(1);

    match acc_prev[0].num {
        0 => {
            module.glwe_copy(&mut acc_prev[0].data, a);
            acc_prev[0].num = 1;
        }
        1 => {
            module.cmux_inplace_neg(
                &mut acc_prev[0].data,
                a,
                &selector.get_bit(i + offset),
                scratch,
            );

            if !acc_next.is_empty() {
                add_core(
                    module,
                    &acc_prev[0].data,
                    acc_next,
                    i + 1,
                    selector,
                    offset,
                    scratch,
                );
            }

            acc_prev[0].num = 0
        }
        _ => {
            panic!("something went wrong")
        }
    }
}

impl<BE: Backend> GLWEBlindRetrieval<BE> for Module<BE> where Self: GLWECopy + Cmux<BE> + Cswap<BE> {}

pub trait GLWEBlindRetrieval<BE: Backend>
where
    Self: GLWECopy + Cmux<BE> + Cswap<BE>,
{
    fn glwe_blind_retrieval_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cswap_tmp_bytes(res_infos, res_infos, k_infos)
    }

    fn glwe_blind_retrieval_statefull<R, K>(
        &self,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        for i in 0..bit_mask {
            let t: usize = 1 << (bit_mask - i - 1);
            let bit: &GGSWPrepared<&[u8], BE> = &bits.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal
            for j in 0..t {
                if j + t < res.len() {
                    let (lo, hi) = res.split_at_mut(j + t);
                    self.cswap(&mut lo[j], &mut hi[0], bit, scratch);
                }
            }
        }
    }

    fn glwe_blind_retrieval_statefull_rev<R, K>(
        &self,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        K: GetGGSWBit<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        for i in (0..bit_mask).rev() {
            let t: usize = 1 << (bit_mask - i - 1);
            let bit: &GGSWPrepared<&[u8], BE> = &bits.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal
            for j in 0..t {
                if j < res.len() && j + t < res.len() {
                    let (lo, hi) = res.split_at_mut(j + t);
                    self.cswap(&mut lo[j], &mut hi[0], bit, scratch);
                }
            }
        }
    }
}
