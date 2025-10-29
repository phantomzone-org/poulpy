use std::marker::PhantomData;

use poulpy_core::{GLWECopy, GLWEPacking, ScratchTakeCore, layouts::GGSWPrepared};
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::tfhe::{
    bdd_arithmetic::{
        BDDKeyPrepared, BitSize, ExecuteBDDCircuit, FheUint, FheUintPrepared, GetBitCircuitInfo, GetGGSWBit, UnsignedInteger,
        circuits,
    },
    blind_rotation::BlindRotationAlgo,
};

impl<T: UnsignedInteger, BE: Backend> ExecuteBDDCircuit2WTo1W<T, BE> for Module<BE> where
    Self: Sized + ExecuteBDDCircuit<T, BE> + GLWEPacking<BE> + GLWECopy
{
}

pub trait ExecuteBDDCircuit2WTo1W<T: UnsignedInteger, BE: Backend>
where
    Self: Sized + ModuleLogN + ExecuteBDDCircuit<T, BE> + GLWEPacking<BE> + GLWECopy,
{
    /// Operations Z x Z -> Z
    fn execute_bdd_circuit_2w_to_1w<R, C, A, B, DK, BRA>(
        &self,
        out: &mut FheUint<R, T>,
        circuit: &C,
        a: &FheUintPrepared<A, T, BE>,
        b: &FheUintPrepared<B, T, BE>,
        key: &BDDKeyPrepared<DK, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        BRA: BlindRotationAlgo,
        DK: DataRef,
        C: GetBitCircuitInfo<T>,
        R: DataMut,
        A: DataRef,
        B: DataRef,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // Collects inputs into a single array
        let inputs: Vec<&dyn GetGGSWBit<BE>> = [a as &dyn GetGGSWBit<BE>, b as &dyn GetGGSWBit<BE>].to_vec();
        let helper: FheUintHelper<'_, T, BE> = FheUintHelper {
            data: inputs,
            _phantom: PhantomData,
        };

        let (mut out_bits, scratch_1) = scratch.take_glwe_slice(T::BITS as usize, out);

        // Evaluates out[i] = circuit[i](a, b)
        self.execute_bdd_circuit(&mut out_bits, &helper, circuit, scratch_1);

        // Repacks the bits
        out.pack(self, out_bits, key, scratch_1);
    }
}

struct FheUintHelper<'a, T: UnsignedInteger, BE: Backend> {
    data: Vec<&'a dyn GetGGSWBit<BE>>,
    _phantom: PhantomData<T>,
}

impl<'a, T: UnsignedInteger, BE: Backend> GetGGSWBit<BE> for FheUintHelper<'a, T, BE> {
    fn get_bit(&self, bit: usize) -> GGSWPrepared<&[u8], BE> {
        let lo: usize = bit % T::BITS as usize;
        let hi: usize = bit / T::BITS as usize;
        self.data[hi].get_bit(lo)
    }
}

impl<'a, T: UnsignedInteger, BE: Backend> BitSize for FheUintHelper<'a, T, BE> {
    fn bit_size(&self) -> usize {
        T::BITS as usize * self.data.len()
    }
}

pub struct JoinedBits<A, B> {
    pub lo: A,
    pub hi: B,
    pub split: usize, // 32 in your example
}

impl<A, B, BE> GetGGSWBit<BE> for JoinedBits<A, B>
where
    BE: Backend,
    A: GetGGSWBit<BE>,
    B: GetGGSWBit<BE>,
{
    fn get_bit(&self, bit: usize) -> GGSWPrepared<&[u8], BE> {
        if bit < self.split {
            self.lo.get_bit(bit)
        } else {
            self.hi.get_bit(bit - self.split)
        }
    }
}

#[macro_export]
macro_rules! define_bdd_2w_to_1w_trait {
    ($(#[$meta:meta])* $vis:vis $trait_name:ident, $method_name:ident) => {
        $(#[$meta])*
        $vis trait $trait_name<T: UnsignedInteger, BE: Backend> {
            fn $method_name<A, M, K, BRA, B>(
                &mut self,
                module: &M,
                a: &FheUintPrepared<A, T, BE>,
                b: &FheUintPrepared<B, T, BE>,
                key: &BDDKeyPrepared<K, BRA, BE>,
                scratch: &mut Scratch<BE>,
            ) where
                M: ExecuteBDDCircuit2WTo1W<T, BE>,
                A: DataRef,
                B: DataRef,
                K: DataRef,
                BRA: BlindRotationAlgo,
                Scratch<BE>: ScratchTakeCore<BE>;
        }
    };
}

#[macro_export]
macro_rules! impl_bdd_2w_to_1w_trait {
    ($trait_name:ident, $method_name:ident, $ty:ty, $n:literal, $circuit_ty:ty, $output_circuits:path) => {
        impl<D: DataMut, BE: Backend> $trait_name<$ty, BE> for FheUint<D, $ty> {
            fn $method_name<A, M, K, BRA, B>(
                &mut self,
                module: &M,
                a: &FheUintPrepared<A, $ty, BE>,
                b: &FheUintPrepared<B, $ty, BE>,
                key: &BDDKeyPrepared<K, BRA, BE>,
                scratch: &mut Scratch<BE>,
            ) where
                M: ExecuteBDDCircuit2WTo1W<$ty, BE>,
                A: DataRef,
                B: DataRef,
                K: DataRef,
                BRA: BlindRotationAlgo,
                Scratch<BE>: ScratchTakeCore<BE>,
            {
                module.execute_bdd_circuit_2w_to_1w(self, &$output_circuits, a, b, key, scratch)
            }
        }
    };
}

define_bdd_2w_to_1w_trait!(pub Add, add);
define_bdd_2w_to_1w_trait!(pub Sub, sub);
define_bdd_2w_to_1w_trait!(pub Sll, sll);
define_bdd_2w_to_1w_trait!(pub Sra, sra);
define_bdd_2w_to_1w_trait!(pub Srl, srl);
define_bdd_2w_to_1w_trait!(pub Slt, slt);
define_bdd_2w_to_1w_trait!(pub Sltu, sltu);
define_bdd_2w_to_1w_trait!(pub Or, or);
define_bdd_2w_to_1w_trait!(pub And, and);
define_bdd_2w_to_1w_trait!(pub Xor, xor);

impl_bdd_2w_to_1w_trait!(
    Add,
    add,
    u32,
    32,
    circuits::u32::add_codegen::AnyBitCircuit,
    circuits::u32::add_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sub,
    sub,
    u32,
    32,
    circuits::u32::sub_codegen::AnyBitCircuit,
    circuits::u32::sub_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sll,
    sll,
    u32,
    32,
    circuits::u32::sll_codegen::AnyBitCircuit,
    circuits::u32::sll_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sra,
    sra,
    u32,
    32,
    circuits::u32::sra_codegen::AnyBitCircuit,
    circuits::u32::sra_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Srl,
    srl,
    u32,
    32,
    circuits::u32::srl_codegen::AnyBitCircuit,
    circuits::u32::srl_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Slt,
    slt,
    u32,
    1,
    circuits::u32::slt_codegen::AnyBitCircuit,
    circuits::u32::slt_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sltu,
    sltu,
    u32,
    1,
    circuits::u32::sltu_codegen::AnyBitCircuit,
    circuits::u32::sltu_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    And,
    and,
    u32,
    32,
    circuits::u32::and_codegen::AnyBitCircuit,
    circuits::u32::and_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Or,
    or,
    u32,
    32,
    circuits::u32::or_codegen::AnyBitCircuit,
    circuits::u32::or_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Xor,
    xor,
    u32,
    32,
    circuits::u32::xor_codegen::AnyBitCircuit,
    circuits::u32::xor_codegen::OUTPUT_CIRCUITS
);
