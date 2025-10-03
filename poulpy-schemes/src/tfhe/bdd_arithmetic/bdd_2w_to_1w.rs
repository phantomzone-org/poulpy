use itertools::Itertools;
use poulpy_core::layouts::prepared::GGSWCiphertextPreparedToRef;
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::tfhe::bdd_arithmetic::{
    BitCircuitInfo, Circuit, CircuitExecute, FheUintBlocks, FheUintBlocksPrep, UnsignedInteger, circuits,
};



/// Operations Z x Z -> Z
pub(crate) struct Circuits2WTo1W<C: BitCircuitInfo + 'static, const WORD_SIZE: usize>(pub &'static Circuit<C, WORD_SIZE>);

pub trait EvalBDD2WTo1W<BE: Backend> {
    fn eval_bdd_2w_to_1w<R, A, B, T: UnsignedInteger>(
        &self,
        module: &Module<BE>,
        out: &mut FheUintBlocks<R, T>,
        a: &FheUintBlocksPrep<A, BE, T>,
        b: &FheUintBlocksPrep<B, BE, T>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;
}

impl<C: BitCircuitInfo + 'static, const WORD_SIZE: usize, BE: Backend> EvalBDD2WTo1W<BE> for Circuits2WTo1W<C, WORD_SIZE>
where
    Circuit<C, WORD_SIZE>: CircuitExecute<BE>,
{
    fn eval_bdd_2w_to_1w<R, A, B, T: UnsignedInteger>(
        &self,
        module: &Module<BE>,
        out: &mut FheUintBlocks<R, T>,
        a: &FheUintBlocksPrep<A, BE, T>,
        b: &FheUintBlocksPrep<B, BE, T>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef,
    {
        eval_bdd_2w_to_1w(module, self.0, out, a, b, scratch);
    }
}

pub fn eval_bdd_2w_to_1w<R: DataMut, A: DataRef, B: DataRef, T: UnsignedInteger, C: CircuitExecute<BE>, BE: Backend>(
    module: &Module<BE>,
    circuit: &C,
    out: &mut FheUintBlocks<R, T>,
    a: &FheUintBlocksPrep<A, BE, T>,
    b: &FheUintBlocksPrep<B, BE, T>,
    scratch: &mut Scratch<BE>,
) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(out.blocks.len(), T::WORD_SIZE);
        assert_eq!(b.blocks.len(), T::WORD_SIZE);
    }

    // Collects inputs into a single array
    let inputs: Vec<&dyn GGSWCiphertextPreparedToRef<BE>> = a
        .blocks
        .iter()
        .map(|x| x as &dyn GGSWCiphertextPreparedToRef<BE>)
        .chain(
            b.blocks
                .iter()
                .map(|x| x as &dyn GGSWCiphertextPreparedToRef<BE>),
        )
        .collect_vec();

    // Evaluates out[i] = circuit[i](a, b)
    circuit.execute(module, &mut out.blocks, &inputs, scratch);
}

#[macro_export]
macro_rules! define_bdd_2w_to_1w_trait {
    ($(#[$meta:meta])* $vis:vis $trait_name:ident, $method_name:ident) => {
        $(#[$meta])*
        $vis trait $trait_name<T: UnsignedInteger, BE: Backend> {
            fn $method_name<A, B>(
                &mut self,
                module: &Module<BE>,
                a: &FheUintBlocksPrep<A, BE, T>,
                b: &FheUintBlocksPrep<B, BE, T>,
                scratch: &mut Scratch<BE>,
            ) where
                A: DataRef,
                B: DataRef;
        }
    };
}

#[macro_export]
macro_rules! impl_2w_to_1w_trait {
    ($trait_name:ident, $method_name:ident, $ty:ty, $circuit_ty:ty, $output_circuits:path) => {
        impl<D: DataMut, BE: Backend> $trait_name<$ty, BE> for FheUintBlocks<D, $ty>
        where
            Circuits2WTo1W<$circuit_ty, { <$ty>::BITS as usize }>: EvalBDD2WTo1W<BE>,
        {
            fn $method_name<A, B>(
                &mut self,
                module: &Module<BE>,
                a: &FheUintBlocksPrep<A, BE, $ty>,
                b: &FheUintBlocksPrep<B, BE, $ty>,
                scratch: &mut Scratch<BE>,
            ) where
                A: DataRef,
                B: DataRef,
            {
                const OP: Circuits2WTo1W<$circuit_ty, { <$ty>::BITS as usize }> =
                    Circuits2WTo1W::<$circuit_ty, { <$ty>::BITS as usize }>(&$output_circuits);

                OP.eval_bdd_2w_to_1w(module, self, a, b, scratch);
            }
        }
    };
}

define_bdd_2w_to_1w_trait!(pub Add, add);
define_bdd_2w_to_1w_trait!(pub Sub, sub);
define_bdd_2w_to_1w_trait!(pub Sll, sll);
define_bdd_2w_to_1w_trait!(pub Sra, sra);
define_bdd_2w_to_1w_trait!(pub Srl, srl);

impl_2w_to_1w_trait!(
    Add,
    add,
    u32,
    circuits::u32::add_codegen::AnyBitCircuit,
    circuits::u32::add_codegen::OUTPUT_CIRCUITS
);

impl_2w_to_1w_trait!(
    Sub,
    sub,
    u32,
    circuits::u32::sub_codegen::AnyBitCircuit,
    circuits::u32::sub_codegen::OUTPUT_CIRCUITS
);

impl_2w_to_1w_trait!(
    Sll,
    sll,
    u32,
    circuits::u32::sll_codegen::AnyBitCircuit,
    circuits::u32::sll_codegen::OUTPUT_CIRCUITS
);

impl_2w_to_1w_trait!(
    Sra,
    sra,
    u32,
    circuits::u32::sra_codegen::AnyBitCircuit,
    circuits::u32::sra_codegen::OUTPUT_CIRCUITS
);

impl_2w_to_1w_trait!(
    Srl,
    srl,
    u32,
    circuits::u32::srl_codegen::AnyBitCircuit,
    circuits::u32::srl_codegen::OUTPUT_CIRCUITS
);