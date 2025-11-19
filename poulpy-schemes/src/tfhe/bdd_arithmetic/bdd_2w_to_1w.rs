use std::marker::PhantomData;

use poulpy_core::{
    GLWECopy, GLWEPacking, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GGSWPrepared, GLWEAutomorphismKeyHelper, GetGaloisElement},
};
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::tfhe::bdd_arithmetic::{
    BitSize, ExecuteBDDCircuit, FheUint, FheUintPrepared, GetBitCircuitInfo, GetGGSWBit, UnsignedInteger, circuits,
};

impl<BE: Backend> ExecuteBDDCircuit2WTo1W<BE> for Module<BE> where Self: Sized + ExecuteBDDCircuit<BE> + GLWEPacking<BE> + GLWECopy
{}

pub trait ExecuteBDDCircuit2WTo1W<BE: Backend>
where
    Self: Sized + ModuleLogN + ExecuteBDDCircuit<BE> + GLWEPacking<BE> + GLWECopy,
{
    fn execute_bdd_circuit_2w_to_1w<R, C, A, B, K, H, T>(
        &self,
        out: &mut FheUint<R, T>,
        circuit: &C,
        a: &FheUintPrepared<A, T, BE>,
        b: &FheUintPrepared<B, T, BE>,
        key: &H,
        scratch: &mut Scratch<BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        R: DataMut,
        A: DataRef,
        B: DataRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.execute_bdd_circuit_2w_to_1w_multi_thread(1, out, circuit, a, b, key, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    /// Operations Z x Z -> Z
    fn execute_bdd_circuit_2w_to_1w_multi_thread<R, C, A, B, K, H, T>(
        &self,
        threads: usize,
        out: &mut FheUint<R, T>,
        circuit: &C,
        a: &FheUintPrepared<A, T, BE>,
        b: &FheUintPrepared<B, T, BE>,
        key: &H,
        scratch: &mut Scratch<BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        R: DataMut,
        A: DataRef,
        B: DataRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
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
        self.execute_bdd_circuit_multi_thread(threads, &mut out_bits, &helper, circuit, scratch_1);

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

#[macro_export]
macro_rules! define_bdd_2w_to_1w_trait {
    ($(#[$meta:meta])* $vis:vis $trait_name:ident, $method_name:ident) => {
        paste::paste! {
            $(#[$meta])*
            $vis trait $trait_name<T: UnsignedInteger, BE: Backend> {

                /// Single-threaded version
                fn $method_name<A, M, K, H, B>(
                    &mut self,
                    module: &M,
                    a: &FheUintPrepared<A, T, BE>,
                    b: &FheUintPrepared<B, T, BE>,
                    key: &H,
                    scratch: &mut Scratch<BE>,
                ) where
                    M: ExecuteBDDCircuit2WTo1W<BE>,
                    A: DataRef,
                    B: DataRef,
                    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
                    H: GLWEAutomorphismKeyHelper<K, BE>,
                    Scratch<BE>: ScratchTakeCore<BE>;

                /// Multithreaded version – same vis, method_name + "_multi_thread"
                fn [<$method_name _multi_thread>]<A, M, K, H, B>(
                    &mut self,
                    threads: usize,
                    module: &M,
                    a: &FheUintPrepared<A, T, BE>,
                    b: &FheUintPrepared<B, T, BE>,
                    key: &H,
                    scratch: &mut Scratch<BE>,
                ) where
                    M: ExecuteBDDCircuit2WTo1W<BE>,
                    A: DataRef,
                    B: DataRef,
                    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
                    H: GLWEAutomorphismKeyHelper<K, BE>,
                    Scratch<BE>: ScratchTakeCore<BE>;
            }
        }
    };
}

#[macro_export]
macro_rules! impl_bdd_2w_to_1w_trait {
    ($trait_name:ident, $method_name:ident, $ty:ty, $circuit_ty:ty, $output_circuits:path) => {
        paste::paste! {
            impl<D: DataMut, BE: Backend> $trait_name<$ty, BE> for FheUint<D, $ty> {

                fn $method_name<A, M, K, H, B>(
                    &mut self,
                    module: &M,
                    a: &FheUintPrepared<A, $ty, BE>,
                    b: &FheUintPrepared<B, $ty, BE>,
                    key: &H,
                    scratch: &mut Scratch<BE>,
                ) where
                    M: ExecuteBDDCircuit2WTo1W<BE>,
                    A: DataRef,
                    B: DataRef,
                    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
                    H: GLWEAutomorphismKeyHelper<K, BE>,
                    Scratch<BE>: ScratchTakeCore<BE>,
                {
                    module.execute_bdd_circuit_2w_to_1w(self, &$output_circuits, a, b, key, scratch)
                }

                fn [<$method_name _multi_thread>]<A, M, K, H, B>(
                    &mut self,
                    threads: usize,
                    module: &M,
                    a: &FheUintPrepared<A, $ty, BE>,
                    b: &FheUintPrepared<B, $ty, BE>,
                    key: &H,
                    scratch: &mut Scratch<BE>,
                ) where
                    M: ExecuteBDDCircuit2WTo1W<BE>,
                    A: DataRef,
                    B: DataRef,
                    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
                    H: GLWEAutomorphismKeyHelper<K, BE>,
                    Scratch<BE>: ScratchTakeCore<BE>,
                {
                    module.execute_bdd_circuit_2w_to_1w_multi_thread(threads, self, &$output_circuits, a, b, key, scratch)
                }
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
    circuits::u32::add_codegen::AnyBitCircuit,
    circuits::u32::add_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sub,
    sub,
    u32,
    circuits::u32::sub_codegen::AnyBitCircuit,
    circuits::u32::sub_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sll,
    sll,
    u32,
    circuits::u32::sll_codegen::AnyBitCircuit,
    circuits::u32::sll_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sra,
    sra,
    u32,
    circuits::u32::sra_codegen::AnyBitCircuit,
    circuits::u32::sra_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Srl,
    srl,
    u32,
    circuits::u32::srl_codegen::AnyBitCircuit,
    circuits::u32::srl_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Slt,
    slt,
    u32,
    circuits::u32::slt_codegen::AnyBitCircuit,
    circuits::u32::slt_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Sltu,
    sltu,
    u32,
    circuits::u32::sltu_codegen::AnyBitCircuit,
    circuits::u32::sltu_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    And,
    and,
    u32,
    circuits::u32::and_codegen::AnyBitCircuit,
    circuits::u32::and_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Or,
    or,
    u32,
    circuits::u32::or_codegen::AnyBitCircuit,
    circuits::u32::or_codegen::OUTPUT_CIRCUITS
);

impl_bdd_2w_to_1w_trait!(
    Xor,
    xor,
    u32,
    circuits::u32::xor_codegen::AnyBitCircuit,
    circuits::u32::xor_codegen::OUTPUT_CIRCUITS
);
