use poulpy_core::{
    GLWECopy, GLWEPacking, ScratchTakeCore,
    layouts::{GGLWEInfos, GGLWEPreparedToRef, GLWEAutomorphismKeyHelper, GetGaloisElement},
};
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::bin_fhe::bdd_arithmetic::{ExecuteBDDCircuit, FheUint, FheUintPrepared, GetBitCircuitInfo, UnsignedInteger, circuits};

impl<BE: Backend> ExecuteBDDCircuit1WTo1W<BE> for Module<BE> where Self: Sized + ExecuteBDDCircuit<BE> + GLWEPacking<BE> + GLWECopy
{}

/// Backend-level executor for single-input BDD circuits (`Z → Z`).
///
/// Evaluates a BDD circuit that reads one encrypted integer and produces
/// one encrypted integer.  After evaluating the per-bit BDD levels, the
/// output bits are repacked into a single [`FheUint`] polynomial via
/// [`GLWEPacking`].
pub trait ExecuteBDDCircuit1WTo1W<BE: Backend>
where
    Self: Sized + ModuleLogN + ExecuteBDDCircuit<BE> + GLWEPacking<BE> + GLWECopy,
{
    fn execute_bdd_circuit_1w_to_1w<R, C, A, K, H, T>(
        &self,
        out: &mut FheUint<R, T>,
        circuit: &C,
        a: &FheUintPrepared<A, T, BE>,
        key: &H,
        scratch: &mut Scratch<BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        R: DataMut,
        A: DataRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.execute_bdd_circuit_1w_to_1w_multi_thread(1, out, circuit, a, key, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    /// Operations Z x Z -> Z
    fn execute_bdd_circuit_1w_to_1w_multi_thread<R, C, A, K, H, T>(
        &self,
        threads: usize,
        out: &mut FheUint<R, T>,
        circuit: &C,
        a: &FheUintPrepared<A, T, BE>,
        key: &H,
        scratch: &mut Scratch<BE>,
    ) where
        T: UnsignedInteger,
        C: GetBitCircuitInfo,
        R: DataMut,
        A: DataRef,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let (mut out_bits, scratch_1) = scratch.take_glwe_slice(T::BITS as usize, out);

        // Evaluates out[i] = circuit[i](a, b)
        self.execute_bdd_circuit_multi_thread(threads, &mut out_bits, a, circuit, scratch_1);

        // Repacks the bits
        out.pack(self, out_bits, key, scratch_1);
    }
}

#[macro_export]
macro_rules! define_bdd_1w_to_1w_trait {
    ($(#[$meta:meta])* $vis:vis $trait_name:ident, $method_name:ident) => {
        paste::paste! {
            $(#[$meta])*
            $vis trait $trait_name<T: UnsignedInteger, BE: Backend> {

                /// Single-threaded version
                fn $method_name<A, M, K, H>(
                    &mut self,
                    module: &M,
                    a: &FheUintPrepared<A, T, BE>,
                    key: &H,
                    scratch: &mut Scratch<BE>,
                ) where
                    M: ExecuteBDDCircuit1WTo1W<BE>,
                    A: DataRef,
                    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
                    H: GLWEAutomorphismKeyHelper<K, BE>,
                    Scratch<BE>: ScratchTakeCore<BE>;

                /// Multithreaded version – same vis, method_name + "_multi_thread"
                fn [<$method_name _multi_thread>]<A, M, K, H>(
                    &mut self,
                    threads: usize,
                    module: &M,
                    a: &FheUintPrepared<A, T, BE>,
                    key: &H,
                    scratch: &mut Scratch<BE>,
                ) where
                    M: ExecuteBDDCircuit1WTo1W<BE>,
                    A: DataRef,
                    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
                    H: GLWEAutomorphismKeyHelper<K, BE>,
                    Scratch<BE>: ScratchTakeCore<BE>;
            }
        }
    };
}

#[macro_export]
macro_rules! impl_bdd_1w_to_1w_trait {
    ($trait_name:ident, $method_name:ident, $ty:ty, $circuit_ty:ty, $output_circuits:path) => {
        paste::paste! {
            impl<D: DataMut, BE: Backend> $trait_name<$ty, BE> for FheUint<D, $ty> {

                fn $method_name<A, M, K, H>(
                    &mut self,
                    module: &M,
                    a: &FheUintPrepared<A, $ty, BE>,
                    key: &H,
                    scratch: &mut Scratch<BE>,
                ) where
                    M: ExecuteBDDCircuit1WTo1W<BE>,
                    A: DataRef,
                    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
                    H: GLWEAutomorphismKeyHelper<K, BE>,
                    Scratch<BE>: ScratchTakeCore<BE>,
                {
                    module.execute_bdd_circuit_1w_to_1w(self, &$output_circuits, a, key, scratch)
                }

                fn [<$method_name _multi_thread>]<A, M, K, H>(
                    &mut self,
                    threads: usize,
                    module: &M,
                    a: &FheUintPrepared<A, $ty, BE>,
                    key: &H,
                    scratch: &mut Scratch<BE>,
                ) where
                    M: ExecuteBDDCircuit1WTo1W<BE>,
                    A: DataRef,
                    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
                    H: GLWEAutomorphismKeyHelper<K, BE>,
                    Scratch<BE>: ScratchTakeCore<BE>,
                {
                    module.execute_bdd_circuit_1w_to_1w_multi_thread(threads, self, &$output_circuits, a, key, scratch)
                }
            }
        }
    };
}
define_bdd_1w_to_1w_trait!(
    /// Homomorphic identity function (`out = a`).
    ///
    /// Re-bootstraps all bits of `a` through the BDD circuit and repacks the
    /// result into a fresh [`FheUint`].  Useful for noise refreshing without
    /// computing any arithmetic.
    pub Identity, identity);

impl_bdd_1w_to_1w_trait!(
    Identity,
    identity,
    u32,
    circuits::u32::identity_codgen::AnyBitCircuit,
    circuits::u32::identity_codgen::OUTPUT_CIRCUITS
);
