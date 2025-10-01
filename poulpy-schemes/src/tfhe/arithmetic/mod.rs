mod bin_ops_2x1;
mod circuits;
mod cmp_ops_2x1;
mod eval;
mod int_ops_2x1;

pub(crate) use bin_ops_2x1::*;
pub(crate) use circuits::*;
pub(crate) use cmp_ops_2x1::*;
pub(crate) use eval::*;
pub(crate) use int_ops_2x1::*;

#[cfg(test)]
mod test;

use itertools::Itertools;
use poulpy_core::{
    GLWEOperations, TakeGGSW, TakeGLWECt, TakeGLWECtSlice, TakeGLWEPt, glwe_packing,
    layouts::{
        GGSWCiphertext, GGSWInfos, GLWECiphertext, GLWECiphertextToRef, GLWEInfos, GLWEPlaintextLayout, LWECiphertext, LWEInfos,
        TorusPrecision,
        prepared::{
            GGLWEAutomorphismKeyPrepared, GGSWCiphertextPrepared, GLWESecretPrepared, GLWEToLWESwitchingKeyPrepared, Prepare,
        },
    },
};
use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAutomorphismInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateInplace, VecZnxCopy,
        VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace,
        VecZnxRshInplace, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes, VmpPrepare,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData};

use crate::tfhe::{
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute},
};

pub trait UnsignedInteger: Copy + 'static {
    const WORD_SIZE: usize;
}

impl UnsignedInteger for u8 {
    const WORD_SIZE: usize = 8;
}
impl UnsignedInteger for u16 {
    const WORD_SIZE: usize = 16;
}
impl UnsignedInteger for u32 {
    const WORD_SIZE: usize = 32;
}
impl UnsignedInteger for u64 {
    const WORD_SIZE: usize = 64;
}
impl UnsignedInteger for u128 {
    const WORD_SIZE: usize = 128;
}

/// A FHE ciphertext encrypting a [UnsignedInteger].
pub struct FheUintWord<D: Data, T: UnsignedInteger>(GLWECiphertext<D>, PhantomData<T>);

impl<D: DataMut, T: UnsignedInteger> FheUintWord<D, T> {
    pub fn add<A, B, ATK, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &FheUintBitsPrep<A, BE, T>,
        b: &FheUintBitsPrep<B, BE, T>,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<ATK, BE>>,
        scratch: &mut Scratch<BE>,
    ) where
        [(); T::WORD_SIZE]:,
        A: DataRef,
        B: DataRef,
        ATK: DataRef,
        Module<BE>: VecZnxSub
            + VecZnxCopy
            + VecZnxNegateInplace
            + VecZnxDftAllocBytes
            + VecZnxAddInplace
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<BE>
            + VmpApplyDftToDft<BE>
            + VmpApplyDftToDftAdd<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxNormalize<BE>
            + VecZnxRotateInplace<BE>
            + VecZnxNormalizeInplace<BE>
            + VecZnxSwitchRing
            + VecZnxBigAutomorphismInplace<BE>
            + VecZnxRshInplace<BE>
            + VecZnxDftCopy<BE>
            + VecZnxIdftApplyTmpA<BE>
            + VecZnxSubInplace
            + VecZnxBigNormalizeTmpBytes
            + VecZnxBigAddSmallInplace<BE>
            + VecZnxAutomorphismInplace<BE>
            + VecZnxBigSubSmallNegateInplace<BE>
            + VecZnxRotate,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeGLWECtSlice,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.0.len(), T::WORD_SIZE);
            assert_eq!(b.0.len(), T::WORD_SIZE);
        }

        // Allocates a temporary list of GLWE (one for each output bit)
        let (mut tmp_res, scratch_1) = scratch.take_glwe_ct_slice(T::WORD_SIZE, self);

        // Calls the appropriate circuit
        match T::WORD_SIZE {
            32 => ADD_OP32.execute(module, &mut tmp_res[..], &a.0, &b.0, scratch_1),
            _ => unimplemented!(),
        }

        // Repacks output
        self.post_process(module, tmp_res, auto_keys, scratch_1);
    }

    fn post_process<ATK, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        mut tmp_res: Vec<GLWECiphertext<&mut [u8]>>,
        auto_keys: &HashMap<i64, GGLWEAutomorphismKeyPrepared<ATK, BE>>,
        scratch: &mut Scratch<BE>,
    ) where
        ATK: DataRef,
        Module<BE>: VecZnxSub
            + VecZnxCopy
            + VecZnxNegateInplace
            + VecZnxDftAllocBytes
            + VecZnxAddInplace
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<BE>
            + VmpApplyDftToDft<BE>
            + VmpApplyDftToDftAdd<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxNormalize<BE>
            + VecZnxRotateInplace<BE>
            + VecZnxNormalizeInplace<BE>
            + VecZnxSwitchRing
            + VecZnxBigAutomorphismInplace<BE>
            + VecZnxRshInplace<BE>
            + VecZnxDftCopy<BE>
            + VecZnxIdftApplyTmpA<BE>
            + VecZnxSubInplace
            + VecZnxBigNormalizeTmpBytes
            + VecZnxBigAddSmallInplace<BE>
            + VecZnxAutomorphismInplace<BE>
            + VecZnxBigSubSmallNegateInplace<BE>
            + VecZnxRotate,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeGLWECtSlice,
    {
        // Repacks the GLWE ciphertexts bits
        let gap: usize = module.n() / T::WORD_SIZE;
        let log_gap: usize = (usize::BITS - (gap - 1).leading_zeros()) as usize;
        let mut cts: HashMap<usize, &mut GLWECiphertext<&mut [u8]>> = HashMap::new();
        for (i, ct) in tmp_res.iter_mut().enumerate().take(T::WORD_SIZE) {
            cts.insert(i * gap, ct);
        }
        glwe_packing(module, &mut cts, log_gap, auto_keys, scratch);

        // And copies the repacked ciphertext on the receiver.
        self.0.copy(module, cts.remove(&0).unwrap())
    }
}

/// An FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintBits<D: Data, T: UnsignedInteger>([GLWECiphertext<D>; T::WORD_SIZE])
where
    [(); T::WORD_SIZE]:;

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintBits<D, T>
where
    [(); T::WORD_SIZE]:,
{
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.0[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.0[0].k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.0[0].n()
    }
}

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintBits<D, T>
where
    [(); T::WORD_SIZE]:,
{
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.0[0].rank()
    }
}

impl<D: DataMut, T: UnsignedInteger> FheUintBits<D, T>
where
    [(); T::WORD_SIZE]:,
{
    pub fn add<A, B, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        a: &FheUintBitsPrep<A, BE, T>,
        b: &FheUintBitsPrep<B, BE, T>,
        scratch: &mut Scratch<BE>,
    ) where
        A: DataRef,
        B: DataRef,
        Module<BE>: VecZnxSub
            + VecZnxCopy
            + VecZnxNegateInplace
            + VecZnxDftAllocBytes
            + VecZnxAddInplace
            + VmpApplyDftToDftTmpBytes
            + VecZnxNormalizeTmpBytes
            + VecZnxDftApply<BE>
            + VmpApplyDftToDft<BE>
            + VmpApplyDftToDftAdd<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxNormalize<BE>,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx,
    {
        match T::WORD_SIZE {
            32 => ADD_OP32.execute(module, &mut self.0, &a.0, &b.0, scratch),
            _ => unimplemented!(),
        }
    }
}

/// A prepared FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintBitsPrep<D: Data, B: Backend, T: UnsignedInteger>([GGSWCiphertextPrepared<D, B>; T::WORD_SIZE])
where
    [(); T::WORD_SIZE]:;

impl<D: DataMut, T: UnsignedInteger, BE: Backend> FheUintBitsPrep<D, BE, T>
where
    [(); T::WORD_SIZE]:,
{
    /// Prepares [FheUintBits] to [FheUintBitsPrep].
    pub fn prepare<BIT, LKS, CBT, BRA>(
        &mut self,
        module: &Module<BE>,
        bits: FheUintBits<BIT, T>,
        lwe_ks: &GLWEToLWESwitchingKeyPrepared<LKS, BE>,
        key: &CircuitBootstrappingKeyPrepared<CBT, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        BIT: DataRef,
        LKS: DataRef,
        CBT: DataRef,
        BRA: BlindRotationAlgo,
        Module<BE>: VmpPrepare<BE>
            + VecZnxRotate
            + VecZnxDftAllocBytes
            + VmpApplyDftToDftTmpBytes
            + VecZnxBigNormalizeTmpBytes
            + VmpApplyDftToDft<BE>
            + VmpApplyDftToDftAdd<BE>
            + VecZnxDftApply<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigAddSmallInplace<BE>
            + VecZnxBigNormalize<BE>
            + VecZnxNormalize<BE>
            + VecZnxNormalizeTmpBytes,
        Scratch<BE>: ScratchAvailable + TakeVecZnxDft<BE> + TakeGLWECt + TakeVecZnx + TakeGGSW,
        CircuitBootstrappingKeyPrepared<CBT, BRA, BE>: CirtuitBootstrappingExecute<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.0.len(), bits.0.len());
        }
        let mut lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(&bits.0[0]); //TODO: add TakeLWE
        let (mut tmp_ggsw, scratch_1) = scratch.take_ggsw(self);
        for (dst, src) in self.0.iter_mut().zip(bits.0.iter()) {
            lwe.from_glwe(module, src, lwe_ks, scratch_1);
            key.execute_to_constant(module, &mut tmp_ggsw, &lwe, 1, 1, scratch_1);
            dst.prepare(module, &tmp_ggsw, scratch_1);
        }
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> LWEInfos for FheUintBitsPrep<D, B, T>
where
    [(); T::WORD_SIZE]:,
{
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.0[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.0[0].k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.0[0].n()
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> GLWEInfos for FheUintBitsPrep<D, B, T>
where
    [(); T::WORD_SIZE]:,
{
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.0[0].rank()
    }
}

impl<D: DataRef, T: UnsignedInteger, B: Backend> GGSWInfos for FheUintBitsPrep<D, B, T>
where
    [(); T::WORD_SIZE]:,
{
    fn digits(&self) -> poulpy_core::layouts::Digits {
        self.0[0].digits()
    }

    fn rows(&self) -> poulpy_core::layouts::Rows {
        self.0[0].rows()
    }
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintWord<D, T> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.0.base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.0.k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.0.n()
    }
}

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintWord<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.0.rank()
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits> FheUintWord<D, T> {
    pub fn encrypt_sk<S: DataRef, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        data: T,
        sk: &GLWESecretPrepared<S, BE>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        Module<BE>: VecZnxAddScalarInplace
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<BE>
            + VecZnxDftApply<BE>
            + SvpApplyDftToDftInplace<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxNormalizeTmpBytes
            + VecZnxFillUniform
            + VecZnxSubInplace
            + VecZnxAddInplace
            + VecZnxNormalizeInplace<BE>
            + VecZnxAddNormal
            + VecZnxNormalize<BE>
            + VecZnxSub,
        Scratch<BE>: TakeVecZnxDft<BE> + ScratchAvailable + TakeVecZnx + TakeGLWEPt<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::WORD_SIZE));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let gap: usize = module.n() / T::WORD_SIZE;

        let mut data_bits: Vec<i64> = vec![0i64; module.n()];

        for i in 0..T::WORD_SIZE {
            data_bits[i * gap] = data.bit(i) as i64
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };

        let (mut pt, scratch_1) = scratch.take_glwe_pt(&pt_infos);

        pt.encode_vec_i64(&data_bits, TorusPrecision(1));
        self.0
            .encrypt_sk(module, &pt, sk, source_xa, source_xe, scratch_1);
    }
}

impl<D: DataRef, T: UnsignedInteger + FromBits> FheUintWord<D, T> {
    pub fn decrypt<S: DataRef, BE: Backend>(
        &self,
        module: &Module<BE>,
        sk: &GLWESecretPrepared<S, BE>,
        scratch: &mut Scratch<BE>,
    ) -> T
    where
        Module<BE>: VecZnxDftApply<BE>
            + SvpApplyDftToDftInplace<BE>
            + VecZnxIdftApplyConsume<BE>
            + VecZnxBigAddInplace<BE>
            + VecZnxBigAddSmallInplace<BE>
            + VecZnxBigNormalize<BE>,
        Scratch<BE>: TakeVecZnxDft<BE> + TakeVecZnxBig<BE> + TakeGLWEPt<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::WORD_SIZE));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let gap: usize = module.n() / T::WORD_SIZE;

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };

        let (mut pt, scratch_1) = scratch.take_glwe_pt(&pt_infos);

        self.0.decrypt(module, &mut pt, sk, scratch_1);

        let mut data: Vec<i64> = vec![0i64; module.n()];

        pt.decode_vec_i64(&mut data, TorusPrecision(1));

        let bits: Vec<u8> = data.iter().step_by(gap).map(|c| *c as u8).collect_vec();
        T::from_bits(&bits)
    }
}

pub trait ToBits {
    fn bit(&self, i: usize) -> u8;
}

macro_rules! impl_tobits {
    ($($t:ty),*) => {
        $(
            impl ToBits for $t {
                fn bit(&self, i: usize) -> u8 {
                    if i >= (std::mem::size_of::<$t>() * 8) {
                        panic!("bit index {} out of range for {}", i, stringify!($t));
                    }
                    ((self >> i) & 1) as u8
                }
            }
        )*
    };
}

impl_tobits!(u8, u16, u32, u64, u128);

pub trait FromBits: Sized {
    fn from_bits(bits: &[u8]) -> Self;
}

macro_rules! impl_from_bits {
    ($($t:ty),*) => {
        $(
            impl FromBits for $t {
                fn from_bits(bits: &[u8]) -> Self {
                    let mut value: $t = 0;
                    let max_bits = std::mem::size_of::<$t>() * 8;
                    let n = bits.len().min(max_bits);

                    for (i, &bit) in bits.iter().take(n).enumerate() {
                        if bit != 0 {
                            value |= 1 << i;
                        }
                    }
                    value
                }
            }
        )*
    };
}

impl_from_bits!(u8, u16, u32, u64, u128);
