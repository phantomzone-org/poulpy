use std::marker::PhantomData;

use poulpy_core::{
    GLWEDecrypt, GLWENoise,
    layouts::{Base2K, GLWE, GLWEInfos, GLWEPlaintextLayout, GLWESecretPreparedToRef, LWEInfos, Rank, TorusPrecision},
};

use poulpy_core::GLWEEncryptSk;
use poulpy_core::ScratchTakeCore;
use poulpy_hal::layouts::{Backend, Data, DataMut, DataRef, Module, Scratch, ZnxZero};
use poulpy_hal::source::Source;

use crate::tfhe::bdd_arithmetic::{FromBits, ToBits, UnsignedInteger};

/// An FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUintBlocks<D: Data, T: UnsignedInteger> {
    pub(crate) blocks: Vec<GLWE<D>>,
    pub(crate) _base: u8,
    pub(crate) _phantom: PhantomData<T>,
}

impl<D: DataRef, T: UnsignedInteger> FheUintBlocks<D, T> {
    pub fn blocks(&self) -> &Vec<GLWE<D>> {
        &self.blocks
    }
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintBlocks<D, T> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.blocks[0].base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.blocks[0].k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.blocks[0].n()
    }
}

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintBlocks<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.blocks[0].rank()
    }
}

impl<D: Data, T: UnsignedInteger> FheUintBlocks<D, T> {
    pub fn new(blocks: Vec<GLWE<D>>) -> Self {
        assert_eq!(blocks.len(), T::WORD_SIZE);
        Self {
            blocks,
            _base: 1,
            _phantom: PhantomData,
        }
    }
}

impl<T: UnsignedInteger> FheUintBlocks<Vec<u8>, T> {
    pub fn alloc_from_infos<A, BE: Backend>(module: &Module<BE>, infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(module, infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc<BE: Backend>(module: &Module<BE>, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        Self {
            blocks: (0..T::WORD_SIZE)
                .map(|_| GLWE::alloc(module.n().into(), base2k, k, rank))
                .collect(),
            _base: 1,
            _phantom: PhantomData,
        }
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits> FheUintBlocks<D, T> {
    pub fn encrypt_sk<S, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        value: T,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Module<BE>: GLWEEncryptSk<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        use poulpy_core::layouts::GLWEPlaintextLayout;
        use poulpy_hal::layouts::ZnxZero;

        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::WORD_SIZE));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(&pt_infos);
        pt.data.zero();

        for i in 0..T::WORD_SIZE {
            pt.encode_coeff_i64(value.bit(i) as i64, TorusPrecision(2), 0);
            self.blocks[i].encrypt_sk(module, &pt, sk, source_xa, source_xe, scratch_1);
        }
    }
}

impl<D: DataRef, T: UnsignedInteger + FromBits + ToBits> FheUintBlocks<D, T> {
    pub fn decrypt<S, BE: Backend>(&self, module: &Module<BE>, sk: &S, scratch: &mut Scratch<BE>) -> T
    where
        Module<BE>: GLWEDecrypt<BE>,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::WORD_SIZE));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.k(),
        };

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(&pt_infos);

        let mut bits: Vec<u8> = vec![0u8; T::WORD_SIZE];

        let base2k: usize = self.base2k().into();
        let scale: f64 = 4.0 / ((1 << base2k) as f64);

        for (i, bit) in bits.iter_mut().enumerate().take(T::WORD_SIZE) {
            self.blocks[i].decrypt(module, &mut pt, sk, scratch_1);
            let value: i64 = pt.decode_coeff_i64(base2k.into(), 0);
            *bit = ((value as f64) * scale).round() as u8;
        }

        T::from_bits(&bits)
    }

    pub fn noise<S, BE: Backend>(&self, module: &Module<BE>, sk: &S, want: T, scratch: &mut Scratch<BE>) -> Vec<f64>
    where
        Module<BE>: GLWENoise<BE>,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::WORD_SIZE));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };

        let (mut pt_want, scratch_1) = scratch.take_glwe_plaintext(&pt_infos);
        pt_want.data.zero();

        let mut noise: Vec<f64> = vec![0f64; T::WORD_SIZE];

        for (i, noise_i) in noise.iter_mut().enumerate().take(T::WORD_SIZE) {
            pt_want.encode_coeff_i64(want.bit(i) as i64, TorusPrecision(2), 0);
            *noise_i = self.blocks[i].noise(module, sk, &pt_want, scratch_1);
        }

        noise
    }
}
