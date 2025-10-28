use itertools::Itertools;
use poulpy_core::{
    GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEPacking, GLWERotate, LWEFromGLWE, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWEPlaintextLayout, GLWESecretPreparedToRef, GLWEToRef,
        LWEInfos, LWEToMut, Rank, TorusPrecision, prepared::GLWEAutomorphismKeyPrepared,
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Data, DataMut, DataRef, Scratch},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData};

use crate::tfhe::bdd_arithmetic::{FromBits, ToBits, UnsignedInteger};

/// An FHE ciphertext encrypting the bits of an [UnsignedInteger].
pub struct FheUint<D: Data, T: UnsignedInteger> {
    pub(crate) bits: GLWE<D>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: UnsignedInteger> FheUint<Vec<u8>, T> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        Self {
            bits: GLWE::alloc(n, base2k, k, rank),
            _phantom: PhantomData,
        }
    }
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUint<D, T> {
    fn base2k(&self) -> poulpy_core::layouts::Base2K {
        self.bits.base2k()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.bits.k()
    }

    fn n(&self) -> poulpy_core::layouts::Degree {
        self.bits.n()
    }
}

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUint<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.bits.rank()
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits> FheUint<D, T> {
    pub fn encrypt_sk<S, M, BE: Backend>(
        &mut self,
        module: &M,
        data: T,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        M: ModuleN + GLWEEncryptSk<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
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

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(&pt_infos);

        pt.encode_vec_i64(&data_bits, TorusPrecision(2));
        self.bits
            .encrypt_sk(module, &pt, sk, source_xa, source_xe, scratch_1);
    }
}

impl<D: DataRef, T: UnsignedInteger + FromBits> FheUint<D, T> {
    pub fn decrypt<S, M, BE: Backend>(&self, module: &M, sk: &S, scratch: &mut Scratch<BE>) -> T
    where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        M: GLWEDecrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
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

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(&pt_infos);

        self.bits.decrypt(module, &mut pt, sk, scratch_1);

        let mut data: Vec<i64> = vec![0i64; module.n()];

        pt.decode_vec_i64(&mut data, TorusPrecision(2));

        let bits: Vec<u8> = data.iter().step_by(gap).map(|c| *c as u8).collect_vec();
        T::from_bits(&bits)
    }
}

impl<D: DataMut, T: UnsignedInteger> FheUint<D, T> {
    #[allow(dead_code)]
    pub(crate) fn pack<D1, ATK, M, BE: Backend>(
        &mut self,
        module: &M,
        mut tmp_res: Vec<GLWE<D1>>,
        auto_keys: &HashMap<i64, GLWEAutomorphismKeyPrepared<ATK, BE>>,
        scratch: &mut Scratch<BE>,
    ) where
        D1: DataMut,
        ATK: DataRef,
        M: GLWEPacking<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // Repacks the GLWE ciphertexts bits
        let gap: usize = module.n() / T::WORD_SIZE;
        let log_gap: usize = (usize::BITS - (gap - 1).leading_zeros()) as usize;
        let mut cts: HashMap<usize, &mut GLWE<D1>> = HashMap::new();
        for (i, ct) in tmp_res.iter_mut().enumerate().take(T::WORD_SIZE) {
            cts.insert(i * gap, ct);
        }

        module.glwe_pack(&mut cts, log_gap, auto_keys, scratch);

        // And copies the repacked ciphertext on the receiver.
        module.glwe_copy(&mut self.bits, cts.remove(&0).unwrap());
    }
}

impl<D: DataRef, T: UnsignedInteger> FheUint<D, T> {
    pub fn get_bit<L, K, M, BE: Backend>(&self, module: &M, bit: usize, res: &mut L, ks: &K, scratch: &mut Scratch<BE>)
    where
        L: LWEToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        M: LWEFromGLWE<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let gap: usize = module.n() / T::WORD_SIZE;
        res.to_mut().from_glwe(module, self, bit * gap, ks, scratch);
    }
}

impl<D: DataRef, T: UnsignedInteger> GLWEToRef for FheUint<D, T> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        self.bits.to_ref()
    }
}
