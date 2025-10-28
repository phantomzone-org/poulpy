use itertools::Itertools;
use poulpy_core::{
    GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEPacking, ScratchTakeCore,
    layouts::{
        GLWE, GLWEInfos, GLWEPlaintextLayout, GLWESecretPreparedToRef, LWEInfos, TorusPrecision,
        prepared::GLWEAutomorphismKeyPrepared,
    },
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Data, DataMut, DataRef, Scratch},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData};

use crate::tfhe::bdd_arithmetic::{FromBits, ToBits, UnsignedInteger};

/// An FHE ciphertext encrypting the bits of an [UnsignedInteger] in compressed format (ideal for decryption/serialization).
pub struct FheUintCompressed<D: Data, T: UnsignedInteger>(pub(crate) GLWE<D>, pub(crate) PhantomData<T>);

impl<D: DataMut, T: UnsignedInteger> FheUintCompressed<D, T> {
    #[allow(dead_code)]
    fn post_process<ATK, M, BE: Backend>(
        &mut self,
        module: &M,
        mut tmp_res: Vec<GLWE<&mut [u8]>>,
        auto_keys: &HashMap<i64, GLWEAutomorphismKeyPrepared<ATK, BE>>,
        scratch: &mut Scratch<BE>,
    ) where
        ATK: DataRef,
        M: GLWEPacking<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // Repacks the GLWE ciphertexts bits
        let gap: usize = module.n() / T::WORD_SIZE;
        let log_gap: usize = (usize::BITS - (gap - 1).leading_zeros()) as usize;
        let mut cts: HashMap<usize, &mut GLWE<&mut [u8]>> = HashMap::new();
        for (i, ct) in tmp_res.iter_mut().enumerate().take(T::WORD_SIZE) {
            cts.insert(i * gap, ct);
        }

        module.glwe_pack(&mut cts, log_gap, auto_keys, scratch);

        // And copies the repacked ciphertext on the receiver.
        module.glwe_copy(&mut self.0, cts.remove(&0).unwrap());
    }
}

impl<D: DataRef, T: UnsignedInteger> LWEInfos for FheUintCompressed<D, T> {
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

impl<D: DataRef, T: UnsignedInteger> GLWEInfos for FheUintCompressed<D, T> {
    fn rank(&self) -> poulpy_core::layouts::Rank {
        self.0.rank()
    }
}

impl<D: DataMut, T: UnsignedInteger + ToBits> FheUintCompressed<D, T> {
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

        pt.encode_vec_i64(&data_bits, TorusPrecision(1));
        self.0
            .encrypt_sk(module, &pt, sk, source_xa, source_xe, scratch_1);
    }
}

impl<D: DataRef, T: UnsignedInteger + FromBits> FheUintCompressed<D, T> {
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

        self.0.decrypt(module, &mut pt, sk, scratch_1);

        let mut data: Vec<i64> = vec![0i64; module.n()];

        pt.decode_vec_i64(&mut data, TorusPrecision(1));

        let bits: Vec<u8> = data.iter().step_by(gap).map(|c| *c as u8).collect_vec();
        T::from_bits(&bits)
    }
}
