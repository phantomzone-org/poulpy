use itertools::Itertools;
use poulpy_core::{
    layouts::{
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared}, GLWEInfos, GLWEPlaintextLayout, LWEInfos, TorusPrecision, GLWE
    }, ScratchTakeCore,
};
use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDftInplace, VecZnxAddInplace, VecZnxAddNormal,
        VecZnxAddScalarInplace, VecZnxAutomorphismInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace,
        VecZnxBigAutomorphismInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateInplace, VecZnxCopy,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA,
        VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace,
        VecZnxRshInplace, VecZnxSub, VecZnxSubInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, Scratch},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData};

use crate::tfhe::bdd_arithmetic::{FromBits, ToBits, UnsignedInteger};

/// A FHE ciphertext encrypting a [UnsignedInteger].
pub struct FheUintWord<D: Data, T: UnsignedInteger>(pub(crate) GLWE<D>, pub(crate) PhantomData<T>);

impl<D: DataMut, T: UnsignedInteger> FheUintWord<D, T> {
    #[allow(dead_code)]
    fn post_process<ATK, BE: Backend>(
        &mut self,
        module: &Module<BE>,
        mut tmp_res: Vec<GLWE<&mut [u8]>>,
        auto_keys: &HashMap<i64, GLWEAutomorphismKeyPrepared<ATK, BE>>,
        scratch: &mut Scratch<BE>,
    ) where
        ATK: DataRef,
        Module<BE>: VecZnxSub
            + VecZnxCopy
            + VecZnxNegateInplace
            + VecZnxDftBytesOf
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
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // Repacks the GLWE ciphertexts bits
        let gap: usize = module.n() / T::WORD_SIZE;
        let log_gap: usize = (usize::BITS - (gap - 1).leading_zeros()) as usize;
        let mut cts: HashMap<usize, &mut GLWE<&mut [u8]>> = HashMap::new();
        for (i, ct) in tmp_res.iter_mut().enumerate().take(T::WORD_SIZE) {
            cts.insert(i * gap, ct);
        }
        glwe_packing(module, &mut cts, log_gap, auto_keys, scratch);

        // And copies the repacked ciphertext on the receiver.
        self.0.copy(module, cts.remove(&0).unwrap())
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
            + VecZnxDftBytesOf
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

        let (mut pt, scratch_1) = scratch.take_glwe_pt(&pt_infos);

        self.0.decrypt(module, &mut pt, sk, scratch_1);

        let mut data: Vec<i64> = vec![0i64; module.n()];

        pt.decode_vec_i64(&mut data, TorusPrecision(1));

        let bits: Vec<u8> = data.iter().step_by(gap).map(|c| *c as u8).collect_vec();
        T::from_bits(&bits)
    }
}
