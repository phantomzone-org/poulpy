use poulpy_core::{
    GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEPacking, GLWERotate, LWEFromGLWE, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEInfos, GLWEPlaintextLayout, GLWESecretPreparedToRef, GLWEToMut,
        GLWEToRef, LWEInfos, LWEToMut, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::ModuleLogN,
    layouts::{Backend, Data, DataMut, DataRef, Scratch},
    source::Source,
};
use std::{collections::HashMap, marker::PhantomData};

use crate::tfhe::{
    bdd_arithmetic::{BDDKeyPrepared, FromBits, ToBits, UnsignedInteger},
    blind_rotation::BlindRotationAlgo,
};

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
        M: ModuleLogN + GLWEEncryptSk<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::BITS as usize));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let mut data_bits: Vec<i64> = vec![0i64; module.n()];

        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;

        // Interleaves bytes
        for i in 0..T::BITS as usize {
            data_bits[T::bit_index(i) << log_gap] = data.bit(i) as i64
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
        M: ModuleLogN + GLWEDecrypt<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(module.n().is_multiple_of(T::BITS as usize));
            assert_eq!(self.n(), module.n() as u32);
            assert_eq!(sk.n(), module.n() as u32);
        }

        let pt_infos = GLWEPlaintextLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: 1_usize.into(),
        };

        let (mut pt, scratch_1) = scratch.take_glwe_plaintext(&pt_infos);

        self.bits.decrypt(module, &mut pt, sk, scratch_1);

        let mut data_bits: Vec<i64> = vec![0i64; module.n()];
        pt.decode_vec_i64(&mut data_bits, TorusPrecision(2));

        let mut bits: Vec<u8> = vec![0u8; T::BITS as usize];

        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;

        // Retrives from interleaved bytes
        for i in 0..T::BITS as usize {
            bits[i] = data_bits[T::bit_index(i) << log_gap] as u8
        }

        T::from_bits(&bits)
    }
}

impl<D: DataMut, T: UnsignedInteger> FheUint<D, T> {
    /// Packs Vec<GLWE(bit[i])> into [FheUint].
    pub fn pack<G, D1, M, BRA: BlindRotationAlgo, BE: Backend>(
        &mut self,
        module: &M,
        mut bits: Vec<G>,
        key: &BDDKeyPrepared<D1, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        G: GLWEToMut + GLWEToRef + GLWEInfos,
        D1: DataRef,
        M: ModuleLogN + GLWEPacking<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // Repacks the GLWE ciphertexts bits
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let mut cts: HashMap<usize, &mut G> = HashMap::new();
        for (i, ct) in bits.iter_mut().enumerate().take(T::BITS as usize) {
            cts.insert(T::bit_index(i) << log_gap, ct);
        }

        module.glwe_pack(&mut cts, log_gap, &key.cbt.atk, scratch);

        // And copies the repacked ciphertext on the receiver.
        module.glwe_copy(&mut self.bits, cts.remove(&0).unwrap());
    }

    // pub fn copy_byte<D0, D1, BRA, M, BE: Backend>(
    // &mut self,
    // module: &M,
    // byte_self: usize,
    // byte_a: usize,
    // a: &FheUint<D1, T>,
    // keys: &BDDKeyPrepared<D0, BRA, BE>,
    // scratch: &mut Scratch<BE>,
    // ) where
    // D0: DataRef,
    // D1: DataRef,
    // BRA: BlindRotationAlgo,
    // M:ModuleLogN + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd,
    // Scratch<BE>: ScratchTakeBDD<T, BE>,
    // {
    // let (mut tmp_fhe_uint_byte, scratch_1) = scratch.take_fhe_uint(a);
    //
    //
    // let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
    // module.glwe_rotate(-((T::bit_index(byte_a << 3) << log_gap) as i64), tmp_fhe_uint_byte, self);
    // module.glwe_trace_inplace(&mut tmp_fhe_uint_byte, module.log_n() - 3, module.log_n(),&keys.cbt.atk, scratch);
    //
    // let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
    // let rot: i64 = (T::bit_index(byte_self << 3) << log_gap) as i64;
    //
    // Move starting byte index to first coefficient
    // module.glwe_rotate_inplace(-rot, &mut self.bits, scratch);
    //
    // Stores this byte (everything else zeroed) into tmp_trace
    // let (mut tmp_trace, scratch_1) = scratch.take_glwe(self);
    // module.glwe_trace(
    // &mut tmp_trace,
    // module.log_n() - 3,
    // module.log_n(),
    // self,
    // &keys.cbt.atk,
    // scratch_1,
    // );
    //
    // Subtracts the byte
    // module.glwe_sub_inplace(&mut self.bits, &tmp_trace);
    //
    // module.glwe_add_inplace(&mut self.bits, &tmp_fhe_uint_byte);
    //
    // Moves back into the original position
    // module.glwe_rotate_inplace(-rot, &mut self.bits, scratch);
    //
    // }
}

impl<D: DataMut, T: UnsignedInteger> GLWEToMut for FheUint<D, T> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        self.bits.to_mut()
    }
}

pub trait ScratchTakeBDD<T: UnsignedInteger, BE: Backend>
where
    Self: ScratchTakeCore<BE>,
{
    fn take_fhe_uint<A>(&mut self, infos: &A) -> (FheUint<&mut [u8], T>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (glwe, scratch) = self.take_glwe(infos);
        (
            FheUint {
                bits: glwe,
                _phantom: PhantomData,
            },
            scratch,
        )
    }
}

impl<T: UnsignedInteger, BE: Backend> ScratchTakeBDD<T, BE> for Scratch<BE> where Self: ScratchTakeCore<BE> {}

impl<D: DataRef, T: UnsignedInteger> FheUint<D, T> {
    pub fn get_bit<R, K, M, BE: Backend>(&self, module: &M, bit: usize, res: &mut R, ks: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        M: ModuleLogN + LWEFromGLWE<BE> + GLWERotate<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        res.to_mut()
            .from_glwe(module, self, T::bit_index(bit) << log_gap, ks, scratch);
    }
}

impl<D: DataRef, T: UnsignedInteger> GLWEToRef for FheUint<D, T> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        self.bits.to_ref()
    }
}
