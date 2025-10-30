use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEPacking, GLWERotate, GLWESub, GLWETrace, LWEFromGLWE, ScratchTakeCore,
    layouts::{
        Base2K, Degree, GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEPlaintextLayout,
        GLWESecretPreparedToRef, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos, LWEToMut, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::ModuleLogN,
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

impl<'a, T: UnsignedInteger> FheUint<&'a mut [u8], T> {
    pub fn from_glwe_to_mut<G>(glwe: &'a mut G) -> Self
    where
        G: GLWEToMut,
    {
        FheUint {
            bits: glwe.to_mut(),
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
    pub fn pack<G, M, K, H, BE: Backend>(&mut self, module: &M, mut bits: Vec<G>, keys: &H, scratch: &mut Scratch<BE>)
    where
        G: GLWEToMut + GLWEToRef + GLWEInfos,
        M: ModuleLogN + GLWEPacking<BE> + GLWECopy,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // Repacks the GLWE ciphertexts bits
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let mut cts: HashMap<usize, &mut G> = HashMap::new();
        for (i, ct) in bits.iter_mut().enumerate().take(T::BITS as usize) {
            cts.insert(T::bit_index(i) << log_gap, ct);
        }

        module.glwe_pack(&mut self.bits, cts, log_gap, keys, scratch);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn splice_u16<A, B, H, K, M, BE: Backend>(
        &mut self,
        module: &M,
        dst: usize,
        src: usize,
        a: &A,
        b: &B,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        B: GLWEToRef + GLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos + GetGaloisElement,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd + GLWECopy,
        Scratch<BE>: ScratchTakeBDD<T, BE>,
    {
        assert!(dst < (T::BITS >> 4) as usize);
        assert!(src < (T::BITS >> 4) as usize);

        let (mut tmp, scratch_1) = scratch.take_fhe_uint(self);
        tmp.splice_u8(module, dst << 1, src << 1, a, b, keys, scratch_1);
        self.splice_u8(
            module,
            (dst << 1) + 1,
            (src << 1) + 1,
            &tmp,
            b,
            keys,
            scratch_1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    // Store on the receiver a where the byte_a-th byte of a has been replaced by byte_src2 of src2.
    pub fn splice_u8<A, B, H, K, M, BE: Backend>(
        &mut self,
        module: &M,
        dst: usize,
        src: usize,
        a: &A,
        b: &B,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        B: GLWEToRef + GLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos + GetGaloisElement,
        M: ModuleLogN + GLWERotate<BE> + GLWETrace<BE> + GLWESub + GLWEAdd + GLWECopy,
        Scratch<BE>: ScratchTakeBDD<T, BE>,
    {
        assert!(dst < (T::BITS >> 3) as usize);
        assert!(src < (T::BITS >> 3) as usize);

        // 1) Zero the byte receiver
        let log_gap: usize = module.log_n() - T::LOG_BITS as usize;
        let trace_start = (T::LOG_BITS - T::LOG_BYTES) as usize;
        let rot: i64 = (T::bit_index(dst << 3) << log_gap) as i64;

        // Move a to self and align byte
        module.glwe_rotate(-rot, &mut self.bits, a);

        // Stores this byte (everything else zeroed) into tmp_trace
        let (mut tmp_trace, scratch_1) = scratch.take_glwe(a);
        module.glwe_trace(
            &mut tmp_trace,
            trace_start,
            module.log_n(),
            self,
            keys,
            scratch_1,
        );

        // Subtracts to self to zero it
        module.glwe_sub_inplace(&mut self.bits, &tmp_trace);

        // Isolate the byte to transfer from a
        let (mut tmp_fhe_uint_byte, scratch_1) = scratch.take_fhe_uint(b);

        // Move a[byte_a] into a[0]
        module.glwe_rotate(
            -((T::bit_index(src << 3) << log_gap) as i64),
            &mut tmp_fhe_uint_byte,
            b,
        );

        // Zeroes all other bytes
        module.glwe_trace_inplace(
            &mut tmp_fhe_uint_byte,
            trace_start,
            module.log_n(),
            keys,
            scratch_1,
        );

        // Add self[0] += a[0]
        module.glwe_add_inplace(&mut self.bits, &tmp_fhe_uint_byte);

        // Moves back self[0] to self[byte_tg]
        module.glwe_rotate_inplace(rot, &mut self.bits, scratch);
    }
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
