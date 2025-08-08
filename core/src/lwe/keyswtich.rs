use backend::hal::{
    api::{
        MatZnxAlloc, ScalarZnxAllocBytes, ScratchAvailable, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace,
        VecZnxAllocBytes, VecZnxAutomorphismInplace, VecZnxSwithcDegree, ZnxView, ZnxViewMut, ZnxZero,
    },
    layouts::{Backend, Data, DataMut, DataRef, Module, ReaderFrom, Scratch, WriterTo},
};
use sampling::source::Source;

use crate::{
    GGLWEEncryptSkFamily, GGLWEExecLayoutFamily, GLWECiphertext, GLWEKeyswitchFamily, GLWESecret, GLWESecretExec,
    GLWESwitchingKey, GLWESwitchingKeyExec, Infos, LWECiphertext, LWESecret, TakeGLWECt, TakeGLWESecret, TakeGLWESecretExec,
};

/// A special [GLWESwitchingKey] required to for the conversion from [GLWECiphertext] to [LWECiphertext].
#[derive(PartialEq, Eq)]
pub struct GLWEToLWESwitchingKey<D: Data>(GLWESwitchingKey<D>);

impl<D: DataMut> ReaderFrom for GLWEToLWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEToLWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

#[derive(PartialEq, Eq)]
pub struct GLWEToLWESwitchingKeyExec<D: Data, B: Backend>(GLWESwitchingKeyExec<D, B>);

impl<B: Backend> GLWEToLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(
            module, basek, k, rows, 1, rank_in, 1,
        ))
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, rank_in, 1)
    }

    pub fn from<DataOther: DataRef>(
        module: &Module<B>,
        other: &GLWEToLWESwitchingKey<DataOther>,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: GLWEToLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.0.basek(),
            other.0.k(),
            other.0.rows(),
            other.0.rank_in(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: DataMut, B: Backend> GLWEToLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWEToLWESwitchingKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        Self(GLWESwitchingKey::alloc(
            module, basek, k, rows, 1, rank_in, 1,
        ))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B> + ScalarZnxAllocBytes + VecZnxAllocBytes,
    {
        GLWESecretExec::bytes_of(module, rank_in)
            + (GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, rank_in, 1) | GLWESecret::bytes_of(module, rank_in))
    }
}

impl<D: DataMut> GLWEToLWESwitchingKey<D> {
    pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DGlwe: DataRef,
        Module<B>: GGLWEEncryptSkFamily<B>
            + VecZnxAutomorphismInplace
            + ScalarZnxAllocBytes
            + VecZnxSwithcDegree
            + VecZnxAllocBytes
            + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx<B> + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.take_glwe_secret(module, 1);
        sk_lwe_as_glwe.data.zero();
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data, 0);

        self.0.encrypt_sk(
            module,
            sk_glwe,
            &sk_lwe_as_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch1,
        );
    }
}

/// A special [GLWESwitchingKey] required to for the conversion from [LWECiphertext] to [GLWECiphertext].
#[derive(PartialEq, Eq)]
pub struct LWEToGLWESwitchingKeyExec<D: Data, B: Backend>(GLWESwitchingKeyExec<D, B>);

impl<B: Backend> LWEToGLWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_out: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(
            module, basek, k, rows, 1, 1, rank_out,
        ))
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, 1, rank_out)
    }

    pub fn from<DataOther: DataRef>(
        module: &Module<B>,
        other: &LWEToGLWESwitchingKey<DataOther>,
        scratch: &mut Scratch<B>,
    ) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: LWEToGLWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(
            module,
            other.0.basek(),
            other.0.k(),
            other.0.rows(),
            other.0.rank(),
        );
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: DataMut, B: Backend> LWEToGLWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &LWEToGLWESwitchingKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}
#[derive(PartialEq, Eq)]
pub struct LWEToGLWESwitchingKey<D: Data>(GLWESwitchingKey<D>);

impl<D: DataMut> ReaderFrom for LWEToGLWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWEToGLWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWEToGLWESwitchingKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, rank_out: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        Self(GLWESwitchingKey::alloc(
            module, basek, k, rows, 1, 1, rank_out,
        ))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B> + ScalarZnxAllocBytes + VecZnxAllocBytes,
    {
        GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, 1, rank_out) + GLWESecret::bytes_of(module, 1)
    }
}

impl<D: DataMut> LWEToGLWESwitchingKey<D> {
    pub fn encrypt_sk<DLwe, DGlwe, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe: &LWESecret<DLwe>,
        sk_glwe: &GLWESecret<DGlwe>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DGlwe: DataRef,
        Module<B>: GGLWEEncryptSkFamily<B>
            + VecZnxAutomorphismInplace
            + ScalarZnxAllocBytes
            + VecZnxSwithcDegree
            + VecZnxAllocBytes
            + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx<B> + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.take_glwe_secret(module, 1);
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data, 0);

        self.0.encrypt_sk(
            module,
            &sk_lwe_as_glwe,
            &sk_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch1,
        );
    }
}

#[derive(PartialEq, Eq)]
pub struct LWESwitchingKeyExec<D: Data, B: Backend>(GLWESwitchingKeyExec<D, B>);

impl<B: Backend> LWESwitchingKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        Self(GLWESwitchingKeyExec::alloc(module, basek, k, rows, 1, 1, 1))
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, 1, 1)
    }

    pub fn from<DataOther: DataRef>(module: &Module<B>, other: &LWESwitchingKey<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut ksk_exec: LWESwitchingKeyExec<Vec<u8>, B> = Self::alloc(module, other.0.basek(), other.0.k(), other.0.rows());
        ksk_exec.prepare(module, other, scratch);
        ksk_exec
    }
}

impl<D: DataMut, B: Backend> LWESwitchingKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &LWESwitchingKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        self.0.prepare(module, &other.0, scratch);
    }
}
#[derive(PartialEq, Eq)]
pub struct LWESwitchingKey<D: Data>(GLWESwitchingKey<D>);

impl<D: DataMut> ReaderFrom for LWESwitchingKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWESwitchingKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl LWESwitchingKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        Self(GLWESwitchingKey::alloc(module, basek, k, rows, 1, 1, 1))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B> + ScalarZnxAllocBytes + VecZnxAllocBytes,
    {
        GLWESecret::bytes_of(module, 1)
            + GLWESecretExec::bytes_of(module, 1)
            + GLWESwitchingKey::encrypt_sk_scratch_space(module, basek, k, 1, 1)
    }
}

impl<D: DataMut> LWESwitchingKey<D> {
    pub fn encrypt_sk<DIn, DOut, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk_lwe_in: &LWESecret<DIn>,
        sk_lwe_out: &LWESecret<DOut>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        DIn: DataRef,
        DOut: DataRef,
        Module<B>: GGLWEEncryptSkFamily<B>
            + VecZnxAutomorphismInplace
            + ScalarZnxAllocBytes
            + VecZnxSwithcDegree
            + VecZnxAllocBytes
            + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx<B> + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe_in.n() <= module.n());
            assert!(sk_lwe_out.n() <= module.n());
        }

        let (mut sk_in_glwe, scratch1) = scratch.take_glwe_secret(module, 1);
        let (mut sk_out_glwe, scratch2) = scratch1.take_glwe_secret(module, 1);

        sk_out_glwe.data.at_mut(0, 0)[..sk_lwe_out.n()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_out_glwe.data.at_mut(0, 0)[sk_lwe_out.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_out_glwe.data, 0);

        sk_in_glwe.data.at_mut(0, 0)[..sk_lwe_in.n()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_in_glwe.data.at_mut(0, 0)[sk_lwe_in.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_in_glwe.data, 0);

        self.0.encrypt_sk(
            module,
            &sk_in_glwe,
            &sk_out_glwe,
            source_xa,
            source_xe,
            sigma,
            scratch2,
        );
    }
}

impl LWECiphertext<Vec<u8>> {
    pub fn from_glwe_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_lwe: usize,
        k_glwe: usize,
        k_ksk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxAllocBytes,
    {
        GLWECiphertext::bytes_of(module, basek, k_lwe, 1)
            + GLWECiphertext::keyswitch_scratch_space(module, basek, k_lwe, k_glwe, k_ksk, 1, rank, 1)
    }

    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_lwe_out: usize,
        k_lwe_in: usize,
        k_ksk: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B> + ScalarZnxAllocBytes + VecZnxAllocBytes,
    {
        GLWECiphertext::bytes_of(module, basek, k_lwe_out.max(k_lwe_in), 1)
            + GLWECiphertext::keyswitch_inplace_scratch_space(module, basek, k_lwe_out, k_ksk, 1, 1)
    }
}

impl<DLwe: DataMut> LWECiphertext<DLwe> {
    pub fn sample_extract<DGlwe: DataRef>(&mut self, a: &GLWECiphertext<DGlwe>) {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= a.n());
        }

        let min_size: usize = self.size().min(a.size());
        let n: usize = self.n();

        self.data.zero();
        (0..min_size).for_each(|i| {
            let data_lwe: &mut [i64] = self.data.at_mut(0, i);
            data_lwe[0] = a.data.at(0, i)[0];
            data_lwe[1..].copy_from_slice(&a.data.at(1, i)[..n]);
        });
    }

    pub fn from_glwe<DGlwe, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &GLWECiphertext<DGlwe>,
        ks: &GLWEToLWESwitchingKeyExec<DKs, B>,
        scratch: &mut Scratch<B>,
    ) where
        DGlwe: DataRef,
        DKs: DataRef,
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.basek(), a.basek());
        }
        let (mut tmp_glwe, scratch1) = scratch.take_glwe_ct(module, a.basek(), self.k(), 1);
        tmp_glwe.keyswitch(module, a, &ks.0, scratch1);
        self.sample_extract(&tmp_glwe);
    }

    pub fn keyswitch<A, DKs, B: Backend>(
        &mut self,
        module: &Module<B>,
        a: &LWECiphertext<A>,
        ksk: &LWESwitchingKeyExec<DKs, B>,
        scratch: &mut Scratch<B>,
    ) where
        A: DataRef,
        DKs: DataRef,
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(self.n() <= module.n());
            assert!(a.n() <= module.n());
            assert_eq!(self.basek(), a.basek());
        }

        let max_k: usize = self.k().max(a.k());
        let basek: usize = self.basek();

        let (mut glwe, scratch1) = scratch.take_glwe_ct(&module, basek, max_k, 1);
        glwe.data.zero();

        let n_lwe: usize = a.n();

        (0..a.size()).for_each(|i| {
            let data_lwe: &[i64] = a.data.at(0, i);
            glwe.data.at_mut(0, i)[0] = data_lwe[0];
            glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        });

        glwe.keyswitch_inplace(module, &ksk.0, scratch1);

        self.sample_extract(&glwe);
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn from_lwe_scratch_space<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k_lwe: usize,
        k_glwe: usize,
        k_ksk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B> + VecZnxAllocBytes,
    {
        GLWECiphertext::keyswitch_scratch_space(module, basek, k_glwe, k_lwe, k_ksk, 1, 1, rank)
            + GLWECiphertext::bytes_of(module, basek, k_lwe, 1)
    }
}

impl<D: DataMut> GLWECiphertext<D> {
    pub fn from_lwe<DLwe, DKsk, B: Backend>(
        &mut self,
        module: &Module<B>,
        lwe: &LWECiphertext<DLwe>,
        ksk: &LWEToGLWESwitchingKeyExec<DKsk, B>,
        scratch: &mut Scratch<B>,
    ) where
        DLwe: DataRef,
        DKsk: DataRef,
        Module<B>: GLWEKeyswitchFamily<B>,
        Scratch<B>: TakeVecZnxDft<B> + ScratchAvailable + TakeVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert!(lwe.n() <= self.n());
            assert_eq!(self.basek(), self.basek());
        }

        let (mut glwe, scratch1) = scratch.take_glwe_ct(module, lwe.basek(), lwe.k(), 1);
        glwe.data.zero();

        let n_lwe: usize = lwe.n();

        (0..lwe.size()).for_each(|i| {
            let data_lwe: &[i64] = lwe.data.at(0, i);
            glwe.data.at_mut(0, i)[0] = data_lwe[0];
            glwe.data.at_mut(1, i)[..n_lwe].copy_from_slice(&data_lwe[1..]);
        });

        self.keyswitch(module, &glwe, &ksk.0, scratch1);
    }
}
