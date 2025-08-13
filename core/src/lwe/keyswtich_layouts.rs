use backend::hal::{
    api::{
        ScratchAvailable, TakeScalarZnx, TakeVecZnx, TakeVecZnxDft, VecZnxAddScalarInplace, VecZnxAutomorphismInplace,
        VecZnxSwithcDegree, ZnxView, ZnxViewMut, ZnxZero,
    },
    layouts::{Backend, Data, DataMut, DataRef, MatZnx, Module, ReaderFrom, Scratch, WriterTo},
};
use sampling::source::Source;

use crate::{
    GGLWEEncryptSkFamily, GLWECiphertext, GLWEKeyswitchFamily, GLWESecret, GLWESecretExec, GLWESwitchingKey, Infos,
    LWECiphertext, LWESecret, TakeGLWESecret, TakeGLWESecretExec,
};

/// A special [GLWESwitchingKey] required to for the conversion from [GLWECiphertext] to [LWECiphertext].
#[derive(PartialEq, Eq)]
pub struct GLWEToLWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: Data> Infos for GLWEToLWESwitchingKey<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<D: Data> GLWEToLWESwitchingKey<D> {
    pub fn digits(&self) -> usize {
        self.0.digits()
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.0.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.rank_out()
    }
}

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

impl GLWEToLWESwitchingKey<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, rank_in: usize) -> Self {
        Self(GLWESwitchingKey::alloc(n, basek, k, rows, 1, rank_in, 1))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank_in: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESecretExec::bytes_of(module, n, rank_in)
            + (GLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, rank_in, 1) | GLWESecret::bytes_of(n, rank_in))
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
        Module<B>: GGLWEEncryptSkFamily<B> + VecZnxAutomorphismInplace + VecZnxSwithcDegree + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.take_glwe_secret(sk_glwe.n(), 1);
        sk_lwe_as_glwe.data.zero();
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0);

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

#[derive(PartialEq, Eq)]
pub struct LWEToGLWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: Data> Infos for LWEToGLWESwitchingKey<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<D: Data> LWEToGLWESwitchingKey<D> {
    pub fn digits(&self) -> usize {
        self.0.digits()
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.0.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.rank_out()
    }
}

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
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, rank_out: usize) -> Self {
        Self(GLWESwitchingKey::alloc(n, basek, k, rows, 1, 1, rank_out))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize, rank_out: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, 1, rank_out) + GLWESecret::bytes_of(n, 1)
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
        Module<B>: GGLWEEncryptSkFamily<B> + VecZnxAutomorphismInplace + VecZnxSwithcDegree + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe.n() <= module.n());
        }

        let (mut sk_lwe_as_glwe, scratch1) = scratch.take_glwe_secret(sk_glwe.n(), 1);
        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0);

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
pub struct LWESwitchingKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: Data> Infos for LWESwitchingKey<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.0.inner()
    }

    fn basek(&self) -> usize {
        self.0.basek()
    }

    fn k(&self) -> usize {
        self.0.k()
    }
}

impl<D: Data> LWESwitchingKey<D> {
    pub fn digits(&self) -> usize {
        self.0.digits()
    }

    pub fn rank(&self) -> usize {
        self.0.rank()
    }

    pub fn rank_in(&self) -> usize {
        self.0.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.0.rank_out()
    }
}

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
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize) -> Self {
        Self(GLWESwitchingKey::alloc(n, basek, k, rows, 1, 1, 1))
    }

    pub fn encrypt_sk_scratch_space<B: Backend>(module: &Module<B>, n: usize, basek: usize, k: usize) -> usize
    where
        Module<B>: GGLWEEncryptSkFamily<B>,
    {
        GLWESecret::bytes_of(n, 1)
            + GLWESecretExec::bytes_of(module, n, 1)
            + GLWESwitchingKey::encrypt_sk_scratch_space(module, n, basek, k, 1, 1)
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
        Module<B>: GGLWEEncryptSkFamily<B> + VecZnxAutomorphismInplace + VecZnxSwithcDegree + VecZnxAddScalarInplace,
        Scratch<B>: ScratchAvailable + TakeScalarZnx + TakeVecZnxDft<B> + TakeGLWESecretExec<B> + TakeVecZnx,
    {
        #[cfg(debug_assertions)]
        {
            assert!(sk_lwe_in.n() <= self.n());
            assert!(sk_lwe_out.n() <= self.n());
            assert!(self.n() <= module.n());
        }

        let (mut sk_in_glwe, scratch1) = scratch.take_glwe_secret(self.n(), 1);
        let (mut sk_out_glwe, scratch2) = scratch1.take_glwe_secret(self.n(), 1);

        sk_out_glwe.data.at_mut(0, 0)[..sk_lwe_out.n()].copy_from_slice(sk_lwe_out.data.at(0, 0));
        sk_out_glwe.data.at_mut(0, 0)[sk_lwe_out.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_out_glwe.data.as_vec_znx_mut(), 0);

        sk_in_glwe.data.at_mut(0, 0)[..sk_lwe_in.n()].copy_from_slice(sk_lwe_in.data.at(0, 0));
        sk_in_glwe.data.at_mut(0, 0)[sk_lwe_in.n()..].fill(0);
        module.vec_znx_automorphism_inplace(-1, &mut sk_in_glwe.data.as_vec_znx_mut(), 0);

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
        n: usize,
        basek: usize,
        k_lwe: usize,
        k_glwe: usize,
        k_ksk: usize,
        rank: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::bytes_of(n, basek, k_lwe, 1)
            + GLWECiphertext::keyswitch_scratch_space(module, n, basek, k_lwe, k_glwe, k_ksk, 1, rank, 1)
    }

    pub fn keyswitch_scratch_space<B: Backend>(
        module: &Module<B>,
        n: usize,
        basek: usize,
        k_lwe_out: usize,
        k_lwe_in: usize,
        k_ksk: usize,
    ) -> usize
    where
        Module<B>: GLWEKeyswitchFamily<B>,
    {
        GLWECiphertext::bytes_of(n, basek, k_lwe_out.max(k_lwe_in), 1)
            + GLWECiphertext::keyswitch_inplace_scratch_space(module, n, basek, k_lwe_out, k_ksk, 1, 1)
    }
}
