use backend::hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftFromVecZnx},
    layouts::{Backend, Data, DataMut, DataRef, Module, ReaderFrom, Scratch, ScratchOwned, VecZnx, VecZnxDft, WriterTo},
    oep::{ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeVecZnxDftImpl, TakeVecZnxImpl},
};
use sampling::source::Source;

use crate::{GLWECiphertext, GLWEEncryptSkFamily, GLWESecretExec, Infos, dist::Distribution};

pub trait GLWEPublicKeyFamily<B: Backend> = GLWEEncryptSkFamily<B>;

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKey<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) dist: Distribution,
}

impl GLWEPublicKey<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: VecZnx::alloc(n, rank + 1, k.div_ceil(basek)),
            basek: basek,
            k: k,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rank: usize) -> usize {
        VecZnx::alloc_bytes(n, rank + 1, k.div_ceil(basek))
    }
}

impl<D: Data> Infos for GLWEPublicKey<D> {
    type Inner = VecZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<D: Data> GLWEPublicKey<D> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<D: DataMut> GLWEPublicKey<D> {
    pub fn generate_from_sk<S: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &GLWESecretExec<S, B>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
    ) where
        Module<B>: GLWEPublicKeyFamily<B>,
        B: ScratchOwnedAllocImpl<B>
            + ScratchOwnedBorrowImpl<B>
            + TakeVecZnxDftImpl<B>
            + ScratchAvailableImpl<B>
            + TakeVecZnxImpl<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), sk.n());

            match sk.dist {
                Distribution::NONE => panic!("invalid sk: SecretDistribution::NONE"),
                _ => {}
            }
        }

        // Its ok to allocate scratch space here since pk is usually generated only once.
        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(GLWECiphertext::encrypt_sk_scratch_space(
            module,
            self.n(),
            self.basek(),
            self.k(),
        ));

        let mut tmp: GLWECiphertext<Vec<u8>> = GLWECiphertext::alloc(self.n(), self.basek(), self.k(), self.rank());
        tmp.encrypt_zero_sk(module, sk, source_xa, source_xe, sigma, scratch.borrow());
        self.dist = sk.dist;
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for GLWEPublicKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        match Distribution::read_from(reader) {
            Ok(dist) => self.dist = dist,
            Err(e) => return Err(e),
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWEPublicKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        match self.dist.write_to(writer) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        self.data.write_to(writer)
    }
}

#[derive(PartialEq, Eq)]
pub struct GLWEPublicKeyExec<D: Data, B: Backend> {
    pub(crate) data: VecZnxDft<D, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) dist: Distribution,
}

impl<D: Data, B: Backend> Infos for GLWEPublicKeyExec<D, B> {
    type Inner = VecZnxDft<D, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<D: Data, B: Backend> GLWEPublicKeyExec<D, B> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<B: Backend> GLWEPublicKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> Self
    where
        Module<B>: VecZnxDftAlloc<B>,
    {
        Self {
            data: module.vec_znx_dft_alloc(n, rank + 1, k.div_ceil(basek)),
            basek: basek,
            k: k,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of(module: &Module<B>, n: usize, basek: usize, k: usize, rank: usize) -> usize
    where
        Module<B>: VecZnxDftAllocBytes,
    {
        module.vec_znx_dft_alloc_bytes(n, rank + 1, k.div_ceil(basek))
    }

    pub fn from<DataOther>(module: &Module<B>, other: &GLWEPublicKey<DataOther>, scratch: &mut Scratch<B>) -> Self
    where
        DataOther: DataRef,
        Module<B>: VecZnxDftAlloc<B> + VecZnxDftFromVecZnx<B>,
    {
        let mut pk_exec: GLWEPublicKeyExec<Vec<u8>, B> =
            GLWEPublicKeyExec::alloc(module, other.n(), other.basek(), other.k(), other.rank());
        pk_exec.prepare(module, other, scratch);
        pk_exec
    }
}

impl<D: DataMut, B: Backend> GLWEPublicKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWEPublicKey<DataOther>, _scratch: &mut Scratch<B>)
    where
        DataOther: DataRef,
        Module<B>: VecZnxDftFromVecZnx<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.n(), other.n());
            assert_eq!(self.size(), other.size());
        }

        (0..self.cols()).for_each(|i| {
            module.vec_znx_dft_from_vec_znx(1, 0, &mut self.data, i, &other.data, i);
        });
        self.k = other.k;
        self.basek = other.basek;
        self.dist = other.dist;
    }
}
