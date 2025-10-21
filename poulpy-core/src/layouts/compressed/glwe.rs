use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEToMut, GetDegree, LWEInfos, Rank, SetGLWEInfos, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GLWECompressed<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) rank: Rank,
    pub(crate) seed: [u8; 32],
}

pub trait GLWECompressedSeedMut {
    fn seed_mut(&mut self) -> &mut [u8; 32];
}

impl<D: DataMut> GLWECompressedSeedMut for GLWECompressed<D> {
    fn seed_mut(&mut self) -> &mut [u8; 32] {
        &mut self.seed
    }
}

pub trait GLWECompressedSeed {
    fn seed(&self) -> &[u8; 32];
}

impl<D: DataRef> GLWECompressedSeed for GLWECompressed<D> {
    fn seed(&self) -> &[u8; 32] {
        &self.seed
    }
}

impl<D: Data> LWEInfos for GLWECompressed<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}
impl<D: Data> GLWEInfos for GLWECompressed<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: DataRef> fmt::Debug for GLWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for GLWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWECompressed: base2k={} k={} rank={} seed={:?}: {}",
            self.base2k(),
            self.k(),
            self.rank(),
            self.seed,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for GLWECompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GLWECompressed<Vec<u8>> {
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        GLWECompressed {
            data: VecZnx::alloc(n.into(), 1, k.0.div_ceil(base2k.0) as usize),
            base2k,
            k,
            rank,
            seed: [0u8; 32],
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::bytes_of(n.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: DataMut> ReaderFrom for GLWECompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.rank = Rank(reader.read_u32::<LittleEndian>()?);
        reader.read_exact(&mut self.seed)?;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWECompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.rank.into())?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

pub trait GLWEDecompress
where
    Self: GetDegree + VecZnxFillUniform + VecZnxCopy,
{
    fn decompress_glwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToMut + SetGLWEInfos,
        O: GLWECompressedToRef + GLWEInfos,
    {
        {
            let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
            let other: &GLWECompressed<&[u8]> = &other.to_ref();
            assert_eq!(
                res.n(),
                self.ring_degree(),
                "invalid receiver: res.n()={} != other.n()={}",
                res.n(),
                self.ring_degree()
            );

            assert_eq!(res.glwe_layout(), other.glwe_layout());

            let mut source: Source = Source::new(other.seed);

            self.vec_znx_copy(&mut res.data, 0, &other.data, 0);
            (1..(other.rank() + 1).into()).for_each(|i| {
                self.vec_znx_fill_uniform(other.base2k.into(), &mut res.data, i, &mut source);
            });
        }

        res.set_base2k(other.base2k());
        res.set_k(other.k());
    }
}

impl<B: Backend> GLWEDecompress for Module<B> where Self: GetDegree + VecZnxFillUniform + VecZnxCopy {}

impl<D: DataMut> GLWE<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: GLWECompressedToRef + GLWEInfos,
        M: GLWEDecompress,
    {
        module.decompress_glwe(self, other);
    }
}

pub trait GLWECompressedToRef {
    fn to_ref(&self) -> GLWECompressed<&[u8]>;
}

impl<D: DataRef> GLWECompressedToRef for GLWECompressed<D> {
    fn to_ref(&self) -> GLWECompressed<&[u8]> {
        GLWECompressed {
            seed: self.seed.clone(),
            base2k: self.base2k,
            k: self.k,
            rank: self.rank,
            data: self.data.to_ref(),
        }
    }
}

pub trait GLWECompressedToMut {
    fn to_mut(&mut self) -> GLWECompressed<&mut [u8]>;
}

impl<D: DataMut> GLWECompressedToMut for GLWECompressed<D> {
    fn to_mut(&mut self) -> GLWECompressed<&mut [u8]> {
        GLWECompressed {
            seed: self.seed.clone(),
            base2k: self.base2k,
            k: self.k,
            rank: self.rank,
            data: self.data.to_mut(),
        }
    }
}
