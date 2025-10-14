use std::fmt;

use poulpy_hal::{
    api::ZnFillUniform,
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo, Zn, ZnToMut, ZnToRef, ZnxInfos, ZnxView,
        ZnxViewMut,
    },
    source::Source,
};

use crate::layouts::{Base2K, Degree, LWE, LWEInfos, LWEToMut, TorusPrecision};

#[derive(PartialEq, Eq, Clone)]
pub struct LWECompressed<D: Data> {
    pub(crate) data: Zn<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) seed: [u8; 32],
}

impl<D: Data> LWEInfos for LWECompressed<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: DataRef> fmt::Debug for LWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for LWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWECompressed: base2k={} k={} seed={:?}: {}",
            self.base2k(),
            self.k(),
            self.seed,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for LWECompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

pub trait LWECompressedAlloc {
    fn alloc_lwe_compressed(&self, base2k: Base2K, k: TorusPrecision) -> LWECompressed<Vec<u8>> {
        LWECompressed {
            data: Zn::alloc(1, 1, k.0.div_ceil(base2k.0) as usize),
            k,
            base2k,
            seed: [0u8; 32],
        }
    }

    fn alloc_lwe_compressed_from_infos<A>(&self, infos: &A) -> LWECompressed<Vec<u8>>
    where
        A: LWEInfos,
    {
        self.alloc_lwe_compressed(infos.base2k(), infos.k())
    }

    fn bytes_of_lwe_compressed(&self, base2k: Base2K, k: TorusPrecision) -> usize {
        Zn::bytes_of(1, 1, k.0.div_ceil(base2k.0) as usize)
    }

    fn bytes_of_lwe_compressed_from_infos<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        self.bytes_of_lwe_compressed(infos.base2k(), infos.k())
    }
}

impl LWECompressed<Vec<u8>> {
    pub fn alloc_from_infos<A, B: Backend>(module: &Module<B>, infos: &A) -> Self
    where
        A: LWEInfos,
        Module<B>: LWECompressedAlloc,
    {
        module.alloc_lwe_compressed_from_infos(infos)
    }

    pub fn alloc<B: Backend>(module: &Module<B>, base2k: Base2K, k: TorusPrecision) -> Self
    where
        Module<B>: LWECompressedAlloc,
    {
        module.alloc_lwe_compressed(base2k, k)
    }

    pub fn bytes_of_from_infos<A, B: Backend>(module: &Module<B>, infos: &A) -> usize
    where
        A: LWEInfos,
        Module<B>: LWECompressedAlloc,
    {
        module.bytes_of_lwe_compressed_from_infos(infos)
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, base2k: Base2K, k: TorusPrecision) -> usize
    where
        Module<B>: LWECompressedAlloc,
    {
        module.bytes_of_lwe_compressed(base2k, k)
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for LWECompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        reader.read_exact(&mut self.seed)?;
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWECompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

pub trait LWEDecompress
where
    Self: ZnFillUniform,
{
    fn decompress_lwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToMut,
        O: LWECompressedToRef,
    {
        let res: &mut LWE<&mut [u8]> = &mut res.to_mut();
        let other: &LWECompressed<&[u8]> = &other.to_ref();

        assert_eq!(res.lwe_layout(), other.lwe_layout());

        let mut source: Source = Source::new(other.seed);
        self.zn_fill_uniform(
            res.n().into(),
            other.base2k().into(),
            &mut res.data,
            0,
            &mut source,
        );
        for i in 0..res.size() {
            res.data.at_mut(0, i)[0] = other.data.at(0, i)[0];
        }
    }
}

impl<B: Backend> LWEDecompress for Module<B> where Self: ZnFillUniform {}

impl<D: DataMut> LWE<D> {
    pub fn decompress<O, B: Backend>(&mut self, module: &Module<B>, other: &O)
    where
        O: LWECompressedToRef,
        Module<B>: LWEDecompress,
    {
        module.decompress_lwe(self, other);
    }
}

pub trait LWECompressedToRef {
    fn to_ref(&self) -> LWECompressed<&[u8]>;
}

impl<D: DataRef> LWECompressedToRef for LWECompressed<D> {
    fn to_ref(&self) -> LWECompressed<&[u8]> {
        LWECompressed {
            k: self.k,
            base2k: self.base2k,
            seed: self.seed,
            data: self.data.to_ref(),
        }
    }
}

pub trait LWECompressedToMut {
    fn to_mut(&mut self) -> LWECompressed<&mut [u8]>;
}

impl<D: DataMut> LWECompressedToMut for LWECompressed<D> {
    fn to_mut(&mut self) -> LWECompressed<&mut [u8]> {
        LWECompressed {
            k: self.k,
            base2k: self.base2k,
            seed: self.seed,
            data: self.data.to_mut(),
        }
    }
}
