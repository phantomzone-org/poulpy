use std::fmt;

use poulpy_hal::{
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo, Zn, ZnToMut, ZnToRef, ZnxInfos},
    source::Source,
};

use crate::layouts::{Base2K, Degree, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

pub trait LWEInfos {
    fn n(&self) -> Degree;
    fn k(&self) -> TorusPrecision;
    fn max_k(&self) -> TorusPrecision {
        TorusPrecision(self.k().0 * self.size() as u32)
    }
    fn base2k(&self) -> Base2K;
    fn size(&self) -> usize {
        self.k().0.div_ceil(self.base2k().0) as usize
    }
    fn lwe_layout(&self) -> LWECiphertextLayout {
        LWECiphertextLayout {
            n: self.n(),
            k: self.k(),
            base2k: self.base2k(),
        }
    }
}

pub trait SetLWEInfos {
    fn set_k(&mut self, k: TorusPrecision);
    fn set_base2k(&mut self, base2k: Base2K);
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWECiphertextLayout {
    pub n: Degree,
    pub k: TorusPrecision,
    pub base2k: Base2K,
}

impl LWEInfos for LWECiphertextLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.n
    }
}
#[derive(PartialEq, Eq, Clone)]
pub struct LWE<D: Data> {
    pub(crate) data: Zn<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

impl<D: Data> LWEInfos for LWE<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32 - 1)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> SetLWEInfos for LWE<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }

    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: DataRef> LWE<D> {
    pub fn data(&self) -> &Zn<D> {
        &self.data
    }
}

impl<D: DataMut> LWE<D> {
    pub fn data_mut(&mut self) -> &Zn<D> {
        &mut self.data
    }
}

impl<D: DataRef> fmt::Debug for LWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef> fmt::Display for LWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWECiphertext: base2k={} k={}: {}",
            self.base2k().0,
            self.k().0,
            self.data
        )
    }
}

impl<D: DataMut> FillUniform for LWE<D>
where
    Zn<D>: FillUniform,
{
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

pub trait LWEAlloc {
    fn alloc_lwe(&self, n: Degree, base2k: Base2K, k: TorusPrecision) -> LWE<Vec<u8>> {
        LWE {
            data: Zn::alloc((n + 1).into(), 1, k.0.div_ceil(base2k.0) as usize),
            k,
            base2k,
        }
    }

    fn alloc_lwe_from_infos<A>(&self, infos: &A) -> LWE<Vec<u8>>
    where
        A: LWEInfos,
    {
        self.alloc_lwe(infos.n(), infos.base2k(), infos.k())
    }

    fn bytes_of_lwe(&self, n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        Zn::bytes_of((n + 1).into(), 1, k.0.div_ceil(base2k.0) as usize)
    }

    fn bytes_of_lwe_from_infos<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        self.bytes_of_lwe(infos.n(), infos.base2k(), infos.k())
    }
}

impl LWE<Vec<u8>> {
    pub fn alloc_from_infos<A, B: Backend>(module: &Module<B>, infos: &A) -> Self
    where
        A: LWEInfos,
        Module<B>: LWEAlloc,
    {
        module.alloc_lwe_from_infos(infos)
    }

    pub fn alloc<B: Backend>(module: &Module<B>, n: Degree, base2k: Base2K, k: TorusPrecision) -> Self
    where
        Module<B>: LWEAlloc,
    {
        module.alloc_lwe(n, base2k, k)
    }

    pub fn bytes_of_from_infos<A, B: Backend>(module: &Module<B>, infos: &A) -> usize
    where
        A: LWEInfos,
        Module<B>: LWEAlloc,
    {
        module.bytes_of_lwe_from_infos(infos)
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, n: Degree, base2k: Base2K, k: TorusPrecision) -> usize
    where
        Module<B>: LWEAlloc,
    {
        module.bytes_of_lwe(n, base2k, k)
    }
}

pub trait LWECiphertextToRef {
    fn to_ref(&self) -> LWE<&[u8]>;
}

impl<D: DataRef> LWECiphertextToRef for LWE<D> {
    fn to_ref(&self) -> LWE<&[u8]> {
        LWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_ref(),
        }
    }
}

pub trait LWEToMut {
    #[allow(dead_code)]
    fn to_mut(&mut self) -> LWE<&mut [u8]>;
}

impl<D: DataMut> LWEToMut for LWE<D> {
    fn to_mut(&mut self) -> LWE<&mut [u8]> {
        LWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_mut(),
        }
    }
}

impl<D: DataMut> ReaderFrom for LWE<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWE<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        self.data.write_to(writer)
    }
}
