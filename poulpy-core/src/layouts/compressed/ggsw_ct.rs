use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, MatZnx, MatZnxToMut, MatZnxToRef, Module, ReaderFrom, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSWCiphertext, GGSWInfos, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{Decompress, GLWECiphertextCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCiphertextCompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
    pub(crate) rank: Rank,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl<D: Data> LWEInfos for GGSWCiphertextCompressed<D> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
    fn size(&self) -> usize {
        self.data.size()
    }
}
impl<D: Data> GLWEInfos for GGSWCiphertextCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data> GGSWInfos for GGSWCiphertextCompressed<D> {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: DataRef> fmt::Debug for GGSWCiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: DataRef> fmt::Display for GGSWCiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSWCiphertextCompressed: base2k={} k={} dsize={}) {}",
            self.base2k, self.k, self.dsize, self.data
        )
    }
}

impl<D: DataMut> FillUniform for GGSWCiphertextCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GGSWCiphertextCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid ggsw: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        Self {
            data: MatZnx::alloc(
                n.into(),
                dnum.into(),
                (rank + 1).into(),
                1,
                k.0.div_ceil(base2k.0) as usize,
            ),
            k,
            base2k,
            dsize,
            rank,
            seed: Vec::new(),
        }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        Self::alloc_bytes_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid ggsw: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        MatZnx::alloc_bytes(
            n.into(),
            dnum.into(),
            (rank + 1).into(),
            1,
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

impl<D: DataRef> GGSWCiphertextCompressed<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECiphertextCompressed<&[u8]> {
        let rank: usize = self.rank().into();
        GLWECiphertextCompressed {
            data: self.data.at(row, col),
            k: self.k,
            base2k: self.base2k,
            rank: self.rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: DataMut> GGSWCiphertextCompressed<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertextCompressed<&mut [u8]> {
        let rank: usize = self.rank().into();
        GLWECiphertextCompressed {
            data: self.data.at_mut(row, col),
            k: self.k,
            base2k: self.base2k,
            rank: self.rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: DataMut> ReaderFrom for GGSWCiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.rank = Rank(reader.read_u32::<LittleEndian>()?);
        let seed_len: usize = reader.read_u32::<LittleEndian>()? as usize;
        self.seed = vec![[0u8; 32]; seed_len];
        for s in &mut self.seed {
            reader.read_exact(s)?;
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGSWCiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.dsize.into())?;
        writer.write_u32::<LittleEndian>(self.rank.into())?;
        writer.write_u32::<LittleEndian>(self.seed.len() as u32)?;
        for s in &self.seed {
            writer.write_all(s)?;
        }
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GGSWCiphertextCompressed<DR>> for GGSWCiphertext<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GGSWCiphertextCompressed<DR>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), other.rank())
        }

        let dnum: usize = self.dnum().into();
        let rank: usize = self.rank().into();
        (0..dnum).for_each(|row_i| {
            (0..rank + 1).for_each(|col_j| {
                self.at_mut(row_i, col_j)
                    .decompress(module, &other.at(row_i, col_j));
            });
        });
    }
}

pub trait GGSWCiphertextCompressedToMut {
    fn to_mut(&mut self) -> GGSWCiphertextCompressed<&mut [u8]>;
}

impl<D: DataMut> GGSWCiphertextCompressedToMut for GGSWCiphertextCompressed<D> {
    fn to_mut(&mut self) -> GGSWCiphertextCompressed<&mut [u8]> {
        GGSWCiphertextCompressed {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: self.data.to_mut(),
        }
    }
}

pub trait GGSWCiphertextCompressedToRef {
    fn to_ref(&self) -> GGSWCiphertextCompressed<&[u8]>;
}

impl<D: DataRef> GGSWCiphertextCompressedToRef for GGSWCiphertextCompressed<D> {
    fn to_ref(&self) -> GGSWCiphertextCompressed<&[u8]> {
        GGSWCiphertextCompressed {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: self.data.to_ref(),
        }
    }
}
