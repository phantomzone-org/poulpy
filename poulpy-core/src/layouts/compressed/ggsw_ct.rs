use poulpy_hal::{
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, MatZnx, MatZnxToMut, MatZnxToRef, Module, ReaderFrom, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSW, GGSWInfos, GGSWToMut, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision,
    compressed::{GLWECompressed, GLWEDecompress},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGSWCompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
    pub(crate) rank: Rank,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl<D: Data> LWEInfos for GGSWCompressed<D> {
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
impl<D: Data> GLWEInfos for GGSWCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl<D: Data> GGSWInfos for GGSWCompressed<D> {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: DataRef> fmt::Debug for GGSWCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: DataRef> fmt::Display for GGSWCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSWCompressed: base2k={} k={} dsize={}) {}",
            self.base2k, self.k, self.dsize, self.data
        )
    }
}

impl<D: DataMut> FillUniform for GGSWCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

pub trait GGSWCompressedAlloc
where
    Self: GetDegree,
{
    fn alloc_ggsw_compressed(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGSWCompressed<Vec<u8>> {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
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

        GGSWCompressed {
            data: MatZnx::alloc(
                self.n().into(),
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

    fn alloc_ggsw_compressed_from_infos<A>(&self, infos: &A) -> GGSWCompressed<Vec<u8>>
    where
        A: GGSWInfos,
    {
        self.alloc_ggsw_compressed(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_ggsw_compressed(&self, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
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

        MatZnx::bytes_of(
            self.n().into(),
            dnum.into(),
            (rank + 1).into(),
            1,
            k.0.div_ceil(base2k.0) as usize,
        )
    }

    fn bytes_of_ggsw_compressed_key_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        self.bytes_of_ggsw_compressed(
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl GGSWCompressed<Vec<u8>> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> Self
    where
        A: GGSWInfos,
        M: GGSWCompressedAlloc,
    {
        module.alloc_ggsw_compressed_from_infos(infos)
    }

    pub fn alloc<M>(module: &M, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self
    where
        M: GGSWCompressedAlloc,
    {
        module.alloc_ggsw_compressed(base2k, k, rank, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, M>(module: &M, infos: &A) -> usize
    where
        A: GGSWInfos,
        M: GGSWCompressedAlloc,
    {
        module.bytes_of_ggsw_compressed_key_from_infos(infos)
    }

    pub fn bytes_of<M>(
        module: &M,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        M: GGSWCompressedAlloc,
    {
        module.bytes_of_ggsw_compressed(base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataRef> GGSWCompressed<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWECompressed<&[u8]> {
        let rank: usize = self.rank().into();
        GLWECompressed {
            data: self.data.at(row, col),
            k: self.k,
            base2k: self.base2k,
            rank: self.rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: DataMut> GGSWCompressed<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWECompressed<&mut [u8]> {
        let rank: usize = self.rank().into();
        GLWECompressed {
            data: self.data.at_mut(row, col),
            k: self.k,
            base2k: self.base2k,
            rank: self.rank,
            seed: self.seed[row * (rank + 1) + col],
        }
    }
}

impl<D: DataMut> ReaderFrom for GGSWCompressed<D> {
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

impl<D: DataRef> WriterTo for GGSWCompressed<D> {
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

pub trait GGSWDecompress
where
    Self: GLWEDecompress,
{
    fn decompress_ggsw<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGSWToMut,
        O: GGSWCompressedToRef,
    {
        let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
        let other: &GGSWCompressed<&[u8]> = &other.to_ref();

        assert_eq!(res.rank(), other.rank());
        let dnum: usize = res.dnum().into();
        let rank: usize = res.rank().into();

        for row_i in 0..dnum {
            for col_j in 0..rank + 1 {
                self.decompress_glwe(&mut res.at_mut(row_i, col_j), &other.at(row_i, col_j));
            }
        }
    }
}

impl<B: Backend> GGSWDecompress for Module<B> where Self: GGSWDecompress {}

impl<D: DataMut> GGSW<D> {
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: GGSWCompressedToRef,
        M: GGSWDecompress,
    {
        module.decompress_ggsw(self, other);
    }
}

pub trait GGSWCompressedToMut {
    fn to_mut(&mut self) -> GGSWCompressed<&mut [u8]>;
}

impl<D: DataMut> GGSWCompressedToMut for GGSWCompressed<D> {
    fn to_mut(&mut self) -> GGSWCompressed<&mut [u8]> {
        GGSWCompressed {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: self.data.to_mut(),
        }
    }
}

pub trait GGSWCompressedToRef {
    fn to_ref(&self) -> GGSWCompressed<&[u8]>;
}

impl<D: DataRef> GGSWCompressedToRef for GGSWCompressed<D> {
    fn to_ref(&self) -> GGSWCompressed<&[u8]> {
        GGSWCompressed {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            rank: self.rank(),
            seed: self.seed.clone(),
            data: self.data.to_ref(),
        }
    }
}
