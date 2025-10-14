use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, MatZnx, MatZnxToMut, MatZnxToRef, Module, ReaderFrom, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToMut, GLWEInfos, GetDegree, LWEInfos, Rank, TorusPrecision,
    compressed::{GLWECompressed, GLWEDecompress},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWECompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) rank_out: Rank,
    pub(crate) dsize: Dsize,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl<D: Data> LWEInfos for GGLWECompressed<D> {
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
impl<D: Data> GLWEInfos for GGLWECompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWECompressed<D> {
    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: DataRef> fmt::Debug for GGLWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWECompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GGLWECompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWECompressed: base2k={} k={} dsize={}) {}",
            self.base2k.0, self.k.0, self.dsize.0, self.data
        )
    }
}

pub trait GGLWECompressedAlloc
where
    Self: GetDegree,
{
    fn alloc_gglwe_compressed(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> GGLWECompressed<Vec<u8>> {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        GGLWECompressed {
            data: MatZnx::alloc(
                self.n().into(),
                dnum.into(),
                rank_in.into(),
                1,
                k.0.div_ceil(base2k.0) as usize,
            ),
            k,
            base2k,
            dsize,
            rank_out,
            seed: vec![[0u8; 32]; (dnum.0 * rank_in.0) as usize],
        }
    }

    fn alloc_gglwe_compressed_from_infos<A>(&self, infos: &A) -> GGLWECompressed<Vec<u8>>
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.n(), self.n());
        self.alloc_gglwe_compressed(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    fn bytes_of_gglwe_compressed(&self, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        debug_assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        MatZnx::bytes_of(
            self.n().into(),
            dnum.into(),
            rank_in.into(),
            1,
            k.0.div_ceil(base2k.0) as usize,
        )
    }

    fn bytes_of_gglwe_compressed_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.n(), self.n());
        self.bytes_of_gglwe_compressed(
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.dnum(),
            infos.dsize(),
        )
    }
}

impl<B: Backend> GGLWECompressedAlloc for Module<B> where Self: GetDegree {}

impl GGLWECompressed<Vec<u8>> {
    pub fn alloc_from_infos<A, B: Backend>(module: &Module<B>, infos: &A) -> Self
    where
        A: GGLWEInfos,
        Module<B>: GGLWECompressedAlloc,
    {
        module.alloc_gglwe_compressed_from_infos(infos)
    }

    pub fn alloc<B: Backend>(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self
    where
        Module<B>: GGLWECompressedAlloc,
    {
        module.alloc_gglwe_compressed(base2k, k, rank_in, rank_out, dnum, dsize)
    }

    pub fn bytes_of_from_infos<A, B: Backend>(module: &Module<B>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<B>: GGLWECompressedAlloc,
    {
        module.bytes_of_gglwe_compressed_from_infos(infos)
    }

    pub fn byte_of<B: Backend>(
        module: &Module<B>,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize
    where
        Module<B>: GGLWECompressedAlloc,
    {
        module.bytes_of_gglwe_compressed(base2k, k, rank_in, dnum, dsize)
    }
}

impl<D: DataRef> GGLWECompressed<D> {
    pub(crate) fn at(&self, row: usize, col: usize) -> GLWECompressed<&[u8]> {
        let rank_in: usize = self.rank_in().into();
        GLWECompressed {
            data: self.data.at(row, col),
            k: self.k,
            base2k: self.base2k,
            rank: self.rank_out,
            seed: self.seed[rank_in * row + col],
        }
    }
}

impl<D: DataMut> GGLWECompressed<D> {
    pub(crate) fn at_mut(&mut self, row: usize, col: usize) -> GLWECompressed<&mut [u8]> {
        let rank_in: usize = self.rank_in().into();
        GLWECompressed {
            k: self.k,
            base2k: self.base2k,
            rank: self.rank_out,
            data: self.data.at_mut(row, col),
            seed: self.seed[rank_in * row + col], // Warning: value is copied and not borrow mut
        }
    }
}

impl<D: DataMut> ReaderFrom for GGLWECompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.rank_out = Rank(reader.read_u32::<LittleEndian>()?);
        let seed_len: u32 = reader.read_u32::<LittleEndian>()?;
        self.seed = vec![[0u8; 32]; seed_len as usize];
        for s in &mut self.seed {
            reader.read_exact(s)?;
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWECompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.dsize.into())?;
        writer.write_u32::<LittleEndian>(self.rank_out.into())?;
        writer.write_u32::<LittleEndian>(self.seed.len() as u32)?;
        for s in &self.seed {
            writer.write_all(s)?;
        }
        self.data.write_to(writer)
    }
}

pub trait GGLWEDecompress
where
    Self: GLWEDecompress,
{
    fn decompress_gglwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToMut,
        O: GGLWECompressedToRef,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let other: &GGLWECompressed<&[u8]> = &other.to_ref();

        assert_eq!(res.gglwe_layout(), other.gglwe_layout());

        let rank_in: usize = res.rank_in().into();
        let dnum: usize = res.dnum().into();

        for row_i in 0..dnum {
            for col_i in 0..rank_in {
                self.decompress_glwe(&mut res.at_mut(row_i, col_i), &other.at(row_i, col_i));
            }
        }
    }
}

impl<B: Backend> GGLWEDecompress for Module<B> where Self: VecZnxFillUniform + VecZnxCopy {}

impl<D: DataMut> GGLWE<D> {
    pub fn decompress<O, B: Backend>(&mut self, module: &Module<B>, other: &O)
    where
        O: GGLWECompressedToRef,
        Module<B>: GGLWEDecompress,
    {
        module.decompress_gglwe(self, other);
    }
}

pub trait GGLWECompressedToMut {
    fn to_mut(&mut self) -> GGLWECompressed<&mut [u8]>;
}

impl<D: DataMut> GGLWECompressedToMut for GGLWECompressed<D> {
    fn to_mut(&mut self) -> GGLWECompressed<&mut [u8]> {
        GGLWECompressed {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            seed: self.seed.clone(),
            rank_out: self.rank_out,
            data: self.data.to_mut(),
        }
    }
}

pub trait GGLWECompressedToRef {
    fn to_ref(&self) -> GGLWECompressed<&[u8]>;
}

impl<D: DataRef> GGLWECompressedToRef for GGLWECompressed<D> {
    fn to_ref(&self) -> GGLWECompressed<&[u8]> {
        GGLWECompressed {
            k: self.k(),
            base2k: self.base2k(),
            dsize: self.dsize(),
            seed: self.seed.clone(),
            rank_out: self.rank_out,
            data: self.data.to_ref(),
        }
    }
}
