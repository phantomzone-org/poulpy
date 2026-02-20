use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{
        Backend, Data, DataMut, DataRef, FillUniform, MatZnx, MatZnxToMut, MatZnxToRef, Module, ReaderFrom, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWEToMut, GLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{GLWECompressed, GLWEDecompress},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

/// Seed-compressed GGLWE (gadget GLWE) ciphertext layout.
///
/// Stores only the body components of a [`GGLWE`] ciphertext matrix;
/// the mask polynomials are regenerated deterministically from 32-byte
/// PRNG seeds during decompression.
#[derive(PartialEq, Eq, Clone)]
pub struct GGLWECompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) rank_out: Rank,
    pub(crate) dsize: Dsize,
    pub(crate) seed: Vec<[u8; 32]>,
}

/// Provides mutable access to the PRNG seeds of a compressed GGLWE.
pub trait GGLWECompressedSeedMut {
    /// Returns a mutable reference to the vector of 32-byte PRNG seeds.
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]>;
}

impl<D: DataMut> GGLWECompressedSeedMut for GGLWECompressed<D> {
    fn seed_mut(&mut self) -> &mut Vec<[u8; 32]> {
        &mut self.seed
    }
}

/// Provides read access to the PRNG seeds of a compressed GGLWE.
pub trait GGLWECompressedSeed {
    /// Returns a reference to the vector of 32-byte PRNG seeds.
    fn seed(&self) -> &Vec<[u8; 32]>;
}

impl<D: DataRef> GGLWECompressedSeed for GGLWECompressed<D> {
    fn seed(&self) -> &Vec<[u8; 32]> {
        &self.seed
    }
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

impl GGLWECompressed<Vec<u8>> {
    /// Allocates a new compressed GGLWE by copying parameters from an existing info provider.
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Allocates a new compressed GGLWE with the given parameters.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank_in: Rank, rank_out: Rank, dnum: Dnum, dsize: Dsize) -> Self {
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
            data: MatZnx::alloc(n.into(), dnum.into(), rank_in.into(), 1, k.0.div_ceil(base2k.0) as usize),
            k,
            base2k,
            dsize,
            rank_out,
            seed: vec![[0u8; 32]; (dnum.0 * rank_in.0) as usize],
        }
    }

    /// Returns the serialized byte size by copying parameters from an existing info provider.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the serialized byte size for a compressed GGLWE with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank_in: Rank, dnum: Dnum, dsize: Dsize) -> usize {
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

        MatZnx::bytes_of(n.into(), dnum.into(), rank_in.into(), 1, k.0.div_ceil(base2k.0) as usize)
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

/// Trait for decompressing a [`GGLWECompressed`] into a standard [`GGLWE`].
///
/// Iterates over every (row, column) entry, decompressing each
/// compressed GLWE row individually via [`GLWEDecompress`].
pub trait GGLWEDecompress
where
    Self: GLWEDecompress,
{
    /// Decompresses `other` into `res`.
    fn decompress_gglwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToMut,
        O: GGLWECompressedToRef,
    {
        let res: &mut GGLWE<&mut [u8]> = &mut res.to_mut();
        let other: &GGLWECompressed<&[u8]> = &other.to_ref();

        assert_eq!(res.dsize(), other.dsize());
        assert!(res.dnum() <= other.dnum());

        let rank_in: usize = res.rank_in().into();
        let dnum: usize = res.dnum().into();
        for col_i in 0..rank_in {
            for row_i in 0..dnum {
                self.decompress_glwe(&mut res.at_mut(row_i, col_i), &other.at(row_i, col_i));
            }
        }
    }
}

impl<B: Backend> GGLWEDecompress for Module<B> where Self: VecZnxFillUniform + VecZnxCopy {}

impl<D: DataMut> GGLWE<D> {
    /// Decompresses a [`GGLWECompressed`] into this standard GGLWE.
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        O: GGLWECompressedToRef,
        M: GGLWEDecompress,
    {
        module.decompress_gglwe(self, other);
    }
}

/// Converts a compressed GGLWE to a mutably-borrowed variant.
pub trait GGLWECompressedToMut {
    /// Returns a mutably-borrowed view of this compressed GGLWE.
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

/// Converts a compressed GGLWE to an immutably-borrowed variant.
pub trait GGLWECompressedToRef {
    /// Returns an immutably-borrowed view of this compressed GGLWE.
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
