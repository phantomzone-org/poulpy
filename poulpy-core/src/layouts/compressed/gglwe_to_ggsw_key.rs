use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedToMut, GGLWECompressedToRef, GGLWEDecompress, GGLWEInfos,
    GGLWEToGGSWKey, GGLWEToGGSWKeyToMut, GLWEInfos, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

/// Seed-compressed GGLWE-to-GGSW key-switching key layout.
///
/// A vector of [`GGLWECompressed`] entries, one per rank element,
/// used for GGLWE-to-GGSW conversion. The mask of each GGLWE is
/// regenerated from its PRNG seed during decompression.
#[derive(PartialEq, Eq, Clone)]
pub struct GGLWEToGGSWKeyCompressed<D: Data> {
    pub(crate) keys: Vec<GGLWECompressed<D>>,
}

impl<D: Data> LWEInfos for GGLWEToGGSWKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }

    fn size(&self) -> usize {
        self.keys[0].size()
    }
}

impl<D: Data> GLWEInfos for GGLWEToGGSWKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.keys[0].rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWEToGGSWKeyCompressed<D> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn dsize(&self) -> Dsize {
        self.keys[0].dsize()
    }

    fn dnum(&self) -> Dnum {
        self.keys[0].dnum()
    }
}

impl<D: DataRef> fmt::Debug for GGLWEToGGSWKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWEToGGSWKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWECompressed<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: DataRef> fmt::Display for GGLWEToGGSWKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GGLWEToGGSWKeyCompressed)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

impl GGLWEToGGSWKeyCompressed<Vec<u8>> {
    /// Allocates a new compressed GGLWE-to-GGSW key by copying parameters from an existing info provider.
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyCompressed"
        );
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Allocates a new compressed GGLWE-to-GGSW key with the given parameters.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        GGLWEToGGSWKeyCompressed {
            keys: (0..rank.as_usize())
                .map(|_| GGLWECompressed::alloc(n, base2k, k, rank, rank, dnum, dsize))
                .collect(),
        }
    }

    /// Returns the serialized byte size by copying parameters from an existing info provider.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyCompressed"
        );
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    /// Returns the serialized byte size for a compressed GGLWE-to-GGSW key with the given parameters.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        rank.as_usize() * GGLWECompressed::bytes_of(n, base2k, k, rank, dnum, dsize)
    }
}

impl<D: DataMut> GGLWEToGGSWKeyCompressed<D> {
    // Returns a mutable reference to GGLWE_{s}([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWECompressed<D> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: DataRef> GGLWEToGGSWKeyCompressed<D> {
    // Returns a reference to GGLWE_{s}(s[i] * s[j])
    pub fn at(&self, i: usize) -> &GGLWECompressed<D> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

impl<D: DataMut> ReaderFrom for GGLWEToGGSWKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;
        if self.keys.len() != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("self.keys.len()={} != read len={}", self.keys.len(), len),
            ));
        }
        for key in &mut self.keys {
            key.read_from(reader)?;
        }
        Ok(())
    }
}

impl<D: DataRef> WriterTo for GGLWEToGGSWKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

/// Trait for decompressing a [`GGLWEToGGSWKeyCompressed`] into a standard [`GGLWEToGGSWKey`].
pub trait GGLWEToGGSWKeyDecompress
where
    Self: GGLWEDecompress,
{
    /// Decompresses `other` into `res` by decompressing each GGLWE entry.
    fn decompress_gglwe_to_ggsw_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToMut,
        O: GGLWEToGGSWKeyCompressedToRef,
    {
        let res: &mut GGLWEToGGSWKey<&mut [u8]> = &mut res.to_mut();
        let other: &GGLWEToGGSWKeyCompressed<&[u8]> = &other.to_ref();

        assert_eq!(res.keys.len(), other.keys.len());

        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            self.decompress_gglwe(a, b);
        }
    }
}

impl<D: DataMut> GGLWEToGGSWKey<D> {
    /// Decompresses a [`GGLWEToGGSWKeyCompressed`] into this standard key.
    pub fn decompress<O, M>(&mut self, module: &M, other: &O)
    where
        M: GGLWEToGGSWKeyDecompress,
        O: GGLWEToGGSWKeyCompressedToRef,
    {
        module.decompress_gglwe_to_ggsw_key(self, other);
    }
}

/// Converts a compressed GGLWE-to-GGSW key to an immutably-borrowed variant.
pub trait GGLWEToGGSWKeyCompressedToRef {
    /// Returns an immutably-borrowed view.
    fn to_ref(&self) -> GGLWEToGGSWKeyCompressed<&[u8]>;
}

impl<D: DataRef> GGLWEToGGSWKeyCompressedToRef for GGLWEToGGSWKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToRef,
{
    fn to_ref(&self) -> GGLWEToGGSWKeyCompressed<&[u8]> {
        GGLWEToGGSWKeyCompressed {
            keys: self.keys.iter().map(|c| c.to_ref()).collect(),
        }
    }
}

/// Converts a compressed GGLWE-to-GGSW key to a mutably-borrowed variant.
pub trait GGLWEToGGSWKeyCompressedToMut {
    /// Returns a mutably-borrowed view.
    fn to_mut(&mut self) -> GGLWEToGGSWKeyCompressed<&mut [u8]>;
}

impl<D: DataMut> GGLWEToGGSWKeyCompressedToMut for GGLWEToGGSWKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToMut,
{
    fn to_mut(&mut self) -> GGLWEToGGSWKeyCompressed<&mut [u8]> {
        GGLWEToGGSWKeyCompressed {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}
