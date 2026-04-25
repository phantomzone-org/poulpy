use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWECompressed, GGLWECompressedBackendMut, GGLWECompressedToBackendMut,
    GGLWECompressedToBackendRef, GGLWECompressedToMut, GGLWECompressedToRef, GGLWEDecompress, GGLWEInfos,
    GGLWEToGGSWKeyToBackendMut, GLWEInfos, LWEInfos, Rank, TorusPrecision,
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

pub type GGLWEToGGSWKeyCompressedBackendRef<'a, BE> = GGLWEToGGSWKeyCompressed<<BE as Backend>::BufRef<'a>>;
pub type GGLWEToGGSWKeyCompressedBackendMut<'a, BE> = GGLWEToGGSWKeyCompressed<<BE as Backend>::BufMut<'a>>;

impl<D: Data> LWEInfos for GGLWEToGGSWKeyCompressed<D> {
    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
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

impl<D: HostDataRef> fmt::Debug for GGLWEToGGSWKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for GGLWEToGGSWKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWECompressed<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: HostDataRef> fmt::Display for GGLWEToGGSWKeyCompressed<D> {
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
            infos.max_k(),
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
            infos.max_k(),
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

impl<D: HostDataMut> GGLWEToGGSWKeyCompressed<D> {
    // Returns a mutable reference to GGLWE_{s}([s[i]*s[0], s[i]*s[1], ..., s[i]*s[rank]])
    pub fn at_mut(&mut self, i: usize) -> &mut GGLWECompressed<D> {
        assert!((i as u32) < self.rank());
        &mut self.keys[i]
    }
}

impl<D: HostDataRef> GGLWEToGGSWKeyCompressed<D> {
    // Returns a reference to GGLWE_{s}(s[i] * s[j])
    pub fn at(&self, i: usize) -> &GGLWECompressed<D> {
        assert!((i as u32) < self.rank());
        &self.keys[i]
    }
}

impl<D: HostDataMut> ReaderFrom for GGLWEToGGSWKeyCompressed<D> {
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

impl<D: HostDataRef> WriterTo for GGLWEToGGSWKeyCompressed<D> {
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
        R: GGLWEToGGSWKeyToBackendMut<Self::Backend>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<Self::Backend>,
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();

        assert_eq!(res.keys.len(), other.keys.len());

        for (a, b) in res.keys.iter_mut().zip(other.keys.iter()) {
            let mut a_ref = a;
            let b_ref = b;
            self.decompress_gglwe(&mut a_ref, &b_ref);
        }
    }
}

// module-only API: decompression is provided by `GGLWEToGGSWKeyDecompress` on `Module`.

/// Converts a compressed GGLWE-to-GGSW key to an immutably-borrowed variant.
pub trait GGLWEToGGSWKeyCompressedToRef {
    /// Returns an immutably-borrowed view.
    fn to_ref(&self) -> GGLWEToGGSWKeyCompressed<&[u8]>;
}

impl<D: HostDataRef> GGLWEToGGSWKeyCompressedToRef for GGLWEToGGSWKeyCompressed<D>
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

impl<D: HostDataMut> GGLWEToGGSWKeyCompressedToMut for GGLWEToGGSWKeyCompressed<D>
where
    GGLWECompressed<D>: GGLWECompressedToMut,
{
    fn to_mut(&mut self) -> GGLWEToGGSWKeyCompressed<&mut [u8]> {
        GGLWEToGGSWKeyCompressed {
            keys: self.keys.iter_mut().map(|c| c.to_mut()).collect(),
        }
    }
}

pub trait GGLWEToGGSWKeyCompressedToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyCompressedBackendRef<'_, BE>;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedToBackendRef<BE> for GGLWEToGGSWKeyCompressed<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GGLWEToGGSWKeyCompressedBackendRef<'_, BE> {
        GGLWEToGGSWKeyCompressed {
            keys: self
                .keys
                .iter()
                .map(GGLWECompressedToBackendRef::<BE>::to_backend_ref)
                .collect(),
        }
    }
}

pub trait GGLWEToGGSWKeyCompressedToBackendMut<BE: Backend>: GGLWEToGGSWKeyCompressedToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyCompressedBackendMut<'_, BE>;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedToBackendMut<BE> for GGLWEToGGSWKeyCompressed<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GGLWEToGGSWKeyCompressedBackendMut<'_, BE> {
        GGLWEToGGSWKeyCompressed {
            keys: self
                .keys
                .iter_mut()
                .map(GGLWECompressedToBackendMut::<BE>::to_backend_mut)
                .collect(),
        }
    }
}

pub trait GGLWEToGGSWKeyCompressedAtBackendMut<BE: Backend> {
    fn at_backend_mut(&mut self, i: usize) -> GGLWECompressedBackendMut<'_, BE>;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedAtBackendMut<BE> for GGLWEToGGSWKeyCompressed<BE::OwnedBuf> {
    fn at_backend_mut(&mut self, i: usize) -> GGLWECompressedBackendMut<'_, BE> {
        assert!((i as u32) < self.rank());
        <GGLWECompressed<BE::OwnedBuf> as GGLWECompressedToBackendMut<BE>>::to_backend_mut(&mut self.keys[i])
    }
}

pub fn gglwe_to_ggsw_key_compressed_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    key: &'a mut GGLWEToGGSWKeyCompressed<BE::BufMut<'b>>,
    i: usize,
) -> GGLWECompressedBackendMut<'a, BE> {
    assert!((i as u32) < key.rank());
    let key_i = &mut key.keys[i];
    GGLWECompressed {
        k: key_i.k,
        base2k: key_i.base2k,
        dsize: key_i.dsize,
        seed: key_i.seed.clone(),
        rank_out: key_i.rank_out,
        data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut key_i.data),
    }
}
