use poulpy_hal::{
    layouts::{
        Data, DataMut, DataRef, FillUniform, ReaderFrom, ToOwnedDeep, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo, ZnxInfos,
    },
    source::Source,
};

use crate::layouts::{Base2K, Degree, LWEInfos, Rank, SetLWEInfos, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

/// Trait providing the parameter accessors for a GLWE (Generalised LWE) ciphertext.
///
/// A GLWE ciphertext is a polynomial-ring LWE ciphertext consisting of
/// a body polynomial and `rank` mask polynomials, all defined over `Z[X]/(X^n + 1)`.
/// Extends [`LWEInfos`] with the GLWE rank.
pub trait GLWEInfos
where
    Self: LWEInfos,
{
    /// Returns the GLWE rank (number of mask polynomials).
    fn rank(&self) -> Rank;
    /// Returns a plain-data [`GLWELayout`] snapshot of the current parameters.
    fn glwe_layout(&self) -> GLWELayout {
        GLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.max_k(),
            rank: self.rank(),
        }
    }
}

/// Plain-data snapshot of the parameters that describe a [`GLWE`] ciphertext.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWELayout {
    /// Ring degree.
    pub n: Degree,
    /// Base-2-log of the limb width.
    pub base2k: Base2K,
    /// Torus precision.
    pub k: TorusPrecision,
    /// Number of mask polynomials.
    pub rank: Rank,
}

impl LWEInfos for GLWELayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GLWELayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

/// A GLWE (Generalised LWE) ciphertext over the polynomial ring `Z[X]/(X^n + 1)`.
///
/// Wraps a [`VecZnx`] with `rank + 1` columns: the first column is the body
/// polynomial, and the remaining `rank` columns are the mask polynomials.
///
/// `D: Data` is the storage backend (e.g. `Vec<u8>`, `&[u8]`, `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GLWE<D: Data, M = ()> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
    pub meta: M,
}

impl<D: DataMut, M> SetLWEInfos for GLWE<D, M> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: DataRef, M> GLWE<D, M> {
    /// Returns a shared reference to the underlying [`VecZnx`].
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: Data, M> GLWE<D, M> {
    /// Returns the allocated limb capacity, which can exceed the active `size()`
    /// after a precision-consuming rescale.
    pub fn max_size(&self) -> usize {
        self.data.max_size
    }
}

impl<D: DataMut, M> GLWE<D, M> {
    /// Returns a mutable reference to the underlying [`VecZnx`].
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: Data, M> LWEInfos for GLWE<D, M> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data, M> GLWEInfos for GLWE<D, M> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: DataRef, M: Clone> ToOwnedDeep for GLWE<D, M> {
    type Owned = GLWE<Vec<u8>, M>;
    fn to_owned_deep(&self) -> Self::Owned {
        GLWE {
            data: self.data.to_owned_deep(),
            base2k: self.base2k,
            meta: self.meta.clone(),
        }
    }
}

impl<D: DataRef, M> fmt::Debug for GLWE<D, M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataRef, M> fmt::Display for GLWE<D, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWE: base2k={} k={}: {}", self.base2k().0, self.max_k().0, self.data)
    }
}

impl<D: DataMut, M> FillUniform for GLWE<D, M> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<M> GLWE<Vec<u8>, M> {
    /// Allocates a new [`GLWE`] with the given parameters.
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
        M: Default,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k(), infos.rank())
    }

    /// Allocates a new [`GLWE`] with the given parameters.
    ///
    /// * `n` -- ring degree.
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    /// * `rank` -- number of mask polynomials.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self
    where
        M: Default,
    {
        GLWE {
            data: VecZnx::alloc(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
            meta: M::default(),
        }
    }

    pub fn alloc_with_meta(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, meta: M) -> Self {
        GLWE {
            data: VecZnx::alloc(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize),
            base2k,
            meta,
        }
    }

    /// Returns the byte count required for a [`GLWE`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k(), infos.rank())
    }

    /// Returns the byte count required for a [`GLWE`] with the given parameters.
    ///
    /// * `n` -- ring degree.
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    /// * `rank` -- number of mask polynomials.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        VecZnx::bytes_of(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }

    /// Reallocates the backing buffer so capacity matches `size` limb count.
    pub fn reallocate_limbs(&mut self, size: usize) {
        self.data.reallocate_limbs(size);
    }
}

impl<D: DataMut, M: Default> ReaderFrom for GLWE<D, M> {
    /// Deserialises a [`GLWE`] in little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)?;
        self.meta = M::default();
        Ok(())
    }
}

impl<D: DataRef, M> WriterTo for GLWE<D, M> {
    /// Serialises the [`GLWE`] in little-endian binary format.
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        self.data.write_to(writer)
    }
}

/// Trait for borrowing a [`GLWE`] as an immutable reference.
pub trait GLWEToRef: Sized {
    /// Borrows the data as `&[u8]`.
    fn to_ref(&self) -> GLWE<&[u8]>;
}

impl<D: DataRef, M> GLWEToRef for GLWE<D, M> {
    fn to_ref(&self) -> GLWE<&[u8]> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_ref(),
            meta: (),
        }
    }
}

/// Trait for borrowing a [`GLWE`] as a mutable reference.
pub trait GLWEToMut: GLWEToRef {
    /// Borrows the data as `&mut [u8]`.
    fn to_mut(&mut self) -> GLWE<&mut [u8]>;
}

impl<D: DataMut, M> GLWEToMut for GLWE<D, M> {
    fn to_mut(&mut self) -> GLWE<&mut [u8]> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_mut(),
            meta: (),
        }
    }
}
