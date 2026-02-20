use std::fmt;

use poulpy_hal::{
    layouts::{Data, DataMut, DataRef, FillUniform, ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo, ZnxInfos},
    source::Source,
};

use crate::layouts::{Base2K, Degree, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Trait providing the parameter accessors for an LWE ciphertext.
///
/// An LWE ciphertext is a scalar (non-polynomial) ciphertext consisting of
/// a body `b` and a mask `(a_1, ..., a_n)`.
pub trait LWEInfos {
    /// Returns the LWE dimension, i.e. the number of mask elements (= GLWE ring degree N).
    fn n(&self) -> Degree;
    /// Returns `log2(n)`.
    fn log_n(&self) -> usize {
        self.n().log2()
    }
    /// Returns the torus precision `k` (number of significant bits).
    fn k(&self) -> TorusPrecision;
    /// Returns the maximum torus precision representable with the current limb decomposition.
    fn max_k(&self) -> TorusPrecision {
        TorusPrecision(self.base2k().0 * self.size() as u32)
    }
    /// Returns the base-2-log of the limb width used for the RNS/CRT representation.
    fn base2k(&self) -> Base2K;
    /// Returns the number of limbs, i.e. `ceil(k / base2k)`.
    fn size(&self) -> usize {
        self.k().0.div_ceil(self.base2k().0) as usize
    }
    /// Returns a plain-data [`LWELayout`] snapshot of the current parameters.
    fn lwe_layout(&self) -> LWELayout {
        LWELayout {
            n: self.n(),
            k: self.k(),
            base2k: self.base2k(),
        }
    }
}

/// Trait for mutating LWE parameters in place.
pub trait SetLWEInfos {
    /// Sets the torus precision `k`.
    fn set_k(&mut self, k: TorusPrecision);
    /// Sets the limb width `base2k`.
    fn set_base2k(&mut self, base2k: Base2K);
}

/// Plain-data snapshot of the parameters that describe an [`LWE`] ciphertext.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWELayout {
    /// Ring degree (LWE dimension).
    pub n: Degree,
    /// Torus precision.
    pub k: TorusPrecision,
    /// Base-2-log of the limb width.
    pub base2k: Base2K,
}

impl LWEInfos for LWELayout {
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
/// A scalar (non-polynomial) LWE ciphertext.
///
/// Stored as a single-column [`VecZnx`] of dimension `n + 1` where
/// the body `b` and the mask `(a_1, ..., a_n)` are packed together.
/// When `rank = 0` the mask column is embedded in the single [`VecZnx`] column.
///
/// `D: Data` is the storage backend (e.g. `Vec<u8>`, `&[u8]`, `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct LWE<D: Data> {
    pub(crate) data: VecZnx<D>,
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
    /// Returns a shared reference to the underlying [`VecZnx`].
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: DataMut> LWE<D> {
    /// Returns a mutable reference to the underlying [`VecZnx`].
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
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
        write!(f, "LWE: base2k={} k={}: {}", self.base2k().0, self.k().0, self.data)
    }
}

impl<D: DataMut> FillUniform for LWE<D>
where
    VecZnx<D>: FillUniform,
{
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl LWE<Vec<u8>> {
    /// Allocates a new [`LWE`] with the given parameters.
    pub fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k())
    }

    /// Allocates a new [`LWE`] with the given parameters.
    ///
    /// * `n` -- ring degree (LWE dimension).
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    pub fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        LWE {
            data: VecZnx::alloc((n + 1).into(), 1, k.0.div_ceil(base2k.0) as usize),
            k,
            base2k,
        }
    }

    /// Returns the byte count required for an [`LWE`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: LWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k())
    }

    /// Returns the byte count required for an [`LWE`] with the given parameters.
    ///
    /// * `n` -- ring degree (LWE dimension).
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::bytes_of((n + 1).into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

/// Trait for borrowing an [`LWE`] as an immutable reference.
pub trait LWEToRef {
    /// Borrows the data as `&[u8]`.
    fn to_ref(&self) -> LWE<&[u8]>;
}

impl<D: DataRef> LWEToRef for LWE<D> {
    fn to_ref(&self) -> LWE<&[u8]> {
        LWE {
            k: self.k,
            base2k: self.base2k,
            data: self.data.to_ref(),
        }
    }
}

/// Trait for borrowing an [`LWE`] as a mutable reference.
pub trait LWEToMut {
    /// Borrows the data as `&mut [u8]`.
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
    /// Deserialises an [`LWE`] in little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for LWE<D> {
    /// Serialises the [`LWE`] in little-endian binary format.
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        self.data.write_to(writer)
    }
}
