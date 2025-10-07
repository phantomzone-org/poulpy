use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, MatZnx, Module, ReaderFrom, WriterTo, ZnxInfos},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dsize, GGLWECiphertext, GGLWEInfos, GLWEInfos, LWEInfos, Rank, Dnum, TorusPrecision,
    compressed::{Decompress, GLWECiphertextCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWECiphertextCompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) rank_out: Rank,
    pub(crate) dsize: Dsize,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl<D: Data> LWEInfos for GGLWECiphertextCompressed<D> {
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
impl<D: Data> GLWEInfos for GGLWECiphertextCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWECiphertextCompressed<D> {
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

impl<D: DataRef> fmt::Debug for GGLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWECiphertextCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: DataRef> fmt::Display for GGLWECiphertextCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWECiphertextCompressed: base2k={} k={} dsize={}) {}",
            self.base2k.0, self.k.0, self.dsize.0, self.data
        )
    }
}

impl GGLWECiphertextCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self {
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

        Self {
            data: MatZnx::alloc(
                n.into(),
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

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::alloc_bytes_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rank_in(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn alloc_bytes_with(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
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

        MatZnx::alloc_bytes(
            n.into(),
            dnum.into(),
            rank_in.into(),
            1,
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

impl<D: DataRef> GGLWECiphertextCompressed<D> {
    pub(crate) fn at(&self, row: usize, col: usize) -> GLWECiphertextCompressed<&[u8]> {
        let rank_in: usize = self.rank_in().into();
        GLWECiphertextCompressed {
            data: self.data.at(row, col),
            k: self.k,
            base2k: self.base2k,
            rank: self.rank_out,
            seed: self.seed[rank_in * row + col],
        }
    }
}

impl<D: DataMut> GGLWECiphertextCompressed<D> {
    pub(crate) fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertextCompressed<&mut [u8]> {
        let rank_in: usize = self.rank_in().into();
        GLWECiphertextCompressed {
            k: self.k,
            base2k: self.base2k,
            rank: self.rank_out,
            data: self.data.at_mut(row, col),
            seed: self.seed[rank_in * row + col], // Warning: value is copied and not borrow mut
        }
    }
}

impl<D: DataMut> ReaderFrom for GGLWECiphertextCompressed<D> {
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

impl<D: DataRef> WriterTo for GGLWECiphertextCompressed<D> {
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

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GGLWECiphertextCompressed<DR>> for GGLWECiphertext<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GGLWECiphertextCompressed<DR>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.n(),
                other.n(),
                "invalid receiver: self.n()={} != other.n()={}",
                self.n(),
                other.n()
            );
            assert_eq!(
                self.size(),
                other.size(),
                "invalid receiver: self.size()={} != other.size()={}",
                self.size(),
                other.size()
            );
            assert_eq!(
                self.rank_in(),
                other.rank_in(),
                "invalid receiver: self.rank_in()={} != other.rank_in()={}",
                self.rank_in(),
                other.rank_in()
            );
            assert_eq!(
                self.rank_out(),
                other.rank_out(),
                "invalid receiver: self.rank_out()={} != other.rank_out()={}",
                self.rank_out(),
                other.rank_out()
            );

            assert_eq!(
                self.dnum(),
                other.dnum(),
                "invalid receiver: self.dnum()={} != other.dnum()={}",
                self.dnum(),
                other.dnum()
            );
        }

        let rank_in: usize = self.rank_in().into();
        let dnum: usize = self.dnum().into();

        (0..rank_in).for_each(|col_i| {
            (0..dnum).for_each(|row_i| {
                self.at_mut(row_i, col_i)
                    .decompress(module, &other.at(row_i, col_i));
            });
        });
    }
}
