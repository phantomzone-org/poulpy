use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, Module, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Digits, GGLWELayoutInfos, GGLWETensorKey, GLWEInfos, LWEInfos, Rank, Rows, TorusPrecision,
    compressed::{Decompress, GGLWESwitchingKeyCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWETensorKeyCompressed<D: Data> {
    pub(crate) keys: Vec<GGLWESwitchingKeyCompressed<D>>,
}

impl<D: Data> LWEInfos for GGLWETensorKeyCompressed<D> {
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
impl<D: Data> GLWEInfos for GGLWETensorKeyCompressed<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWELayoutInfos for GGLWETensorKeyCompressed<D> {
    fn rank_in(&self) -> Rank {
        self.rank_out()
    }

    fn rank_out(&self) -> Rank {
        self.keys[0].rank_out()
    }

    fn digits(&self) -> Digits {
        self.keys[0].digits()
    }

    fn rows(&self) -> Rows {
        self.keys[0].rows()
    }
}

impl<D: DataRef> fmt::Debug for GGLWETensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: DataMut> FillUniform for GGLWETensorKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWESwitchingKeyCompressed<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: DataRef> fmt::Display for GGLWETensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKeyCompressed)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{i}: {key}")?;
        }
        Ok(())
    }
}

impl GGLWETensorKeyCompressed<Vec<u8>> {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GGLWELayoutInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyCompressed"
        );
        Self::alloc_with(
            infos.n(),
            infos.base2k(),
            infos.k(),
            infos.rows(),
            infos.digits(),
            infos.rank_out(),
        )
    }

    pub fn alloc_with(n: Degree, base2k: Base2K, k: TorusPrecision, rows: Rows, digits: Digits, rank: Rank) -> Self {
        let mut keys: Vec<GGLWESwitchingKeyCompressed<Vec<u8>>> = Vec::new();
        let pairs: u32 = (((rank.0 + 1) * rank.0) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GGLWESwitchingKeyCompressed::alloc_with(
                n,
                base2k,
                k,
                rows,
                digits,
                Rank(1),
                rank,
            ));
        });
        Self { keys }
    }

    pub fn alloc_bytes<A>(infos: &A) -> usize
    where
        A: GGLWELayoutInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyCompressed"
        );
        let rank_out: usize = infos.rank_out().into();
        let pairs: usize = (((rank_out + 1) * rank_out) >> 1).max(1);
        pairs
            * GGLWESwitchingKeyCompressed::alloc_bytes_with(
                infos.n(),
                infos.base2k(),
                infos.k(),
                infos.rows(),
                infos.digits(),
                Rank(1),
                infos.rank_out(),
            )
    }

    pub fn alloc_bytes_with(n: Degree, base2k: Base2K, k: TorusPrecision, rows: Rows, digits: Digits, rank: Rank) -> usize {
        let pairs: usize = (((rank.0 + 1) * rank.0) >> 1).max(1) as usize;
        pairs * GGLWESwitchingKeyCompressed::alloc_bytes_with(n, base2k, k, rows, digits, Rank(1), rank)
    }
}

impl<D: DataMut> ReaderFrom for GGLWETensorKeyCompressed<D> {
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

impl<D: DataRef> WriterTo for GGLWETensorKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

impl<D: DataMut> GGLWETensorKeyCompressed<D> {
    pub(crate) fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GGLWESwitchingKeyCompressed<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank_out().into();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataMut, DR: DataRef, B: Backend> Decompress<B, GGLWETensorKeyCompressed<DR>> for GGLWETensorKey<D>
where
    Module<B>: VecZnxFillUniform + VecZnxCopy,
{
    fn decompress(&mut self, module: &Module<B>, other: &GGLWETensorKeyCompressed<DR>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.keys.len(),
                other.keys.len(),
                "invalid receiver: self.keys.len()={} != other.keys.len()={}",
                self.keys.len(),
                other.keys.len()
            );
        }

        self.keys
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(a, b)| {
                a.decompress(module, b);
            });
    }
}
