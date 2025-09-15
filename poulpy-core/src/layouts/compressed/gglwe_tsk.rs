use poulpy_hal::{
    api::{VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, FillUniform, MatZnx, Module, ReaderFrom, Reset, WriterTo},
    source::Source,
};

use crate::layouts::{
    GGLWETensorKey, Infos,
    compressed::{Decompress, GGLWESwitchingKeyCompressed},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWETensorKeyCompressed<D: Data> {
    pub(crate) keys: Vec<GGLWESwitchingKeyCompressed<D>>,
}

impl<D: DataRef> fmt::Debug for GGLWETensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GGLWETensorKeyCompressed<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWESwitchingKeyCompressed<D>| key.fill_uniform(log_bound, source))
    }
}

impl<D: DataMut> Reset for GGLWETensorKeyCompressed<D>
where
    MatZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWESwitchingKeyCompressed<D>| key.reset())
    }
}

impl<D: DataRef> fmt::Display for GGLWETensorKeyCompressed<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKeyCompressed)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{}: {}", i, key)?;
        }
        Ok(())
    }
}

impl GGLWETensorKeyCompressed<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        let mut keys: Vec<GGLWESwitchingKeyCompressed<Vec<u8>>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GGLWESwitchingKeyCompressed::alloc(
                n, basek, k, rows, digits, 1, rank,
            ));
        });
        Self { keys }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        pairs * GGLWESwitchingKeyCompressed::bytes_of(n, basek, k, rows, digits, 1)
    }
}

impl<D: Data> Infos for GGLWETensorKeyCompressed<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        self.keys[0].inner()
    }

    fn basek(&self) -> usize {
        self.keys[0].basek()
    }

    fn k(&self) -> usize {
        self.keys[0].k()
    }
}

impl<D: Data> GGLWETensorKeyCompressed<D> {
    pub fn rank(&self) -> usize {
        self.keys[0].rank()
    }

    pub fn digits(&self) -> usize {
        self.keys[0].digits()
    }

    pub fn rank_in(&self) -> usize {
        self.keys[0].rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.keys[0].rank_out()
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
        let rank: usize = self.rank();
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
