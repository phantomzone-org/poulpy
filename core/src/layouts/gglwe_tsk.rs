use backend::hal::{
    api::{FillUniform, Reset},
    layouts::{Data, DataMut, DataRef, MatZnx, ReaderFrom, WriterTo},
};

use crate::layouts::{GGLWESwitchingKey, Infos};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fmt;

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWETensorKey<D: Data> {
    pub(crate) keys: Vec<GGLWESwitchingKey<D>>,
}

impl<D: DataRef> fmt::Debug for GGLWETensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<D: DataMut> FillUniform for GGLWETensorKey<D> {
    fn fill_uniform(&mut self, source: &mut sampling::source::Source) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWESwitchingKey<D>| key.fill_uniform(source))
    }
}

impl<D: DataMut> Reset for GGLWETensorKey<D>
where
    MatZnx<D>: Reset,
{
    fn reset(&mut self) {
        self.keys
            .iter_mut()
            .for_each(|key: &mut GGLWESwitchingKey<D>| key.reset())
    }
}

impl<D: DataRef> fmt::Display for GGLWETensorKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "(GLWETensorKey)",)?;
        for (i, key) in self.keys.iter().enumerate() {
            write!(f, "{}: {}", i, key)?;
        }
        Ok(())
    }
}

impl GGLWETensorKey<Vec<u8>> {
    pub fn alloc(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        let mut keys: Vec<GGLWESwitchingKey<Vec<u8>>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GGLWESwitchingKey::alloc(n, basek, k, rows, digits, 1, rank));
        });
        Self { keys: keys }
    }

    pub fn bytes_of(n: usize, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        pairs * GGLWESwitchingKey::<Vec<u8>>::bytes_of(n, basek, k, rows, digits, 1, rank)
    }
}

impl<D: Data> Infos for GGLWETensorKey<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.keys[0].inner()
    }

    fn basek(&self) -> usize {
        self.keys[0].basek()
    }

    fn k(&self) -> usize {
        self.keys[0].k()
    }
}

impl<D: Data> GGLWETensorKey<D> {
    pub fn rank(&self) -> usize {
        self.keys[0].rank()
    }

    pub fn rank_in(&self) -> usize {
        self.keys[0].rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.keys[0].rank_out()
    }

    pub fn digits(&self) -> usize {
        self.keys[0].digits()
    }
}

impl<D: DataMut> GGLWETensorKey<D> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GGLWESwitchingKey<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataRef> GGLWETensorKey<D> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GGLWESwitchingKey<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataMut> ReaderFrom for GGLWETensorKey<D> {
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

impl<D: DataRef> WriterTo for GGLWETensorKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}
