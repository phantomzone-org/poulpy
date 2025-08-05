use backend::hal::{
    api::{MatZnxAlloc, MatZnxAllocBytes},
    layouts::{Backend, MatZnx, Module, ReaderFrom, Scratch, VmpPMat, WriterTo},
};

use crate::{GGLWEExecLayoutFamily, GLWESwitchingKey, GLWESwitchingKeyExec, Infos};

pub struct GLWETensorKey<D> {
    pub(crate) keys: Vec<GLWESwitchingKey<D>>,
}

impl GLWETensorKey<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        let mut keys: Vec<GLWESwitchingKey<Vec<u8>>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKey::alloc(
                module, basek, k, rows, digits, 1, rank,
            ));
        });
        Self { keys: keys }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: MatZnxAllocBytes,
    {
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        pairs * GLWESwitchingKey::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, 1, rank)
    }
}

impl<D> Infos for GLWETensorKey<D> {
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

impl<D> GLWETensorKey<D> {
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

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWETensorKey<D> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKey<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: AsRef<[u8]>> GLWETensorKey<D> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKey<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: AsRef<[u8]> + AsMut<[u8]>> ReaderFrom for GLWETensorKey<D> {
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

impl<D: AsRef<[u8]>> WriterTo for GLWETensorKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

pub struct GLWETensorKeyExec<D, B: Backend> {
    pub(crate) keys: Vec<GLWESwitchingKeyExec<D, B>>,
}

impl<B: Backend> GLWETensorKeyExec<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let mut keys: Vec<GLWESwitchingKeyExec<Vec<u8>, B>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKeyExec::alloc(
                module, basek, k, rows, digits, 1, rank,
            ));
        });
        Self { keys }
    }

    pub fn bytes_of(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        pairs * GLWESwitchingKeyExec::<Vec<u8>, B>::bytes_of(module, basek, k, rows, digits, 1, rank)
    }
}

impl<D, B: Backend> Infos for GLWETensorKeyExec<D, B> {
    type Inner = VmpPMat<D, B>;

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

impl<D, B: Backend> GLWETensorKeyExec<D, B> {
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

impl<D: AsMut<[u8]> + AsRef<[u8]>, B: Backend> GLWETensorKeyExec<D, B> {
    // Returns a mutable reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKeyExec<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: AsRef<[u8]>, B: Backend> GLWETensorKeyExec<D, B> {
    // Returns a reference to GLWESwitchingKey_{s}(s[i] * s[j])
    pub fn at(&self, mut i: usize, mut j: usize) -> &GLWESwitchingKeyExec<D, B> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>, B: Backend> GLWETensorKeyExec<D, B> {
    pub fn prepare<DataOther>(&mut self, module: &Module<B>, other: &GLWETensorKey<DataOther>, scratch: &mut Scratch<B>)
    where
        DataOther: AsRef<[u8]>,
        Module<B>: GGLWEExecLayoutFamily<B>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.keys.len(), other.keys.len());
        }
        self.keys
            .iter_mut()
            .zip(other.keys.iter())
            .for_each(|(a, b)| {
                a.prepare(module, b, scratch);
            });
    }
}
