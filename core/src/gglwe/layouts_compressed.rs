use backend::hal::{
    api::{MatZnxAlloc, MatZnxAllocBytes, VecZnxCopy, VecZnxFillUniform},
    layouts::{Backend, Data, DataMut, DataRef, MatZnx, Module, ReaderFrom, WriterTo},
};

use crate::{AutomorphismKey, Decompress, GGLWECiphertext, GLWECiphertextCompressed, GLWESwitchingKey, GLWETensorKey, Infos};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

#[derive(PartialEq, Eq)]
pub struct GGLWECiphertextCompressed<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) rank_out: usize,
    pub(crate) digits: usize,
    pub(crate) seed: Vec<[u8; 32]>,
}

impl GGLWECiphertextCompressed<Vec<u8>> {
    pub fn alloc<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        Self {
            data: module.mat_znx_alloc(rows, rank_in, 1, size),
            basek: basek,
            k,
            rank_out,
            digits,
            seed: vec![[0u8; 32]; rows * rank_in],
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize) -> usize
    where
        Module<B>: MatZnxAllocBytes,
    {
        let size: usize = k.div_ceil(basek);
        debug_assert!(
            size > digits,
            "invalid gglwe: ceil(k/basek): {} <= digits: {}",
            size,
            digits
        );

        assert!(
            rows * digits <= size,
            "invalid gglwe: rows: {} * digits:{} > ceil(k/basek): {}",
            rows,
            digits,
            size
        );

        module.mat_znx_alloc_bytes(rows, rank_in, 1, rows)
    }
}

impl<D: Data> Infos for GGLWECiphertextCompressed<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<D: Data> GGLWECiphertextCompressed<D> {
    pub fn rank(&self) -> usize {
        self.rank_out
    }

    pub fn digits(&self) -> usize {
        self.digits
    }

    pub fn rank_in(&self) -> usize {
        self.data.cols_in()
    }

    pub fn rank_out(&self) -> usize {
        self.rank_out
    }
}

impl<D: DataRef> GGLWECiphertextCompressed<D> {
    pub(crate) fn at(&self, row: usize, col: usize) -> GLWECiphertextCompressed<&[u8]> {
        GLWECiphertextCompressed {
            data: self.data.at(row, col),
            basek: self.basek,
            k: self.k,
            rank: self.rank_out,
            seed: self.seed[self.rank_in() * row + col],
        }
    }
}

impl<D: DataMut> GGLWECiphertextCompressed<D> {
    pub(crate) fn at_mut(&mut self, row: usize, col: usize) -> GLWECiphertextCompressed<&mut [u8]> {
        let rank_in: usize = self.rank_in();
        GLWECiphertextCompressed {
            data: self.data.at_mut(row, col),
            basek: self.basek,
            k: self.k,
            rank: self.rank_out,
            seed: self.seed[rank_in * row + col], // Warning: value is copied and not borrow mut
        }
    }
}

impl<D: DataMut> ReaderFrom for GGLWECiphertextCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.k = reader.read_u64::<LittleEndian>()? as usize;
        self.basek = reader.read_u64::<LittleEndian>()? as usize;
        self.digits = reader.read_u64::<LittleEndian>()? as usize;
        self.rank_out = reader.read_u64::<LittleEndian>()? as usize;
        let seed_len = reader.read_u64::<LittleEndian>()? as usize;
        if seed_len != self.seed.len() {
        } else {
            for s in &mut self.seed {
                reader.read_exact(s)?;
            }
        }
        self.data.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GGLWECiphertextCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.k as u64)?;
        writer.write_u64::<LittleEndian>(self.basek as u64)?;
        writer.write_u64::<LittleEndian>(self.digits as u64)?;
        writer.write_u64::<LittleEndian>(self.rank_out as u64)?;
        writer.write_u64::<LittleEndian>(self.seed.len() as u64)?;
        for s in &self.seed {
            writer.write_all(s)?;
        }
        self.data.write_to(writer)
    }
}

impl<D: DataMut, B: Backend, DR: DataRef> Decompress<B, GGLWECiphertextCompressed<DR>> for GGLWECiphertext<D> {
    fn decompress(&mut self, module: &Module<B>, other: &GGLWECiphertextCompressed<DR>)
    where
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
        #[cfg(debug_assertions)]
        {
            use backend::hal::api::ZnxInfos;

            assert_eq!(
                self.n(),
                other.data.n(),
                "invalid receiver: self.n()={} != other.n()={}",
                self.n(),
                other.data.n()
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
                self.rows(),
                other.rows(),
                "invalid receiver: self.rows()={} != other.rows()={}",
                self.rows(),
                other.rows()
            );
        }

        let rank_in: usize = self.rank_in();
        let rows: usize = self.rows();

        (0..rank_in).for_each(|col_i| {
            (0..rows).for_each(|row_i| {
                self.at_mut(row_i, col_i)
                    .decompress(module, &other.at(row_i, col_i));
            });
        });
    }
}

#[derive(PartialEq, Eq)]
pub struct GLWESwitchingKeyCompressed<D: Data> {
    pub(crate) key: GGLWECiphertextCompressed<D>,
    pub(crate) sk_in_n: usize,  // Degree of sk_in
    pub(crate) sk_out_n: usize, // Degree of sk_out
}

impl<D: Data> Infos for GLWESwitchingKeyCompressed<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<D: Data> GLWESwitchingKeyCompressed<D> {
    pub fn rank(&self) -> usize {
        self.key.rank()
    }

    pub fn digits(&self) -> usize {
        self.key.digits()
    }

    pub fn rank_in(&self) -> usize {
        self.key.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.rank_out()
    }
}

impl GLWESwitchingKeyCompressed<Vec<u8>> {
    pub fn alloc<B: Backend>(
        module: &Module<B>,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        GLWESwitchingKeyCompressed {
            key: GGLWECiphertextCompressed::alloc(module, basek, k, rows, digits, rank_in, rank_out),
            sk_in_n: 0,
            sk_out_n: 0,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank_in: usize) -> usize
    where
        Module<B>: MatZnxAllocBytes,
    {
        GGLWECiphertextCompressed::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, rank_in)
    }
}

impl<D: DataMut> ReaderFrom for GLWESwitchingKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.sk_in_n = reader.read_u64::<LittleEndian>()? as usize;
        self.sk_out_n = reader.read_u64::<LittleEndian>()? as usize;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for GLWESwitchingKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.sk_in_n as u64)?;
        writer.write_u64::<LittleEndian>(self.sk_out_n as u64)?;
        self.key.write_to(writer)
    }
}

impl<D: DataMut> GLWESwitchingKey<D> {
    pub fn decompress<DataOther: DataRef, B: Backend>(
        &mut self,
        module: &Module<B>,
        other: &GLWESwitchingKeyCompressed<DataOther>,
    ) where
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
        self.key.decompress(module, &other.key);
        self.sk_in_n = other.sk_in_n;
        self.sk_out_n = other.sk_out_n;
    }
}

#[derive(PartialEq, Eq)]
pub struct AutomorphismKeyCompressed<D: Data> {
    pub(crate) key: GLWESwitchingKeyCompressed<D>,
    pub(crate) p: i64,
}

impl AutomorphismKeyCompressed<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        AutomorphismKeyCompressed {
            key: GLWESwitchingKeyCompressed::alloc(module, basek, k, rows, digits, rank, rank),
            p: 0,
        }
    }

    pub fn bytes_of<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize
    where
        Module<B>: MatZnxAllocBytes,
    {
        GLWESwitchingKeyCompressed::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, rank)
    }
}

impl<D: Data> Infos for AutomorphismKeyCompressed<D> {
    type Inner = MatZnx<D>;

    fn inner(&self) -> &Self::Inner {
        &self.key.inner()
    }

    fn basek(&self) -> usize {
        self.key.basek()
    }

    fn k(&self) -> usize {
        self.key.k()
    }
}

impl<D: Data> AutomorphismKeyCompressed<D> {
    pub fn rank(&self) -> usize {
        self.key.rank()
    }

    pub fn digits(&self) -> usize {
        self.key.digits()
    }

    pub fn rank_in(&self) -> usize {
        self.key.rank_in()
    }

    pub fn rank_out(&self) -> usize {
        self.key.rank_out()
    }
}

impl<D: DataMut> ReaderFrom for AutomorphismKeyCompressed<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.p = reader.read_u64::<LittleEndian>()? as i64;
        self.key.read_from(reader)
    }
}

impl<D: DataRef> WriterTo for AutomorphismKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.p as u64)?;
        self.key.write_to(writer)
    }
}

impl<D: DataMut> AutomorphismKey<D> {
    pub fn decompress<DataOther: DataRef, B: Backend>(&mut self, module: &Module<B>, other: &AutomorphismKeyCompressed<DataOther>)
    where
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
        self.key.decompress(module, &other.key);
        self.p = other.p;
    }
}

#[derive(PartialEq, Eq)]
pub struct GLWETensorKeyCompressed<D: Data> {
    pub(crate) keys: Vec<GLWESwitchingKeyCompressed<D>>,
}

impl GLWETensorKeyCompressed<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self
    where
        Module<B>: MatZnxAlloc,
    {
        let mut keys: Vec<GLWESwitchingKeyCompressed<Vec<u8>>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);
        (0..pairs).for_each(|_| {
            keys.push(GLWESwitchingKeyCompressed::alloc(
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
        pairs * GLWESwitchingKeyCompressed::<Vec<u8>>::bytes_of(module, basek, k, rows, digits, 1)
    }
}

impl<D: Data> Infos for GLWETensorKeyCompressed<D> {
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

impl<D: Data> GLWETensorKeyCompressed<D> {
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

impl<D: DataMut> ReaderFrom for GLWETensorKeyCompressed<D> {
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

impl<D: DataRef> WriterTo for GLWETensorKeyCompressed<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.keys.len() as u64)?;
        for key in &self.keys {
            key.write_to(writer)?;
        }
        Ok(())
    }
}

impl<D: DataMut> GLWETensorKeyCompressed<D> {
    pub(crate) fn at_mut(&mut self, mut i: usize, mut j: usize) -> &mut GLWESwitchingKeyCompressed<D> {
        if i > j {
            std::mem::swap(&mut i, &mut j);
        };
        let rank: usize = self.rank();
        &mut self.keys[i * rank + j - (i * (i + 1) / 2)]
    }
}

impl<D: DataMut> GLWETensorKey<D> {
    pub fn decompress<DataOther: DataRef, B: Backend>(&mut self, module: &Module<B>, other: &GLWETensorKeyCompressed<DataOther>)
    where
        Module<B>: VecZnxFillUniform + VecZnxCopy,
    {
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
