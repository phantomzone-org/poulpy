use std::hash::{DefaultHasher, Hasher};

use rand::seq::SliceRandom;
use rand_core::RngCore;
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::{
    alloc_aligned,
    layouts::{
        Data, DataMut, DataRef, DataView, DataViewMut, DigestU64, FillUniform, ReaderFrom, ToOwnedDeep, VecZnx, WriterTo,
        ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

#[repr(C)]
#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct ScalarZnx<D: Data> {
    pub data: D,
    pub n: usize,
    pub cols: usize,
}

impl<D: DataRef> DigestU64 for ScalarZnx<D> {
    fn digest_u64(&self) -> u64 {
        let mut h: DefaultHasher = DefaultHasher::new();
        h.write(self.data.as_ref());
        h.write_usize(self.n);
        h.write_usize(self.cols);
        h.finish()
    }
}

impl<D: DataRef> ToOwnedDeep for ScalarZnx<D> {
    type Owned = ScalarZnx<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        ScalarZnx {
            data: self.data.as_ref().to_vec(),
            n: self.n,
            cols: self.cols,
        }
    }
}

impl<D: Data> ZnxInfos for ScalarZnx<D> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> usize {
        1
    }

    fn n(&self) -> usize {
        self.n
    }

    fn size(&self) -> usize {
        1
    }
}

impl<D: Data> ZnxSliceSize for ScalarZnx<D> {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl<D: Data> DataView for ScalarZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D: Data> DataViewMut for ScalarZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: DataRef> ZnxView for ScalarZnx<D> {
    type Scalar = i64;
}

impl<D: DataMut> ScalarZnx<D> {
    pub fn fill_ternary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [i64; 3] = [-1, 0, 1];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    pub fn fill_ternary_hw(&mut self, col: usize, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (((source.next_u32() & 1) as i64) << 1) - 1);
        self.at_mut(col, 0).shuffle(source);
    }

    pub fn fill_binary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [i64; 2] = [0, 1];
        let weights: [f64; 2] = [1.0 - prob, prob];
        let dist: WeightedIndex<f64> = WeightedIndex::new(weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    pub fn fill_binary_hw(&mut self, col: usize, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (source.next_u32() & 1) as i64);
        self.at_mut(col, 0).shuffle(source);
    }

    pub fn fill_binary_block(&mut self, col: usize, block_size: usize, source: &mut Source) {
        assert!(self.n().is_multiple_of(block_size));
        let max_idx: u64 = (block_size + 1) as u64;
        let mask_idx: u64 = (1 << ((u64::BITS - max_idx.leading_zeros()) as u64)) - 1;
        for block in self.at_mut(col, 0).chunks_mut(block_size) {
            let idx: usize = source.next_u64n(max_idx, mask_idx) as usize;
            if idx != block_size {
                block[idx] = 1;
            }
        }
    }
}

impl ScalarZnx<Vec<u8>> {
    pub fn bytes_of(n: usize, cols: usize) -> usize {
        n * cols * size_of::<i64>()
    }

    pub fn alloc(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols));
        Self { data, n, cols }
    }

    pub fn from_bytes(n: usize, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols));
        Self { data, n, cols }
    }
}

impl<D: DataMut> ZnxZero for ScalarZnx<D> {
    fn zero(&mut self) {
        self.raw_mut().fill(0)
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(0);
    }
}

impl<D: DataMut> FillUniform for ScalarZnx<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        match log_bound {
            64 => source.fill_bytes(self.data.as_mut()),
            0 => panic!("invalid log_bound, cannot be zero"),
            _ => {
                let mask: u64 = (1u64 << log_bound) - 1;
                for x in self.raw_mut().iter_mut() {
                    let r = source.next_u64() & mask;
                    *x = ((r << (64 - log_bound)) as i64) >> (64 - log_bound);
                }
            }
        }
    }
}

pub type ScalarZnxOwned = ScalarZnx<Vec<u8>>;

impl<D: Data> ScalarZnx<D> {
    pub fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self { data, n, cols }
    }
}

pub trait ScalarZnxToRef {
    fn to_ref(&self) -> ScalarZnx<&[u8]>;
}

impl<D: DataRef> ScalarZnxToRef for ScalarZnx<D> {
    fn to_ref(&self) -> ScalarZnx<&[u8]> {
        ScalarZnx {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
        }
    }
}

pub trait ScalarZnxToMut {
    fn to_mut(&mut self) -> ScalarZnx<&mut [u8]>;
}

impl<D: DataMut> ScalarZnxToMut for ScalarZnx<D> {
    fn to_mut(&mut self) -> ScalarZnx<&mut [u8]> {
        ScalarZnx {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
        }
    }
}

impl<D: DataRef> ScalarZnx<D> {
    pub fn as_vec_znx(&self) -> VecZnx<&[u8]> {
        VecZnx {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: 1,
            max_size: 1,
        }
    }
}

impl<D: DataMut> ScalarZnx<D> {
    pub fn as_vec_znx_mut(&mut self) -> VecZnx<&mut [u8]> {
        VecZnx {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
            size: 1,
            max_size: 1,
        }
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: DataMut> ReaderFrom for ScalarZnx<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.n = reader.read_u64::<LittleEndian>()? as usize;
        self.cols = reader.read_u64::<LittleEndian>()? as usize;
        let len: usize = reader.read_u64::<LittleEndian>()? as usize;
        let buf: &mut [u8] = self.data.as_mut();
        if buf.len() != len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("self.data.len()={} != read len={}", buf.len(), len),
            ));
        }
        reader.read_exact(&mut buf[..len])?;
        Ok(())
    }
}

impl<D: DataRef> WriterTo for ScalarZnx<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n as u64)?;
        writer.write_u64::<LittleEndian>(self.cols as u64)?;
        let buf: &[u8] = self.data.as_ref();
        writer.write_u64::<LittleEndian>(buf.len() as u64)?;
        writer.write_all(buf)?;
        Ok(())
    }
}
