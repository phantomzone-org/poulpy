use rand::seq::SliceRandom;
use rand_core::RngCore;
use rand_distr::{Distribution, weighted::WeightedIndex};
use sampling::source::Source;

use crate::{
    alloc_aligned,
    hal::{
        api::{DataView, DataViewMut, ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut, ZnxZero},
        layouts::{ReaderFrom, VecZnx, VecZnxToMut, VecZnxToRef, WriterTo},
    },
};

#[derive(PartialEq, Eq)]
pub struct ScalarZnx<D> {
    pub(crate) data: D,
    pub(crate) n: usize,
    pub(crate) cols: usize,
}

impl<D> ZnxInfos for ScalarZnx<D> {
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

impl<D> ZnxSliceSize for ScalarZnx<D> {
    fn sl(&self) -> usize {
        self.n()
    }
}

impl<D> DataView for ScalarZnx<D> {
    type D = D;
    fn data(&self) -> &Self::D {
        &self.data
    }
}

impl<D> DataViewMut for ScalarZnx<D> {
    fn data_mut(&mut self) -> &mut Self::D {
        &mut self.data
    }
}

impl<D: AsRef<[u8]>> ZnxView for ScalarZnx<D> {
    type Scalar = i64;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> ScalarZnx<D> {
    pub fn fill_ternary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [i64; 3] = [-1, 0, 1];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(&weights).unwrap();
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
        let dist: WeightedIndex<f64> = WeightedIndex::new(&weights).unwrap();
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
        assert!(self.n() % block_size == 0);
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

impl<D: AsRef<[u8]>> ScalarZnx<D> {
    pub fn bytes_of(n: usize, cols: usize) -> usize {
        n * cols * size_of::<i64>()
    }
}

impl<D: From<Vec<u8>> + AsRef<[u8]>> ScalarZnx<D> {
    pub fn new(n: usize, cols: usize) -> Self {
        let data: Vec<u8> = alloc_aligned::<u8>(Self::bytes_of(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
        }
    }

    pub(crate) fn new_from_bytes(n: usize, cols: usize, bytes: impl Into<Vec<u8>>) -> Self {
        let data: Vec<u8> = bytes.into();
        assert!(data.len() == Self::bytes_of(n, cols));
        Self {
            data: data.into(),
            n,
            cols,
        }
    }
}

impl<D: AsRef<[u8]> + AsMut<[u8]>> ZnxZero for ScalarZnx<D> {
    fn zero(&mut self) {
        self.raw_mut().fill(0)
    }
    fn zero_at(&mut self, i: usize, j: usize) {
        self.at_mut(i, j).fill(0);
    }
}

pub type ScalarZnxOwned = ScalarZnx<Vec<u8>>;

impl<D> ScalarZnx<D> {
    pub(crate) fn from_data(data: D, n: usize, cols: usize) -> Self {
        Self { data, n, cols }
    }
}

pub trait ScalarZnxToRef {
    fn to_ref(&self) -> ScalarZnx<&[u8]>;
}

impl<D> ScalarZnxToRef for ScalarZnx<D>
where
    D: AsRef<[u8]>,
{
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

impl<D> ScalarZnxToMut for ScalarZnx<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
{
    fn to_mut(&mut self) -> ScalarZnx<&mut [u8]> {
        ScalarZnx {
            data: self.data.as_mut(),
            n: self.n,
            cols: self.cols,
        }
    }
}

impl<D> VecZnxToRef for ScalarZnx<D>
where
    D: AsRef<[u8]>,
{
    fn to_ref(&self) -> VecZnx<&[u8]> {
        VecZnx {
            data: self.data.as_ref(),
            n: self.n,
            cols: self.cols,
            size: 1,
            max_size: 1,
        }
    }
}

impl<D> VecZnxToMut for ScalarZnx<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
{
    fn to_mut(&mut self) -> VecZnx<&mut [u8]> {
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

impl<D: AsRef<[u8]> + AsMut<[u8]>> ReaderFrom for ScalarZnx<D> {
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

impl<D: AsRef<[u8]>> WriterTo for ScalarZnx<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u64::<LittleEndian>(self.n as u64)?;
        writer.write_u64::<LittleEndian>(self.cols as u64)?;
        let buf: &[u8] = self.data.as_ref();
        writer.write_u64::<LittleEndian>(buf.len() as u64)?;
        writer.write_all(buf)?;
        Ok(())
    }
}
