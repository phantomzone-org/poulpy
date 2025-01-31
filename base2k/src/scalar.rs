use crate::Module;
use rand::seq::SliceRandom;
use rand_core::RngCore;
use rand_distr::{Distribution, WeightedIndex};
use sampling::source::Source;

pub struct Scalar(pub Vec<i64>);

impl Module {
    pub fn new_scalar(&self) -> Scalar {
        Scalar::new(self.n())
    }
}

impl Scalar {
    pub fn new(n: usize) -> Self {
        Self(vec![i64::default(); Self::buffer_size(n)])
    }

    pub fn buffer_size(n: usize) -> usize {
        n
    }

    pub fn from_buffer(&mut self, n: usize, buf: &[i64]) {
        let size = Self::buffer_size(n);
        assert!(
            buf.len() >= size,
            "invalid buffer: buf.len()={} < self.buffer_size(n={})={}",
            buf.len(),
            n,
            size
        );
        self.0 = Vec::from(&buf[..size])
    }

    pub fn as_ptr(&self) -> *const i64 {
        self.0.as_ptr()
    }

    pub fn fill_ternary_prob(&mut self, prob: f64, source: &mut Source) {
        let choices: [i64; 3] = [-1, 0, 1];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(&weights).unwrap();
        self.0
            .iter_mut()
            .for_each(|x: &mut i64| *x = choices[dist.sample(source)]);
    }

    pub fn fill_ternary_hw(&mut self, hw: usize, source: &mut Source) {
        self.0[..hw]
            .iter_mut()
            .for_each(|x: &mut i64| *x = (((source.next_u32() & 1) as i64) << 1) - 1);
        self.0.shuffle(source);
    }
}
