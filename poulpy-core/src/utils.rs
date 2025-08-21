use crate::layouts::{GLWEPlaintext, Infos, LWEPlaintext};
use poulpy_hal::layouts::{DataMut, DataRef};
use rug::Float;

impl<D: DataMut> GLWEPlaintext<D> {
    pub fn encode_vec_i64(&mut self, data: &[i64], k: usize) {
        let basek: usize = self.basek();
        self.data
            .encode_vec_i64(basek, 0, k, data, i64::BITS as usize);
    }

    pub fn encode_coeff_i64(&mut self, data: i64, k: usize, idx: usize) {
        let basek: usize = self.basek();
        self.data
            .encode_coeff_i64(basek, 0, k, idx, data, i64::BITS as usize);
    }
}

impl<D: DataRef> GLWEPlaintext<D> {
    pub fn decode_vec_i64(&self, data: &mut [i64], k: usize) {
        self.data.decode_vec_i64(self.basek(), 0, k, data);
    }

    pub fn decode_coeff_i64(&self, k: usize, idx: usize) -> i64 {
        self.data.decode_coeff_i64(self.basek(), 0, k, idx)
    }

    pub fn decode_vec_float(&self, data: &mut [Float]) {
        self.data.decode_vec_float(self.basek(), 0, data);
    }

    pub fn std(&self) -> f64 {
        self.data.std(self.basek(), 0)
    }
}

impl<D: DataMut> LWEPlaintext<D> {
    pub fn encode_i64(&mut self, data: i64, k: usize) {
        let basek: usize = self.basek();
        self.data.encode_i64(basek, k, data, i64::BITS as usize);
    }
}

impl<D: DataRef> LWEPlaintext<D> {
    pub fn decode_i64(&self, k: usize) -> i64 {
        self.data.decode_i64(self.basek(), k)
    }

    pub fn decode_float(&self) -> Float {
        self.data.decode_float(self.basek())
    }
}
