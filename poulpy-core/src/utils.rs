use crate::layouts::{GLWEPlaintext, LWEInfos, LWEPlaintext, TorusPrecision};
use poulpy_hal::layouts::{DataMut, DataRef};
use rug::Float;

impl<D: DataMut> GLWEPlaintext<D> {
    pub fn encode_vec_i64(&mut self, data: &[i64], k: TorusPrecision) {
        let base2k: usize = self.base2k().into();
        self.data
            .encode_vec_i64(base2k, 0, k.into(), data, i64::BITS as usize);
    }

    pub fn encode_coeff_i64(&mut self, data: i64, k: TorusPrecision, idx: usize) {
        let base2k: usize = self.base2k().into();
        self.data
            .encode_coeff_i64(base2k, 0, k.into(), idx, data, i64::BITS as usize);
    }
}

impl<D: DataRef> GLWEPlaintext<D> {
    pub fn decode_vec_i64(&self, data: &mut [i64], k: TorusPrecision) {
        self.data
            .decode_vec_i64(self.base2k().into(), 0, k.into(), data);
    }

    pub fn decode_coeff_i64(&self, k: TorusPrecision, idx: usize) -> i64 {
        self.data
            .decode_coeff_i64(self.base2k().into(), 0, k.into(), idx)
    }

    pub fn decode_vec_float(&self, data: &mut [Float]) {
        self.data.decode_vec_float(self.base2k().into(), 0, data);
    }

    pub fn std(&self) -> f64 {
        self.data.std(self.base2k().into(), 0)
    }
}

impl<D: DataMut> LWEPlaintext<D> {
    pub fn encode_i64(&mut self, data: i64, k: TorusPrecision) {
        let base2k: usize = self.base2k().into();
        self.data
            .encode_i64(base2k, k.into(), data, i64::BITS as usize);
    }
}

impl<D: DataRef> LWEPlaintext<D> {
    pub fn decode_i64(&self, k: TorusPrecision) -> i64 {
        self.data.decode_i64(self.base2k().into(), k.into())
    }

    pub fn decode_float(&self) -> Float {
        self.data.decode_float(self.base2k().into())
    }
}
