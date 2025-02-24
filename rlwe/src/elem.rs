use base2k::{Infos, Module, VecZnx, VecZnxApi, VecZnxBorrow, VecZnxOps, VmpPMat, VmpPMatOps};

use crate::parameters::Parameters;

impl Parameters {
    pub fn elem_from_bytes<T>(&self, log_q: usize, size: usize, bytes: &mut [u8]) -> Elem<T>
    where
        T: VecZnxCommon<Owned = T>,
        Elem<T>: ElemVecZnx<T>,
    {
        Elem::<T>::from_bytes(self.module(), self.log_base2k(), log_q, size, bytes)
    }
}

pub struct Elem<T> {
    pub value: Vec<T>,
    pub log_base2k: usize,
    pub log_q: usize,
    pub log_scale: usize,
}

pub trait VecZnxCommon: VecZnxApi + Infos {}
impl VecZnxCommon for VecZnx {}
impl VecZnxCommon for VecZnxBorrow {}

pub trait ElemVecZnx<T: VecZnxCommon<Owned = T>> {
    fn from_bytes(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        size: usize,
        bytes: &mut [u8],
    ) -> Elem<T>;
    fn bytes_of(module: &Module, log_base2k: usize, log_q: usize, size: usize) -> usize;
    fn zero(&mut self);
}

impl<T> ElemVecZnx<T> for Elem<T>
where
    T: VecZnxCommon<Owned = T>,
{
    fn bytes_of(module: &Module, log_base2k: usize, log_q: usize, size: usize) -> usize {
        let cols = (log_q + log_base2k - 1) / log_base2k;
        module.n() * cols * size * 8
    }

    fn from_bytes(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        size: usize,
        bytes: &mut [u8],
    ) -> Elem<T> {
        assert!(size > 0);
        let n: usize = module.n();
        assert!(bytes.len() >= Self::bytes_of(module, log_base2k, log_q, size));
        let mut value: Vec<T> = Vec::new();
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let elem_size = T::bytes_of(n, limbs);
        let mut ptr: usize = 0;
        (0..size).for_each(|_| {
            value.push(T::from_bytes(n, limbs, &mut bytes[ptr..]));
            ptr += elem_size
        });
        Self {
            value,
            log_q,
            log_base2k,
            log_scale: 0,
        }
    }

    fn zero(&mut self) {
        self.value.iter_mut().for_each(|i| i.zero());
    }
}

pub trait ElemCommon<T> {
    fn n(&self) -> usize;
    fn log_n(&self) -> usize;
    fn elem(&self) -> &Elem<T>;
    fn elem_mut(&mut self) -> &mut Elem<T>;
    fn size(&self) -> usize;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn log_base2k(&self) -> usize;
    fn log_q(&self) -> usize;
    fn log_scale(&self) -> usize;
    fn at(&self, i: usize) -> &T;
    fn at_mut(&mut self, i: usize) -> &mut T;
}

impl<T: Infos> ElemCommon<T> for Elem<T> {
    fn n(&self) -> usize {
        self.value[0].n()
    }

    fn log_n(&self) -> usize {
        self.value[0].log_n()
    }

    fn elem(&self) -> &Elem<T> {
        self
    }

    fn elem_mut(&mut self) -> &mut Elem<T> {
        self
    }

    fn size(&self) -> usize {
        self.value.len()
    }

    fn rows(&self) -> usize {
        self.value[0].rows()
    }

    fn cols(&self) -> usize {
        self.value[0].cols()
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_q(&self) -> usize {
        self.log_q
    }

    fn log_scale(&self) -> usize {
        self.log_scale
    }

    fn at(&self, i: usize) -> &T {
        assert!(i < self.size());
        &self.value[i]
    }

    fn at_mut(&mut self, i: usize) -> &mut T {
        assert!(i < self.size());
        &mut self.value[i]
    }
}

impl Elem<VecZnx> {
    pub fn new(module: &Module, log_base2k: usize, log_q: usize, rows: usize) -> Self {
        assert!(rows > 0);
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let mut value: Vec<VecZnx> = Vec::new();
        (0..rows).for_each(|_| value.push(module.new_vec_znx(limbs)));
        Self {
            value,
            log_q,
            log_base2k,
            log_scale: 0,
        }
    }
}

impl Elem<VmpPMat> {
    pub fn new(module: &Module, log_base2k: usize, size: usize, rows: usize, cols: usize) -> Self {
        assert!(rows > 0);
        assert!(cols > 0);
        let mut value: Vec<VmpPMat> = Vec::new();
        (0..size).for_each(|_| value.push(module.new_vmp_pmat(rows, cols)));
        Self {
            value: value,
            log_q: 0,
            log_base2k: log_base2k,
            log_scale: 0,
        }
    }
}
