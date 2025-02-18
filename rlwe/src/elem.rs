use base2k::{Infos, Module, VecZnx, VecZnxApi, VecZnxBorrow, VecZnxOps, VmpPMat, VmpPMatOps};

use crate::parameters::Parameters;

impl Parameters {
    pub fn elem_from_bytes<T>(&self, log_q: usize, rows: usize, bytes: &mut [u8]) -> Elem<T>
    where
        T: VecZnxCommon,
        Elem<T>: Infos + ElemVecZnx<T>,
    {
        Elem::<T>::from_bytes(self.module(), self.log_base2k(), log_q, rows, bytes)
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

pub trait ElemVecZnx<T: VecZnxCommon> {
    fn from_bytes(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        rows: usize,
        bytes: &mut [u8],
    ) -> Elem<T>;
    fn bytes_of(module: &Module, log_base2k: usize, log_q: usize, rows: usize) -> usize;
    fn at(&self, i: usize) -> &T;
    fn at_mut(&mut self, i: usize) -> &mut T;
    fn zero(&mut self);
}

impl<T> ElemVecZnx<T> for Elem<T>
where
    T: VecZnxCommon<Owned = T>,
    Elem<T>: Infos,
{
    fn bytes_of(module: &Module, log_base2k: usize, log_q: usize, rows: usize) -> usize {
        let cols = (log_q + log_base2k - 1) / log_base2k;
        module.n() * cols * (rows + 1) * 8
    }

    fn from_bytes(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        rows: usize,
        bytes: &mut [u8],
    ) -> Elem<T> {
        assert!(rows > 0);
        let n: usize = module.n();
        assert!(bytes.len() >= Self::bytes_of(module, log_base2k, log_q, rows));
        let mut value: Vec<T> = Vec::new();
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let size = T::bytes_of(n, limbs);
        let mut ptr: usize = 0;
        (0..rows).for_each(|_| {
            value.push(T::from_bytes(n, limbs, &mut bytes[ptr..]));
            ptr += size
        });
        Self {
            value,
            log_q,
            log_base2k,
            log_scale: 0,
        }
    }

    fn at(&self, i: usize) -> &T {
        assert!(i < self.rows());
        &self.value[i]
    }

    fn at_mut(&mut self, i: usize) -> &mut T {
        assert!(i < self.rows());
        &mut self.value[i]
    }

    fn zero(&mut self) {
        self.value.iter_mut().for_each(|i| i.zero());
    }
}

impl<T> Elem<T> {
    pub fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn log_scale(&self) -> usize {
        self.log_scale
    }
}

impl Infos for Elem<VecZnx> {
    fn n(&self) -> usize {
        self.value[0].n()
    }

    fn log_n(&self) -> usize {
        self.value[0].log_n()
    }

    fn rows(&self) -> usize {
        self.value.len()
    }
    fn cols(&self) -> usize {
        self.value[0].cols()
    }
}

impl Infos for Elem<VecZnxBorrow> {
    fn n(&self) -> usize {
        self.value[0].n()
    }

    fn log_n(&self) -> usize {
        self.value[0].log_n()
    }

    fn rows(&self) -> usize {
        self.value.len()
    }
    fn cols(&self) -> usize {
        self.value[0].cols()
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

impl Infos for Elem<VmpPMat> {
    fn n(&self) -> usize {
        self.value[0].n()
    }

    fn log_n(&self) -> usize {
        self.value[0].log_n()
    }

    fn rows(&self) -> usize {
        self.value[0].rows()
    }

    fn cols(&self) -> usize {
        self.value[0].cols()
    }
}

impl Elem<VmpPMat> {
    pub fn new(module: &Module, log_base2k: usize, rows: usize, cols: usize) -> Self {
        assert!(rows > 0);
        assert!(cols > 0);
        Self {
            value: Vec::from([module.new_vmp_pmat(rows, cols); 1]),
            log_q: 0,
            log_base2k: log_base2k,
            log_scale: 0,
        }
    }
}
