use crate::parameters::Parameters;
use base2k::{Infos, Module, VecZnx, VecZnxApi, VecZnxBorrow, VecZnxOps};


impl Parameters {
    pub fn bytes_of_elem(&self, log_q: usize, degree: usize) -> usize {
        Elem::<VecZnx>::bytes_of(self.module(), self.log_base2k(), log_q, degree)
    }

    pub fn elem_from_bytes(&self, log_q: usize, degree: usize, bytes: &mut [u8]) -> Elem<VecZnx> {
        Elem::<VecZnx>::from_bytes(self.module(), self.log_base2k(), log_q, degree, bytes)
    }

    pub fn elem_borrow_from_bytes(&self, log_q: usize, degree: usize, bytes: &mut [u8]) -> Elem<VecZnxBorrow> {
        Elem::<VecZnxBorrow>::from_bytes(self.module(), self.log_base2k(), log_q, degree, bytes)
    }
}

pub struct Elem<T: VecZnxApi + Infos> {
    pub value: Vec<T>,
    pub log_base2k: usize,
    pub log_q: usize,
    pub log_scale: usize,
}

pub trait ElemBasics<T>
where
    T: VecZnxApi + Infos,
{
    fn n(&self) -> usize;
    fn degree(&self) -> usize;
    fn limbs(&self) -> usize;
    fn log_base2k(&self) -> usize;
    fn log_scale(&self) -> usize;
    fn log_q(&self) -> usize;
    fn at(&self, i: usize) -> &T;
    fn at_mut(&mut self, i: usize) -> &mut T;
    fn zero(&mut self);
}

impl Elem<VecZnx> {
    pub fn new(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        degree: usize,
        log_scale: usize,
    ) -> Self {
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let mut value: Vec<VecZnx> = Vec::new();
        (0..degree + 1).for_each(|_| value.push(module.new_vec_znx(limbs)));
        Self {
            value,
            log_q,
            log_base2k,
            log_scale: log_scale,
        }
    }

    pub fn bytes_of(module: &Module, log_base2k: usize, log_q: usize, degree: usize) -> usize {
        let cols = (log_q + log_base2k - 1) / log_base2k;
        module.n() * cols * (degree + 1) * 8
    }

    pub fn from_bytes(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        degree: usize,
        bytes: &mut [u8],
    ) -> Self {
        let n: usize = module.n();
        assert!(bytes.len() >= Self::bytes_of(module, log_base2k, log_q, degree));
        let mut value: Vec<VecZnx> = Vec::new();
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let size = VecZnx::bytes_of(n, limbs);
        let mut ptr: usize = 0;
        (0..degree + 1).for_each(|_| {
            value.push(VecZnx::from_bytes(n, limbs, &mut bytes[ptr..]));
            ptr += size
        });
        Self {
            value,
            log_q,
            log_base2k,
            log_scale: 0,
        }
    }
}

impl Elem<VecZnxBorrow> {

    pub fn bytes_of(module: &Module, log_base2k: usize, log_q: usize, degree: usize) -> usize {
        let cols = (log_q + log_base2k - 1) / log_base2k;
        module.n() * cols * (degree + 1) * 8
    }

    pub fn from_bytes(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        degree: usize,
        bytes: &mut [u8],
    ) -> Self {
        let n: usize = module.n();
        assert!(bytes.len() >= Self::bytes_of(module, log_base2k, log_q, degree));
        let mut value: Vec<VecZnxBorrow> = Vec::new();
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let size = VecZnxBorrow::bytes_of(n, limbs);
        let mut ptr: usize = 0;
        (0..degree + 1).for_each(|_| {
            value.push(VecZnxBorrow::from_bytes(n, limbs, &mut bytes[ptr..]));
            ptr += size
        });
        Self {
            value,
            log_q,
            log_base2k,
            log_scale: 0,
        }
    }
}


impl<T: VecZnxApi + Infos> ElemBasics<T> for Elem<T> {
    fn n(&self) -> usize {
        self.value[0].n()
    }

    fn degree(&self) -> usize {
        self.value.len()
    }

    fn limbs(&self) -> usize {
        self.value[0].limbs()
    }

    fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    fn log_scale(&self) -> usize {
        self.log_scale
    }

    fn log_q(&self) -> usize {
        self.log_q
    }

    fn at(&self, i: usize) -> &T {
        assert!(i <= self.degree());
        &self.value[i]
    }

    fn at_mut(&mut self, i: usize) -> &mut T {
        assert!(i <= self.degree());
        &mut self.value[i]
    }

    fn zero(&mut self) {
        self.value.iter_mut().for_each(|i| i.zero());
    }
}
