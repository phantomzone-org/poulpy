use crate::elem::{Elem, ElemVecZnx, VecZnxCommon};
use crate::parameters::Parameters;
use crate::plaintext::Plaintext;
use base2k::{Infos, Module, VecZnx, VecZnxApi, VmpPMat};

pub struct Ciphertext<T>(pub Elem<T>);

impl Ciphertext<VecZnx> {
    pub fn new(module: &Module, log_base2k: usize, log_q: usize, rows: usize) -> Self {
        Self(Elem::<VecZnx>::new(module, log_base2k, log_q, rows))
    }
}

impl<T> Ciphertext<T>
where
    T: VecZnxCommon,
    Elem<T>: Infos + ElemVecZnx<T>,
{
    pub fn n(&self) -> usize {
        self.0.n()
    }

    pub fn log_q(&self) -> usize {
        self.0.log_q
    }

    pub fn rows(&self) -> usize {
        self.0.rows()
    }

    pub fn cols(&self) -> usize {
        self.0.cols()
    }

    pub fn at(&self, i: usize) -> &T {
        self.0.at(i)
    }

    pub fn at_mut(&mut self, i: usize) -> &mut T {
        self.0.at_mut(i)
    }

    pub fn log_base2k(&self) -> usize {
        self.0.log_base2k
    }

    pub fn log_scale(&self) -> usize {
        self.0.log_scale
    }

    pub fn zero(&mut self) {
        self.0.zero()
    }

    pub fn as_plaintext(&self) -> Plaintext<T> {
        unsafe { Plaintext::<T>(std::ptr::read(&self.0)) }
    }
}

impl Parameters {
    pub fn new_ciphertext(&self, log_q: usize) -> Ciphertext<VecZnx> {
        Ciphertext::new(self.module(), self.log_base2k(), log_q, 2)
    }
}

pub fn new_gadget_ciphertext(
    module: &Module,
    log_base2k: usize,
    rows: usize,
    log_q: usize,
) -> Ciphertext<VmpPMat> {
    let cols: usize = (log_q + log_base2k - 1) / log_base2k;
    let mut elem: Elem<VmpPMat> = Elem::<VmpPMat>::new(module, log_base2k, rows, 2 * cols);
    elem.log_q = log_q;
    Ciphertext(elem)
}

pub fn new_rgsw_ciphertext(
    module: &Module,
    log_base2k: usize,
    rows: usize,
    log_q: usize,
) -> Ciphertext<VmpPMat> {
    let cols: usize = (log_q + log_base2k - 1) / log_base2k;
    let mut elem: Elem<VmpPMat> = Elem::<VmpPMat>::new(module, log_base2k, 2 * rows, 2 * cols);
    elem.log_q = log_q;
    Ciphertext(elem)
}

impl Ciphertext<VmpPMat> {
    pub fn n(&self) -> usize {
        self.0.n()
    }

    pub fn rows(&self) -> usize {
        self.0.rows()
    }

    pub fn cols(&self) -> usize {
        self.0.cols()
    }

    pub fn log_base2k(&self) -> usize {
        self.0.log_base2k
    }

    pub fn log_q(&self) -> usize {
        self.0.log_q
    }
}
