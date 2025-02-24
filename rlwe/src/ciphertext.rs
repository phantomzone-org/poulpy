use crate::elem::{Elem, ElemCommon};
use crate::parameters::Parameters;
use base2k::{Infos, Module, VecZnx, VmpPMat};

pub struct Ciphertext<T>(pub Elem<T>);

impl Parameters {
    pub fn new_ciphertext(&self, log_q: usize) -> Ciphertext<VecZnx> {
        Ciphertext::new(self.module(), self.log_base2k(), log_q, 2)
    }
}

impl<T> ElemCommon<T> for Ciphertext<T>
where
    T: Infos,
{
    fn n(&self) -> usize {
        self.elem().n()
    }

    fn log_n(&self) -> usize {
        self.elem().log_n()
    }

    fn log_q(&self) -> usize {
        self.elem().log_q()
    }

    fn elem(&self) -> &Elem<T> {
        &self.0
    }

    fn elem_mut(&mut self) -> &mut Elem<T> {
        &mut self.0
    }

    fn size(&self) -> usize {
        self.elem().size()
    }

    fn rows(&self) -> usize {
        self.elem().rows()
    }

    fn cols(&self) -> usize {
        self.elem().cols()
    }

    fn at(&self, i: usize) -> &T {
        self.elem().at(i)
    }

    fn at_mut(&mut self, i: usize) -> &mut T {
        self.elem_mut().at_mut(i)
    }

    fn log_base2k(&self) -> usize {
        self.elem().log_base2k()
    }

    fn log_scale(&self) -> usize {
        self.elem().log_scale()
    }
}

impl Ciphertext<VecZnx> {
    pub fn new(module: &Module, log_base2k: usize, log_q: usize, rows: usize) -> Self {
        Self(Elem::<VecZnx>::new(module, log_base2k, log_q, rows))
    }
}

pub fn new_rlwe_ciphertext(module: &Module, log_base2k: usize, log_q: usize) -> Ciphertext<VecZnx> {
    let rows: usize = 2;
    Ciphertext::<VecZnx>::new(module, log_base2k, log_q, rows)
}

pub fn new_gadget_ciphertext(
    module: &Module,
    log_base2k: usize,
    rows: usize,
    log_q: usize,
) -> Ciphertext<VmpPMat> {
    let cols: usize = (log_q + log_base2k - 1) / log_base2k;
    let mut elem: Elem<VmpPMat> = Elem::<VmpPMat>::new(module, log_base2k, 2, rows, cols);
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
    let mut elem: Elem<VmpPMat> = Elem::<VmpPMat>::new(module, log_base2k, 4, rows, cols);
    elem.log_q = log_q;
    Ciphertext(elem)
}
