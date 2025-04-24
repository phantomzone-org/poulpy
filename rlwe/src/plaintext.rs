use crate::ciphertext::Ciphertext;
use crate::elem::{Elem, ElemCommon, ElemVecZnx};
use crate::parameters::Parameters;
use base2k::{Module, VecZnx};

pub struct Plaintext(pub Elem<VecZnx>);

impl Parameters {
    pub fn new_plaintext(&self, log_q: usize) -> Plaintext {
        Plaintext::new(self.module(), self.log_base2k(), log_q)
    }

    pub fn bytes_of_plaintext(&self, log_q: usize) -> usize
where {
        Elem::<VecZnx>::bytes_of(self.module(), self.log_base2k(), log_q, 1)
    }

    pub fn plaintext_from_bytes(&self, log_q: usize, bytes: &mut [u8]) -> Plaintext {
        Plaintext(Elem::<VecZnx>::from_bytes(
            self.module(),
            self.log_base2k(),
            log_q,
            1,
            bytes,
        ))
    }
}

impl Plaintext {
    pub fn new(module: &Module, log_base2k: usize, log_q: usize) -> Self {
        Self(Elem::<VecZnx>::new(module, log_base2k, log_q, 1))
    }
}

impl Plaintext {
    pub fn bytes_of(module: &Module, log_base2k: usize, log_q: usize) -> usize {
        Elem::<VecZnx>::bytes_of(module, log_base2k, log_q, 1)
    }

    pub fn from_bytes(module: &Module, log_base2k: usize, log_q: usize, bytes: &mut [u8]) -> Self {
        Self(Elem::<VecZnx>::from_bytes(
            module, log_base2k, log_q, 1, bytes,
        ))
    }

    pub fn from_bytes_borrow(module: &Module, log_base2k: usize, log_q: usize, bytes: &mut [u8]) -> Self {
        Self(Elem::<VecZnx>::from_bytes_borrow(
            module, log_base2k, log_q, 1, bytes,
        ))
    }

    pub fn as_ciphertext(&self) -> Ciphertext<VecZnx> {
        unsafe { Ciphertext::<VecZnx>(std::ptr::read(&self.0)) }
    }
}

impl ElemCommon<VecZnx> for Plaintext {
    fn n(&self) -> usize {
        self.0.n()
    }

    fn log_n(&self) -> usize {
        self.elem().log_n()
    }

    fn log_q(&self) -> usize {
        self.0.log_q
    }

    fn elem(&self) -> &Elem<VecZnx> {
        &self.0
    }

    fn elem_mut(&mut self) -> &mut Elem<VecZnx> {
        &mut self.0
    }

    fn size(&self) -> usize {
        self.elem().size()
    }

    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn at(&self, i: usize) -> &VecZnx {
        self.0.at(i)
    }

    fn at_mut(&mut self, i: usize) -> &mut VecZnx {
        self.0.at_mut(i)
    }

    fn log_base2k(&self) -> usize {
        self.0.log_base2k()
    }

    fn log_scale(&self) -> usize {
        self.0.log_scale()
    }
}
