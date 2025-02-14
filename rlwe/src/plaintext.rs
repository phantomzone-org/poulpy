use crate::ciphertext::Ciphertext;
use crate::elem::{Elem, ElemBasics};
use crate::parameters::Parameters;
use base2k::{Infos, Module, VecZnx, VecZnxApi, VecZnxBorrow};

pub struct Plaintext<T: VecZnxApi + Infos>(pub Elem<T>);

impl Parameters {
    pub fn new_plaintext(&self, log_q: usize) -> Plaintext<VecZnx> {
        Plaintext::new(self.module(), self.log_base2k(), log_q, self.log_scale())
    }

    pub fn bytes_of_plaintext(&self, log_q: usize) -> usize {
        Elem::<VecZnx>::bytes_of(self.module(), self.log_base2k(), log_q, 0)
    }

    pub fn plaintext_from_bytes(&self, log_q: usize, bytes: &mut [u8]) -> Plaintext<VecZnx> {
        Plaintext(self.elem_from_bytes(log_q, 0, bytes))
    }

    pub fn plaintext_borrow_from_bytes(&self, log_q: usize, bytes: &mut [u8]) -> Plaintext<VecZnxBorrow> {
        Plaintext(self.elem_borrow_from_bytes(log_q, 0, bytes))
    }
}

impl Plaintext<VecZnx> {
    pub fn new(module: &Module, log_base2k: usize, log_q: usize, log_scale: usize) -> Self {
        Self(Elem::<VecZnx>::new(module, log_base2k, log_q, 0, log_scale))
    }

    pub fn bytes_of(module: &Module, log_base2k: usize, log_q: usize) -> usize {
        Elem::<VecZnx>::bytes_of(module, log_base2k, log_q, 0)
    }

    pub fn from_bytes(module: &Module, log_base2k: usize, log_q: usize, bytes: &mut [u8]) -> Self {
        Self(Elem::<VecZnx>::from_bytes(module, log_base2k, log_q, 0, bytes))
    }

    pub fn n(&self) -> usize {
        self.0.n()
    }

    pub fn degree(&self) -> usize {
        self.0.degree()
    }

    pub fn log_q(&self) -> usize {
        self.0.log_q()
    }

    pub fn limbs(&self) -> usize {
        self.0.limbs()
    }

    pub fn at(&self, i: usize) -> &VecZnx {
        self.0.at(i)
    }

    pub fn at_mut(&mut self, i: usize) -> &mut VecZnx {
        self.0.at_mut(i)
    }

    pub fn log_base2k(&self) -> usize {
        self.0.log_base2k()
    }

    pub fn log_scale(&self) -> usize {
        self.0.log_scale()
    }

    pub fn zero(&mut self) {
        self.0.zero()
    }

    pub fn as_ciphertext(&self) -> Ciphertext {
        unsafe { Ciphertext(std::ptr::read(&self.0)) }
    }

}

impl Plaintext<VecZnxBorrow> {

    pub fn bytes_of(module: &Module, log_base2k: usize, log_q: usize) -> usize {
        Elem::<VecZnxBorrow>::bytes_of(module, log_base2k, log_q, 0)
    }

    pub fn from_bytes(module: &Module, log_base2k: usize, log_q: usize, bytes: &mut [u8]) -> Self {
        Self(Elem::<VecZnxBorrow>::from_bytes(module, log_base2k, log_q, 0, bytes))
    }

    pub fn n(&self) -> usize {
        self.0.n()
    }

    pub fn degree(&self) -> usize {
        self.0.degree()
    }

    pub fn log_q(&self) -> usize {
        self.0.log_q()
    }

    pub fn limbs(&self) -> usize {
        self.0.limbs()
    }

    pub fn at(&self, i: usize) -> &VecZnxBorrow {
        self.0.at(i)
    }

    pub fn at_mut(&mut self, i: usize) -> &mut VecZnxBorrow {
        self.0.at_mut(i)
    }

    pub fn log_base2k(&self) -> usize {
        self.0.log_base2k()
    }

    pub fn log_scale(&self) -> usize {
        self.0.log_scale()
    }

    pub fn zero(&mut self) {
        self.0.zero()
    }

    /* 
    pub fn as_ciphertext(&self) -> Ciphertext {
        unsafe { Ciphertext(std::ptr::read(&self.0)) }
    }
    */
}
