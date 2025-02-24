use crate::ciphertext::Ciphertext;
use crate::elem::{Elem, ElemCommon, ElemVecZnx, VecZnxCommon};
use crate::parameters::Parameters;
use base2k::{Module, VecZnx};

pub struct Plaintext<T>(pub Elem<T>);

impl Parameters {
    pub fn new_plaintext(&self, log_q: usize) -> Plaintext<VecZnx> {
        Plaintext::new(self.module(), self.log_base2k(), log_q)
    }

    pub fn bytes_of_plaintext<T>(&self, log_q: usize) -> usize
    where
        T: VecZnxCommon<Owned = T>,
        Elem<T>: ElemVecZnx<T>,
    {
        Elem::<T>::bytes_of(self.module(), self.log_base2k(), log_q, 1)
    }

    pub fn plaintext_from_bytes<T>(&self, log_q: usize, bytes: &mut [u8]) -> Plaintext<T>
    where
        T: VecZnxCommon<Owned = T>,
        Elem<T>: ElemVecZnx<T>,
    {
        Plaintext::<T>(self.elem_from_bytes::<T>(log_q, 1, bytes))
    }
}

impl Plaintext<VecZnx> {
    pub fn new(module: &Module, log_base2k: usize, log_q: usize) -> Self {
        Self(Elem::<VecZnx>::new(module, log_base2k, log_q, 1))
    }
}

impl<T> Plaintext<T>
where
    T: VecZnxCommon<Owned = T>,
    Elem<T>: ElemVecZnx<T>,
{
    pub fn bytes_of(module: &Module, log_base2k: usize, log_q: usize) -> usize {
        Elem::<T>::bytes_of(module, log_base2k, log_q, 1)
    }

    pub fn from_bytes(module: &Module, log_base2k: usize, log_q: usize, bytes: &mut [u8]) -> Self {
        Self(Elem::<T>::from_bytes(module, log_base2k, log_q, 1, bytes))
    }

    pub fn as_ciphertext(&self) -> Ciphertext<T> {
        unsafe { Ciphertext::<T>(std::ptr::read(&self.0)) }
    }
}

impl<T> ElemCommon<T> for Plaintext<T>
where
    T: VecZnxCommon<Owned = T>,
    Elem<T>: ElemVecZnx<T>,
{
    fn n(&self) -> usize {
        self.0.n()
    }

    fn log_n(&self) -> usize {
        self.elem().log_n()
    }

    fn log_q(&self) -> usize {
        self.0.log_q
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
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn at(&self, i: usize) -> &T {
        self.0.at(i)
    }

    fn at_mut(&mut self, i: usize) -> &mut T {
        self.0.at_mut(i)
    }

    fn log_base2k(&self) -> usize {
        self.0.log_base2k()
    }

    fn log_scale(&self) -> usize {
        self.0.log_scale()
    }
}
