use crate::elem::Elem;
use crate::plaintext::Plaintext;
use base2k::VecZnx;

pub struct Ciphertext(pub Elem);

/*
impl Parameters {
    pub fn new_ciphertext(&self, degree: usize, log_base2k: usize, log_q: usize) -> Ciphertext {
        Ciphertext(self.new_elem(degree, log_base2k, log_q))
    }
}
 */

impl Ciphertext {
    pub fn new(n: usize, log_base2k: usize, log_q: usize, degree: usize) -> Self {
        Self(Elem::new(n, log_base2k, log_q, degree))
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

    pub fn as_plaintext(&self) -> Plaintext {
        unsafe { Plaintext(std::ptr::read(&self.0)) }
    }
}
