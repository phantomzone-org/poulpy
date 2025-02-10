use crate::ciphertext::Ciphertext;
use crate::elem::Elem;
use base2k::VecZnx;

pub struct Plaintext(pub Elem);

/*
impl Parameters {
    pub fn new_plaintext(&self, log_q: usize) -> Plaintext {
        Plaintext(self.new_elem(0, log_q))
    }
}
*/

impl Plaintext {
    pub fn new(n: usize, log_base2k: usize, log_q: usize) -> Self {
        Self(Elem::new(n, log_base2k, log_q, 0))
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

    pub fn as_ciphertext(&self) -> Ciphertext {
        unsafe { Ciphertext(std::ptr::read(&self.0)) }
    }
}
