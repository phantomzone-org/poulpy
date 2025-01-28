use crate::elem::Elem;
use crate::parameters::Parameters;
use crate::plaintext::Plaintext;

pub struct Ciphertext(pub Elem);

impl Parameters {
    pub fn new_ciphertext(&self, degree: usize, log_q: usize) -> Ciphertext {
        Ciphertext(self.new_elem(degree, log_q))
    }
}

impl Ciphertext {
    pub fn as_plaintext(&self) -> Plaintext {
        unsafe { Plaintext(std::ptr::read(&self.0)) }
    }
}
