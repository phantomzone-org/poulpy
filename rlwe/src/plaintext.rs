use crate::ciphertext::Ciphertext;
use crate::elem::Elem;
use crate::parameters::Parameters;

pub struct Plaintext(pub Elem);

impl Parameters {
    pub fn new_plaintext(&self, log_q: usize) -> Plaintext {
        Plaintext(self.new_elem(0, log_q))
    }
}

impl Plaintext {
    pub fn as_ciphertext(&self) -> Ciphertext {
        unsafe { Ciphertext(std::ptr::read(&self.0)) }
    }
}
