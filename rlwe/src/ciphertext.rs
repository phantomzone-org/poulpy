use crate::elem::Elem;
use crate::parameters::Parameters;
use crate::plaintext::Plaintext;
use base2k::{Module, VecZnx, VmpPMat, VmpPMatOps};

pub struct Ciphertext(pub Elem);

impl Ciphertext {
    pub fn new(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        degree: usize,
        log_scale: usize,
    ) -> Self {
        Self(Elem::new(module, log_base2k, log_q, degree, log_scale))
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
        self.0.log_scale
    }

    pub fn as_plaintext(&self) -> Plaintext {
        unsafe { Plaintext(std::ptr::read(&self.0)) }
    }
}

impl Parameters {
    pub fn new_ciphertext(&self, log_q: usize) -> Ciphertext {
        Ciphertext::new(self.module(), self.log_base2k(), log_q, self.log_scale(), 1)
    }
}

pub struct GadgetCiphertext {
    pub value: Vec<VmpPMat>,
    pub log_base2k: usize,
    pub log_q: usize,
}

impl GadgetCiphertext {
    pub fn new(module: &Module, log_base2k: usize, rows: usize, log_q: usize) -> Self {
        let cols: usize = (log_q + log_base2k - 1) / log_base2k;
        let mut value: Vec<VmpPMat> = Vec::new();
        (0..rows).for_each(|_| value.push(module.new_vmp_pmat(rows, cols)));
        Self {
            value,
            log_base2k,
            log_q,
        }
    }

    pub fn n(&self) -> usize {
        self.value[0].n
    }

    pub fn rows(&self) -> usize {
        self.value[0].rows
    }

    pub fn cols(&self) -> usize {
        self.value[0].cols
    }

    pub fn degree(&self) -> usize {
        self.value.len() - 1
    }

    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn log_base2k(&self) -> usize {
        self.log_base2k
    }
}

pub struct RGSWCiphertext {
    pub value: [GadgetCiphertext; 2],
    pub log_base2k: usize,
    pub log_q: usize,
    pub log_p: usize,
}
