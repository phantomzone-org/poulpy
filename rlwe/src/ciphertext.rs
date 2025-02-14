use crate::elem::{Elem, ElemBasics};
use crate::parameters::Parameters;
use crate::plaintext::Plaintext;
use base2k::{Infos, Module, VecZnx, VecZnxApi, VmpPMat, VmpPMatOps};

pub struct Ciphertext(pub Elem<VecZnx>);

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

    pub fn at(&self, i: usize) -> &(impl VecZnxApi + Infos) {
        self.0.at(i)
    }

    pub fn at_mut(&mut self, i: usize) -> &mut (impl VecZnxApi + Infos) {
        self.0.at_mut(i)
    }

    pub fn log_base2k(&self) -> usize {
        self.0.log_base2k()
    }

    pub fn log_scale(&self) -> usize {
        self.0.log_scale
    }

    pub fn zero(&mut self) {
        self.0.zero()
    }

    pub fn as_plaintext(&self) -> Plaintext<VecZnx> {
        unsafe { Plaintext(std::ptr::read(&self.0)) }
    }
}

impl Parameters {
    pub fn new_ciphertext(&self, log_q: usize) -> Ciphertext {
        Ciphertext::new(self.module(), self.log_base2k(), log_q, self.log_scale(), 1)
    }
}

pub struct GadgetCiphertext {
    pub value: VmpPMat,
    pub log_base2k: usize,
    pub log_q: usize,
}

impl GadgetCiphertext {
    pub fn new(module: &Module, log_base2k: usize, rows: usize, log_q: usize) -> Self {
        let cols: usize = (log_q + log_base2k - 1) / log_base2k;
        Self {
            value: module.new_vmp_pmat(rows, cols * 2),
            log_base2k,
            log_q,
        }
    }

    pub fn n(&self) -> usize {
        self.value.n
    }

    pub fn rows(&self) -> usize {
        self.value.rows
    }

    pub fn cols(&self) -> usize {
        self.value.cols
    }

    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn log_base2k(&self) -> usize {
        self.log_base2k
    }
}

pub struct RGSWCiphertext {
    pub value: VmpPMat,
    pub log_base2k: usize,
    pub log_q: usize,
}

impl RGSWCiphertext {
    pub fn new(module: &Module, log_base2k: usize, rows: usize, log_q: usize) -> Self {
        let cols: usize = (log_q + log_base2k - 1) / log_base2k;
        Self {
            value: module.new_vmp_pmat(rows * 2, cols * 2),
            log_base2k,
            log_q,
        }
    }

    pub fn n(&self) -> usize {
        self.value.n
    }

    pub fn rows(&self) -> usize {
        self.value.rows
    }

    pub fn cols(&self) -> usize {
        self.value.cols
    }

    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn log_base2k(&self) -> usize {
        self.log_base2k
    }
}
