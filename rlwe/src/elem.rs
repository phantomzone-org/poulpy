use crate::parameters::Parameters;
use base2k::{Infos, Module, VecZnx, VecZnxOps};

pub struct Elem {
    pub value: Vec<VecZnx>,
    pub log_base2k: usize,
    pub log_q: usize,
    pub log_scale: usize,
}

impl Elem {
    pub fn new(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        degree: usize,
        log_scale: usize,
    ) -> Self {
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let mut value: Vec<VecZnx> = Vec::new();
        (0..degree + 1).for_each(|_| value.push(module.new_vec_znx(limbs)));
        Self {
            value,
            log_q,
            log_base2k,
            log_scale: log_scale,
        }
    }

    pub fn bytes_of(module: &Module, log_base2k: usize, log_q: usize, degree: usize) -> usize {
        let cols = (log_q + log_base2k - 1) / log_base2k;
        module.n() * cols * (degree + 1) * 8
    }

    pub fn from_bytes(
        module: &Module,
        log_base2k: usize,
        log_q: usize,
        degree: usize,
        bytes: &mut [u8],
    ) -> Self {
        let n: usize = module.n();
        assert!(bytes.len() >= Self::bytes_of(module, log_base2k, log_q, degree));
        let mut value: Vec<VecZnx> = Vec::new();
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let size = VecZnx::bytes(n, limbs);
        let mut ptr: usize = 0;
        (0..degree + 1).for_each(|_| {
            value.push(VecZnx::from_bytes(n, limbs, &mut bytes[ptr..]));
            ptr += size
        });
        Self {
            value,
            log_q,
            log_base2k,
            log_scale: 0,
        }
    }

    pub fn n(&self) -> usize {
        self.value[0].n()
    }

    pub fn degree(&self) -> usize {
        self.value.len()
    }

    pub fn limbs(&self) -> usize {
        self.value[0].limbs()
    }

    pub fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    pub fn log_scale(&self) -> usize {
        self.log_scale
    }

    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn at(&self, i: usize) -> &VecZnx {
        assert!(i <= self.degree());
        &self.value[i]
    }

    pub fn at_mut(&mut self, i: usize) -> &mut VecZnx {
        assert!(i <= self.degree());
        &mut self.value[i]
    }

    pub fn zero(&mut self) {
        self.value.iter_mut().for_each(|i| i.zero());
    }
}

impl Parameters {
    pub fn bytes_of_elem(&self, log_q: usize, degree: usize) -> usize {
        Elem::bytes_of(self.module(), self.log_base2k(), log_q, degree)
    }

    pub fn elem_from_bytes(&self, log_q: usize, degree: usize, bytes: &mut [u8]) -> Elem {
        Elem::from_bytes(self.module(), self.log_base2k(), log_q, degree, bytes)
    }
}
