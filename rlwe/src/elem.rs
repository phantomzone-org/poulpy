use crate::parameters::Parameters;
use base2k::{Infos, VecZnx};

pub struct Elem {
    pub value: Vec<VecZnx>,
    pub log_base2k: usize,
    pub log_q: usize,
}

impl Elem {
    pub fn new(n: usize, log_base2k: usize, log_q: usize, degree: usize) -> Self {
        let limbs: usize = (log_q + log_base2k - 1) / log_base2k;
        let mut value: Vec<VecZnx> = Vec::new();
        (0..degree + 1).for_each(|_| value.push(VecZnx::new(n, limbs)));
        Self {
            value,
            log_base2k,
            log_q,
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
}
