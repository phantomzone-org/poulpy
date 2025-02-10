use base2k::module::{MODULETYPE, Module};

pub struct ParametersLiteral {
    pub log_n: usize,
    pub log_q: usize,
    pub log_p: usize,
    pub log_base2k: usize,
    pub log_scale: usize,
    pub xe: f64,
    pub xs: usize,
}

pub struct Parameters {
    log_n: usize,
    log_q: usize,
    log_p: usize,
    log_scale: usize,
    log_base2k: usize,
    xe: f64,
    xs: usize,
    module: Module,
}

impl Parameters {
    pub fn new<const MTYPE: MODULETYPE>(p: &ParametersLiteral) -> Self {
        assert!(
            p.log_n + 2 * p.log_base2k <= 53,
            "invalid parameters: p.log_n + 2*p.log_base2k > 53"
        );
        Self {
            log_n: p.log_n,
            log_q: p.log_q,
            log_p: p.log_p,
            log_scale: p.log_scale,
            log_base2k: p.log_base2k,
            xe: p.xe,
            xs: p.xs,
            module: Module::new::<MTYPE>(1 << p.log_n),
        }
    }

    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    pub fn log_scale(&self) -> usize {
        self.log_scale
    }

    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn log_p(&self) -> usize {
        self.log_p
    }

    pub fn log_qp(&self) -> usize {
        self.log_q + self.log_p
    }

    pub fn limbs_q(&self) -> usize {
        (self.log_q + self.log_base2k - 1) / self.log_base2k
    }

    pub fn limbs_qp(&self) -> usize {
        (self.log_q + self.log_p + self.log_base2k - 1) / self.log_base2k
    }

    pub fn log_base2k(&self) -> usize {
        self.log_base2k
    }

    pub fn module(&self) -> &Module {
        &self.module
    }

    pub fn xe(&self) -> f64 {
        self.xe
    }

    pub fn xs(&self) -> usize {
        self.xs
    }
}
