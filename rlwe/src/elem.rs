use crate::parameters::Parameters;
use base2k::VecZnx;

pub struct Elem {
    pub value: Vec<VecZnx>,
    pub log_scale: usize,
}

impl Parameters {
    pub fn new_elem(&self, degree: usize, log_q: usize) -> Elem {
        let mut value: Vec<VecZnx> = Vec::new();
        (0..degree + 1).for_each(|_| value.push(VecZnx::new(self.n(), self.log_base2k(), log_q)));
        Elem {
            value: value,
            log_scale: self.log_scale(),
        }
    }
}
