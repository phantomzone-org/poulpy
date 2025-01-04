#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Barrett<O>(pub O, pub O);

impl<O> Barrett<O> {
    
    #[inline(always)]
    pub fn value(&self) -> &O {
        &self.0
    }

    #[inline(always)]
    pub fn quotient(&self) -> &O {
        &self.1
    }
}

pub struct BarrettRNS<O>(pub Vec<Barrett<O>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BarrettPrecomp<O>{
    pub q: O,
    pub two_q:O,
    pub four_q:O,
    pub lo:O,
    pub hi:O,
    pub one: Barrett<O>,
}

impl<O> BarrettPrecomp<O>{

    #[inline(always)]
    pub fn value_hi(&self) -> &O{
        &self.hi
    }

    #[inline(always)]
    pub fn value_lo(&self) -> &O{
        &self.lo
    }
}

