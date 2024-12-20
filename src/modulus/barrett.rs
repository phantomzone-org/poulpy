#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BarrettPrecomp<O>{
    pub q: O,
    pub lo:O,
    pub hi:O,
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

