pub struct BarrettPrecomp<O>(O, O);

impl<O> BarrettPrecomp<O>{

    #[inline(always)]
    pub fn new(a:O, b: O) -> Self{
        Self(a, b)
    }

    #[inline(always)]
    pub fn value_hi(&self) -> &O{
        &self.0
    }

    #[inline(always)]
    pub fn value_lo(&self) -> &O{
        &self.1
    }
}