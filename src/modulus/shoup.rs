#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shoup<O>(pub O, pub O);

impl<O> Shoup<O> {
    
    #[inline(always)]
    pub fn value(&self) -> &O {
        &self.0
    }

    #[inline(always)]
    pub fn quotient(&self) -> &O {
        &self.1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShoupPrecomp<O>{
    pub q: O,
    pub one: Shoup<O>,
}

