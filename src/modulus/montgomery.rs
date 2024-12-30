use crate::modulus::barrett::BarrettPrecomp;

/// Montgomery is a generic struct storing
/// an element in the Montgomery domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Montgomery<O>(pub O);

/// Implements helper methods on the struct Montgomery<O>.
impl<O> Montgomery<O>{

    #[inline(always)]
    pub fn new(lhs: O) -> Self{
        Self(lhs)
    }

    #[inline(always)]
    pub fn value(&self) -> &O{
        &self.0
    }

    pub fn value_mut(&mut self) -> &mut O{
        &mut self.0
    }
}

/// Default instantiation.
impl<O> Default for  Montgomery<O> where O:Default {
    fn default() -> Self {
        Self {
            0: O::default(),
        }
    }
}

/// MontgomeryPrecomp is a generic struct storing 
/// precomputations for Montgomery arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MontgomeryPrecomp<O>{
    pub q: O,
    pub two_q: O,
    pub four_q: O,
    pub barrett: BarrettPrecomp<O>,
    pub q_inv: O,
    pub one: Montgomery<O>,
    pub minus_one: Montgomery<O>,
}
