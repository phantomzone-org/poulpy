use crate::modulus::barrett::BarrettPrecomp;

/// Montgomery is a generic struct storing
/// an element in the Montgomery domain.
pub type Montgomery<O> = O;

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
