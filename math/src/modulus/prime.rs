use crate::modulus::barrett::BarrettPrecomp;
use crate::modulus::montgomery::MontgomeryPrecomp;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prime<O> {
    pub q: O,
    /// q_base^q_powers
    pub two_q: O,
    pub four_q: O,
    pub q_base: O,
    pub q_power: usize,
    pub factors: Vec<O>,
    /// distinct factors of q-1
    pub montgomery: MontgomeryPrecomp<O>,
    pub barrett: BarrettPrecomp<O>,
    pub phi: O,
}

pub struct NTTFriendlyPrimesGenerator<O> {
    pub size: f64,
    pub next_prime: O,
    pub prev_prime: O,
    pub check_next_prime: bool,
    pub check_prev_prime: bool,
}
