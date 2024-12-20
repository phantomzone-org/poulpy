use crate::modulus::montgomery::{MontgomeryPrecomp, Montgomery};
use crate::modulus::shoup::{ShoupPrecomp};


pub struct Prime<O> {
	pub q: O, /// q_base^q_powers
    pub q_base: O,
    pub q_power: O,
    pub factors: Vec<O>, /// distinct factors of q-1
    pub montgomery: MontgomeryPrecomp<O>,
    pub shoup:ShoupPrecomp<O>,
    pub phi: O,
}

pub struct NTTFriendlyPrimesGenerator<O>{
	pub size: f64,
	pub next_prime: O,
	pub prev_prime: O,
	pub check_next_prime: bool,
	pub check_prev_prime: bool,
}
