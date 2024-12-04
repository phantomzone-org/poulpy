use primality_test::is_prime;
use prime_factorization::Factorization;

pub struct Prime<O> {
	q: O, /// q_base^q_powers
    q_base: O,
    q_powers: O,
    factors: Vec<O>, /// distinct factors of q-1
    nth_root: O,
}

impl Prime<u64>{
    pub fn new(q_base: u64, q_power: u64) -> Self{
		assert!(is_prime(q) && q > 2);
        assert!()

        q_exp
        for i in 0..q_power{

        }

        Self::new_unchecked(q)
	}

	pub fn new_unchecked(q: u64) -> Self {
        assert!(q.next_power_of_two().ilog2() <= 61);
        Self {
            q,
        }
    }

    /// Returns returns Phi(BaseModulus^BaseModulusPower)
    pub fn phi() -> u64 {

    }

    /// Returns the smallest primitive root. The unique factors
    /// can be given as argument to avoid factorization of q-1.
    pub fn primitive_root(&self) -> u64{
        if self.factors.len() != 0{
            self.check_factors();
        }else{
            let factors = Factorization::run(q).prime_factor_repr();
            let mut distincts_factors: Vec<u64> = Vec::with_capacity(factors.len());
            for factor in factors.iter(){
                distincts_factors.push(factor.0)
            }
            self.factors = distincts_factors
        }

        let log_nth_root = 64 - self.q.leading_zeros() as usize;

        0
    }

    pub fn check_factors(&self){

        if self.factors.len() == 0{
            panic!("invalid factor list: empty")
        }

        let mut q = self.q;

        for &factor in &self.factors{
            if !is_prime(factor){
                panic!("invalid factor list: factor {} is not prime", factor)
            }

            while q%factor != 0{
                q /= factor
            }
        }

        if q != 1{
            panic!("invalid factor list: does not fully divide q: q % (alll factors) = {}", q)
        }
    }
}