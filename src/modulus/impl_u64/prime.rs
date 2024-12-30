use crate::modulus::prime::Prime;
use crate::modulus::montgomery::{Montgomery, MontgomeryPrecomp};
use crate::modulus::shoup::{ShoupPrecomp};
use crate::modulus::ONCE;
use primality_test::is_prime;
use prime_factorization::Factorization;

impl Prime<u64>{

    /// Returns a new instance of Prime<u64>.
    /// Panics if q_base is not a prime > 2 and
    /// if q_base^q_power would overflow u64.
    pub fn new(q_base: u64, q_power: usize) -> Self{
		assert!(is_prime(q_base) && q_base > 2);
        Self::new_unchecked(q_base, q_power)
	}

    /// Returns a new instance of Prime<u64>.
    /// Does not check if q_base is a prime > 2.
    /// Panics if q_base^q_power would overflow u64.
	pub fn new_unchecked(q_base: u64, q_power: usize) -> Self {

        let mut q = q_base;
        for _i in 1..q_power{
            q *= q_base
        }

        assert!(q.next_power_of_two().ilog2() <= 61);

        let mut phi = q_base-1;
        for _i in 1..q_power{
            phi *= q_base
        }

        let mut prime: Prime<u64> = Self {
            q:q,
            q_base:q_base,
            q_power:q_power,
            factors: Vec::new(),
            montgomery:MontgomeryPrecomp::new(q),
            shoup:ShoupPrecomp::new(q),
            phi:phi,
        };

        prime.check_factors();

        prime

    }

    pub fn q(&self) -> u64{
        self.q
    }

    pub fn q_base(&self) -> u64{
        self.q_base
    }

    pub fn q_power(&self) -> usize{
        self.q_power
    }

    /// Returns x^exponen mod q.
    #[inline(always)]
    pub fn pow(&self, x: u64, exponent: u64) -> u64{
        let mut y_mont: Montgomery<u64> = self.montgomery.one();
        let mut x_mont: Montgomery<u64> = self.montgomery.prepare::<ONCE>(x);
        let mut i: u64 = exponent;
        while i > 0{
            if i & 1 == 1{
                self.montgomery.mul_internal_assign::<ONCE>(x_mont, &mut y_mont);
            }

            self.montgomery.mul_internal_assign::<ONCE>(x_mont, &mut x_mont);

            i >>= 1;

        }
        
        self.montgomery.unprepare::<ONCE>(y_mont)
    }

    /// Returns x^-1 mod q.
    /// User must ensure that x is not divisible by q_base.
    #[inline(always)]
    pub fn inv(&self, x: u64) -> u64{
        self.pow(x, self.phi-1)
    }
}

impl Prime<u64>{
    /// Returns the smallest nth primitive root of q_base.
    pub fn primitive_root(&self) -> u64{

        let mut candidate: u64 = 1u64;
        let mut not_found: bool = true;

        while not_found{

            candidate += 1;

            for &factor in &self.factors{

                if  Pow(candidate, (self.q_base-1)/factor, self.q_base) == 1{
                    not_found = true;
                    break
                }
                not_found = false;
            }
        }

        if not_found{
            panic!("failed to find a primitive root for q_base={}", self.q_base)
        }

        candidate
    }

    /// Returns an nth primitive root of q = q_base^q_power in Montgomery.
    pub fn primitive_nth_root(&self, nth_root:u64) -> u64{

        assert!(self.q & (nth_root-1) == 1, "invalid prime: q = {} % nth_root = {} = {} != 1", self.q, nth_root, self.q & (nth_root-1));

        let psi: u64 = self.primitive_root();

        // nth primitive root mod q_base: psi_nth^(prime.q_base-1)/nth_root mod q_base
        let psi_nth_q_base: u64 = Pow(psi, (self.q_base-1)/nth_root, self.q_base);

        // lifts nth primitive root mod q_base to q = q_base^q_power
        let psi_nth_q: u64 = self.hensel_lift(psi_nth_q_base, nth_root);

        assert!(self.pow(psi_nth_q, nth_root) == 1, "invalid nth primitive root: psi^nth_root != 1 mod q");
        assert!(self.pow(psi_nth_q, nth_root>>1) == self.q-1, "invalid nth primitive root: psi^(nth_root/2) != -1 mod q");

        psi_nth_q
    }

    /// Checks if the field self.factor is populated.
    /// If not, factorize q_base-1 and populates self.factor.
    /// If yes, checks that it contains the unique factors of q_base-1.
    pub fn check_factors(&mut self){

        if self.factors.len() == 0{

            let factors = Factorization::run(self.q_base-1).prime_factor_repr();
            let mut distincts_factors: Vec<u64> = Vec::with_capacity(factors.len());
            for factor in factors.iter(){
                distincts_factors.push(factor.0)
            }
            self.factors = distincts_factors
            
        }else{
            let mut q_base: u64 = self.q_base;

            for &factor in &self.factors{
                if !is_prime(factor){
                    panic!("invalid factor list: factor {} is not prime", factor)
                }
    
                while q_base%factor != 0{
                    q_base /= factor
                }
            }
    
            if q_base != 1{
                panic!("invalid factor list: does not fully divide q_base: q_base % (all factors) = {}", q_base)
            }
        }        
    }

    /// Returns (psi + a * q_base)^{nth_root} = 1 mod q = q_base^q_power given psi^{nth_root} = 1 mod q_base.
    /// Panics if psi^{nth_root} != 1 mod q_base.
    fn hensel_lift(&self, psi: u64, nth_root: u64) -> u64{
        assert!(Pow(psi, nth_root, self.q_base)==1, "invalid argument psi: psi^nth_root = {} != 1", Pow(psi, nth_root, self.q_base));

        let mut psi_mont: Montgomery<u64> = self.montgomery.prepare::<ONCE>(psi);
        let nth_root_mont: Montgomery<u64> = self.montgomery.prepare::<ONCE>(nth_root);

        for _i in 1..self.q_power{

            let psi_pow: Montgomery<u64> = self.montgomery.pow(psi_mont, nth_root-1);

            let num: Montgomery<u64> = Montgomery(self.montgomery.one().value() + self.q - self.montgomery.mul_internal::<ONCE>(psi_pow, psi_mont).value());

            let mut den: Montgomery<u64> = self.montgomery.mul_internal::<ONCE>(nth_root_mont, psi_pow);

            den = self.montgomery.pow(den, self.phi-1);

            psi_mont = self.montgomery.add_internal(psi_mont, self.montgomery.mul_internal::<ONCE>(num, den));
        }
        
        self.montgomery.unprepare::<ONCE>(psi_mont)
    }
}

/// Returns x^exponent mod q.
/// This function internally instantiate a new MontgomeryPrecomp<u64>
/// To be used when called only a few times and if there 
/// is no Prime instantiated with q.
pub fn Pow(x:u64, exponent:u64, q:u64) -> u64{
    let montgomery: MontgomeryPrecomp<u64> = MontgomeryPrecomp::<u64>::new(q);
    let mut y_mont: Montgomery<u64> = montgomery.one();
    let mut x_mont: Montgomery<u64> = montgomery.prepare::<ONCE>(x);
    let mut i: u64 = exponent;
    while i > 0{
        if i & 1 == 1{
            montgomery.mul_internal_assign::<ONCE>(x_mont, &mut y_mont);
        }

        montgomery.mul_internal_assign::<ONCE>(x_mont, &mut x_mont);

        i >>= 1;
    }
    
    montgomery.unprepare::<ONCE>(y_mont)
}