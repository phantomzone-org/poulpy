use crate::modulus::prime;

use prime::Prime;
use primality_test::is_prime;

pub struct NTTFriendlyPrimesGenerator{
	size: f64,
	next_prime: u64,
	prev_prime: u64,
	nth_root: u64,
	check_next_prime: bool,
	check_prev_prime: bool,
}

impl NTTFriendlyPrimesGenerator {
	pub fn new(bit_size: u64, nth_root: u64) -> Self{
		let mut check_next_prime: bool = true;
		let mut check_prev_prime: bool = true;
		let next_prime = (1<<bit_size) + 1;
		let mut prev_prime = next_prime;

		if next_prime > nth_root.wrapping_neg(){
			check_next_prime = false;
		}

		if prev_prime < nth_root{
			check_prev_prime = false
		}

		prev_prime -= nth_root;

		Self{
			size: bit_size as f64,
			check_next_prime,
			check_prev_prime,
			nth_root,
			next_prime,
			prev_prime,
		}
	}

	pub fn next_upstream_primes(&mut self, k: usize) -> Vec<Prime>{
		let mut primes: Vec<Prime> = Vec::with_capacity(k);
		for i in 0..k{
			primes.push(self.next_upstream_prime())
		}
		primes
	}

	pub fn next_downstream_primes(&mut self, k: usize) -> Vec<Prime>{
		let mut primes: Vec<Prime> = Vec::with_capacity(k);
		for i in 0..k{
			primes.push(self.next_downstream_prime())
		}
		primes
	}

	pub fn next_alternating_primes(&mut self, k: usize) -> Vec<Prime>{
		let mut primes: Vec<Prime> = Vec::with_capacity(k);
		for i in 0..k{
			primes.push(self.next_alternating_prime())
		}
		primes
	}

	pub fn next_upstream_prime(&mut self) -> Prime{
		loop  {
			if self.check_next_prime{
				if (self.next_prime as f64).log2() - self.size >= 0.5 || self.next_prime > 0xffff_ffff_ffff_ffff-self.nth_root{
					self.check_next_prime = false;
					panic!("prime list for upstream primes is exhausted (overlap with next bit-size or prime > 2^64)");
				}
			}else{
				if is_prime(self.next_prime) {
					let prime = Prime::new_unchecked(self.next_prime);
					self.next_prime += self.nth_root;
					return prime
				}
				self.next_prime += self.nth_root;
			}
		}
	}

	pub fn next_downstream_prime(&mut self) -> Prime{
		loop  {
			if self.size - (self.prev_prime as f64).log2() >= 0.5 || self.prev_prime < self.nth_root{
				self.check_next_prime = false;
				panic!("prime list for downstream primes is exhausted (overlap with previous bit-size or prime < nth_root)")
			}else{
				if is_prime(self.prev_prime){
					let prime = Prime::new_unchecked(self.next_prime);
					self.prev_prime -= self.nth_root;
					return prime
				}
				self.prev_prime -= self.nth_root; 
			}
		}
	}

	pub fn next_alternating_prime(&mut self) -> Prime{
		loop {
			if !(self.check_next_prime || self.check_prev_prime){
				panic!("prime list for upstream and downstream prime is exhausted for the (overlap with previous/next bit-size or NthRoot > prime > 2^64)")
			}

			if self.check_next_prime{
				if (self.next_prime as f64).log2() - self.size >= 0.5 || self.next_prime > 0xffff_ffff_ffff_ffff-self.nth_root{
					self.check_next_prime = false;
				}else{
					if is_prime(self.next_prime){
						let prime = Prime::new_unchecked(self.next_prime);
						self.next_prime += self.nth_root;
						return prime
					}
					self.next_prime += self.nth_root;
				}
			}

			if self.check_prev_prime {
				if self.size - (self.prev_prime as f64).log2() >= 0.5 || self.prev_prime < self.nth_root{
					self.check_prev_prime = false;
				}else{
					if is_prime(self.prev_prime){
						let prime = Prime::new_unchecked(self.prev_prime);
						self.prev_prime -= self.nth_root;
						return prime
					}
					self.prev_prime -= self.nth_root;
				}
			}
		}
	}
}


#[cfg(test)]
mod test {
    use crate::modulus::prime_generator;

    #[test]
    fn prime_generation() {
    	let nth_root: u64 = 1<<16 ;
    	let mut g: prime_generator::NTTFriendlyPrimesGenerator = prime_generator::NTTFriendlyPrimesGenerator::new(30, nth_root);

    	let primes = g.next_alternating_primes(10);
    	println!("{:?}", primes);
    	for prime in primes.iter(){
    		assert!(prime.q() % nth_root == 1);
    	}
    }
}