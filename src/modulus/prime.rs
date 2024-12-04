use primality_test::is_prime;

pub struct Prime {
	q: u64,
}

impl Prime {
	pub fn new(q: u64) -> Self{
		assert!(is_prime(q) && q > 2);
        Self::new_unchecked(q)
	}

	pub fn new_unchecked(q: u64) -> Self {
        assert!(q.next_power_of_two().ilog2() <= 61);
        Self {
            q,
        }
    }
}