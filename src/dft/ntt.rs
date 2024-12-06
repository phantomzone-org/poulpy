use crate::modulus::montgomery::Montgomery;
use crate::modulus::prime::Prime;

pub struct Table<'a, O>{
    prime:&'a Prime<O>,
    pub psi_forward_rev:Vec<Montgomery<O>>,
    psi_backward_rev: Vec<Montgomery<O>>,
    n_inv: Montgomery<O>,
}

impl<'a> Table<'a, u64> {
    pub fn new(prime: &'a mut Prime<u64>, n: u64, nth_root: u64)->Self{

        assert!(n&(n-1) == 0, "invalid argument: n = {} is not a power of two", n);
        assert!(n&(n-1) == 0, "invalid argument: nth_root = {} is not a power of two", nth_root);
        assert!(n < nth_root, "invalid argument: n = {} cannot be greater or equal to nth_root = {}", n, nth_root);

        let psi: u64 = prime.primitive_nth_root(nth_root);

        let psi_mont: Montgomery<u64> = prime.montgomery.prepare(psi);
        let psi_inv_mont: Montgomery<u64> = prime.montgomery.pow(psi_mont, prime.phi-1);
        
        let mut psi_forward_rev: Vec<Montgomery<u64>> = vec![Montgomery(0); (nth_root >> 1) as usize];
        let mut psi_backward_rev: Vec<Montgomery<u64>> = vec![Montgomery(0); (nth_root >> 1) as usize];

        psi_forward_rev[0] = prime.montgomery.one();
        psi_backward_rev[0] = prime.montgomery.one();

        let log_nth_root_half: usize = (usize::MAX - ((nth_root>>1 as usize)-1).leading_zeros() as usize) as usize;

        for i in 1..(nth_root>>1) as usize{

            let i_rev_prev: usize = (i-1).reverse_bits() >> (usize::MAX - log_nth_root_half) as usize;
            let i_rev_next: usize = i.reverse_bits() >> (usize::MAX - log_nth_root_half) as usize;

            psi_forward_rev[i_rev_next] = prime.montgomery.mul_internal(psi_forward_rev[i_rev_prev], psi_mont);
            psi_backward_rev[i_rev_next] = prime.montgomery.mul_internal(psi_backward_rev[i_rev_prev], psi_inv_mont);
        }

        let n_inv: Montgomery<u64> = prime.montgomery.pow(prime.montgomery.prepare(nth_root>>1), prime.phi-1);

        Self{ 
            prime: prime, 
            psi_forward_rev: psi_forward_rev, 
            psi_backward_rev: psi_backward_rev, 
            n_inv: n_inv,
        }
    }
}