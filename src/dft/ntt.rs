use crate::modulus::montgomery::Montgomery;
use crate::modulus::shoup::Shoup;
use crate::modulus::prime::Prime;
use crate::modulus::ReduceOnce;
use crate::modulus::WordOps;
use crate::dft::DFT;
use itertools::izip;

pub struct Table<'a, O>{
    prime:&'a Prime<O>,
    psi_forward_rev:Vec<Shoup<u64>>,
    psi_backward_rev: Vec<Shoup<u64>>,
    q:O,
    two_q:O,
    four_q:O,
}

impl<'a> Table<'a, u64> {
    pub fn new(prime: &'a mut Prime<u64>, nth_root: u64)->Self{

        assert!(nth_root&(nth_root-1) == 0, "invalid argument: nth_root = {} is not a power of two", nth_root);

        let psi: u64 = prime.primitive_nth_root(nth_root);

        let psi_mont: Montgomery<u64> = prime.montgomery.prepare(psi);
        let psi_inv_mont: Montgomery<u64> = prime.montgomery.pow(psi_mont, prime.phi-1);
        
        let mut psi_forward_rev: Vec<Shoup<u64>> = vec![Shoup(0, 0); (nth_root >> 1) as usize];
        let mut psi_backward_rev: Vec<Shoup<u64>> = vec![Shoup(0, 0); (nth_root >> 1) as usize];

        psi_forward_rev[0] = prime.shoup.prepare(1);
        psi_backward_rev[0] = prime.shoup.prepare(1);

        let log_nth_root_half: u32 = (nth_root>>1).log2() as _;

        let mut powers_forward: u64 = 1u64;
        let mut powers_backward: u64 = 1u64;

        for i in 1..(nth_root>>1) as usize{

            let i_rev: usize = i.reverse_bits_msb(log_nth_root_half);

            prime.montgomery.mul_external_assign(psi_mont, &mut powers_forward);
            prime.montgomery.mul_external_assign(psi_inv_mont, &mut powers_backward);

            psi_forward_rev[i_rev] = prime.shoup.prepare(powers_forward);
            psi_backward_rev[i_rev] = prime.shoup.prepare(powers_backward); 
        }

        Self{ 
            prime: prime, 
            psi_forward_rev: psi_forward_rev, 
            psi_backward_rev: psi_backward_rev,
            q:prime.q(),
            two_q:prime.q()<<1,
            four_q:prime.q()<<2,
        }
    }

    // Returns n^-1 mod q in Montgomery.
    fn inv(&self, n:u64) -> Montgomery<u64>{
        self.prime.montgomery.pow(self.prime.montgomery.prepare(n), self.prime.phi-1)
    }
}


impl<'a> DFT<u64> for Table<'a,u64>{
    fn forward_inplace(&self, a: &mut [u64]){
        self.forward_inplace(a)
    }

    fn forward_inplace_lazy(&self, a: &mut [u64]){
        self.forward_inplace_lazy(a)
    }

    fn backward_inplace(&self, a: &mut [u64]){
        self.backward_inplace(a)
    }

    fn backward_inplace_lazy(&self, a: &mut [u64]){
        self.backward_inplace_lazy(a)
    }
}

impl<'a> Table<'a,u64>{

    pub fn forward_inplace_lazy(&self, a: &mut [u64]){
        self.forward_inplace_core::<true>(a);
    }

    pub fn forward_inplace(&self, a: &mut [u64]){
        self.forward_inplace_core::<false>(a);
    }

    pub fn forward_inplace_core<const LAZY: bool>(&self, a: &mut [u64]) {

        let n: usize = a.len();
        assert!(n & n-1 == 0, "invalid x.len()= {} must be a power of two", n);
        let log_n: u32 = usize::BITS - ((n as usize)-1).leading_zeros();

        for layer in 0..log_n {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            let t: usize = 2*size;
            if layer == log_n - 1 {
                if LAZY{
                    izip!(a.chunks_exact_mut(t), &self.psi_forward_rev[m..]).for_each(|(a, psi)| {
                        let (a, b) = a.split_at_mut(size);
                        self.dit::<false>(&mut a[0], &mut b[0], *psi);
                        debug_assert!(a[0] < self.two_q, "forward_inplace_core::<LAZY=true> output {} > {} (2q-1)", a[0], self.two_q-1);
                        debug_assert!(b[0] < self.two_q, "forward_inplace_core::<LAZY=true> output {} > {} (2q-1)", b[0], self.two_q-1);
                    });
                }else{
                    izip!(a.chunks_exact_mut(t), &self.psi_forward_rev[m..]).for_each(|(a, psi)| {
                        let (a, b) = a.split_at_mut(size);
                        self.dit::<true>(&mut a[0], &mut b[0], *psi);
                        self.prime.shoup.reduce_assign(&mut a[0]);
                        self.prime.shoup.reduce_assign(&mut b[0]);
                        debug_assert!(a[0] < self.q, "forward_inplace_core::<LAZY=false> output {} > {} (q-1)", a[0], self.q-1);
                        debug_assert!(b[0] < self.q, "forward_inplace_core::<LAZY=false> output {} > {} (q-1)", b[0], self.q-1);
                    });
                }
                
            } else if t >= 16{
                izip!(a.chunks_exact_mut(t), &self.psi_forward_rev[m..]).for_each(|(a, psi)| {
                    let (a, b) = a.split_at_mut(size);
                    izip!(a.chunks_exact_mut(8), b.chunks_exact_mut(8)).for_each(|(a, b)| {
                        self.dit::<true>(&mut a[0], &mut b[0], *psi);
                        self.dit::<true>(&mut a[1], &mut b[1], *psi);
                        self.dit::<true>(&mut a[2], &mut b[2], *psi);
                        self.dit::<true>(&mut a[3], &mut b[3], *psi);
                        self.dit::<true>(&mut a[4], &mut b[4], *psi);
                        self.dit::<true>(&mut a[5], &mut b[5], *psi);
                        self.dit::<true>(&mut a[6], &mut b[6], *psi);
                        self.dit::<true>(&mut a[7], &mut b[7], *psi);
                    });
                });
            }else{
                izip!(a.chunks_exact_mut(t), &self.psi_forward_rev[m..]).for_each(|(a, psi)| {
                    let (a, b) = a.split_at_mut(size);
                    izip!(a, b).for_each(|(a, b)| self.dit::<true>(a, b, *psi));
                });
            }
        }
    }

    #[inline(always)]
    fn dit<const LAZY: bool>(&self, a: &mut u64, b: &mut u64, t: Shoup<u64>) {
        debug_assert!(*a < self.four_q, "a:{} q:{}", a, self.four_q);
        debug_assert!(*b < self.four_q, "b:{} q:{}", b, self.four_q);
        a.reduce_once_assign(self.two_q);
        let bt: u64 = self.prime.shoup.mul_external_lazy(t, *b);
        *b = a.wrapping_add(self.two_q-bt);
        *a = a.wrapping_add(bt);
        if !LAZY {
            a.reduce_once_assign(self.two_q);
            b.reduce_once_assign(self.two_q);
        }
    }

    pub fn backward_inplace_lazy(&self, a: &mut [u64]){
        self.backward_inplace_core::<true>(a);
    }

    pub fn backward_inplace(&self, a: &mut [u64]){
        self.backward_inplace_core::<false>(a);
    }

    pub fn backward_inplace_core<const LAZY:bool>(&self, a: &mut [u64]) {
        let n: usize = a.len();
        assert!(n & n-1 == 0, "invalid x.len()= {} must be a power of two", n);
        let log_n = usize::BITS - ((n as usize)-1).leading_zeros();

        for layer in (0..log_n).rev() {
            let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
            let t: usize = 2*size;
            if layer == 0 {

                let n_inv: Shoup<u64> = self.prime.shoup.prepare(self.prime.inv(n as u64));
                let psi: Shoup<u64> = self.prime.shoup.prepare(self.prime.shoup.mul_external(n_inv, self.psi_backward_rev[1].0));

                izip!(a.chunks_exact_mut(2 * size)).for_each(
                    |a| {
                        let (a, b) = a.split_at_mut(size);
                        izip!(a.chunks_exact_mut(8), b.chunks_exact_mut(8)).for_each(|(a, b)| {
                            self.dif_last::<LAZY>(&mut a[0], &mut b[0], psi, n_inv);
                            self.dif_last::<LAZY>(&mut a[1], &mut b[1], psi, n_inv);
                            self.dif_last::<LAZY>(&mut a[2], &mut b[2], psi, n_inv);
                            self.dif_last::<LAZY>(&mut a[3], &mut b[3], psi, n_inv);
                            self.dif_last::<LAZY>(&mut a[4], &mut b[4], psi, n_inv);
                            self.dif_last::<LAZY>(&mut a[5], &mut b[5], psi, n_inv);
                            self.dif_last::<LAZY>(&mut a[6], &mut b[6], psi, n_inv);
                            self.dif_last::<LAZY>(&mut a[7], &mut b[7], psi, n_inv);
                        });
                    },
                );

            } else if t >= 16{
                izip!(a.chunks_exact_mut(t), &self.psi_backward_rev[m..]).for_each(
                    |(a, psi)| {
                        let (a, b) = a.split_at_mut(size);
                        izip!(a.chunks_exact_mut(8), b.chunks_exact_mut(8)).for_each(|(a, b)| {
                            self.dif::<true>(&mut a[0], &mut b[0], *psi);
                            self.dif::<true>(&mut a[1], &mut b[1], *psi);
                            self.dif::<true>(&mut a[2], &mut b[2], *psi);
                            self.dif::<true>(&mut a[3], &mut b[3], *psi);
                            self.dif::<true>(&mut a[4], &mut b[4], *psi);
                            self.dif::<true>(&mut a[5], &mut b[5], *psi);
                            self.dif::<true>(&mut a[6], &mut b[6], *psi);
                            self.dif::<true>(&mut a[7], &mut b[7], *psi);
                        });
                    },
                );
            } else {
                izip!(a.chunks_exact_mut(2 * size), &self.psi_backward_rev[m..]).for_each(
                    |(a, psi)| {
                        let (a, b) = a.split_at_mut(size);
                        izip!(a, b).for_each(|(a, b)| self.dif::<true>(a, b, *psi));
                    },
                );
            }
        }
    }

    #[inline(always)]
    fn dif<const LAZY: bool>(&self, a: &mut u64, b: &mut u64, t: Shoup<u64>) {
        debug_assert!(*a < self.two_q);
        debug_assert!(*b < self.two_q);
        let d: u64 = self.prime.shoup.mul_external_lazy(t, *a + self.two_q - *b);
        *a = a.wrapping_add(*b);
        a.reduce_once_assign(self.two_q);
        *b = d;
        if !LAZY {
            a.reduce_once_assign(self.q);
            b.reduce_once_assign(self.q);
        }
    }

    fn dif_last<const LAZY:bool>(&self, a: &mut u64, b: &mut u64, psi: Shoup<u64>, n_inv: Shoup<u64>){
        debug_assert!(*a < self.two_q);
        debug_assert!(*b < self.two_q);
        if LAZY{
            let d: u64 = self.prime.shoup.mul_external_lazy(psi, *a + self.two_q - *b);
            *a = self.prime.shoup.mul_external_lazy(n_inv, *a + *b);
            *b = d;
        }else{
            let d: u64 = self.prime.shoup.mul_external(psi, *a + self.two_q - *b);
            *a = self.prime.shoup.mul_external(n_inv, *a + *b);
            *b = d;
        }
    }
}
