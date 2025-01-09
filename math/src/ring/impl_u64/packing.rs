use std::collections::HashMap;
use crate::poly::Poly;
use crate::ring::Ring;
use crate::modulus::{ONCE, WordOps};
use crate::modulus::barrett::Barrett;
use std::cmp::min;
use std::mem::transmute;


impl Ring<u64>{

    // Generates a vector storing {X^{2^0}, X^{2^1}, .., X^{2^log_n}}.
    pub fn gen_x_pow_2<const NTT: bool, const INV: bool>(&self, log_n: usize) -> Vec<Poly<u64>>{
        let mut x_pow: Vec<Poly<u64>> = Vec::<Poly<u64>>::with_capacity(log_n);

        (0..log_n).for_each(|i|{
            let mut idx: usize = 1<<i;

            if INV{
                idx = self.n() - idx;
            }

            x_pow.push(self.new_poly());

            if i == 0{
                x_pow[i].0[idx] = self.modulus.montgomery.one();
                self.ntt_inplace::<false>(&mut x_pow[i]);
            }else{
                let (left, right) = x_pow.split_at_mut(i);
                self.a_mul_b_montgomery_into_c::<ONCE>(&left[i-1], &left[i-1], &mut right[0]);
            }
        });

        if INV{
            self.a_neg_into_a::<1, ONCE>(&mut x_pow[0]);
        }

        if !NTT{
            x_pow.iter_mut().for_each(|x| self.intt_inplace::<false>(x));
        }

        x_pow
    }

    pub fn pack<'a, const ZEROGARBAGE: bool, const NTT: bool>(&self, polys: &'a mut HashMap<usize, &'a mut Poly<u64>>, log_gap: usize) -> &'a Poly<u64>{
        
        let log_n: usize = self.log_n();
        let log_nth_root: usize = log_n+1;
        let nth_root: usize = 1<<log_nth_root;
        let log_start: usize = log_n - log_gap;
        let mut log_end: usize = log_n;

        let mut keys: Vec<usize> = polys.keys().copied().collect();
        keys.sort();

        let mut gap = 0usize;

        if keys.len() > 1{
            gap = max_pow2_gap(&keys);
        }else{
            gap = 1<<log_n;
        }

        let log_gap: usize = gap.log2();

        if !ZEROGARBAGE{
            if gap > 0 {
                log_end -= log_gap;
            }
        }
        
        let n_inv: Barrett<u64> = self.modulus.barrett.prepare(self.modulus.inv(1<<(log_end - log_start)));

        for (_, poly) in polys.iter_mut() {
            if !NTT{
                self.ntt_inplace::<true>(poly);
            }

            self.a_mul_b_scalar_barrett_into_a::<ONCE>(&n_inv,  poly);
        }

        let x_pow2: Vec<Poly<u64>> = self.gen_x_pow_2::<true, false>(log_n);
        let mut tmpa: Poly<u64> = self.new_poly();
        let mut tmpb: Poly<u64> = self.new_poly();

        for i in log_start..log_end{

            let t: usize = 1<<(log_n-1-i);

            for j in 0..t{

                let option_lo: Option<&&mut Poly<u64>> = polys.get(&i);
                let option_hi: Option<&&mut Poly<u64>> = polys.get(&(i+t));
                let mut hi_exists: bool = false;

                match option_hi{
                    Some(hi) =>{

                            // Unsafe code is necessary because two mutables references are
                            // accessed from the map.
                            unsafe{
                                self.a_mul_b_montgomery_into_a::<ONCE>(&x_pow2[log_n-i-1],  transmute(*hi as *const Poly<u64> as *mut Poly<u64>));
                            }

                            hi_exists = true;

                            match option_lo{
                                Some(lo) =>{

                                    self.a_sub_b_into_c::<1, ONCE>(lo, hi, &mut tmpa);
                                    
                                    // Ensures unsafe blocks are "safe".
                                    let ptr_hi: *mut Poly<u64> = *hi as *const Poly<u64> as *mut Poly<u64>;
                                    let ptr_lo: *mut Poly<u64> = *lo as *const Poly<u64> as *mut Poly<u64>;
                                    assert!(ptr_hi != ptr_lo, "something went seriously wrong");

                                    unsafe{
                                        self.a_add_b_into_b::<ONCE>(hi, transmute(ptr_lo));
                                    }
                                }
                                None =>{
                                    unsafe{
                                        polys.insert(j, transmute(*hi as *const Poly<u64> as *mut Poly<u64>));
                                    }
                                },
                            }

                            polys.remove(&(j+t));
                        }

                    None =>{},  
                }

                let option_lo: Option<&&mut Poly<u64>> = polys.get(&i);
                let option_hi: Option<&&mut Poly<u64>> = polys.get(&(i+t));

                match option_lo{
                    Some(lo) =>{
                        
                        let gal_el: usize = self.galois_element(1<<(i-1), i == 0, log_nth_root);

                        if hi_exists{
                            self.automorphism::<true>(&tmpa, gal_el, 2<<self.log_n(), &mut tmpb);
                        }else{
                            self.automorphism::<true>(*lo, gal_el, nth_root, &mut tmpa);
                        }
                        unsafe{
                            self.a_add_b_into_b::<ONCE>(&tmpa, transmute(*lo as *const Poly<u64> as *mut Poly<u64>));
                        }
                    }

                    None =>{
                        match option_hi{
                            Some(hi) =>{
                                let gal_el: usize = self.galois_element(1<<(i-1), i == 0, log_nth_root);

                                self.automorphism::<true>(*hi, gal_el, nth_root, &mut tmpa);

                                unsafe{
                                    self.a_sub_b_into_a::<1, ONCE>(&tmpa, transmute(*hi as *const Poly<u64> as *mut Poly<u64>))
                                }
                            }

                            None =>{}
                        }
                    }
                }
            }
        }

        *polys.get(&0).unwrap()
    }
}


// Returns the largest
fn max_pow2_gap(vec: &[usize]) -> usize{
    let mut gap: usize = usize::MAX;
    for i in 1..vec.len(){
        let (l, r) = (vec[i-1], vec[i]);
        assert!(l > r, "invalid input vec: not sorted");
        gap = min(gap, r-l);
        if gap == 1{
            break;
        }
    };
    1 << gap.trailing_zeros()
}