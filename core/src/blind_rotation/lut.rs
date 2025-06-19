use backend::{FFT64, Module, ScratchOwned, VecZnx, VecZnxAlloc, VecZnxOps, ZnxInfos, ZnxViewMut, alloc_aligned};

pub struct LookUpTable {
    pub(crate) data: Vec<VecZnx<Vec<u8>>>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
}

impl LookUpTable {
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, extend_factor: usize) -> Self {
        let size: usize = k.div_ceil(basek);
        let mut data: Vec<VecZnx<Vec<u8>>> = Vec::with_capacity(extend_factor);
        (0..extend_factor).for_each(|_| {
            data.push(module.new_vec_znx(1, size));
        });
        Self { data, basek, k }
    }

    pub fn set(&mut self, module: &Module<FFT64>, f: fn(i64) -> i64, message_modulus: usize) {
        let basek: usize = self.basek;

        // Get the number minimum limb to store the message modulus
        let limbs: usize = message_modulus.div_ceil(1 << basek);

        // Scaling factor
        let scale: i64 = (1 << (basek * limbs - 1)).div_round(message_modulus) as i64;

        // Updates function
        let f_scaled = |x: i64| (f(x) % message_modulus as i64) * scale;

        // If LUT size > module.n()
        let domain_size: usize = self.data[0].n() * self.data.len();

        let size: usize = self.k.div_ceil(self.basek);

        // Equivalent to AUTO([f(0), f(1), ..., f(n-1)], -1)
        let mut lut_full: VecZnx<Vec<u8>> = VecZnx::new::<i64>(domain_size, 1, size);
        {
            let lut_at: &mut [i64] = lut_full.at_mut(0, limbs - 1);

            let start: usize = 0;
            let end: usize = (domain_size).div_round(message_modulus);

            let y: i64 = f_scaled(0);
            (start..end).for_each(|i| {
                lut_at[i] = y;
            });

            (1..message_modulus).for_each(|x| {
                let start: usize = (x * domain_size).div_round(message_modulus);
                let end: usize = ((x + 1) * domain_size).div_round(message_modulus);
                let y: i64 = f_scaled(x as i64);
                (start..end).for_each(|i| {
                    lut_at[domain_size - i] = -y;
                })
            });
        }

        // Rotates half the step to the left
        let half_step: usize = domain_size.div_round(message_modulus << 1);
        module.vec_znx_rotate_inplace(-(half_step as i64), &mut lut_full, 0);

        let mut tmp_bytes: Vec<u8> = alloc_aligned(lut_full.n() * size_of::<i64>());
        lut_full.normalize(self.basek, 0, &mut tmp_bytes);

        if self.data.len() > 1 {
            let mut scratch: ScratchOwned = ScratchOwned::new(module.bytes_of_vec_znx(1, size));
            module.vec_znx_split(&mut self.data, 0, &lut_full, 0, scratch.borrow());
        } else {
            module.vec_znx_copy(&mut self.data[0], 0, &lut_full, 0);
        }
    }
}

pub trait DivRound {
    fn div_round(self, rhs: Self) -> Self;
}

impl DivRound for usize {
    #[inline]
    fn div_round(self, rhs: Self) -> Self {
        (self + rhs / 2) / rhs
    }
}
