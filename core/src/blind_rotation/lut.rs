use backend::{FFT64, Module, ScalarZnx, VecZnx, VecZnxAlloc, VecZnxOps, ZnxInfos, ZnxView, ZnxViewMut, alloc_aligned};

pub struct LookUpTable {
    pub(crate) data: Vec<VecZnx<Vec<u8>>>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
}

impl LookUpTable {
    pub fn alloc(module: &Module<FFT64>, basek: usize, k: usize, extension_factor: usize) -> Self {
        let size: usize = k.div_ceil(basek);
        let mut data: Vec<VecZnx<Vec<u8>>> = Vec::with_capacity(extension_factor);
        (0..extension_factor).for_each(|_| {
            data.push(module.new_vec_znx(1, size));
        });
        Self { data, basek, k }
    }

    pub fn extension_factor(&self) -> usize {
        self.data.len()
    }

    pub fn domain_size(&self) -> usize {
        self.data.len() * self.data[0].n()
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
        let domain_size: usize = self.domain_size();

        let size: usize = self.k.div_ceil(self.basek);

        // Equivalent to AUTO([f(0), -f(n-1), -f(n-2), ..., -f(1)], -1)
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
                    lut_at[i] = y;
                })
            });
        }

        // Rotates half the step to the left
        let half_step: usize = domain_size.div_round(message_modulus << 1);

        lut_full.rotate(-(half_step as i64));

        let mut tmp_bytes: Vec<u8> = alloc_aligned(lut_full.n() * size_of::<i64>());
        lut_full.normalize(self.basek, 0, &mut tmp_bytes);

        if self.extension_factor() > 1 {
            (0..self.extension_factor()).for_each(|i| {
                module.switch_degree(&mut self.data[i], 0, &lut_full, 0);
                if i < self.extension_factor() {
                    lut_full.rotate(-1);
                }
            });
        } else {
            module.vec_znx_copy(&mut self.data[0], 0, &lut_full, 0);
        }
    }

    pub fn set_raw<D>(&mut self, module: &Module<FFT64>, lut: &ScalarZnx<D>)
    where
        D: AsRef<[u8]>,
    {
        let domain_size: usize = self.domain_size();

        let size: usize = self.k.div_ceil(self.basek);

        let mut lut_full: VecZnx<Vec<u8>> = VecZnx::new::<i64>(domain_size, 1, size);

        lut_full.at_mut(0, 0).copy_from_slice(lut.raw());

        if self.extension_factor() > 1 {
            (0..self.extension_factor()).for_each(|i| {
                module.switch_degree(&mut self.data[i], 0, &lut_full, 0);
                if i < self.extension_factor() {
                    lut_full.rotate(-1);
                }
            });
        } else {
            module.vec_znx_copy(&mut self.data[0], 0, &lut_full, 0);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rotate(&mut self, k: i64) {
        let extension_factor: usize = self.extension_factor();
        let two_n: usize = 2 * self.data[0].n();
        let two_n_ext: usize = two_n * extension_factor;

        let k_pos: usize = ((k + two_n_ext as i64) % two_n_ext as i64) as usize;

        let k_hi: usize = k_pos / extension_factor;
        let k_lo: usize = k_pos % extension_factor;

        (0..extension_factor - k_lo).for_each(|i| {
            self.data[i].rotate(k_hi as i64);
        });

        (extension_factor - k_lo..extension_factor).for_each(|i| {
            self.data[i].rotate(k_hi as i64 + 1);
        });

        self.data.rotate_right(k_lo as usize);
    }
}

pub(crate) trait DivRound {
    fn div_round(self, rhs: Self) -> Self;
}

impl DivRound for usize {
    #[inline]
    fn div_round(self, rhs: Self) -> Self {
        (self + rhs / 2) / rhs
    }
}
