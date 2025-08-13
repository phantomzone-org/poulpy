use backend::hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotateInplace,
        VecZnxSwithcDegree, ZnxInfos, ZnxViewMut,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnx},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

pub struct LookUpTable {
    pub(crate) data: Vec<VecZnx<Vec<u8>>>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
}

impl LookUpTable {
    pub fn alloc(n: usize, basek: usize, k: usize, extension_factor: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(
                extension_factor & (extension_factor - 1) == 0,
                "extension_factor must be a power of two but is: {}",
                extension_factor
            );
        }
        let size: usize = k.div_ceil(basek);
        let mut data: Vec<VecZnx<Vec<u8>>> = Vec::with_capacity(extension_factor);
        (0..extension_factor).for_each(|_| {
            data.push(VecZnx::alloc(n, 1, size));
        });
        Self { data, basek, k }
    }

    pub fn log_extension_factor(&self) -> usize {
        (usize::BITS - (self.extension_factor() - 1).leading_zeros()) as _
    }

    pub fn extension_factor(&self) -> usize {
        self.data.len()
    }

    pub fn domain_size(&self) -> usize {
        self.data.len() * self.data[0].n()
    }

    pub fn set<B: Backend>(&mut self, module: &Module<B>, f: &Vec<i64>, k: usize)
    where
        Module<B>: VecZnxRotateInplace + VecZnxNormalizeInplace<B> + VecZnxNormalizeTmpBytes + VecZnxSwithcDegree + VecZnxCopy,
        B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
    {
        assert!(f.len() <= module.n());

        let basek: usize = self.basek;

        // Get the number minimum limb to store the message modulus
        let limbs: usize = k.div_ceil(1 << basek);

        #[cfg(debug_assertions)]
        {
            assert!(limbs <= self.data[0].size());
        }

        // Scaling factor
        let scale: i64 = 1 << (k % basek) as i64;

        // #elements in lookup table
        let f_len: usize = f.len();

        // If LUT size > TakeScalarZnx
        let domain_size: usize = self.domain_size();

        let size: usize = self.k.div_ceil(self.basek);

        // Equivalent to AUTO([f(0), -f(n-1), -f(n-2), ..., -f(1)], -1)
        let mut lut_full: VecZnx<Vec<u8>> = VecZnx::alloc(domain_size, 1, size);

        let lut_at: &mut [i64] = lut_full.at_mut(0, limbs - 1);

        f.iter().enumerate().for_each(|(i, fi)| {
            let start: usize = (i * domain_size).div_round(f_len);
            let end: usize = ((i + 1) * domain_size).div_round(f_len);
            lut_at[start..end].fill(fi * scale);
        });

        // Rotates half the step to the left
        let half_step: usize = domain_size.div_round(f_len << 1);

        module.vec_znx_rotate_inplace(-(half_step as i64), &mut lut_full, 0);

        let n_large: usize = lut_full.n();

        module.vec_znx_normalize_inplace(
            self.basek,
            &mut lut_full,
            0,
            ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes(n_large)).borrow(),
        );

        if self.extension_factor() > 1 {
            (0..self.extension_factor()).for_each(|i| {
                module.vec_znx_switch_degree(&mut self.data[i], 0, &lut_full, 0);
                if i < self.extension_factor() {
                    module.vec_znx_rotate_inplace(-1, &mut lut_full, 0);
                }
            });
        } else {
            module.vec_znx_copy(&mut self.data[0], 0, &lut_full, 0);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn rotate<B: Backend>(&mut self, module: &Module<B>, k: i64)
    where
        Module<B>: VecZnxRotateInplace,
    {
        let extension_factor: usize = self.extension_factor();
        let two_n: usize = 2 * self.data[0].n();
        let two_n_ext: usize = two_n * extension_factor;

        let k_pos: usize = ((k + two_n_ext as i64) % two_n_ext as i64) as usize;

        let k_hi: usize = k_pos / extension_factor;
        let k_lo: usize = k_pos % extension_factor;

        (0..extension_factor - k_lo).for_each(|i| {
            module.vec_znx_rotate_inplace(k_hi as i64, &mut self.data[i], 0);
        });

        (extension_factor - k_lo..extension_factor).for_each(|i| {
            module.vec_znx_rotate_inplace(k_hi as i64 + 1, &mut self.data[i], 0);
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
