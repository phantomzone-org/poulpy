use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxCopy, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotateInplace,
        VecZnxRotateInplaceTmpBytes, VecZnxSwitchRing,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnx, ZnxInfos, ZnxViewMut},
};

#[derive(Debug, Clone, Copy)]
pub enum LookUpTableRotationDirection {
    Left,
    Right,
}

pub struct LookUpTable {
    pub(crate) data: Vec<VecZnx<Vec<u8>>>,
    pub(crate) rot_dir: LookUpTableRotationDirection,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) drift: usize,
}

impl LookUpTable {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, extension_factor: usize) -> Self {
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
            data.push(VecZnx::alloc(module.n(), 1, size));
        });
        Self {
            data,
            basek,
            k,
            drift: 0,
            rot_dir: LookUpTableRotationDirection::Left,
        }
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

    pub fn rotation_direction(&self) -> LookUpTableRotationDirection {
        self.rot_dir
    }

    // By default X^{-dec(lwe)} is computed during the blind rotation.
    // Setting [reverse_rotation] to true will reverse the sign of
    // rotation of the LUT by instead evaluating X^{dec(lwe)} during
    // the blind rotation.
    pub fn set_rotation_direction(&mut self, rot_dir: LookUpTableRotationDirection) {
        self.rot_dir = rot_dir
    }

    pub fn set<B: Backend>(&mut self, module: &Module<B>, f: &[i64], k: usize)
    where
        Module<B>: VecZnxRotateInplace<B>
            + VecZnxNormalizeInplace<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxSwitchRing<B>
            + VecZnxCopy
            + VecZnxRotateInplaceTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        assert!(f.len() <= module.n());

        let basek: usize = self.basek;

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        // Get the number minimum limb to store the message modulus
        let limbs: usize = k.div_ceil(basek);

        #[cfg(debug_assertions)]
        {
            assert!(f.len() <= module.n());
            assert!(
                (max_bit_size(f) + (k % basek) as u32) < i64::BITS,
                "overflow: max(|f|) << (k%basek) > i64::BITS"
            );
            assert!(limbs <= self.data[0].size());
        }

        // Scaling factor
        let mut scale = 1;
        if !k.is_multiple_of(basek) {
            scale <<= basek - (k % basek);
        }

        // #elements in lookup table
        let f_len: usize = f.len();

        // If LUT size > TakeScalarZnx
        let domain_size: usize = self.domain_size();

        let size: usize = self.k.div_ceil(self.basek);

        // Equivalent to AUTO([f(0), -f(n-1), -f(n-2), ..., -f(1)], -1)
        let mut lut_full: VecZnx<Vec<u8>> = VecZnx::alloc(domain_size, 1, size);

        let lut_at: &mut [i64] = lut_full.at_mut(0, limbs - 1);

        let step: usize = domain_size.div_round(f_len);

        f.iter().enumerate().for_each(|(i, fi)| {
            let start: usize = i * step;
            let end: usize = start + step;
            lut_at[start..end].fill(fi * scale);
        });

        let drift: usize = step >> 1;

        // Rotates half the step to the left

        if self.extension_factor() > 1 {
            (0..self.extension_factor()).for_each(|i| {
                module.vec_znx_switch_ring(&mut self.data[i], 0, &lut_full, 0);
                if i < self.extension_factor() {
                    module.vec_znx_rotate_inplace(-1, &mut lut_full, 0, scratch.borrow());
                }
            });
        } else {
            module.vec_znx_copy(&mut self.data[0], 0, &lut_full, 0);
        }

        self.data.iter_mut().for_each(|a| {
            module.vec_znx_normalize_inplace(self.basek, a, 0, scratch.borrow());
        });

        self.rotate(module, -(drift as i64));

        self.drift = drift
    }

    #[allow(dead_code)]
    pub(crate) fn rotate<B: Backend>(&mut self, module: &Module<B>, k: i64)
    where
        Module<B>: VecZnxRotateInplace<B> + VecZnxRotateInplaceTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let extension_factor: usize = self.extension_factor();
        let two_n: usize = 2 * self.data[0].n();
        let two_n_ext: usize = two_n * extension_factor;

        let mut scratch: ScratchOwned<_> = ScratchOwned::alloc(module.vec_znx_rotate_inplace_tmp_bytes());

        let k_pos: usize = ((k + two_n_ext as i64) % two_n_ext as i64) as usize;

        let k_hi: usize = k_pos / extension_factor;
        let k_lo: usize = k_pos % extension_factor;

        (0..extension_factor - k_lo).for_each(|i| {
            module.vec_znx_rotate_inplace(k_hi as i64, &mut self.data[i], 0, scratch.borrow());
        });

        (extension_factor - k_lo..extension_factor).for_each(|i| {
            module.vec_znx_rotate_inplace(k_hi as i64 + 1, &mut self.data[i], 0, scratch.borrow());
        });

        self.data.rotate_right(k_lo);
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

#[allow(dead_code)]
fn max_bit_size(vec: &[i64]) -> u32 {
    vec.iter()
        .map(|&v| {
            if v == 0 {
                0
            } else {
                v.unsigned_abs().ilog2() + 1
            }
        })
        .max()
        .unwrap_or(0)
}
