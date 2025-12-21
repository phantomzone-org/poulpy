use poulpy_core::layouts::{Base2K, Degree, TorusPrecision};
use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, TakeSlice, VecZnxCopy, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes,
        VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxSwitchRing,
    },
    layouts::{Backend, Module, Scratch, ScratchOwned, VecZnx, ZnxInfos, ZnxViewMut},
    reference::{vec_znx::vec_znx_rotate_inplace, znx::ZnxRef},
};

#[derive(Debug, Clone, Copy)]
pub enum LookUpTableRotationDirection {
    Left,
    Right,
}

pub struct LookUpTableLayout {
    pub n: Degree,
    pub extension_factor: usize,
    pub k: TorusPrecision,
    pub base2k: Base2K,
}

pub trait LookupTableInfos {
    fn n(&self) -> Degree;
    fn extension_factor(&self) -> usize;
    fn k(&self) -> TorusPrecision;
    fn base2k(&self) -> Base2K;
    fn size(&self) -> usize;
}

impl LookupTableInfos for LookUpTableLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn extension_factor(&self) -> usize {
        self.extension_factor
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn size(&self) -> usize {
        self.k().as_usize().div_ceil(self.base2k().as_usize())
    }

    fn n(&self) -> Degree {
        self.n
    }
}

pub struct LookupTable {
    pub(crate) data: Vec<VecZnx<Vec<u8>>>,
    pub(crate) rot_dir: LookUpTableRotationDirection,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) drift: usize,
}

impl LookupTableInfos for LookupTable {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn extension_factor(&self) -> usize {
        self.data.len()
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }

    fn n(&self) -> Degree {
        self.data[0].n().into()
    }

    fn size(&self) -> usize {
        self.data[0].size()
    }
}

pub trait LookupTableFactory {
    fn lookup_table_set(&self, res: &mut LookupTable, f: &[i64], k: usize);
    fn lookup_table_rotate(&self, k: i64, res: &mut LookupTable);
}

impl LookupTable {
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: LookupTableInfos,
    {
        #[cfg(debug_assertions)]
        {
            assert!(
                infos.extension_factor() & (infos.extension_factor() - 1) == 0,
                "extension_factor must be a power of two but is: {}",
                infos.extension_factor()
            );
        }
        Self {
            data: (0..infos.extension_factor())
                .map(|_| VecZnx::alloc(infos.n().into(), 1, infos.size()))
                .collect(),
            base2k: infos.base2k(),
            k: infos.k(),
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

    pub fn set<M>(&mut self, module: &M, f: &[i64], k: usize)
    where
        M: LookupTableFactory,
    {
        module.lookup_table_set(self, f, k);
    }

    pub(crate) fn rotate<M>(&mut self, module: &M, k: i64)
    where
        M: LookupTableFactory,
    {
        module.lookup_table_rotate(k, self);
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
        .map(|&v| if v == 0 { 0 } else { v.unsigned_abs().ilog2() + 1 })
        .max()
        .unwrap_or(0)
}

impl<BE: Backend> LookupTableFactory for Module<BE>
where
    Self: VecZnxRotateInplace<BE>
        + VecZnxNormalizeInplace<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxCopy
        + VecZnxRotateInplaceTmpBytes
        + VecZnxRotateInplace<BE>
        + VecZnxRotateInplaceTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: TakeSlice,
{
    fn lookup_table_set(&self, res: &mut LookupTable, f: &[i64], k: usize) {
        assert!(f.len() <= self.n());

        let base2k: usize = res.base2k.into();

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.vec_znx_normalize_tmp_bytes().max(res.domain_size() << 3));

        // Get the number minimum limb to store the message modulus
        let limbs: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            assert!(f.len() <= self.n());
            assert!(
                (max_bit_size(f) + (k % base2k) as u32) < i64::BITS,
                "overflow: max(|f|) << (k%base2k) > i64::BITS"
            );
            assert!(limbs <= res.data[0].size());
        }

        // Scaling factor
        let mut scale = 1;
        if !k.is_multiple_of(base2k) {
            scale <<= base2k - (k % base2k);
        }

        // #elements in lookup table
        let f_len: usize = f.len();

        // If LUT size > TakeScalarZnx
        let domain_size: usize = res.domain_size();

        let size: usize = res.k.as_usize().div_ceil(base2k);

        // Equivalent to AUTO([f(0), -f(n-1), -f(n-2), ..., -f(1)], -1)
        let mut lut_full: VecZnx<Vec<u8>> = VecZnx::alloc(domain_size, 1, size);

        let lut_at: &mut [i64] = lut_full.at_mut(0, limbs - 1);

        let step: usize = domain_size.div_round(f_len);

        for (i, fi) in f.iter().enumerate() {
            let start: usize = i * step;
            let end: usize = start + step;
            lut_at[start..end].fill(fi * scale);
        }

        let drift: usize = step >> 1;

        // Rotates half the step to the left
        if res.extension_factor() > 1 {
            let (tmp, _) = scratch.borrow().take_slice(lut_full.n());

            for i in 0..res.extension_factor() {
                self.vec_znx_switch_ring(&mut res.data[i], 0, &lut_full, 0);
                if i < res.extension_factor() {
                    vec_znx_rotate_inplace::<_, ZnxRef>(-1, &mut lut_full, 0, tmp);
                }
            }
        } else {
            self.vec_znx_copy(&mut res.data[0], 0, &lut_full, 0);
        }

        for a in res.data.iter_mut() {
            self.vec_znx_normalize_inplace(res.base2k.into(), a, 0, scratch.borrow());
        }

        res.rotate(self, -(drift as i64));

        res.drift = drift
    }

    fn lookup_table_rotate(&self, k: i64, res: &mut LookupTable) {
        let extension_factor: usize = res.extension_factor();
        let two_n: usize = 2 * res.data[0].n();
        let two_n_ext: usize = two_n * extension_factor;

        let mut scratch: ScratchOwned<_> = ScratchOwned::alloc(self.vec_znx_rotate_inplace_tmp_bytes());

        let k_pos: usize = ((k + two_n_ext as i64) % two_n_ext as i64) as usize;

        let k_hi: usize = k_pos / extension_factor;
        let k_lo: usize = k_pos % extension_factor;

        (0..extension_factor - k_lo).for_each(|i| {
            self.vec_znx_rotate_inplace(k_hi as i64, &mut res.data[i], 0, scratch.borrow());
        });

        (extension_factor - k_lo..extension_factor).for_each(|i| {
            self.vec_znx_rotate_inplace(k_hi as i64 + 1, &mut res.data[i], 0, scratch.borrow());
        });

        res.data.rotate_right(k_lo);
    }
}
