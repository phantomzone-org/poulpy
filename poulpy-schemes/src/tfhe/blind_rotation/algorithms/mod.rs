mod cggi;

pub use cggi::*;

use itertools::izip;
use poulpy_core::{
    ScratchTakeCore,
    layouts::{GGSWInfos, GLWE, GLWEInfos, LWE, LWEInfos},
};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Scratch, ZnxView};

use crate::tfhe::blind_rotation::{BlindRotationKeyInfos, BlindRotationKeyPrepared, LookUpTableRotationDirection, LookupTable};

pub trait BlindRotationAlgo: Sync {}

pub trait BlindRotationExecute<BRA: BlindRotationAlgo, BE: Backend> {
    fn blind_rotation_execute_tmp_bytes<G, B>(
        &self,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &G,
        brk_infos: &B,
    ) -> usize
    where
        G: GLWEInfos,
        B: GGSWInfos;

    fn blind_rotation_execute<DR, DL, DB>(
        &self,
        res: &mut GLWE<DR>,
        lwe: &LWE<DL>,
        lut: &LookupTable,
        brk: &BlindRotationKeyPrepared<DB, BRA, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        DR: DataMut,
        DL: DataRef,
        DB: DataRef;
}

impl<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> BlindRotationKeyPrepared<D, BRA, BE>
where
    Scratch<BE>: ScratchTakeCore<BE>,
{
    pub fn execute<DR: DataMut, DI: DataRef, M>(
        &self,
        module: &M,
        res: &mut GLWE<DR>,
        lwe: &LWE<DI>,
        lut: &LookupTable,
        scratch: &mut Scratch<BE>,
    ) where
        M: BlindRotationExecute<BRA, BE>,
    {
        module.blind_rotation_execute(res, lwe, lut, self, scratch);
    }
}

impl<BE: Backend, BRA: BlindRotationAlgo> BlindRotationKeyPrepared<Vec<u8>, BRA, BE> {
    pub fn execute_tmp_bytes<A, B, M>(
        module: &M,
        block_size: usize,
        extension_factor: usize,
        glwe_infos: &A,
        brk_infos: &B,
    ) -> usize
    where
        A: GLWEInfos,
        B: BlindRotationKeyInfos,
        M: BlindRotationExecute<BRA, BE>,
    {
        module.blind_rotation_execute_tmp_bytes(block_size, extension_factor, glwe_infos, brk_infos)
    }
}

pub fn mod_switch_2n(n: usize, res: &mut [i64], lwe: &LWE<&[u8]>, rot_dir: LookUpTableRotationDirection) {
    let base2k: usize = lwe.base2k().into();

    let log2n: usize = usize::BITS as usize - (n - 1).leading_zeros() as usize + 1;

    res.copy_from_slice(lwe.data().at(0, 0));

    match rot_dir {
        LookUpTableRotationDirection::Left => {
            res.iter_mut().for_each(|x| *x = -*x);
        }
        LookUpTableRotationDirection::Right => {}
    }

    if base2k > log2n {
        let diff: usize = base2k - (log2n - 1); // additional -1 because we map to [-N/2, N/2) instead of [0, N)
        res.iter_mut().for_each(|x| {
            *x = div_round_by_pow2(x, diff);
        })
    } else {
        let rem: usize = base2k - (log2n % base2k);
        let size: usize = log2n.div_ceil(base2k);
        (1..size).for_each(|i| {
            if i == size - 1 && rem != base2k {
                let k_rem: usize = base2k - rem;
                izip!(lwe.data().at(0, i).iter(), res.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << k_rem) + (x >> rem);
                });
            } else {
                izip!(lwe.data().at(0, i).iter(), res.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << base2k) + x;
                });
            }
        })
    }
}

#[inline(always)]
fn div_round_by_pow2(x: &i64, k: usize) -> i64 {
    (x + (1 << (k - 1))) >> k
}
