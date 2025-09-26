use std::vec;

use poulpy_hal::{
    api::{
        VecZnxCopy, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes,
        VecZnxSwitchRing,
    },
    layouts::{Backend, Module},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use crate::tfhe::blind_rotation::{DivRound, LookUpTable};

pub fn test_lut_standard<B>(module: &Module<B>)
where
    Module<B>: VecZnxRotateInplace<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxCopy
        + VecZnxRotateInplaceTmpBytes,
    B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
{
    let base2k: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 1;

    let log_scale: usize = base2k + 1;

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i as i64) - 8);

    let mut lut: LookUpTable = LookUpTable::alloc(module, base2k, k_lut, extension_factor);
    lut.set(module, &f, log_scale);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(module, half_step);

    let step: usize = lut.domain_size().div_round(message_modulus);

    let mut lut_dec: Vec<i64> = vec![0i64; module.n()];
    lut.data[0].decode_vec_i64(base2k, 0, log_scale, &mut lut_dec);

    (0..lut.domain_size()).step_by(step).for_each(|i| {
        (0..step).for_each(|_| {
            assert_eq!(f[i / step] % message_modulus as i64, lut_dec[i]);
        });
    });
}

pub fn test_lut_extended<B>(module: &Module<B>)
where
    Module<B>: VecZnxRotateInplace<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSwitchRing
        + VecZnxCopy
        + VecZnxRotateInplaceTmpBytes,
    B: Backend + ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
{
    let base2k: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 4;

    let log_scale: usize = base2k + 1;

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i as i64) - 8);

    let mut lut: LookUpTable = LookUpTable::alloc(module, base2k, k_lut, extension_factor);
    lut.set(module, &f, log_scale);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(module, half_step);

    let step: usize = module.n().div_round(message_modulus);

    let mut lut_dec: Vec<i64> = vec![0i64; module.n()];

    (0..extension_factor).for_each(|ext| {
        lut.data[ext].decode_vec_i64(base2k, 0, log_scale, &mut lut_dec);
        (0..module.n()).step_by(step).for_each(|i| {
            (0..step).for_each(|_| {
                assert_eq!(f[i / step] % message_modulus as i64, lut_dec[i]);
            });
        });
    });
}
