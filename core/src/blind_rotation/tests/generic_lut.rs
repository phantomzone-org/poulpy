use std::vec;

use backend::hal::{
    api::{VecZnxCopy, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotateInplace, VecZnxSwithcDegree, ZnxView},
    layouts::{Backend, Module},
    oep::{ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl},
};

use crate::{DivRound, LookUpTable};

pub(crate) fn test_lut_standard<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxRotateInplace + VecZnxNormalizeInplace<B> + VecZnxNormalizeTmpBytes + VecZnxSwithcDegree + VecZnxCopy,
    B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
{
    let n: usize = module.n();
    let basek: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 1;

    let log_scale: usize = basek + 1;

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i as i64) - 8);

    let mut lut: LookUpTable = LookUpTable::alloc(n, basek, k_lut, extension_factor);
    lut.set(module, &f, log_scale);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(module, half_step);

    let step: usize = lut.domain_size().div_round(message_modulus);

    (0..lut.domain_size()).step_by(step).for_each(|i| {
        (0..step).for_each(|_| {
            assert_eq!(
                f[i / step] % message_modulus as i64,
                lut.data[0].raw()[0] / (1 << (log_scale % basek)) as i64
            );
            lut.rotate(module, -1);
        });
    });
}

pub(crate) fn test_lut_extended<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxRotateInplace + VecZnxNormalizeInplace<B> + VecZnxNormalizeTmpBytes + VecZnxSwithcDegree + VecZnxCopy,
    B: ScratchOwnedAllocImpl<B> + ScratchOwnedBorrowImpl<B>,
{
    let n: usize = module.n();
    let basek: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 4;

    let log_scale: usize = basek + 1;

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i as i64) - 8);

    let mut lut: LookUpTable = LookUpTable::alloc(n, basek, k_lut, extension_factor);
    lut.set(&module, &f, log_scale);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(&module, half_step);

    let step: usize = lut.domain_size().div_round(message_modulus);

    (0..lut.domain_size()).step_by(step).for_each(|i| {
        (0..step).for_each(|_| {
            assert_eq!(
                f[i / step] % message_modulus as i64,
                lut.data[0].raw()[0] / (1 << (log_scale % basek)) as i64
            );
            lut.rotate(&module, -1);
        });
    });
}
