use std::vec;

use backend::{FFT64, Module, ModuleNew, ZnxView};

use crate::blind_rotation::lut::{DivRound, LookUpTable};

#[test]
fn standard() {
    let module: Module<FFT64> = Module::<FFT64>::new(32);
    let basek: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 1;

    let log_scale: usize = basek + 1;

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i as i64) - 8);

    let mut lut: LookUpTable = LookUpTable::alloc(&module, basek, k_lut, extension_factor);
    lut.set(&module, &f, log_scale);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(half_step);

    let step: usize = lut.domain_size().div_round(message_modulus);

    (0..lut.domain_size()).step_by(step).for_each(|i| {
        (0..step).for_each(|_| {
            assert_eq!(
                f[i / step] % message_modulus as i64,
                lut.data[0].raw()[0] / (1 << (log_scale % basek)) as i64
            );
            lut.rotate(-1);
        });
    });
}

#[test]
fn extended() {
    let module: Module<FFT64> = Module::<FFT64>::new(32);
    let basek: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 4;

    let log_scale: usize = basek + 1;

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i as i64) - 8);

    let mut lut: LookUpTable = LookUpTable::alloc(&module, basek, k_lut, extension_factor);
    lut.set(&module, &f, log_scale);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(half_step);

    let step: usize = lut.domain_size().div_round(message_modulus);

    (0..lut.domain_size()).step_by(step).for_each(|i| {
        (0..step).for_each(|_| {
            assert_eq!(
                f[i / step] % message_modulus as i64,
                lut.data[0].raw()[0] / (1 << (log_scale % basek)) as i64
            );
            lut.rotate(-1);
        });
    });
}
