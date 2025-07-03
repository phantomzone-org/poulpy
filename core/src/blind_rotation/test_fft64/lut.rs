use backend::{FFT64, Module, ZnxView};

use crate::blind_rotation::lut::{DivRound, LookUpTable};

#[test]
fn standard() {
    let module: Module<FFT64> = Module::<FFT64>::new(32);
    let basek: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 1;

    let scale: usize = (1 << (basek - 1)) / message_modulus;

    fn lut_fn(x: i64) -> i64 {
        x - 8
    }

    let mut lut: LookUpTable = LookUpTable::alloc(&module, basek, k_lut, extension_factor);
    lut.set(&module, lut_fn, message_modulus);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(half_step);

    let step: usize = lut.domain_size().div_round(message_modulus);

    (0..lut.domain_size()).step_by(step).for_each(|i| {
        (0..step).for_each(|_| {
            assert_eq!(
                lut_fn((i / step) as i64) % message_modulus as i64,
                lut.data[0].raw()[0] / scale as i64
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

    let scale: usize = (1 << (basek - 1)) / message_modulus;

    fn lut_fn(x: i64) -> i64 {
        x - 8
    }

    let mut lut: LookUpTable = LookUpTable::alloc(&module, basek, k_lut, extension_factor);
    lut.set(&module, lut_fn, message_modulus);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(half_step);

    let step: usize = lut.domain_size().div_round(message_modulus);

    (0..lut.domain_size()).step_by(step).for_each(|i| {
        (0..step).for_each(|_| {
            assert_eq!(
                lut_fn((i / step) as i64) % message_modulus as i64,
                lut.data[0].raw()[0] / scale as i64
            );
            lut.rotate(-1);
        });
    });
}
