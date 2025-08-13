use itertools::izip;
use sampling::source::Source;

use crate::hal::{
    api::{VecZnxAddNormal, VecZnxDecodeVeci64, VecZnxEncodeVeci64, VecZnxFillUniform, VecZnxStd, ZnxInfos, ZnxView, ZnxViewMut},
    layouts::{Backend, Module, VecZnx},
};

pub fn test_vec_znx_fill_uniform<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxFillUniform + VecZnxStd,
{
    let n: usize = module.n();
    let basek: usize = 17;
    let size: usize = 5;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let one_12_sqrt: f64 = 0.28867513459481287;
    (0..cols).for_each(|col_i| {
        let mut a: VecZnx<_> = VecZnx::alloc(n, cols, size);
        module.vec_znx_fill_uniform(basek, &mut a, col_i, size * basek, &mut source);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = module.vec_znx_std(basek, &a, col_i);
                assert!(
                    (std - one_12_sqrt).abs() < 0.01,
                    "std={} ~!= {}",
                    std,
                    one_12_sqrt
                );
            }
        })
    });
}

pub fn test_vec_znx_add_normal<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAddNormal + VecZnxStd,
{
    let n: usize = module.n();
    let basek: usize = 17;
    let k: usize = 2 * 17;
    let size: usize = 5;
    let sigma: f64 = 3.2;
    let bound: f64 = 6.0 * sigma;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << k as u64) as f64;
    (0..cols).for_each(|col_i| {
        let mut a: VecZnx<_> = VecZnx::alloc(n, cols, size);
        module.vec_znx_add_normal(basek, &mut a, col_i, k, &mut source, sigma, bound);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = module.vec_znx_std(basek, &a, col_i) * k_f64;
                assert!((std - sigma).abs() < 0.1, "std={} ~!= {}", std, sigma);
            }
        })
    });
}

pub fn test_vec_znx_encode_vec_i64_lo_norm<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxEncodeVeci64 + VecZnxDecodeVeci64,
{
    let n: usize = module.n();
    let basek: usize = 17;
    let size: usize = 5;
    let k: usize = size * basek - 5;
    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, 2, size);
    let mut source: Source = Source::new([0u8; 32]);
    let raw: &mut [i64] = a.raw_mut();
    raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
    (0..a.cols()).for_each(|col_i| {
        let mut have: Vec<i64> = vec![i64::default(); n];
        have.iter_mut()
            .for_each(|x| *x = (source.next_i64() << 56) >> 56);
        module.encode_vec_i64(basek, &mut a, col_i, k, &have, 10);
        let mut want: Vec<i64> = vec![i64::default(); n];
        module.decode_vec_i64(basek, &a, col_i, k, &mut want);
        izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
    });
}

pub fn test_vec_znx_encode_vec_i64_hi_norm<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxEncodeVeci64 + VecZnxDecodeVeci64,
{
    let n: usize = module.n();
    let basek: usize = 17;
    let size: usize = 5;
    for k in [1, basek / 2, size * basek - 5] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, 2, size);
        let mut source = Source::new([0u8; 32]);
        let raw: &mut [i64] = a.raw_mut();
        raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        (0..a.cols()).for_each(|col_i| {
            let mut have: Vec<i64> = vec![i64::default(); n];
            have.iter_mut().for_each(|x| {
                if k < 64 {
                    *x = source.next_u64n(1 << k, (1 << k) - 1) as i64;
                } else {
                    *x = source.next_i64();
                }
            });
            module.encode_vec_i64(basek, &mut a, col_i, k, &have, 63);
            let mut want: Vec<i64> = vec![i64::default(); n];
            module.decode_vec_i64(basek, &a, col_i, k, &mut want);
            izip!(want, have).for_each(|(a, b)| assert_eq!(a, b, "{} != {}", a, b));
        })
    }
}
