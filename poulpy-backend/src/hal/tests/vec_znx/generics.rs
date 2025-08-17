use crate::hal::{
    api::{VecZnxAddNormal, VecZnxFillUniform, ZnxView},
    layouts::{Backend, Module, VecZnx},
    source::Source,
};

pub fn test_vec_znx_fill_uniform<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxFillUniform,
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
                let std: f64 = a.std(basek, col_i);
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
    Module<B>: VecZnxAddNormal,
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
                let std: f64 = a.std(basek, col_i) * k_f64;
                assert!((std - sigma).abs() < 0.1, "std={} ~!= {}", std, sigma);
            }
        })
    });
}
