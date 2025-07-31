

#[cfg(test)]
mod tests {
    use super::{AddNormal, FillUniform};
    use crate::{Module, ModuleNew, VecZnx, VecZnxStd, FFT64};
    use crate::{VecZnxAlloc, znx_base::*};
    use sampling::source::Source;

    #[test]
    fn vec_znx_fill_uniform() {
        let n: usize = 4096;
        let module: Module<FFT64> = Module::<FFT64>::new(n as u64);
        let basek: usize = 17;
        let size: usize = 5;
        let mut source: Source = Source::new([0u8; 32]);
        let cols: usize = 2;
        let zero: Vec<i64> = vec![0; n];
        let one_12_sqrt: f64 = 0.28867513459481287;
        (0..cols).for_each(|col_i| {
            let mut a: VecZnx<_> = module.vec_znx_alloc(cols, size);
            a.fill_uniform(basek, col_i, size, &mut source);
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

    #[test]
    fn vec_znx_add_normal() {
        let n: usize = 4096;
        let module: Module<FFT64> = Module::<FFT64>::new(n as u64);
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
            let mut a: VecZnx<_> = module.vec_znx_alloc(cols, size);
            a.add_normal(basek, col_i, k, &mut source, sigma, bound);
            (0..cols).for_each(|col_j| {
                if col_j != col_i {
                    (0..size).for_each(|limb_i| {
                        assert_eq!(a.at(col_j, limb_i), zero);
                    })
                } else {
                    let std: f64 = module.vec_znx_std(basek, a, col_i) * k_f64;
                    assert!((std - sigma).abs() < 0.1, "std={} ~!= {}", std, sigma);
                }
            })
        });
    }
}
