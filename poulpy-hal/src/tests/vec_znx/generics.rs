use crate::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAdd, VecZnxAddNormal, VecZnxFillUniform, VecZnxNormalize,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxSub,
    },
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    source::Source,
};

use itertools::izip;
use rand::RngCore;

pub fn test_vec_znx_normalize<B: Backend>(module: &Module<B>)
where
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    Module<B>: VecZnxNormalizeTmpBytes + VecZnxNormalize<B>,
{
    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let basek: usize = 17;
    let zero: Vec<i64> = vec![0i64; module.n()];

    let get_digit = |x: i64| -> i64 { (x << (i64::BITS - basek as u32)) >> (i64::BITS - basek as u32) };
    let get_carry = |x: i64, digit: i64| -> i64 { (x - digit) >> basek };

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        // Raw copy on b
        b.raw_mut().copy_from_slice(a.raw());

        // Reference
        let mut carry: Vec<i64> = vec![0i64; module.n()];
        for i in 0..cols {
            for j in (0..a_size).rev() {
                if j == a_size - 1 {
                    b.at_mut(i, j)
                        .iter_mut()
                        .zip(carry.iter_mut())
                        .for_each(|(x, c)| {
                            let digit: i64 = get_digit(*x);
                            *c = get_carry(*x, digit);
                            *x = digit;
                        });
                } else if j == 0 {
                    b.at_mut(i, j)
                        .iter_mut()
                        .zip(carry.iter_mut())
                        .for_each(|(x, c)| {
                            *x = get_digit(get_digit(*x) + *c);
                        });
                } else {
                    b.at_mut(i, j)
                        .iter_mut()
                        .zip(carry.iter_mut())
                        .for_each(|(x, c)| {
                            let digit: i64 = get_digit(*x);
                            let carry: i64 = get_carry(*x, digit);
                            let digit_plus_c: i64 = digit + *c;
                            *x = get_digit(digit_plus_c);
                            *c = carry + get_carry(digit_plus_c, *x);
                        });
                }
            }
        }

        for c_size in [1, 2, 6, 11] {
            let mut c: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, c_size);

            // Set scratch to garbage
            source.fill_bytes(&mut scratch.data);

            // Set c to garbage
            c.fill_uniform(&mut source);

            // Normalize on c
            for i in 0..cols {
                module.vec_znx_normalize(basek, &mut c, i, &a, i, scratch.borrow());
            }

            let min_size: usize = a_size.min(c_size);

            for i in 0..cols {
                for j in 0..min_size {
                    assert_eq!(c.at(i, j), b.at(i, j));
                }

                for j in min_size..c_size {
                    assert_eq!(c.at(i, j), zero);
                }
            }
        }
    }
}

pub fn test_vec_znx_add<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAdd + VecZnxNormalizeInplace<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0i64; module.n()];
    let basek = 12;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        for b_size in [1, 2, 6, 11] {
            let min_size: usize = a_size.min(b_size);
            let max_size: usize = a_size.max(b_size);

            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, b_size);
            let mut c: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, max_size);

            // Fill a with random i64
            a.fill_uniform(&mut source);
            b.fill_uniform(&mut source);

            // Normalize to avoid i64 overflow
            for i in 0..cols {
                module.vec_znx_normalize_inplace(basek, &mut a, i, scratch.borrow());
                module.vec_znx_normalize_inplace(basek, &mut b, i, scratch.borrow());
            }

            // Reference
            for i in 0..cols {
                for j in 0..min_size {
                    izip!(
                        a.at(i, j).iter(),
                        b.at(i, j).iter(),
                        c.at_mut(i, j).iter_mut()
                    )
                    .for_each(|(ai, bi, ci)| *ci = *ai + *bi);
                }

                for j in min_size..max_size {
                    if a_size > b_size {
                        c.at_mut(i, j).copy_from_slice(a.at(i, j));
                    } else if b_size > a_size {
                        c.at_mut(i, j).copy_from_slice(b.at(i, j));
                    }
                }
            }

            for d_size in [1, 2, 6, 11] {
                let mut d: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, d_size);

                // Set d to garbage
                d.fill_uniform(&mut source);

                // Normalize on c
                for i in 0..cols {
                    module.vec_znx_add(&mut d, i, &a, i, &b, i);
                }

                let min_size: usize = (a_size.max(b_size)).min(d_size);

                for i in 0..cols {
                    for j in 0..min_size {
                        assert_eq!(d.at(i, j), c.at(i, j));
                    }

                    for j in min_size..d_size {
                        assert_eq!(d.at(i, j), zero);
                    }
                }
            }
        }
    }
}

pub fn test_vec_znx_sub<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxSub + VecZnxNormalizeInplace<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0i64; module.n()];
    let basek = 12;

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        for b_size in [1, 2, 6, 11] {
            let min_size: usize = a_size.min(b_size);
            let max_size: usize = a_size.max(b_size);

            let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, b_size);
            let mut c: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, max_size);

            // Fill a with random i64
            a.fill_uniform(&mut source);
            b.fill_uniform(&mut source);

            // Normalize to avoid i64 overflow
            for i in 0..cols {
                module.vec_znx_normalize_inplace(basek, &mut a, i, scratch.borrow());
                module.vec_znx_normalize_inplace(basek, &mut b, i, scratch.borrow());
            }

            // Reference
            for i in 0..cols {
                for j in 0..min_size {
                    izip!(
                        a.at(i, j).iter(),
                        b.at(i, j).iter(),
                        c.at_mut(i, j).iter_mut()
                    )
                    .for_each(|(ai, bi, ci)| *ci = *ai - *bi);
                }

                for j in min_size..max_size {
                    if a_size > b_size {
                        izip!(c.at_mut(i, j), a.at(i, j)).for_each(|(ci, ai)| *ci = *ai);
                    } else if b_size > a_size {
                        izip!(c.at_mut(i, j), b.at(i, j)).for_each(|(ci, bi)| *ci = -*bi);
                    }
                }
            }

            for d_size in [1, 2, 6, 11] {
                let mut d: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, d_size);

                // Set d to garbage
                d.fill_uniform(&mut source);

                // Normalize on c
                for i in 0..cols {
                    module.vec_znx_sub(&mut d, i, &a, i, &b, i);
                }

                let min_size: usize = (a_size.max(b_size)).min(d_size);

                for i in 0..cols {
                    for j in 0..min_size {
                        assert_eq!(d.at(i, j), c.at(i, j));
                    }

                    for j in min_size..d_size {
                        assert_eq!(d.at(i, j), zero);
                    }
                }
            }
        }
    }
}

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
