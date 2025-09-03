use crate::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddNormal, VecZnxAutomorphism, VecZnxFillUniform, VecZnxLshInplace,
        VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate,
    },
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    reference::vec_znx::{vec_znx_automorphism_ref, vec_znx_lsh_inplace_ref, vec_znx_rotate_ref},
    source::Source,
};

pub fn test_vec_znx_automorphism<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxAutomorphism,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0i64; module.n()];

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        let p: i64 = -5;

        // Reference
        for i in 0..cols {
            vec_znx_automorphism_ref(p, &mut b, i, &a, i);
        }

        for d_size in [1, 2, 6, 11] {
            let mut d: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, d_size);

            // Set d to garbage
            d.fill_uniform(&mut source);

            // Normalize on c
            for i in 0..cols {
                module.vec_znx_automorphism(p, &mut d, i, &a, i);
            }

            let min_size: usize = a_size.min(d_size);

            for i in 0..cols {
                for j in 0..min_size {
                    assert_eq!(d.at(i, j), b.at(i, j));
                }

                for j in min_size..d_size {
                    assert_eq!(d.at(i, j), zero);
                }
            }
        }
    }
}

pub fn test_vec_znx_lsh<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxLshInplace<B> + VecZnxNormalizeInplace<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let basek = 12;
    let k: usize = 38;

    for size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        // Normalize to avoid i64 overflow
        for i in 0..cols {
            module.vec_znx_normalize_inplace(basek, &mut a, i, scratch.borrow());
        }

        b.raw_mut().copy_from_slice(a.raw());

        let mut carry: Vec<i64> = vec![0i64; module.n()];

        // Normalize on c
        for i in 0..cols {
            module.vec_znx_lsh_inplace(basek, k, &mut a, i, scratch.borrow());
            vec_znx_lsh_inplace_ref(basek, k, &mut b, i, &mut carry);
        }

        for i in 0..cols {
            for j in 0..size {
                assert_eq!(a.at(i, j), b.at(i, j));
            }
        }
    }
}

pub fn test_vec_znx_rotate<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxRotate + VecZnxNormalizeInplace<B> + VecZnxNormalizeTmpBytes,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut r0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);
        let mut r1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        let p: i64 = -7;

        // Normalize on c
        for i in 0..cols {
            module.vec_znx_rotate(p, &mut r0, i, &a, i);
            vec_znx_rotate_ref(p, &mut r1, i, &a, i);
        }

        for i in 0..cols {
            for j in 0..size {
                assert_eq!(r0.at(i, j), r1.at(i, j));
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
