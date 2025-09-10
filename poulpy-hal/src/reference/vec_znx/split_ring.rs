use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxSplitRing, VecZnxSplitRingTmpBytes},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxArithmetic, ZnxArithmeticRef},
    source::Source,
};

pub fn vec_znx_split_ring_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_split_ring<R, A, ZNXARI>(res: &mut [R], res_col: usize, a: &A, a_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
    ZNXARI: ZnxArithmetic,
{
    let a: VecZnx<&[u8]> = a.to_ref();
    let a_size = a.size();

    let (n_in, n_out) = (a.n(), res[0].to_mut().n());

    #[cfg(debug_assertions)]
    {
        assert_eq!(tmp.len(), a.n());

        assert!(
            n_out < n_in,
            "invalid a: output ring degree should be smaller"
        );

        res[1..].iter_mut().for_each(|bi| {
            assert_eq!(
                bi.to_mut().n(),
                n_out,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        assert!(n_in.is_multiple_of(n_out));
        assert_eq!(res.len(), n_in / n_out);
    }

    res.iter_mut().enumerate().for_each(|(i, bi)| {
        let mut bi: VecZnx<&mut [u8]> = bi.to_mut();

        let min_size = bi.size().min(a_size);

        if i == 0 {
            for j in 0..min_size {
                ZNXARI::znx_switch_ring(bi.at_mut(res_col, j), a.at(a_col, j));
            }
        } else {
            for j in 0..min_size {
                ZNXARI::znx_rotate(-(i as i64), tmp, a.at(a_col, j));
                ZNXARI::znx_switch_ring(bi.at_mut(res_col, j), tmp);
            }
        }

        for j in min_size..bi.size() {
            ZNXARI::znx_zero(bi.at_mut(res_col, j));
        }
    })
}

pub fn test_vec_znx_split_ring<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxSplitRing<B> + ModuleNew<B> + VecZnxSplitRingTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let mut scratch: ScratchOwned<_> = ScratchOwned::alloc(module.vec_znx_split_ring_tmp_bytes());
    let mut tmp: Vec<i64> = vec![0i64; module.n()];

    for a_size in [1, 2, 6, 11] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, a_size);

        // Fill a with random i64
        a.fill_uniform(&mut source);

        for res_size in [1, 2, 6, 11] {
            let mut r0: [VecZnx<Vec<u8>>; 2] = [
                VecZnx::alloc(module.n() >> 1, cols, res_size),
                VecZnx::alloc(module.n() >> 1, cols, res_size),
            ];

            let mut r1: [VecZnx<Vec<u8>>; 2] = [
                VecZnx::alloc(module.n() >> 1, cols, res_size),
                VecZnx::alloc(module.n() >> 1, cols, res_size),
            ];

            r0.iter_mut().for_each(|ri| {
                ri.fill_uniform(&mut source);
            });

            r1.iter_mut().for_each(|ri| {
                ri.fill_uniform(&mut source);
            });

            for i in 0..cols {
                module.vec_znx_split_ring(&mut r0, i, &a, i, scratch.borrow());
            }

            for i in 0..cols {
                vec_znx_split_ring::<_, _, ZnxArithmeticRef>(&mut r1, i, &a, i, &mut tmp);
            }

            r0.iter().zip(r1.iter()).for_each(|(r0, r1)| {
                for i in 0..cols {
                    for j in 0..res_size {
                        assert_eq!(r0.at(i, j), r1.at(i, j));
                    }
                }
            });
        }
    }
}
