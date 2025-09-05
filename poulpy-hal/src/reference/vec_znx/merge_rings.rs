use crate::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxMergeRings, VecZnxMergeRingsTmpBytes},
    layouts::{Backend, FillUniform, Module, ScratchOwned, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView},
    reference::vec_znx::{vec_znx_rotate_inplace_ref, vec_znx_switch_ring_ref},
    source::Source,
};

pub fn vec_znx_merge_rings_tmp_bytes_ref(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_merge_rings_ref<R, A>(res: &mut R, res_col: usize, a: &[A], a_col: usize, tmp: &mut [i64])
where
    R: VecZnxToMut,
    A: VecZnxToRef,
{
    let mut res: VecZnx<&mut [u8]> = res.to_mut();

    let (n_out, n_in) = (res.n(), a[0].to_ref().n());

    #[cfg(debug_assertions)]
    {
        assert_eq!(tmp.len(), res.n());

        debug_assert!(
            n_out > n_in,
            "invalid a: output ring degree should be greater"
        );
        a[1..].iter().for_each(|ai| {
            debug_assert_eq!(
                ai.to_ref().n(),
                n_in,
                "invalid input a: all VecZnx must have the same degree"
            )
        });

        assert!(n_out.is_multiple_of(n_in));
        assert_eq!(a.len(), n_out / n_in);
    }

    a.iter().for_each(|ai| {
        vec_znx_switch_ring_ref(&mut res, res_col, ai, a_col);
        vec_znx_rotate_inplace_ref(-1, &mut res, res_col, tmp);
    });

    vec_znx_rotate_inplace_ref(a.len() as i64, &mut res, res_col, tmp);
}

pub fn test_vec_znx_merge_rings<B: Backend>(module: &Module<B>)
where
    Module<B>: VecZnxMergeRings<B> + ModuleNew<B> + VecZnxMergeRingsTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let mut scratch: ScratchOwned<_> = ScratchOwned::alloc(module.vec_znx_merge_rings_tmp_bytes());
    let mut tmp: Vec<i64> = vec![0i64; module.n()];

    for a_size in [1, 2, 6, 11] {
        let mut a: [VecZnx<Vec<u8>>; 2] = [
            VecZnx::alloc(module.n() >> 1, cols, a_size),
            VecZnx::alloc(module.n() >> 1, cols, a_size),
        ];

        a.iter_mut().for_each(|ai| {
            ai.fill_uniform(&mut source);
        });

        for res_size in [1, 2, 6, 11] {
            let mut res_0: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = VecZnx::alloc(module.n(), cols, res_size);

            res_0.fill_uniform(&mut source);
            res_1.fill_uniform(&mut source);

            for i in 0..cols {
                module.vec_znx_merge_rings(&mut res_0, i, &a, i, scratch.borrow());
                vec_znx_merge_rings_ref(&mut res_1, i, &a, i, &mut tmp);
            }

            for i in 0..cols {
                for j in 0..res_size {
                    assert_eq!(res_0.at(i, j), res_1.at(i, j));
                }
            }
        }
    }
}
