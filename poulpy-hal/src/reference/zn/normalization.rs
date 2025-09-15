use crate::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, ZnNormalizeInplace, ZnNormalizeTmpBytes},
    layouts::{Backend, Module, ScratchOwned, Zn, ZnToMut, ZnxInfos, ZnxView, ZnxViewMut},
    reference::znx::{ZnxNormalizeFinalStepInplace, ZnxNormalizeFirstStepInplace, ZnxNormalizeMiddleStepInplace, ZnxRef},
    source::Source,
};

pub fn zn_normalize_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn zn_normalize_inplace<R, ARI>(n: usize, basek: usize, res: &mut R, res_col: usize, carry: &mut [i64])
where
    R: ZnToMut,
    ARI: ZnxNormalizeFirstStepInplace + ZnxNormalizeFinalStepInplace + ZnxNormalizeMiddleStepInplace,
{
    let mut res: Zn<&mut [u8]> = res.to_mut();

    #[cfg(debug_assertions)]
    {
        assert_eq!(carry.len(), res.n());
    }

    let res_size: usize = res.size();

    for j in (0..res_size).rev() {
        let out = &mut res.at_mut(res_col, j)[..n];

        if j == res_size - 1 {
            ARI::znx_normalize_first_step_inplace(basek, 0, out, carry);
        } else if j == 0 {
            ARI::znx_normalize_final_step_inplace(basek, 0, out, carry);
        } else {
            ARI::znx_normalize_middle_step_inplace(basek, 0, out, carry);
        }
    }
}

pub fn test_zn_normalize_inplace<B: Backend>(module: &Module<B>)
where
    Module<B>: ZnNormalizeInplace<B> + ZnNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let basek: usize = 12;

    let n = 33;

    let mut carry: Vec<i64> = vec![0i64; zn_normalize_tmp_bytes(n)];

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.zn_normalize_tmp_bytes(module.n()));

    for res_size in [1, 2, 6, 11] {
        let mut res_0: Zn<Vec<u8>> = Zn::alloc(n, cols, res_size);
        let mut res_1: Zn<Vec<u8>> = Zn::alloc(n, cols, res_size);

        res_0
            .raw_mut()
            .iter_mut()
            .for_each(|x| *x = source.next_i32() as i64);
        res_1.raw_mut().copy_from_slice(res_0.raw());

        // Reference
        for i in 0..cols {
            zn_normalize_inplace::<_, ZnxRef>(n, basek, &mut res_0, i, &mut carry);
            module.zn_normalize_inplace(n, basek, &mut res_1, i, scratch.borrow());
        }

        assert_eq!(res_0.raw(), res_1.raw());
    }
}
