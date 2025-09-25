use itertools::izip;
use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDft, SvpPPolAllocBytes, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, TakeVecZnxDftSlice,
        TakeVecZnxSlice, VecZnxAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAdd, VecZnxDftAddInplace, VecZnxDftAllocBytes, VecZnxDftApply,
        VecZnxDftSubInplace, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyConsume, VecZnxIdftApplyTmpBytes,
        VecZnxMulXpMinusOneInplace, VecZnxNormalize, VecZnxNormalizeInplace, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxSubInplace, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, SvpPPol, VecZnx, ZnxView, ZnxZero},
};

use poulpy_core::{
    Distribution, GLWEOperations, TakeGLWECt,
    layouts::{GGSWInfos, GLWECiphertext, GLWECiphertextToMut, GLWEInfos, LWECiphertext, LWECiphertextToRef, LWEInfos},
};

use crate::tfhe::blind_rotation::{
    BlincRotationExecute, BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI, LookUpTable, LookUpTableRotationDirection,
};

#[allow(clippy::too_many_arguments)]
pub fn cggi_blind_rotate_scratch_space<B: Backend, OUT, GGSW>(
    module: &Module<B>,
    block_size: usize,
    extension_factor: usize,
    glwe_infos: &OUT,
    brk_infos: &GGSW,
) -> usize
where
    OUT: GLWEInfos,
    GGSW: GGSWInfos,
    Module<B>: VecZnxDftAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAllocBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
{
    let brk_size: usize = brk_infos.size();

    if block_size > 1 {
        let cols: usize = (brk_infos.rank() + 1).into();
        let rows: usize = brk_infos.rows().into();
        let acc_dft: usize = module.vec_znx_dft_alloc_bytes(cols, rows) * extension_factor;
        let acc_big: usize = module.vec_znx_big_alloc_bytes(1, brk_size);
        let vmp_res: usize = module.vec_znx_dft_alloc_bytes(cols, brk_size) * extension_factor;
        let vmp_xai: usize = module.vec_znx_dft_alloc_bytes(1, brk_size);
        let acc_dft_add: usize = vmp_res;
        let vmp: usize = module.vmp_apply_dft_to_dft_tmp_bytes(brk_size, rows, rows, 2, 2, brk_size); // GGSW product: (1 x 2) x (2 x 2)
        let acc: usize = if extension_factor > 1 {
            VecZnx::alloc_bytes(module.n(), cols, glwe_infos.size()) * extension_factor
        } else {
            0
        };

        acc + acc_dft
            + acc_dft_add
            + vmp_res
            + vmp_xai
            + (vmp | (acc_big + (module.vec_znx_big_normalize_tmp_bytes() | module.vec_znx_idft_apply_tmp_bytes())))
    } else {
        GLWECiphertext::alloc_bytes(glwe_infos)
            + GLWECiphertext::external_product_inplace_scratch_space(module, glwe_infos, brk_infos)
    }
}

impl<D: DataRef, B: Backend> BlincRotationExecute<B> for BlindRotationKeyPrepared<D, CGGI, B>
where
    Module<B>: VecZnxBigAllocBytes
        + VecZnxDftAllocBytes
        + SvpPPolAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApply<B>
        + VecZnxDftAdd<B>
        + VecZnxDftAddInplace<B>
        + VecZnxDftApply<B>
        + VecZnxDftZero<B>
        + SvpApplyDftToDft<B>
        + VecZnxDftSubInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxRotate
        + VecZnxAddInplace
        + VecZnxSubInplace
        + VecZnxNormalize<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxCopy
        + VecZnxMulXpMinusOneInplace<B>
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes,
    Scratch<B>: TakeVecZnxDftSlice<B> + TakeVecZnxDft<B> + TakeVecZnxBig<B> + TakeVecZnxSlice + TakeVecZnx + ScratchAvailable,
{
    fn execute<DR: DataMut, DI: DataRef>(
        &self,
        module: &Module<B>,
        res: &mut GLWECiphertext<DR>,
        lwe: &LWECiphertext<DI>,
        lut: &LookUpTable,
        scratch: &mut Scratch<B>,
    ) {
        match self.dist {
            Distribution::BinaryBlock(_) | Distribution::BinaryFixed(_) | Distribution::BinaryProb(_) | Distribution::ZERO => {
                if lut.extension_factor() > 1 {
                    execute_block_binary_extended(module, res, lwe, lut, self, scratch)
                } else if self.block_size() > 1 {
                    execute_block_binary(module, res, lwe, lut, self, scratch);
                } else {
                    execute_standard(module, res, lwe, lut, self, scratch);
                }
            }
            _ => panic!("invalid CGGI distribution"),
        }
    }
}

fn execute_block_binary_extended<DataRes, DataIn, DataBrk, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyPrepared<DataBrk, CGGI, B>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    Module<B>: VecZnxBigAllocBytes
        + VecZnxDftAllocBytes
        + SvpPPolAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApply<B>
        + VecZnxDftAdd<B>
        + VecZnxDftAddInplace<B>
        + VecZnxDftApply<B>
        + VecZnxDftZero<B>
        + SvpApplyDftToDft<B>
        + VecZnxDftSubInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxRotate
        + VecZnxAddInplace
        + VecZnxSubInplace
        + VecZnxNormalize<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxCopy
        + VecZnxMulXpMinusOneInplace<B>
        + VecZnxBigNormalize<B>
        + VmpApplyDftToDft<B>,
    Scratch<B>: TakeVecZnxDftSlice<B> + TakeVecZnxDft<B> + TakeVecZnxBig<B> + TakeVecZnxSlice + ScratchAvailable + TakeVecZnx,
{
    let n_glwe: usize = brk.n_glwe().into();
    let extension_factor: usize = lut.extension_factor();
    let base2k: usize = res.base2k().into();
    let rows: usize = brk.rows().into();
    let cols: usize = (res.rank() + 1).into();

    let (mut acc, scratch_1) = scratch.take_vec_znx_slice(extension_factor, n_glwe, cols, res.size());
    let (mut acc_dft, scratch_2) = scratch_1.take_vec_znx_dft_slice(extension_factor, n_glwe, cols, rows);
    let (mut vmp_res, scratch_3) = scratch_2.take_vec_znx_dft_slice(extension_factor, n_glwe, cols, brk.size());
    let (mut acc_add_dft, scratch_4) = scratch_3.take_vec_znx_dft_slice(extension_factor, n_glwe, cols, brk.size());
    let (mut vmp_xai, scratch_5) = scratch_4.take_vec_znx_dft(n_glwe, 1, brk.size());

    (0..extension_factor).for_each(|i| {
        acc[i].zero();
    });

    let x_pow_a: &Vec<SvpPPol<Vec<u8>, B>>;
    if let Some(b) = &brk.x_pow_a {
        x_pow_a = b
    } else {
        panic!("invalid key: x_pow_a has not been initialized")
    }

    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).as_usize()]; // TODO: from scratch space
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();

    let two_n: usize = 2 * n_glwe;
    let two_n_ext: usize = 2 * lut.domain_size();

    mod_switch_2n(two_n_ext, &mut lwe_2n, &lwe_ref, lut.rotation_direction());

    let a: &[i64] = &lwe_2n[1..];
    let b_pos: usize = ((lwe_2n[0] + two_n_ext as i64) & (two_n_ext - 1) as i64) as usize;

    let b_hi: usize = b_pos / extension_factor;
    let b_lo: usize = b_pos & (extension_factor - 1);

    for (i, j) in (0..b_lo).zip(extension_factor - b_lo..extension_factor) {
        module.vec_znx_rotate(b_hi as i64 + 1, &mut acc[i], 0, &lut.data[j], 0);
    }
    for (i, j) in (b_lo..extension_factor).zip(0..extension_factor - b_lo) {
        module.vec_znx_rotate(b_hi as i64, &mut acc[i], 0, &lut.data[j], 0);
    }

    let block_size: usize = brk.block_size();

    izip!(
        a.chunks_exact(block_size),
        brk.data.chunks_exact(block_size)
    )
    .for_each(|(ai, ski)| {
        (0..extension_factor).for_each(|i| {
            (0..cols).for_each(|j| {
                module.vec_znx_dft_apply(1, 0, &mut acc_dft[i], j, &acc[i], j);
            });
            module.vec_znx_dft_zero(&mut acc_add_dft[i])
        });

        // TODO: first & last iterations can be optimized
        izip!(ai.iter(), ski.iter()).for_each(|(aii, skii)| {
            let ai_pos: usize = ((aii + two_n_ext as i64) & (two_n_ext - 1) as i64) as usize;
            let ai_hi: usize = ai_pos / extension_factor;
            let ai_lo: usize = ai_pos & (extension_factor - 1);

            // vmp_res = DFT(acc) * BRK[i]
            (0..extension_factor).for_each(|i| {
                module.vmp_apply_dft_to_dft(&mut vmp_res[i], &acc_dft[i], skii.data(), scratch_5);
            });

            // Trivial case: no rotation between polynomials, we can directly multiply with (X^{-ai} - 1)
            if ai_lo == 0 {
                // Sets acc_add_dft[i] = (acc[i] * sk) * X^{-ai} - (acc[i] * sk)
                if ai_hi != 0 {
                    // DFT X^{-ai}
                    (0..extension_factor).for_each(|j| {
                        (0..cols).for_each(|i| {
                            module.svp_apply_dft_to_dft(&mut vmp_xai, 0, &x_pow_a[ai_hi], 0, &vmp_res[j], i);
                            module.vec_znx_dft_add_inplace(&mut acc_add_dft[j], i, &vmp_xai, 0);
                            module.vec_znx_dft_sub_inplace(&mut acc_add_dft[j], i, &vmp_res[j], i);
                        });
                    });
                }

            // Non trivial case: rotation between polynomials
            // In this case we can't directly multiply with (X^{-ai} - 1) because of the
            // ring homomorphism R^{N} -> prod R^{N/extension_factor}, so we split the
            // computation in two steps: acc_add_dft = (acc * sk) * (-1) + (acc * sk) * X^{-ai}
            } else {
                // Sets acc_add_dft[0..ai_lo] += (acc[extension_factor - ai_lo..extension_factor] * sk) * X^{-ai+1}
                if (ai_hi + 1) & (two_n - 1) != 0 {
                    for (i, j) in (0..ai_lo).zip(extension_factor - ai_lo..extension_factor) {
                        (0..cols).for_each(|k| {
                            module.svp_apply_dft_to_dft(&mut vmp_xai, 0, &x_pow_a[ai_hi + 1], 0, &vmp_res[j], k);
                            module.vec_znx_dft_add_inplace(&mut acc_add_dft[i], k, &vmp_xai, 0);
                            module.vec_znx_dft_sub_inplace(&mut acc_add_dft[i], k, &vmp_res[i], k);
                        });
                    }
                }

                // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                if ai_hi != 0 {
                    // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                    for (i, j) in (ai_lo..extension_factor).zip(0..extension_factor - ai_lo) {
                        (0..cols).for_each(|k| {
                            module.svp_apply_dft_to_dft(&mut vmp_xai, 0, &x_pow_a[ai_hi], 0, &vmp_res[j], k);
                            module.vec_znx_dft_add_inplace(&mut acc_add_dft[i], k, &vmp_xai, 0);
                            module.vec_znx_dft_sub_inplace(&mut acc_add_dft[i], k, &vmp_res[i], k);
                        });
                    }
                }
            }
        });

        {
            let (mut acc_add_big, scratch7) = scratch_5.take_vec_znx_big(n_glwe, 1, brk.size());

            (0..extension_factor).for_each(|j| {
                (0..cols).for_each(|i| {
                    module.vec_znx_idft_apply(&mut acc_add_big, 0, &acc_add_dft[j], i, scratch7);
                    module.vec_znx_big_add_small_inplace(&mut acc_add_big, 0, &acc[j], i);
                    module.vec_znx_big_normalize(base2k, &mut acc[j], i, base2k, &acc_add_big, 0, scratch7);
                });
            });
        }
    });

    (0..cols).for_each(|i| {
        module.vec_znx_copy(res.data_mut(), i, &acc[0], i);
    });
}

fn execute_block_binary<DataRes, DataIn, DataBrk, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyPrepared<DataBrk, CGGI, B>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    Module<B>: VecZnxBigAllocBytes
        + VecZnxDftAllocBytes
        + SvpPPolAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApply<B>
        + VecZnxDftAdd<B>
        + VecZnxDftAddInplace<B>
        + VecZnxDftApply<B>
        + VecZnxDftZero<B>
        + SvpApplyDftToDft<B>
        + VecZnxDftSubInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxRotate
        + VecZnxAddInplace
        + VecZnxSubInplace
        + VecZnxNormalize<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxCopy
        + VecZnxMulXpMinusOneInplace<B>
        + VmpApplyDftToDft<B>
        + VecZnxBigNormalize<B>,
    Scratch<B>: TakeVecZnxDftSlice<B> + TakeVecZnxDft<B> + TakeVecZnxBig<B> + TakeVecZnxSlice + ScratchAvailable + TakeVecZnx,
{
    let n_glwe: usize = brk.n_glwe().into();
    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).into()]; // TODO: from scratch space
    let mut out_mut: GLWECiphertext<&mut [u8]> = res.to_mut();
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();
    let two_n: usize = n_glwe << 1;
    let base2k: usize = brk.base2k().into();
    let rows: usize = brk.rows().into();

    let cols: usize = (out_mut.rank() + 1).into();

    mod_switch_2n(
        2 * lut.domain_size(),
        &mut lwe_2n,
        &lwe_ref,
        lut.rotation_direction(),
    );

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_mut.data_mut().zero();

    // Initialize out to X^{b} * LUT(X)
    module.vec_znx_rotate(b, out_mut.data_mut(), 0, &lut.data[0], 0);

    let block_size: usize = brk.block_size();

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]

    let (mut acc_dft, scratch_1) = scratch.take_vec_znx_dft(n_glwe, cols, rows);
    let (mut vmp_res, scratch_2) = scratch_1.take_vec_znx_dft(n_glwe, cols, brk.size());
    let (mut acc_add_dft, scratch_3) = scratch_2.take_vec_znx_dft(n_glwe, cols, brk.size());
    let (mut vmp_xai, scratch_4) = scratch_3.take_vec_znx_dft(n_glwe, 1, brk.size());

    let x_pow_a: &Vec<SvpPPol<Vec<u8>, B>>;
    if let Some(b) = &brk.x_pow_a {
        x_pow_a = b
    } else {
        panic!("invalid key: x_pow_a has not been initialized")
    }

    izip!(
        a.chunks_exact(block_size),
        brk.data.chunks_exact(block_size)
    )
    .for_each(|(ai, ski)| {
        (0..cols).for_each(|j| {
            module.vec_znx_dft_apply(1, 0, &mut acc_dft, j, out_mut.data_mut(), j);
        });

        module.vec_znx_dft_zero(&mut acc_add_dft);

        izip!(ai.iter(), ski.iter()).for_each(|(aii, skii)| {
            let ai_pos: usize = ((aii + two_n as i64) & (two_n - 1) as i64) as usize;

            // vmp_res = DFT(acc) * BRK[i]
            module.vmp_apply_dft_to_dft(&mut vmp_res, &acc_dft, skii.data(), scratch_4);

            // DFT(X^ai -1) * (DFT(acc) * BRK[i])
            (0..cols).for_each(|i| {
                module.svp_apply_dft_to_dft(&mut vmp_xai, 0, &x_pow_a[ai_pos], 0, &vmp_res, i);
                module.vec_znx_dft_add_inplace(&mut acc_add_dft, i, &vmp_xai, 0);
                module.vec_znx_dft_sub_inplace(&mut acc_add_dft, i, &vmp_res, i);
            });
        });

        {
            let (mut acc_add_big, scratch_5) = scratch_4.take_vec_znx_big(n_glwe, 1, brk.size());

            (0..cols).for_each(|i| {
                module.vec_znx_idft_apply(&mut acc_add_big, 0, &acc_add_dft, i, scratch_5);
                module.vec_znx_big_add_small_inplace(&mut acc_add_big, 0, out_mut.data_mut(), i);
                module.vec_znx_big_normalize(
                    base2k,
                    out_mut.data_mut(),
                    i,
                    base2k,
                    &acc_add_big,
                    0,
                    scratch_5,
                );
            });
        }
    });
}

fn execute_standard<DataRes, DataIn, DataBrk, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyPrepared<DataBrk, CGGI, B>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    Module<B>: VecZnxBigAllocBytes
        + VecZnxDftAllocBytes
        + SvpPPolAllocBytes
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxIdftApply<B>
        + VecZnxDftAdd<B>
        + VecZnxDftAddInplace<B>
        + VecZnxDftApply<B>
        + VecZnxDftZero<B>
        + SvpApplyDftToDft<B>
        + VecZnxDftSubInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxRotate
        + VecZnxAddInplace
        + VecZnxSubInplace
        + VecZnxNormalize<B>
        + VecZnxNormalizeInplace<B>
        + VecZnxCopy
        + VecZnxMulXpMinusOneInplace<B>
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigNormalize<B>
        + VecZnxNormalizeTmpBytes,
    Scratch<B>: TakeVecZnxDftSlice<B> + TakeVecZnxDft<B> + TakeVecZnxBig<B> + TakeVecZnxSlice + ScratchAvailable + TakeVecZnx,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(
            res.n(),
            brk.n(),
            "res.n(): {} != brk.n(): {}",
            res.n(),
            brk.n()
        );
        assert_eq!(
            lut.domain_size(),
            brk.n_glwe().as_usize(),
            "lut.n(): {} != brk.n(): {}",
            lut.domain_size(),
            brk.n_glwe().as_usize()
        );
        assert_eq!(
            res.rank(),
            brk.rank(),
            "res.rank(): {} != brk.rank(): {}",
            res.rank(),
            brk.rank()
        );
        assert_eq!(
            lwe.n(),
            brk.n_lwe(),
            "lwe.n(): {} != brk.data.len(): {}",
            lwe.n(),
            brk.n_lwe()
        );
    }

    let mut lwe_2n: Vec<i64> = vec![0i64; (lwe.n() + 1).into()]; // TODO: from scratch space
    let mut out_mut: GLWECiphertext<&mut [u8]> = res.to_mut();
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();

    mod_switch_2n(
        2 * lut.domain_size(),
        &mut lwe_2n,
        &lwe_ref,
        lut.rotation_direction(),
    );

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_mut.data_mut().zero();

    // Initialize out to X^{b} * LUT(X)
    module.vec_znx_rotate(b, out_mut.data_mut(), 0, &lut.data[0], 0);

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]
    let (mut acc_tmp, scratch_1) = scratch.take_glwe_ct(&out_mut);

    // TODO: see if faster by skipping normalization in external product and keeping acc in big coeffs
    // TODO: first iteration can be optimized to be a gglwe product
    izip!(a.iter(), brk.data.iter()).for_each(|(ai, ski)| {
        // acc_tmp = sk[i] * acc
        acc_tmp.external_product(module, &out_mut, ski, scratch_1);

        // acc_tmp = (sk[i] * acc) * (X^{ai} - 1)
        acc_tmp.mul_xp_minus_one_inplace(module, *ai, scratch_1);

        // acc = acc + (sk[i] * acc) * (X^{ai} - 1)
        out_mut.add_inplace(module, &acc_tmp);
    });

    // We can normalize only at the end because we add normalized values in [-2^{base2k-1}, 2^{base2k-1}]
    // on top of each others, thus ~ 2^{63-base2k} additions are supported before overflow.
    out_mut.normalize_inplace(module, scratch_1);
}

pub fn mod_switch_2n(n: usize, res: &mut [i64], lwe: &LWECiphertext<&[u8]>, rot_dir: LookUpTableRotationDirection) {
    let base2k: usize = lwe.base2k().into();

    let log2n: usize = usize::BITS as usize - (n - 1).leading_zeros() as usize + 1;

    res.copy_from_slice(lwe.data().at(0, 0));

    match rot_dir {
        LookUpTableRotationDirection::Left => {
            res.iter_mut().for_each(|x| *x = -*x);
        }
        LookUpTableRotationDirection::Right => {}
    }

    if base2k > log2n {
        let diff: usize = base2k - (log2n - 1); // additional -1 because we map to [-N/2, N/2) instead of [0, N)
        res.iter_mut().for_each(|x| {
            *x = div_round_by_pow2(x, diff);
        })
    } else {
        let rem: usize = base2k - (log2n % base2k);
        let size: usize = log2n.div_ceil(base2k);
        (1..size).for_each(|i| {
            if i == size - 1 && rem != base2k {
                let k_rem: usize = base2k - rem;
                izip!(lwe.data().at(0, i).iter(), res.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << k_rem) + (x >> rem);
                });
            } else {
                izip!(lwe.data().at(0, i).iter(), res.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << base2k) + x;
                });
            }
        })
    }
}

#[inline(always)]
fn div_round_by_pow2(x: &i64, k: usize) -> i64 {
    (x + (1 << (k - 1))) >> k
}
