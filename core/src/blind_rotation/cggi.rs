use backend::hal::{
    api::{
        ScratchAvailable, SvpApply, SvpPPolAllocBytes, TakeVecZnx, TakeVecZnxBig, TakeVecZnxDft, TakeVecZnxDftSlice,
        TakeVecZnxSlice, VecZnxAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAllocBytes, VecZnxBigNormalizeTmpBytes, VecZnxCopy,
        VecZnxDftAdd, VecZnxDftAddInplace, VecZnxDftAllocBytes, VecZnxDftFromVecZnx, VecZnxDftSubABInplace, VecZnxDftToVecZnxBig,
        VecZnxDftToVecZnxBigTmpBytes, VecZnxDftZero, VecZnxMulXpMinusOneInplace, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxRotate, VecZnxSubABInplace, VmpApplyTmpBytes, ZnxView, ZnxZero,
    },
    layouts::{Backend, DataMut, DataRef, Module, Scratch, SvpPPol, VecZnx},
};
use itertools::izip;

use crate::{
    GLWECiphertext, GLWECiphertextToMut, GLWEExternalProductFamily, GLWEOps, Infos, LWECiphertext, LWECiphertextToRef,
    TakeGLWECt,
    blind_rotation::{key::BlindRotationKeyCGGIExec, lut::LookUpTable},
    dist::Distribution,
};

pub trait CCGIBlindRotationFamily<B: Backend> = VecZnxBigAllocBytes
    + VecZnxDftAllocBytes
    + SvpPPolAllocBytes
    + VmpApplyTmpBytes
    + VecZnxBigNormalizeTmpBytes
    + VecZnxDftToVecZnxBigTmpBytes
    + VecZnxDftToVecZnxBig<B>
    + VecZnxDftAdd<B>
    + VecZnxDftAddInplace<B>
    + VecZnxDftFromVecZnx<B>
    + VecZnxDftZero<B>
    + SvpApply<B>
    + VecZnxDftSubABInplace<B>
    + VecZnxBigAddSmallInplace<B>
    + GLWEExternalProductFamily<B>
    + VecZnxRotate
    + VecZnxAddInplace
    + VecZnxSubABInplace
    + VecZnxNormalize<B>
    + VecZnxNormalizeInplace<B>
    + VecZnxCopy
    + VecZnxMulXpMinusOneInplace;

pub fn cggi_blind_rotate_scratch_space<B: Backend>(
    module: &Module<B>,
    n: usize,
    block_size: usize,
    extension_factor: usize,
    basek: usize,
    k_res: usize,
    k_brk: usize,
    rows: usize,
    rank: usize,
) -> usize
where
    Module<B>: CCGIBlindRotationFamily<B>,
{
    let brk_size: usize = k_brk.div_ceil(basek);

    if block_size > 1 {
        let cols: usize = rank + 1;
        let acc_dft: usize = module.vec_znx_dft_alloc_bytes(n, cols, rows) * extension_factor;
        let acc_big: usize = module.vec_znx_big_alloc_bytes(n, 1, brk_size);
        let vmp_res: usize = module.vec_znx_dft_alloc_bytes(n, cols, brk_size) * extension_factor;
        let vmp_xai: usize = module.vec_znx_dft_alloc_bytes(n, 1, brk_size);
        let acc_dft_add: usize = vmp_res;
        let vmp: usize = module.vmp_apply_tmp_bytes(n, brk_size, rows, rows, 2, 2, brk_size); // GGSW product: (1 x 2) x (2 x 2)

        let acc: usize;
        if extension_factor > 1 {
            acc = VecZnx::alloc_bytes(n, cols, k_res.div_ceil(basek)) * extension_factor;
        } else {
            acc = 0;
        }

        return acc
            + acc_dft
            + acc_dft_add
            + vmp_res
            + vmp_xai
            + (vmp | (acc_big + (module.vec_znx_big_normalize_tmp_bytes(n) | module.vec_znx_dft_to_vec_znx_big_tmp_bytes(n))));
    } else {
        GLWECiphertext::bytes_of(n, basek, k_res, rank)
            + GLWECiphertext::external_product_scratch_space(module, n, basek, k_res, k_res, k_brk, 1, rank)
    }
}

pub fn cggi_blind_rotate<DataRes, DataIn, DataBrk, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyCGGIExec<DataBrk, B>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    Module<B>: CCGIBlindRotationFamily<B>,
    Scratch<B>: TakeVecZnxDftSlice<B> + TakeVecZnxDft<B> + TakeVecZnxBig<B> + TakeVecZnx + ScratchAvailable + TakeVecZnxSlice,
{
    match brk.dist {
        Distribution::BinaryBlock(_) | Distribution::BinaryFixed(_) | Distribution::BinaryProb(_) | Distribution::ZERO => {
            if lut.extension_factor() > 1 {
                cggi_blind_rotate_block_binary_extended(module, res, lwe, lut, brk, scratch);
            } else if brk.block_size() > 1 {
                cggi_blind_rotate_block_binary(module, res, lwe, lut, brk, scratch);
            } else {
                cggi_blind_rotate_binary_standard(module, res, lwe, lut, brk, scratch);
            }
        }
        // TODO: ternary distribution ?
        _ => panic!(
            "invalid BlindRotationKeyCGGI distribution: must be BinaryBlock, BinaryFixed or BinaryProb (or ZERO for debugging)"
        ),
    }
}

pub(crate) fn cggi_blind_rotate_block_binary_extended<DataRes, DataIn, DataBrk, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyCGGIExec<DataBrk, B>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    Module<B>: CCGIBlindRotationFamily<B>,
    Scratch<B>: TakeVecZnxDftSlice<B> + TakeVecZnxDft<B> + TakeVecZnxBig<B> + TakeVecZnxSlice,
{
    let n_glwe: usize = brk.n();
    let extension_factor: usize = lut.extension_factor();
    let basek: usize = res.basek();
    let rows: usize = brk.rows();
    let cols: usize = res.rank() + 1;

    let (mut acc, scratch1) = scratch.take_vec_znx_slice(extension_factor, n_glwe, cols, res.size());
    let (mut acc_dft, scratch2) = scratch1.take_vec_znx_dft_slice(extension_factor, n_glwe, cols, rows);
    let (mut vmp_res, scratch3) = scratch2.take_vec_znx_dft_slice(extension_factor, n_glwe, cols, brk.size());
    let (mut acc_add_dft, scratch4) = scratch3.take_vec_znx_dft_slice(extension_factor, n_glwe, cols, brk.size());
    let (mut vmp_xai, scratch5) = scratch4.take_vec_znx_dft(n_glwe, 1, brk.size());

    (0..extension_factor).for_each(|i| {
        acc[i].zero();
    });

    let x_pow_a: &Vec<SvpPPol<Vec<u8>, B>>;
    if let Some(b) = &brk.x_pow_a {
        x_pow_a = b
    } else {
        panic!("invalid key: x_pow_a has not been initialized")
    }

    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();

    let two_n: usize = 2 * n_glwe;
    let two_n_ext: usize = 2 * lut.domain_size();

    negate_and_mod_switch_2n(two_n_ext, &mut lwe_2n, &lwe_ref);

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
                module.vec_znx_dft_from_vec_znx(1, 0, &mut acc_dft[i], j, &acc[i], j);
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
                module.vmp_apply(&mut vmp_res[i], &acc_dft[i], &skii.data, scratch5);
            });

            // Trivial case: no rotation between polynomials, we can directly multiply with (X^{-ai} - 1)
            if ai_lo == 0 {
                // Sets acc_add_dft[i] = (acc[i] * sk) * X^{-ai} - (acc[i] * sk)
                if ai_hi != 0 {
                    // DFT X^{-ai}
                    (0..extension_factor).for_each(|j| {
                        (0..cols).for_each(|i| {
                            module.svp_apply(&mut vmp_xai, 0, &x_pow_a[ai_hi], 0, &vmp_res[j], i);
                            module.vec_znx_dft_add_inplace(&mut acc_add_dft[j], i, &vmp_xai, 0);
                            module.vec_znx_dft_sub_ab_inplace(&mut acc_add_dft[j], i, &vmp_res[j], i);
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
                            module.svp_apply(&mut vmp_xai, 0, &x_pow_a[ai_hi + 1], 0, &vmp_res[j], k);
                            module.vec_znx_dft_add_inplace(&mut acc_add_dft[i], k, &vmp_xai, 0);
                            module.vec_znx_dft_sub_ab_inplace(&mut acc_add_dft[i], k, &vmp_res[i], k);
                        });
                    }
                }

                // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                if ai_hi != 0 {
                    // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                    for (i, j) in (ai_lo..extension_factor).zip(0..extension_factor - ai_lo) {
                        (0..cols).for_each(|k| {
                            module.svp_apply(&mut vmp_xai, 0, &x_pow_a[ai_hi], 0, &vmp_res[j], k);
                            module.vec_znx_dft_add_inplace(&mut acc_add_dft[i], k, &vmp_xai, 0);
                            module.vec_znx_dft_sub_ab_inplace(&mut acc_add_dft[i], k, &vmp_res[i], k);
                        });
                    }
                }
            }
        });

        {
            let (mut acc_add_big, scratch7) = scratch5.take_vec_znx_big(n_glwe, 1, brk.size());

            (0..extension_factor).for_each(|j| {
                (0..cols).for_each(|i| {
                    module.vec_znx_dft_to_vec_znx_big(&mut acc_add_big, 0, &acc_add_dft[j], i, scratch7);
                    module.vec_znx_big_add_small_inplace(&mut acc_add_big, 0, &acc[j], i);
                    module.vec_znx_big_normalize(basek, &mut acc[j], i, &acc_add_big, 0, scratch7);
                });
            });
        }
    });

    (0..cols).for_each(|i| {
        module.vec_znx_copy(&mut res.data, i, &acc[0], i);
    });
}

pub(crate) fn cggi_blind_rotate_block_binary<DataRes, DataIn, DataBrk, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyCGGIExec<DataBrk, B>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    Module<B>: CCGIBlindRotationFamily<B>,
    Scratch<B>: TakeVecZnxDft<B> + TakeVecZnxBig<B>,
{
    let n_glwe: usize = brk.n();
    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space
    let mut out_mut: GLWECiphertext<&mut [u8]> = res.to_mut();
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();
    let two_n: usize = n_glwe << 1;
    let basek: usize = brk.basek();
    let rows: usize = brk.rows();

    let cols: usize = out_mut.rank() + 1;

    negate_and_mod_switch_2n(2 * lut.domain_size(), &mut lwe_2n, &lwe_ref);

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_mut.data.zero();

    // Initialize out to X^{b} * LUT(X)
    module.vec_znx_rotate(b, &mut out_mut.data, 0, &lut.data[0], 0);

    let block_size: usize = brk.block_size();

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]

    let (mut acc_dft, scratch1) = scratch.take_vec_znx_dft(n_glwe, cols, rows);
    let (mut vmp_res, scratch2) = scratch1.take_vec_znx_dft(n_glwe, cols, brk.size());
    let (mut acc_add_dft, scratch3) = scratch2.take_vec_znx_dft(n_glwe, cols, brk.size());
    let (mut vmp_xai, scratch4) = scratch3.take_vec_znx_dft(n_glwe, 1, brk.size());

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
            module.vec_znx_dft_from_vec_znx(1, 0, &mut acc_dft, j, &out_mut.data, j);
        });

        module.vec_znx_dft_zero(&mut acc_add_dft);

        izip!(ai.iter(), ski.iter()).for_each(|(aii, skii)| {
            let ai_pos: usize = ((aii + two_n as i64) & (two_n - 1) as i64) as usize;

            // vmp_res = DFT(acc) * BRK[i]
            module.vmp_apply(&mut vmp_res, &acc_dft, &skii.data, scratch4);

            // DFT(X^ai -1) * (DFT(acc) * BRK[i])
            (0..cols).for_each(|i| {
                module.svp_apply(&mut vmp_xai, 0, &x_pow_a[ai_pos], 0, &vmp_res, i);
                module.vec_znx_dft_add_inplace(&mut acc_add_dft, i, &vmp_xai, 0);
                module.vec_znx_dft_sub_ab_inplace(&mut acc_add_dft, i, &vmp_res, i);
            });
        });

        {
            let (mut acc_add_big, scratch5) = scratch4.take_vec_znx_big(n_glwe, 1, brk.size());

            (0..cols).for_each(|i| {
                module.vec_znx_dft_to_vec_znx_big(&mut acc_add_big, 0, &acc_add_dft, i, scratch5);
                module.vec_znx_big_add_small_inplace(&mut acc_add_big, 0, &out_mut.data, i);
                module.vec_znx_big_normalize(basek, &mut out_mut.data, i, &acc_add_big, 0, scratch5);
            });
        }
    });
}

pub(crate) fn cggi_blind_rotate_binary_standard<DataRes, DataIn, DataBrk, B: Backend>(
    module: &Module<B>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyCGGIExec<DataBrk, B>,
    scratch: &mut Scratch<B>,
) where
    DataRes: DataMut,
    DataIn: DataRef,
    DataBrk: DataRef,
    Module<B>: CCGIBlindRotationFamily<B>,
    Scratch<B>: TakeVecZnxDft<B> + TakeVecZnxBig<B> + TakeVecZnx + ScratchAvailable,
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
            brk.n(),
            "lut.n(): {} != brk.n(): {}",
            lut.domain_size(),
            brk.n()
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
            brk.data.len(),
            "lwe.n(): {} != brk.data.len(): {}",
            lwe.n(),
            brk.data.len()
        );
    }

    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space
    let mut out_mut: GLWECiphertext<&mut [u8]> = res.to_mut();
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();
    let basek: usize = brk.basek();

    negate_and_mod_switch_2n(2 * lut.domain_size(), &mut lwe_2n, &lwe_ref);

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_mut.data.zero();

    // Initialize out to X^{b} * LUT(X)
    module.vec_znx_rotate(b, &mut out_mut.data, 0, &lut.data[0], 0);

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]
    let (mut acc_tmp, scratch1) = scratch.take_glwe_ct(out_mut.n(), basek, out_mut.k(), out_mut.rank());

    // TODO: see if faster by skipping normalization in external product and keeping acc in big coeffs
    // TODO: first iteration can be optimized to be a gglwe product
    izip!(a.iter(), brk.data.iter()).for_each(|(ai, ski)| {
        // acc_tmp = sk[i] * acc
        acc_tmp.external_product(module, &out_mut, ski, scratch1);

        // acc_tmp = (sk[i] * acc) * (X^{ai} - 1)
        acc_tmp.mul_xp_minus_one_inplace(module, *ai);

        // acc = acc + (sk[i] * acc) * (X^{ai} - 1)
        out_mut.add_inplace(module, &acc_tmp);
    });

    // We can normalize only at the end because we add normalized values in [-2^{basek-1}, 2^{basek-1}]
    // on top of each others, thus ~ 2^{63-basek} additions are supported before overflow.
    out_mut.normalize_inplace(module, scratch1);
}

pub(crate) fn negate_and_mod_switch_2n(n: usize, res: &mut [i64], lwe: &LWECiphertext<&[u8]>) {
    let basek: usize = lwe.basek();

    let log2n: usize = usize::BITS as usize - (n - 1).leading_zeros() as usize + 1;

    res.copy_from_slice(&lwe.data.at(0, 0));
    res.iter_mut().for_each(|x| *x = -*x);

    if basek > log2n {
        let diff: usize = basek - log2n;
        res.iter_mut().for_each(|x| {
            *x = div_round_by_pow2(x, diff);
        })
    } else {
        let rem: usize = basek - (log2n % basek);
        let size: usize = log2n.div_ceil(basek);
        (1..size).for_each(|i| {
            if i == size - 1 && rem != basek {
                let k_rem: usize = basek - rem;
                izip!(lwe.data.at(0, i).iter(), res.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << k_rem) + (x >> rem);
                });
            } else {
                izip!(lwe.data.at(0, i).iter(), res.iter_mut()).for_each(|(x, y)| {
                    *y = (*y << basek) + x;
                });
            }
        })
    }
}

#[inline(always)]
fn div_round_by_pow2(x: &i64, k: usize) -> i64 {
    (x + (1 << (k - 1))) >> k
}
