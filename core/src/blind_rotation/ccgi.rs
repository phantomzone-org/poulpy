use std::time::Instant;

use backend::{
    FFT64, MatZnxDftOps, MatZnxDftScratch, Module, ScalarZnx, ScalarZnxAlloc, ScalarZnxDft, ScalarZnxDftAlloc, ScalarZnxDftOps,
    Scratch, VecZnxBigAlloc, VecZnxBigOps, VecZnxDftOps, VecZnxOps, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
};
use itertools::izip;

use crate::{
    FourierGLWECiphertext, FourierGLWESecret, GLWECiphertext, GLWECiphertextToMut, GLWEPlaintext, Infos, LWECiphertext,
    ScratchCore,
    blind_rotation::{key::BlindRotationKeyCGGI, lut::LookUpTable},
    lwe::ciphertext::LWECiphertextToRef,
};

pub fn cggi_blind_rotate_scratch_space(
    module: &Module<FFT64>,
    extension_factor: usize,
    basek: usize,
    k_lut: usize,
    k_brk: usize,
    rows: usize,
    rank: usize,
) -> usize {
    let lut_size: usize = k_lut.div_ceil(basek);
    let brk_size: usize = k_brk.div_ceil(basek);

    let acc_dft: usize = FourierGLWECiphertext::bytes_of(module, basek, k_brk, rank) * extension_factor;
    let acc_big: usize = module.bytes_of_vec_znx_big(rank + 1, brk_size);
    let acc_dft_add: usize = FourierGLWECiphertext::bytes_of(module, basek, k_brk, rank) * extension_factor;
    let vmp_res: usize = FourierGLWECiphertext::bytes_of(module, basek, k_brk, rank) * extension_factor;
    let xai_plus_y: usize = module.bytes_of_scalar_znx(1);
    let xai_plus_y_dft: usize = module.bytes_of_scalar_znx_dft(1);
    let vmp: usize = module.vmp_apply_tmp_bytes(lut_size, lut_size, rows, 2, 2, brk_size); // GGSW product: (1 x 2) x (2 x 2)

    let acc: usize;
    if extension_factor > 1 {
        acc = GLWECiphertext::bytes_of(module, basek, k_lut, rank) * extension_factor;
    } else {
        acc = 0;
    }

    return acc + acc_dft + acc_dft_add + vmp_res + xai_plus_y + xai_plus_y_dft + (vmp | acc_big);
}

pub fn cggi_blind_rotate<DataRes, DataIn>(
    module: &Module<FFT64>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyCGGI<FFT64>,
    scratch: &mut Scratch,
) where
    DataRes: AsRef<[u8]> + AsMut<[u8]>,
    DataIn: AsRef<[u8]>,
{
    if lut.extension_factor() > 1 {
        cggi_blind_rotate_block_binary_extended(module, res, lwe, lut, brk, scratch);
    } else if brk.block_size() > 1 {
        cggi_blind_rotate_block_binary(module, res, lwe, lut, brk, scratch);
    } else {
        todo!("implement this case")
    }
}

pub(crate) fn cggi_blind_rotate_block_binary_extended<DataRes, DataIn>(
    module: &Module<FFT64>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyCGGI<FFT64>,
    scratch: &mut Scratch,
) where
    DataRes: AsRef<[u8]> + AsMut<[u8]>,
    DataIn: AsRef<[u8]>,
{
    let extension_factor: usize = lut.extension_factor();
    let basek: usize = res.basek();

    let (mut acc, scratch1) = scratch.tmp_vec_glwe_ct(extension_factor, module, basek, res.k(), res.rank());
    let (mut acc_dft, scratch2) = scratch1.tmp_vec_fourier_glwe_ct(extension_factor, module, basek, brk.k(), res.rank());
    let (mut vmp_res, scratch3) = scratch2.tmp_vec_fourier_glwe_ct(extension_factor, module, basek, brk.k(), res.rank());
    let (mut acc_add_dft, scratch4) = scratch3.tmp_vec_fourier_glwe_ct(extension_factor, module, basek, brk.k(), res.rank());

    (0..extension_factor).for_each(|i| {
        acc[i].data.zero();
    });

    let (mut xai_plus_y, scratch5) = scratch4.tmp_scalar_znx(module, 1);
    let (mut xai_plus_y_dft, scratch6) = scratch5.tmp_scalar_znx_dft(module, 1);
    let (mut acc_add_big, scratch7) = scratch6.tmp_vec_znx_big(module, 1, brk.size());

    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();

    let two_n_ext: usize = 2 * lut.domain_size();

    let cols: usize = res.rank() + 1;

    negate_and_mod_switch_2n(two_n_ext, &mut lwe_2n, &lwe_ref);

    let a: &[i64] = &lwe_2n[1..];
    let b_pos: usize = ((lwe_2n[0] + two_n_ext as i64) % two_n_ext as i64) as usize;

    let b_hi: usize = b_pos / extension_factor;
    let b_lo: usize = b_pos % extension_factor;

    for (i, j) in (0..b_lo).zip(extension_factor - b_lo..extension_factor) {
        module.vec_znx_rotate(b_hi as i64 + 1, &mut acc[i].data, 0, &lut.data[j], 0);
    }
    for (i, j) in (b_lo..extension_factor).zip(0..extension_factor - b_lo) {
        module.vec_znx_rotate(b_hi as i64, &mut acc[i].data, 0, &lut.data[j], 0);
    }

    let block_size: usize = brk.block_size();

    izip!(
        a.chunks_exact(block_size),
        brk.data.chunks_exact(block_size)
    )
    .enumerate()
    .for_each(|(i, (ai, ski))| {
        (0..extension_factor).for_each(|i| {
            (0..cols).for_each(|j| {
                module.vec_znx_dft(1, 0, &mut acc_dft[i].data, j, &acc[i].data, j);
            });
            acc_add_dft[i].data.zero();
        });

        // TODO: first & last iterations can be optimized
        izip!(ai.iter(), ski.iter()).for_each(|(aii, skii)| {
            let ai_pos: usize = ((aii + two_n_ext as i64) % two_n_ext as i64) as usize;
            let ai_hi: usize = ai_pos / extension_factor;
            let ai_lo: usize = ai_pos % extension_factor;

            // vmp_res = DFT(acc) * BRK[i]
            (0..extension_factor).for_each(|i| {
                module.vmp_apply(&mut vmp_res[i].data, &acc_dft[i].data, &skii.data, scratch7);
            });

            // Trivial case: no rotation between polynomials, we can directly multiply with (X^{-ai} - 1)
            if ai_lo == 0 {
                // DFT X^{-ai}
                set_xai_plus_y(
                    module,
                    ai_hi as i64,
                    -1,
                    &mut xai_plus_y_dft,
                    &mut xai_plus_y,
                );

                // Sets acc_add_dft[i] = (acc[i] * sk) * (X^{-ai} - 1)
                (0..extension_factor).for_each(|j| {
                    (0..cols).for_each(|i| {
                        module.svp_apply_inplace(&mut vmp_res[j].data, i, &xai_plus_y_dft, 0);
                        module.vec_znx_dft_add_inplace(&mut acc_add_dft[j].data, i, &vmp_res[j].data, i);
                    });
                });
            // Non trivial case: rotation between polynomials
            // In this case we can't directly multiply with (X^{-ai} - 1) because of the
            // ring homomorphism R^{N} -> prod R^{N/extension_factor}, so we split the
            // computation in two steps: acc_add_dft = (acc * sk) * (-1) + (acc * sk) * X^{-ai}
            } else {
                // Sets acc_add_dft[i] = acc[i] * sk
                (0..extension_factor).for_each(|i| {
                    (0..cols).for_each(|k| {
                        module.vec_znx_dft_sub_ab_inplace(&mut acc_add_dft[i].data, k, &vmp_res[i].data, k);
                    })
                });

                // DFT X^{-ai+1}
                set_xai_plus_y(
                    module,
                    ai_hi as i64 + 1,
                    0,
                    &mut xai_plus_y_dft,
                    &mut xai_plus_y,
                );

                // Sets acc_add_dft[0..ai_lo] += (acc[extension_factor - ai_lo..extension_factor] * sk) * X^{-ai+1}
                for (i, j) in (0..ai_lo).zip(extension_factor - ai_lo..extension_factor) {
                    module.vec_znx_rotate(b_hi as i64 + 1, &mut acc[i].data, 0, &lut.data[j], 0);
                    (0..cols).for_each(|k| {
                        module.svp_apply_inplace(&mut vmp_res[j].data, k, &xai_plus_y_dft, 0);
                        module.vec_znx_dft_add_inplace(&mut acc_add_dft[i].data, k, &vmp_res[j].data, k);
                    });
                }

                // DFT X^{-ai}
                set_xai_plus_y(
                    module,
                    ai_hi as i64,
                    0,
                    &mut xai_plus_y_dft,
                    &mut xai_plus_y,
                );

                // Sets acc_add_dft[ai_lo..extension_factor] += (acc[0..extension_factor - ai_lo] * sk) * X^{-ai}
                for (i, j) in (ai_lo..extension_factor).zip(0..extension_factor - ai_lo) {
                    module.vec_znx_rotate(b_hi as i64, &mut acc[i].data, 0, &lut.data[j], 0);
                    (0..cols).for_each(|k| {
                        module.svp_apply_inplace(&mut vmp_res[j].data, k, &xai_plus_y_dft, 0);
                        module.vec_znx_dft_add_inplace(&mut acc_add_dft[i].data, k, &vmp_res[j].data, k);
                    });
                }
            }
        });

        (0..extension_factor).for_each(|j| {
            (0..cols).for_each(|i| {
                module.vec_znx_dft_add_inplace(&mut acc_dft[j].data, i, &acc_add_dft[j].data, i);
                module.vec_znx_idft(&mut acc_add_big, 0, &acc_dft[j].data, i, scratch7);
                module.vec_znx_big_normalize(basek, &mut acc[j].data, i, &acc_add_big, 0, scratch7);
            });
        });
    });

    (0..cols).for_each(|i| {
        module.vec_znx_copy(&mut res.data, i, &acc[0].data, i);
    });
}

fn set_xai_plus_y(
    module: &Module<FFT64>,
    k: i64,
    y: i64,
    res: &mut ScalarZnxDft<&mut [u8], FFT64>,
    buf: &mut ScalarZnx<&mut [u8]>,
) {
    buf.zero();
    buf.at_mut(0, 0)[0] = 1;
    module.vec_znx_rotate_inplace(k, buf, 0);
    buf.at_mut(0, 0)[0] += y;
    module.svp_prepare(res, 0, buf, 0);
}

pub(crate) fn cggi_blind_rotate_block_binary<DataRes, DataIn>(
    module: &Module<FFT64>,
    res: &mut GLWECiphertext<DataRes>,
    lwe: &LWECiphertext<DataIn>,
    lut: &LookUpTable,
    brk: &BlindRotationKeyCGGI<FFT64>,
    scratch: &mut Scratch,
) where
    DataRes: AsRef<[u8]> + AsMut<[u8]>,
    DataIn: AsRef<[u8]>,
{
    let basek: usize = res.basek();

    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space
    let mut out_mut: GLWECiphertext<&mut [u8]> = res.to_mut();
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();

    let cols: usize = out_mut.rank() + 1;

    negate_and_mod_switch_2n(2 * lut.domain_size(), &mut lwe_2n, &lwe_ref);

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_mut.data.zero();

    // Initialize out to X^{b} * LUT(X)
    module.vec_znx_rotate(-b, &mut out_mut.data, 0, &lut.data[0], 0);

    let block_size: usize = brk.block_size();

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]

    let (mut acc_dft, scratch1) = scratch.tmp_fourier_glwe_ct(module, brk.basek(), out_mut.k(), out_mut.rank());
    let (mut acc_add_dft, scratch2) = scratch1.tmp_fourier_glwe_ct(module, brk.basek(), out_mut.k(), out_mut.rank());
    let (mut vmp_res, scratch3) = scratch2.tmp_fourier_glwe_ct(module, basek, out_mut.k(), out_mut.rank());
    let (mut xai_plus_y, scratch4) = scratch3.tmp_scalar_znx(module, 1);
    let (mut xai_plus_y_dft, scratch5) = scratch4.tmp_scalar_znx_dft(module, 1);

    let start: Instant = Instant::now();
    izip!(
        a.chunks_exact(block_size),
        brk.data.chunks_exact(block_size)
    )
    .for_each(|(ai, ski)| {
        out_mut.dft(module, &mut acc_dft);
        acc_add_dft.data.zero();

        izip!(ai.iter(), ski.iter()).for_each(|(aii, skii)| {
            // vmp_res = DFT(acc) * BRK[i]
            module.vmp_apply(&mut vmp_res.data, &acc_dft.data, &skii.data, scratch5);

            // DFT(X^ai -1)
            set_xai_plus_y(module, *aii, -1, &mut xai_plus_y_dft, &mut xai_plus_y);

            // DFT(X^ai -1) * (DFT(acc) * BRK[i])
            (0..cols).for_each(|i| {
                module.svp_apply_inplace(&mut vmp_res.data, i, &xai_plus_y_dft, 0);
                module.vec_znx_dft_add_inplace(&mut acc_add_dft.data, i, &vmp_res.data, i);
            });
        });

        (0..cols).for_each(|i| {
            module.vec_znx_dft_add_inplace(&mut acc_dft.data, i, &acc_add_dft.data, i);
        });

        acc_dft.idft(module, &mut out_mut, scratch5);
    });
    let duration: std::time::Duration = start.elapsed();
}

pub(crate) fn negate_and_mod_switch_2n(n: usize, res: &mut [i64], lwe: &LWECiphertext<&[u8]>) {
    let basek: usize = lwe.basek();

    let log2n: usize = usize::BITS as usize - (n - 1).leading_zeros() as usize + 1;

    res.copy_from_slice(&lwe.data.at(0, 0));
    res.iter_mut().for_each(|x| *x = -*x);

    if basek > log2n {
        let diff: usize = basek - log2n;
        res.iter_mut().for_each(|x| {
            *x = div_signed_by_pow2(x, diff);
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
fn div_signed_by_pow2(x: &i64, k: usize) -> i64 {
    let bias: i64 = (1 << k) - 1;
    (x + ((x >> 63) & bias)) >> k
}
