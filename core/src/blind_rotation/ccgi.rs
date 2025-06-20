use std::time::Instant;

use backend::{
    FFT64, MatZnxDftOps, MatZnxDftScratch, Module, ScalarZnxAlloc, ScalarZnxDftAlloc, ScalarZnxDftOps, Scratch, VecZnxDftOps,
    VecZnxOps, ZnxView, ZnxViewMut, ZnxZero,
};
use itertools::izip;

use crate::{
    FourierGLWECiphertext, GGSWCiphertext, GLWECiphertext, GLWECiphertextToMut, Infos, LWECiphertext, ScratchCore,
    blind_rotation::{key::BlindRotationKeyCGGI, lut::LookUpTable},
    lwe::ciphertext::LWECiphertextToRef,
};

pub fn cggi_blind_rotate_scratch_space(
    module: &Module<FFT64>,
    basek: usize,
    k_lut: usize,
    k_brk: usize,
    rows: usize,
    rank: usize,
) -> usize {
    let size = k_brk.div_ceil(basek);
    GGSWCiphertext::<Vec<u8>, FFT64>::bytes_of(module, basek, k_brk, rows, 1, rank)
        + (module.mat_znx_dft_mul_x_pow_minus_one_scratch_space(size, rank + 1)
            | GLWECiphertext::external_product_inplace_scratch_space(module, basek, k_lut, k_brk, 1, rank))
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
    if lut.data.len() > 1 {
        cggi_blind_rotate_block_binary_exnteded(module, res, lwe, lut, brk, scratch);
    } else if brk.block_size() > 1 {
        cggi_blind_rotate_block_binary(module, res, lwe, lut, brk, scratch);
    } else {
        todo!("implement this case")
    }
}

pub(crate) fn cggi_blind_rotate_block_binary_exnteded<DataRes, DataIn>(
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
    let mut lwe_2n: Vec<i64> = vec![0i64; lwe.n() + 1]; // TODO: from scratch space
    let mut out_mut: GLWECiphertext<&mut [u8]> = res.to_mut();
    let lwe_ref: LWECiphertext<&[u8]> = lwe.to_ref();
    let basek: usize = out_mut.basek();

    let cols: usize = out_mut.rank() + 1;

    mod_switch_2n(
        2 * module.n() * lut.extension_factor(),
        &mut lwe_2n,
        &lwe_ref,
    );

    let extension_factor: i64 = lut.extension_factor() as i64;

    let mut acc: Vec<GLWECiphertext<Vec<u8>>> = Vec::with_capacity(lut.extension_factor());

    for _ in 0..extension_factor {
        acc.push(GLWECiphertext::alloc(
            module,
            basek,
            out_mut.k(),
            out_mut.rank(),
        ));
    }

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    let b_inner: i64 = b / extension_factor;
    let b_outer: i64 = b % extension_factor;

    for (i, j) in (0..b_outer).zip(extension_factor - b_outer..extension_factor) {
        module.vec_znx_rotate(
            b_inner + 1,
            &mut acc[j as usize].data,
            0,
            &lut.data[i as usize],
            0,
        );
    }
    for (i, j) in (b_outer..extension_factor).zip(0..extension_factor - b_outer) {
        module.vec_znx_rotate(
            b_inner,
            &mut acc[j as usize].data,
            0,
            &lut.data[i as usize],
            0,
        );
    }

    let block_size: usize = brk.block_size();

    let mut acc_dft: Vec<FourierGLWECiphertext<Vec<u8>, FFT64>> = Vec::with_capacity(lut.extension_factor());

    for _ in 0..extension_factor {
        acc_dft.push(FourierGLWECiphertext::alloc(
            module,
            basek,
            out_mut.k(),
            out_mut.rank(),
        ));
    }

    let mut vmp_res: Vec<FourierGLWECiphertext<Vec<u8>, FFT64>> = Vec::with_capacity(lut.extension_factor());

    for _ in 0..extension_factor {
        vmp_res.push(FourierGLWECiphertext::alloc(
            module,
            basek,
            out_mut.k(),
            out_mut.rank(),
        ));
    }

    let mut acc_add_dft: Vec<FourierGLWECiphertext<Vec<u8>, FFT64>> = Vec::with_capacity(lut.extension_factor());

    for _ in 0..extension_factor {
        acc_add_dft.push(FourierGLWECiphertext::alloc(
            module,
            basek,
            out_mut.k(),
            out_mut.rank(),
        ));
    }

    let mut xai_minus_one: backend::ScalarZnx<Vec<u8>> = module.new_scalar_znx(1);
    let mut xai_minus_one_dft: backend::ScalarZnxDft<Vec<u8>, FFT64> = module.new_scalar_znx_dft(1);

    izip!(
        a.chunks_exact(block_size),
        brk.data.chunks_exact(block_size)
    )
    .enumerate()
    .for_each(|(i, (ai, ski))| {
        (0..lut.extension_factor()).for_each(|i| {
            acc[i].dft(module, &mut acc_dft[i]);
            acc_add_dft[i].data.zero();
        });

        izip!(ai.iter(), ski.iter()).for_each(|(aii, skii)| {
            let aii_inner: i64 = aii / extension_factor;
            let aii_outer: i64 = aii % extension_factor;

            // vmp_res = DFT(acc) * BRK[i]
            (0..lut.extension_factor()).for_each(|i| {
                module.vmp_apply(&mut vmp_res[i].data, &acc_dft[i].data, &skii.data, scratch);
            });

            if aii_outer == 0 {
                xai_minus_one.zero();
                xai_minus_one.at_mut(0, 0)[0] = 1;
                module.vec_znx_rotate_inplace(aii_inner, &mut xai_minus_one, 0);
                xai_minus_one.at_mut(0, 0)[0] -= 1;
                module.svp_prepare(&mut xai_minus_one_dft, 0, &xai_minus_one, 0);

                (0..lut.extension_factor()).for_each(|j| {
                    (0..cols).for_each(|i| {
                        module.svp_apply_inplace(&mut vmp_res[j].data, i, &xai_minus_one_dft, 0);
                        module.vec_znx_dft_add_inplace(&mut acc_add_dft[j].data, i, &vmp_res[j].data, i);
                    });
                })
            } else {
                xai_minus_one.zero();
                xai_minus_one.at_mut(0, 0)[0] = 1;
                module.vec_znx_rotate_inplace(aii_inner + 1, &mut xai_minus_one, 0);
                xai_minus_one.at_mut(0, 0)[0] -= 1;
                module.svp_prepare(&mut xai_minus_one_dft, 0, &xai_minus_one, 0);

                for (i, j) in (0..aii_outer).zip(extension_factor - aii_outer..extension_factor) {
                    module.vec_znx_rotate(
                        b_inner + 1,
                        &mut acc[j as usize].data,
                        0,
                        &lut.data[i as usize],
                        0,
                    );

                    (0..cols).for_each(|k| {
                        module.svp_apply_inplace(&mut vmp_res[i as usize].data, k, &xai_minus_one_dft, 0);
                        module.vec_znx_dft_add_inplace(
                            &mut acc_add_dft[j as usize].data,
                            k,
                            &vmp_res[i as usize].data,
                            k,
                        );
                    });
                }

                xai_minus_one.zero();
                xai_minus_one.at_mut(0, 0)[0] = 1;
                module.vec_znx_rotate_inplace(aii_inner, &mut xai_minus_one, 0);
                xai_minus_one.at_mut(0, 0)[0] -= 1;
                module.svp_prepare(&mut xai_minus_one_dft, 0, &xai_minus_one, 0);

                for (i, j) in (aii_outer..extension_factor).zip(0..extension_factor - aii_outer) {
                    module.vec_znx_rotate(
                        b_inner,
                        &mut acc[j as usize].data,
                        0,
                        &lut.data[i as usize],
                        0,
                    );

                    (0..cols).for_each(|k| {
                        module.svp_apply_inplace(&mut vmp_res[i as usize].data, k, &xai_minus_one_dft, 0);
                        module.vec_znx_dft_add_inplace(
                            &mut acc_add_dft[j as usize].data,
                            k,
                            &vmp_res[i as usize].data,
                            k,
                        );
                    });
                }
            }
        });

        if i == lwe.n() - block_size {
            (0..cols).for_each(|i| {
                module.vec_znx_dft_add_inplace(&mut acc_dft[0].data, i, &acc_add_dft[0].data, i);
            });
            acc_dft[0].idft(module, &mut out_mut, scratch);
        } else {
            (0..lut.extension_factor()).for_each(|j| {
                (0..cols).for_each(|i| {
                    module.vec_znx_dft_add_inplace(&mut acc_dft[j].data, i, &acc_add_dft[j].data, i);
                });

                acc_dft[j].idft(module, &mut acc[j], scratch);
            })
        }
    });
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

    mod_switch_2n(2 * module.n(), &mut lwe_2n, &lwe_ref);

    let a: &[i64] = &lwe_2n[1..];
    let b: i64 = lwe_2n[0];

    out_mut.data.zero();

    // Initialize out to X^{b} * LUT(X)
    module.vec_znx_rotate(b, &mut out_mut.data, 0, &lut.data[0], 0);

    let block_size: usize = brk.block_size();

    // ACC + [sum DFT(X^ai -1) * (DFT(ACC) x BRKi)]

    let (mut acc_dft, scratch1) = scratch.tmp_glwe_fourier(module, brk.basek(), out_mut.k(), out_mut.rank());
    let (mut acc_add_dft, scratch2) = scratch1.tmp_glwe_fourier(module, brk.basek(), out_mut.k(), out_mut.rank());
    let (mut vmp_res, scratch3) = scratch2.tmp_glwe_fourier(module, basek, out_mut.k(), out_mut.rank());
    let (mut xai_minus_one, scratch4) = scratch3.tmp_scalar_znx(module, 1);
    let (mut xai_minus_one_dft, scratch5) = scratch4.tmp_scalar_znx_dft(module, 1);

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
            xai_minus_one.zero();
            xai_minus_one.at_mut(0, 0)[0] = 1;
            module.vec_znx_rotate_inplace(*aii, &mut xai_minus_one, 0);
            xai_minus_one.at_mut(0, 0)[0] -= 1;
            module.svp_prepare(&mut xai_minus_one_dft, 0, &xai_minus_one, 0);

            // DFT(X^ai -1) * (DFT(acc) * BRK[i])
            (0..cols).for_each(|i| {
                module.svp_apply_inplace(&mut vmp_res.data, i, &xai_minus_one_dft, 0);
                module.vec_znx_dft_add_inplace(&mut acc_add_dft.data, i, &vmp_res.data, i);
            });
        });

        (0..cols).for_each(|i| {
            module.vec_znx_dft_add_inplace(&mut acc_dft.data, i, &acc_add_dft.data, i);
        });

        acc_dft.idft(module, &mut out_mut, scratch5);
    });
    let duration: std::time::Duration = start.elapsed();
    println!("external products: {} us", duration.as_micros());
}

pub(crate) fn mod_switch_2n(n: usize, res: &mut [i64], lwe: &LWECiphertext<&[u8]>) {
    let basek: usize = lwe.basek();

    let log2n: usize = usize::BITS as usize - (n - 1).leading_zeros() as usize + 1;

    res.copy_from_slice(&lwe.data.at(0, 0));

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
