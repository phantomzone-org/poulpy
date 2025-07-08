use backend::{
    FFT64, Module, ScalarZnx, ScalarZnxDftOps, ScratchOwned, Stats, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps, VecZnxDft,
    VecZnxDftAlloc, VecZnxDftOps, VecZnxOps, ZnxZero,
};

use crate::{FourierGLWECiphertext, FourierGLWESecret, GGSWCiphertext, GLWEPlaintext, GetRow, Infos};

pub fn get_ggsw_noise<DCt, DPt, DSk>(
    module: &Module<FFT64>,
    ct_ggsw: &GGSWCiphertext<DCt, FFT64>,
    pt_ggsw: &ScalarZnx<DPt>,
    sk: &FourierGLWESecret<DSk, FFT64>,
) -> Vec<f64>
where
    DCt: AsRef<[u8]>,
    DPt: AsRef<[u8]>,
    DSk: AsRef<[u8]>,
{
    let basek: usize = ct_ggsw.basek();
    let k: usize = ct_ggsw.k();
    let size: usize = ct_ggsw.size();
    let rank: usize = ct_ggsw.rank();
    let digits: usize = ct_ggsw.digits();

    let mut scratch: ScratchOwned = ScratchOwned::new(FourierGLWECiphertext::decrypt_scratch_space(
        &module, basek, k,
    ));

    let mut ct_glwe_fourier: FourierGLWECiphertext<Vec<u8>, FFT64> = FourierGLWECiphertext::alloc(&module, basek, k, rank);
    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);
    let mut pt_dft: VecZnxDft<Vec<u8>, FFT64> = module.new_vec_znx_dft(1, size);
    let mut pt_big: VecZnxBig<Vec<u8>, FFT64> = module.new_vec_znx_big(1, size);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc(&module, basek, k);

    let mut noise: Vec<f64> = Vec::new();

    (0..ct_ggsw.rank() + 1).for_each(|col_j| {
        (0..ct_ggsw.rows()).for_each(|row_i| {
            module.vec_znx_add_scalar_inplace(
                &mut pt_want.data,
                0,
                (digits - 1) + row_i * digits,
                pt_ggsw,
                0,
            );

            if col_j > 0 {
                module.vec_znx_dft(1, 0, &mut pt_dft, 0, &pt_want.data, 0);
                module.svp_apply_inplace(&mut pt_dft, 0, &sk.data, col_j - 1);
                module.vec_znx_idft_tmp_a(&mut pt_big, 0, &mut pt_dft, 0);
                module.vec_znx_big_normalize(basek, &mut pt_want.data, 0, &pt_big, 0, scratch.borrow());
            }

            ct_ggsw.get_row(&module, row_i, col_j, &mut ct_glwe_fourier);
            ct_glwe_fourier.decrypt(&module, &mut pt, &sk, scratch.borrow());

            module.vec_znx_sub_ab_inplace(&mut pt.data, 0, &pt_want.data, 0);

            noise.push(pt.data.std(0, basek).log2());

            pt_want.data.zero();
        });
    });

    noise
}
