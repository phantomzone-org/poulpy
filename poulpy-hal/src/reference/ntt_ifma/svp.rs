//! SVP operations for the IFMA backend.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, ScalarZnx, ScalarZnxToRef, SvpPPol, SvpPPolToMut, SvpPPolToRef, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef,
        ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::ntt120::types::Q120bScalar,
};

use super::{
    NttIfmaCFromB, NttIfmaDFTExecute, NttIfmaFromZnx64, NttIfmaMulBbc, NttIfmaZero, ntt::NttIfmaTable, primes::Primes40,
    vec_znx_dft::NttIfmaModuleHandle,
};

/// Encode a scalar polynomial into IFMA prepared format.
pub fn ntt_ifma_svp_prepare<R, A, BE>(module: &impl NttIfmaModuleHandle, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaDFTExecute<NttIfmaTable<Primes40>> + NttIfmaFromZnx64 + NttIfmaCFromB,
    R: SvpPPolToMut<BE>,
    A: ScalarZnxToRef,
{
    let mut res: SvpPPol<&mut [u8], BE> = res.to_mut();
    let a: ScalarZnx<&[u8]> = a.to_ref();
    let n = res.n();

    let mut tmp = vec![0u64; 4 * n];
    BE::ntt_ifma_from_znx64(&mut tmp, a.at(a_col, 0));
    BE::ntt_ifma_dft_execute(module.get_ntt_ifma_table(), &mut tmp);

    let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(res_col, 0));
    BE::ntt_ifma_c_from_b(n, res_u32, &tmp);
}

/// Pointwise DFT-domain multiply: `res = a ⊙ b`.
pub fn ntt_ifma_svp_apply_dft_to_dft<R, A, C, BE>(
    module: &impl NttIfmaModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &C,
    b_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaMulBbc + NttIfmaZero,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
    C: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let meta = module.get_bbc_ifma_meta();
    let n = res.n();
    let res_size = res.size();
    let b_size = b.size();
    let min_size = res_size.min(b_size);

    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    for j in 0..min_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        let b_u32: &[u32] = cast_slice(b.at(b_col, j));
        for n_i in 0..n {
            BE::ntt_ifma_mul_bbc(
                meta,
                1,
                &mut res_u64[4 * n_i..4 * n_i + 4],
                &b_u32[8 * n_i..8 * n_i + 8],
                &a_u32[8 * n_i..8 * n_i + 8],
            );
        }
    }

    for j in min_size..res_size {
        BE::ntt_ifma_zero(cast_slice_mut(res.at_mut(res_col, j)));
    }
}

/// Pointwise DFT-domain multiply in place: `res = a ⊙ res`.
pub fn ntt_ifma_svp_apply_dft_to_dft_inplace<R, A, BE>(
    module: &impl NttIfmaModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttIfmaMulBbc,
    R: VecZnxDftToMut<BE>,
    A: SvpPPolToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: SvpPPol<&[u8], BE> = a.to_ref();

    let meta = module.get_bbc_ifma_meta();
    let n = res.n();
    let res_size = res.size();

    let a_u32: &[u32] = cast_slice(a.at(a_col, 0));

    for j in 0..res_size {
        let res_slice: &mut [Q120bScalar] = res.at_mut(res_col, j);
        let mut product = [0u64; 4];
        for n_i in 0..n {
            let x_elem: Q120bScalar = res_slice[n_i];
            let x_u32: &[u32] = cast_slice(std::slice::from_ref(&x_elem));
            BE::ntt_ifma_mul_bbc(meta, 1, &mut product, x_u32, &a_u32[8 * n_i..8 * n_i + 8]);
            res_slice[n_i] = Q120bScalar(product);
        }
    }
}
