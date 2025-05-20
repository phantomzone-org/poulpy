use base2k::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx, ScalarZnxDft,
    ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnxAlloc, VecZnxBigOps, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut,
    VecZnxDftToRef, VecZnxOps, ZnxInfos, ZnxZero,
};
use sampling::source::Source;

use crate::{
    automorphism::AutomorphismKey,
    elem::{GetRow, Infos, SetRow},
    glwe_ciphertext::GLWECiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_plaintext::GLWEPlaintext,
    keys::SecretKeyFourier,
    keyswitch_key::GLWESwitchingKey,
    tensor_key::TensorKey,
    utils::derive_size,
};

pub struct GGSWCiphertext<C, B: Backend> {
    pub data: MatZnxDft<C, B>,
    pub basek: usize,
    pub k: usize,
}

impl<B: Backend> GGSWCiphertext<Vec<u8>, B> {
    pub fn new(module: &Module<B>, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(rows, rank + 1, rank + 1, derive_size(basek, k)),
            basek: basek,
            k: k,
        }
    }
}

impl<T, B: Backend> Infos for GGSWCiphertext<T, B> {
    type Inner = MatZnxDft<T, B>;

    fn inner(&self) -> &Self::Inner {
        &self.data
    }

    fn basek(&self) -> usize {
        self.basek
    }

    fn k(&self) -> usize {
        self.k
    }
}

impl<T, B: Backend> GGSWCiphertext<T, B> {
    pub fn rank(&self) -> usize {
        self.data.cols_out() - 1
    }
}

impl<C, B: Backend> MatZnxDftToMut<B> for GGSWCiphertext<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToMut<B>,
{
    fn to_mut(&mut self) -> MatZnxDft<&mut [u8], B> {
        self.data.to_mut()
    }
}

impl<C, B: Backend> MatZnxDftToRef<B> for GGSWCiphertext<C, B>
where
    MatZnxDft<C, B>: MatZnxDftToRef<B>,
{
    fn to_ref(&self) -> MatZnxDft<&[u8], B> {
        self.data.to_ref()
    }
}

impl GGSWCiphertext<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, rank: usize, size: usize) -> usize {
        GLWECiphertext::encrypt_sk_scratch_space(module, size)
            + module.bytes_of_vec_znx(rank + 1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(rank + 1, size)
    }

    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        auto_key_size: usize,
        rank: usize,
    ) -> usize {
        let tmp_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, auto_key_size);
        let tmp_idft: usize = module.bytes_of_vec_znx(rank + 1, out_size);
        let vmp: usize = GLWECiphertext::keyswitch_from_fourier_scratch_space(module, out_size, rank, in_size, rank, auto_key_size);
        tmp_dft + tmp_idft + vmp
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        let tmp_in: usize = module.bytes_of_vec_znx_dft(rank + 1, in_size);
        let tmp_out: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        let ggsw: usize = GLWECiphertextFourier::external_product_scratch_space(module, out_size, in_size, ggsw_size, rank);
        tmp_in + tmp_out + ggsw
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        ggsw_size: usize,
        rank: usize,
    ) -> usize {
        let tmp: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        let ggsw: usize = GLWECiphertextFourier::external_product_inplace_scratch_space(module, out_size, ggsw_size, rank);
        tmp + ggsw
    }
}

impl<DataSelf> GGSWCiphertext<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64> + MatZnxDftToRef<FFT64>,
{
    pub fn encrypt_sk<DataPt, DataSk>(
        &mut self,
        module: &Module<FFT64>,
        pt: &ScalarZnx<DataPt>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) where
        ScalarZnx<DataPt>: ScalarZnxToRef,
        ScalarZnxDft<DataSk, FFT64>: ScalarZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk_dft.n(), module.n());
        }

        let size: usize = self.size();
        let basek: usize = self.basek();
        let k: usize = self.k();
        let cols: usize = self.rank() + 1;

        let (tmp_znx_pt, scratch_1) = scratch.tmp_vec_znx(module, 1, size);
        let (tmp_znx_ct, scrach_2) = scratch_1.tmp_vec_znx(module, cols, size);

        let mut vec_znx_pt: GLWEPlaintext<&mut [u8]> = GLWEPlaintext {
            data: tmp_znx_pt,
            basek: basek,
            k: k,
        };

        let mut vec_znx_ct: GLWECiphertext<&mut [u8]> = GLWECiphertext {
            data: tmp_znx_ct,
            basek: basek,
            k,
        };

        (0..self.rows()).for_each(|row_j| {
            vec_znx_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_j, pt, 0);
            module.vec_znx_normalize_inplace(basek, &mut vec_znx_pt, 0, scrach_2);

            (0..cols).for_each(|col_i| {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                vec_znx_ct.encrypt_sk_private(
                    module,
                    Some((&vec_znx_pt, col_i)),
                    sk_dft,
                    source_xa,
                    source_xe,
                    sigma,
                    scrach_2,
                );

                // Switch vec_znx_ct into DFT domain
                {
                    let (mut vec_znx_dft_ct, _) = scrach_2.tmp_vec_znx_dft(module, cols, size);

                    (0..cols).for_each(|i| {
                        module.vec_znx_dft(&mut vec_znx_dft_ct, i, &vec_znx_ct, i);
                    });

                    module.vmp_prepare_row(self, row_j, col_i, &vec_znx_dft_ct);
                }
            });
        });
    }


    pub fn keyswitch<DataLhs, DataRhs0, DataRhs1>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        ksk: &GLWESwitchingKey<DataRhs0, FFT64>,
        tsk: &TensorKey<DataRhs1, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs0, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs1, FFT64>: MatZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                lhs.rank(),
                ksk.rank(),
                "ggsw_in rank: {} != ksk rank: {}",
                lhs.rank(),
                ksk.rank()
            );
            assert_eq!(
                lhs.rank(),
                tsk.rank(),
                "ggsw_in rank: {} != tsk rank: {}",
                lhs.rank(),
                tsk.rank()
            );
        }

        let cols: usize = self.rank() + 1;

        // Example for rank 3:
        // 
        // Note: M is a vector (m, Bm, B^2m, B^3m, ...), so each column is
        // actually composed of that many rows.
        //
        // # Input
        //
        // col 0: (-(a0s0 + a1s1 + a2s2) + M, a0    , a1    , a2    )
        // col 1: (-(b0s0 + b1s1 + b2s2)    , b0 + M, b1    , b2    )
        // col 2: (-(c0s0 + c1s1 + c2s2)    , c0    , c1 + M, c2    )
        // col 3: (-(d0s0 + d1s1 + d2s2)    , d0    , d1    , d2 + M)
        //
        // # Output 
        //
        // col 0: (-(a0s0' + a1s1' + a2s2') + M, a0    , a1    , a2    )
        // col 1: (-(b0s0' + b1s1' + b2s2')    , b0 + M, b1    , b2    )
        // col 2: (-(c0s0' + c1s1' + c2s2')    , c0    , c1 + M, c2    )
        // col 3: (-(d0s0' + d1s1' + d2s2')    , d0    , d1    , d2 + M)
        (0..self.rows()).for_each(|row_j| {

            let (tmp_dft_out_data, scratch1) = scratch.tmp_vec_znx_dft(module, self.rank() + 1, self.size());

            let mut tmp_dft_out: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
                data: tmp_dft_out_data,
                basek: lhs.basek(),
                k: lhs.k(),
            };

            {
                let (tmp_dft_in_data, scratch2) = scratch1.tmp_vec_znx_dft(module, lhs.rank() + 1, lhs.size());

                let mut tmp_dft_in: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
                    data: tmp_dft_in_data,
                    basek: lhs.basek(),
                    k: lhs.k(),
                };

                // 1) Applies key-switching to GGSW[i][0]: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2)
                lhs.get_row(module, row_j, 0, &mut tmp_dft_in);
                // (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
                tmp_dft_out.keyswitch(module, &tmp_dft_in, ksk, scratch2);
                self.set_row(module, row_j, 0, &tmp_dft_out);
            }

            // 2) Isolates IDFT(-(a0s0' + a1s1' + a2s2') + M[i])
            let (mut tmp_c0_data, scratch2) = scratch1.tmp_vec_znx_big(module, 1, self.size());
            module.vec_znx_idft_tmp_a(&mut tmp_c0_data, 0, &mut tmp_dft_out, 0);

            // 3) Expands the i-th row of the other columns using the tensor key
            // col 1: (-(b0s0' + b1s1' + b2s2')    , b0 + M[i], b1       , b2       ) = KS_{s0's0', s0's1', s0's2'}(a0) + (0, -(a0s0' + a1s1' + a2s2') + M[i], 0, 0)
            // col 2: (-(c0s0' + c1s1' + c2s2')    , c0       , c1 + M[i], c2       ) = KS_{s1's0', s1's1', s1's2'}(a1) + (0, 0, -(a0s0' + a1s1' + a2s2') + M[i], 0)
            // col 3: (-(d0s0' + d1s1' + d2s2')    , d0       , d1       , d2 + M[i]) = KS_{s2's0', s2's1', s2's2'}(a2) + (0, 0, 0, -(a0s0' + a1s1' + a2s2') + M[i])
            (1..cols).for_each(|col_i| {

                let (tmp_dft_i_data, scratch3) = scratch2.tmp_vec_znx_dft(module, cols, tsk.size());
                let mut tmp_dft_i: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
                    data: tmp_dft_i_data,
                    basek: lhs.basek(),
                    k: lhs.k(),
                };

                // 5) Performs a key-switch for each combination of s[i]*s[j], i.e. for a0, a1, a2
                //
                // # Example for col=1
                //
                // a0 * (-(f0s0 + f1s1 + f1s2) + s0^2, f0, f1, f2) = (-(a0f0s0 + a0f1s1 + a0f1s2) + a0s0^2, a0f0, a0f1, a0f2)
                // +
                // a1 * (-(g0s0 + g1s1 + g1s2) + s0s1, g0, g1, g2) = (-(a1g0s0 + a1g1s1 + a1g1s2) + a1s0s1, a1g0, a1g1, a1g2)
                // +
                // a2 * (-(h0s0 + h1s1 + h1s2) + s0s2, h0, h1, h2) = (-(a2h0s0 + a2h1s1 + a2h1s2) + a2s0s2, a2h0, a2h1, a2h2)
                // =
                // (-(x0s0' + x1s1' + x2s2') + s0'(a0s0' + a1s1' + a2s2'), x0, x1, x2)
                (1..cols).for_each(|col_j| {

                    // Extracts a[i] and multipies with Enc(s'[i]s'[j])
                    let (mut tmp_dft_col_data, scratch4) = scratch3.tmp_vec_znx_dft(module, 1, self.size());
                    tmp_dft_col_data.extract_column(0, &tmp_dft_out.data, col_j);

                    if col_j == 1 {
                        module.vmp_apply(
                            &mut tmp_dft_i,
                            &tmp_dft_col_data,
                            tsk.at(col_i - 1, col_j - 1), // Selects Enc(s'[i]s'[j])
                            scratch4,
                        );
                    } else {
                        module.vmp_apply_add(
                            &mut tmp_dft_i,
                            &tmp_dft_col_data,
                            tsk.at(col_i - 1, col_j - 1), // Selects Enc(s'[i]s'[j])
                            scratch4,
                        );
                    }
                });

                // Adds -(sum a[i] * s[i]) + m)  on the i-th column of tmp_idft_i
                //
                // (-(x0s0' + x1s1' + x2s2') + a0s0's0' + a1s0's1' + a2s0's2', x0, x1, x2)
                // + 
                // (0, -(a0s0' + a1s1' + a2s2') + M[i], 0, 0)
                // =
                // (-(x0s0' + x1s1' + x2s2') + s0'(a0s0' + a1s1' + a2s2'), x0 -(a0s0' + a1s1' + a2s2') + M[i], x1, x2)
                // =
                // (-(x0s0' + x1s1' + x2s2'), x0 + M[i], x1, x2)
                {
                    let (mut tmp_idft, scratch3) = scratch3.tmp_vec_znx_big(module, 1, tsk.size());
                    let (mut tmp_znx_small, scratch5) = scratch3.tmp_vec_znx(module, 1, self.size());
                    (0..cols).for_each(|i| {
                        module.vec_znx_idft_tmp_a(&mut tmp_idft, 0, &mut tmp_dft_i, i);
                        module.vec_znx_big_add_inplace(&mut tmp_idft, col_i, &tmp_c0_data, 0);
                        module.vec_znx_big_normalize(self.basek(), &mut tmp_znx_small, 0, &tmp_idft, 0, scratch5);
                        module.vec_znx_dft(&mut tmp_dft_i, i, &tmp_znx_small, 0);
                    });
                }

                // Stores (-(x0s0' + x1s1' + x2s2'), x0 + M[i], x1, x2)
                self.set_row(module, row_j, col_i, &tmp_dft_i);
            })
        })
    }

    pub fn automorphism<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_in rank: {} != auto_key rank: {}",
                self.rank(),
                rhs.rank()
            );
        }
;
        let cols: usize = self.rank() + 1;

        let (tmp_dft_data, scratch1) = scratch.tmp_vec_znx_dft(module, cols, rhs.size()); //TODO optimize

        let mut tmp_dft: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_dft_data,
            basek: lhs.basek(),
            k: lhs.k(),
        };

        let (tmp_idft_data, scratch2) = scratch1.tmp_vec_znx(module, cols, self.size());

        let mut tmp_idft: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: tmp_idft_data,
            basek: self.basek(),
            k: self.k(),
        };

        (0..cols).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                lhs.get_row(module, row_j, col_i, &mut tmp_dft);
                tmp_idft.keyswitch_from_fourier(module, &tmp_dft, &rhs.key, scratch2);
                (0..cols).for_each(|i| {
                    module.vec_znx_automorphism_inplace(rhs.p(), &mut tmp_idft, i);
                });
                self.set_row(module, row_j, col_i, &tmp_dft);
            });
        });

        tmp_dft.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank() + 1).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_dft);
            });
        });
    }

    pub fn external_product<DataLhs, DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                lhs.rank(),
                "ggsw_out rank: {} != ggsw_in rank: {}",
                self.rank(),
                lhs.rank()
            );
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_in rank: {} != ggsw_apply rank: {}",
                self.rank(),
                rhs.rank()
            );
        }

        let (tmp_in_data, scratch1) = scratch.tmp_vec_znx_dft(module, lhs.rank() + 1, lhs.size());

        let mut tmp_in: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_in_data,
            basek: lhs.basek(),
            k: lhs.k(),
        };

        let (tmp_out_data, scratch2) = scratch1.tmp_vec_znx_dft(module, self.rank() + 1, self.size());

        let mut tmp_out: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_out_data,
            basek: self.basek(),
            k: self.k(),
        };

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                lhs.get_row(module, row_j, col_i, &mut tmp_in);
                tmp_out.external_product(module, &tmp_in, rhs, scratch2);
                self.set_row(module, row_j, col_i, &tmp_out);
            });
        });

        tmp_out.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank() + 1).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_out);
            });
        });
    }

    pub fn external_product_inplace<DataRhs>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataRhs, FFT64>: MatZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                self.rank(),
                rhs.rank(),
                "ggsw_out rank: {} != ggsw_apply: {}",
                self.rank(),
                rhs.rank()
            );
        }

        let (tmp_data, scratch1) = scratch.tmp_vec_znx_dft(module, self.rank() + 1, self.size());

        let mut tmp: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_data,
            basek: self.basek(),
            k: self.k(),
        };

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.get_row(module, row_j, col_i, &mut tmp);
                tmp.external_product_inplace(module, rhs, scratch1);
                self.set_row(module, row_j, col_i, &tmp);
            });
        });
    }
}

impl<DataSelf> GetRow<FFT64> for GGSWCiphertext<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToRef<FFT64>,
{
    fn get_row<R>(&self, module: &Module<FFT64>, row_i: usize, col_j: usize, res: &mut R)
    where
        R: VecZnxDftToMut<FFT64>,
    {
        module.vmp_extract_row(res, self, row_i, col_j);
    }
}

impl<DataSelf> SetRow<FFT64> for GGSWCiphertext<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToMut<FFT64>,
{
    fn set_row<R>(&mut self, module: &Module<FFT64>, row_i: usize, col_j: usize, a: &R)
    where
        R: VecZnxDftToRef<FFT64>,
    {
        module.vmp_prepare_row(self, row_i, col_j, a);
    }
}
