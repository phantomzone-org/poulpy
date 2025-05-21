use backend::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftScratch, MatZnxDftToMut, MatZnxDftToRef, Module, ScalarZnx,
    ScalarZnxDft, ScalarZnxDftToRef, ScalarZnxToRef, Scratch, VecZnx, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigOps,
    VecZnxBigScratch, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxDftToMut, VecZnxDftToRef, VecZnxOps, VecZnxToMut,
    VecZnxToRef, ZnxInfos, ZnxZero,
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
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, rank: usize) -> Self {
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

    pub(crate) fn expand_row_scratch_space(
        module: &Module<FFT64>,
        self_size: usize,
        tensor_key_size: usize,
        rank: usize,
    ) -> usize {
        let tmp_dft_i: usize = module.bytes_of_vec_znx_dft(rank + 1, tensor_key_size);
        let tmp_dft_col_data: usize = module.bytes_of_vec_znx_dft(1, self_size);
        let vmp: usize =
            tmp_dft_col_data + module.vmp_apply_tmp_bytes(self_size, self_size, self_size, rank, rank, tensor_key_size);
        let tmp_idft: usize = module.bytes_of_vec_znx_big(1, tensor_key_size);
        let norm: usize = module.vec_znx_big_normalize_tmp_bytes();
        tmp_dft_i + ((tmp_dft_col_data + vmp) | (tmp_idft + norm))
    }

    pub(crate) fn keyswitch_internal_col0_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ksk_size: usize,
        rank: usize,
    ) -> usize {
        GLWECiphertext::keyswitch_from_fourier_scratch_space(module, out_size, rank, in_size, rank, ksk_size)
            + module.bytes_of_vec_znx_dft(rank + 1, in_size)
    }

    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        ksk_size: usize,
        tensor_key_size: usize,
        rank: usize,
    ) -> usize {
        let res_znx: usize = module.bytes_of_vec_znx(rank + 1, out_size);
        let ci_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        let ks: usize = GGSWCiphertext::keyswitch_internal_col0_scratch_space(module, out_size, in_size, ksk_size, rank);
        let expand_rows: usize = GGSWCiphertext::expand_row_scratch_space(module, out_size, tensor_key_size, rank);
        let res_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        res_znx + ci_dft + (ks | expand_rows | res_dft)
    }

    pub fn keyswitch_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        ksk_size: usize,
        tensor_key_size: usize,
        rank: usize,
    ) -> usize {
        GGSWCiphertext::keyswitch_scratch_space(module, out_size, out_size, ksk_size, tensor_key_size, rank)
    }

    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        in_size: usize,
        auto_key_size: usize,
        tensor_key_size: usize,
        rank: usize,
    ) -> usize {
        GGSWCiphertext::keyswitch_scratch_space(
            module,
            out_size,
            in_size,
            auto_key_size,
            tensor_key_size,
            rank,
        )
    }

    pub fn automorphism_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        auto_key_size: usize,
        tensor_key_size: usize,
        rank: usize,
    ) -> usize {
        GGSWCiphertext::automorphism_scratch_space(
            module,
            out_size,
            out_size,
            auto_key_size,
            tensor_key_size,
            rank,
        )
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

        (0..self.rows()).for_each(|row_i| {
            vec_znx_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut vec_znx_pt, 0, row_i, pt, 0);
            module.vec_znx_normalize_inplace(basek, &mut vec_znx_pt, 0, scrach_2);

            (0..cols).for_each(|col_j| {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                vec_znx_ct.encrypt_sk_private(
                    module,
                    Some((&vec_znx_pt, col_j)),
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

                    self.set_row(module, row_i, col_j, &vec_znx_dft_ct);
                }
            });
        });
    }

    pub(crate) fn expand_row<R, DataCi, DataTsk>(
        &mut self,
        module: &Module<FFT64>,
        col_j: usize,
        res: &mut R,
        ci_dft: &VecZnxDft<DataCi, FFT64>,
        tsk: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) where
        R: VecZnxToMut,
        VecZnxDft<DataCi, FFT64>: VecZnxDftToRef<FFT64>,
        MatZnxDft<DataTsk, FFT64>: MatZnxDftToRef<FFT64>,
    {
        let cols: usize = self.rank() + 1;

        // Example for rank 3:
        //
        // Note: M is a vector (m, Bm, B^2m, B^3m, ...), so each column is
        // actually composed of that many rows and we focus on a specific row here
        // implicitely given ci_dft.
        //
        // # Input
        //
        // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0    , a1    , a2    )
        // col 1: (0, 0, 0, 0)
        // col 2: (0, 0, 0, 0)
        // col 3: (0, 0, 0, 0)
        //
        // # Output
        //
        // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0       , a1       , a2       )
        // col 1: (-(b0s0 + b1s1 + b2s2)       , b0 + M[i], b1       , b2       )
        // col 2: (-(c0s0 + c1s1 + c2s2)       , c0       , c1 + M[i], c2       )
        // col 3: (-(d0s0 + d1s1 + d2s2)       , d0       , d1       , d2 + M[i])

        let (mut tmp_dft_i, scratch1) = scratch.tmp_vec_znx_dft(module, cols, tsk.size());
        {
            let (mut tmp_dft_col_data, scratch2) = scratch1.tmp_vec_znx_dft(module, 1, self.size());

            // Performs a key-switch for each combination of s[i]*s[j], i.e. for a0, a1, a2
            //
            // # Example for col=1
            //
            // a0 * (-(f0s0 + f1s1 + f1s2) + s0^2, f0, f1, f2) = (-(a0f0s0 + a0f1s1 + a0f1s2) + a0s0^2, a0f0, a0f1, a0f2)
            // +
            // a1 * (-(g0s0 + g1s1 + g1s2) + s0s1, g0, g1, g2) = (-(a1g0s0 + a1g1s1 + a1g1s2) + a1s0s1, a1g0, a1g1, a1g2)
            // +
            // a2 * (-(h0s0 + h1s1 + h1s2) + s0s2, h0, h1, h2) = (-(a2h0s0 + a2h1s1 + a2h1s2) + a2s0s2, a2h0, a2h1, a2h2)
            // =
            // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0, x1, x2)
            (1..cols).for_each(|col_i| {
                // Extracts a[i] and multipies with Enc(s[i]s[j])
                tmp_dft_col_data.extract_column(0, ci_dft, col_i);

                if col_i == 1 {
                    module.vmp_apply(
                        &mut tmp_dft_i,
                        &tmp_dft_col_data,
                        tsk.at(col_i - 1, col_j - 1), // Selects Enc(s[i]s[j])
                        scratch2,
                    );
                } else {
                    module.vmp_apply_add(
                        &mut tmp_dft_i,
                        &tmp_dft_col_data,
                        tsk.at(col_i - 1, col_j - 1), // Selects Enc(s[i]s[j])
                        scratch2,
                    );
                }
            });
        }

        // Adds -(sum a[i] * s[i]) + m)  on the i-th column of tmp_idft_i
        //
        // (-(x0s0 + x1s1 + x2s2) + a0s0s0 + a1s0s1 + a2s0s2, x0, x1, x2)
        // +
        // (0, -(a0s0 + a1s1 + a2s2) + M[i], 0, 0)
        // =
        // (-(x0s0 + x1s1 + x2s2) + s0(a0s0 + a1s1 + a2s2), x0 -(a0s0 + a1s1 + a2s2) + M[i], x1, x2)
        // =
        // (-(x0s0 + x1s1 + x2s2), x0 + M[i], x1, x2)
        module.vec_znx_dft_add_inplace(&mut tmp_dft_i, col_j, ci_dft, 0);
        let (mut tmp_idft, scratch2) = scratch1.tmp_vec_znx_big(module, 1, tsk.size());
        (0..cols).for_each(|i| {
            module.vec_znx_idft_tmp_a(&mut tmp_idft, 0, &mut tmp_dft_i, i);
            module.vec_znx_big_normalize(self.basek(), res, i, &tmp_idft, 0, scratch2);
        });
    }

    pub fn keyswitch<DataLhs, DataKsk, DataTsk>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        ksk: &GLWESwitchingKey<DataKsk, FFT64>,
        tsk: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataKsk, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataTsk, FFT64>: MatZnxDftToRef<FFT64>,
    {
        let cols: usize = self.rank() + 1;

        let (res_data, scratch1) = scratch.tmp_vec_znx(&module, cols, self.size());
        let mut res: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: res_data,
            basek: self.basek(),
            k: self.k(),
        };

        let (mut ci_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, self.size());

        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            lhs.keyswitch_internal_col0(module, row_i, &mut res, ksk, scratch2);

            // Isolates DFT(a[i])
            (0..cols).for_each(|col_i| {
                module.vec_znx_dft(&mut ci_dft, col_i, &res, col_i);
            });

            self.set_row(module, row_i, 0, &ci_dft);

            // Generates
            //
            // col 1: (-(b0s0' + b1s1' + b2s2')    , b0 + M[i], b1       , b2       )
            // col 2: (-(c0s0' + c1s1' + c2s2')    , c0       , c1 + M[i], c2       )
            // col 3: (-(d0s0' + d1s1' + d2s2')    , d0       , d1       , d2 + M[i])
            (1..cols).for_each(|col_j| {
                self.expand_row(module, col_j, &mut res, &ci_dft, tsk, scratch2);

                let (mut res_dft, _) = scratch2.tmp_vec_znx_dft(module, cols, self.size());
                (0..cols).for_each(|i| {
                    module.vec_znx_dft(&mut res_dft, i, &res, i);
                });

                self.set_row(module, row_i, col_j, &res_dft);
            })
        })
    }

    pub fn keyswitch_inplace<DataKsk, DataTsk>(
        &mut self,
        module: &Module<FFT64>,
        ksk: &GLWESwitchingKey<DataKsk, FFT64>,
        tsk: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataKsk, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataTsk, FFT64>: MatZnxDftToRef<FFT64>,
    {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf, FFT64> = self as *mut GGSWCiphertext<DataSelf, FFT64>;
            self.keyswitch(module, &*self_ptr, ksk, tsk, scratch);
        }
    }

    pub fn automorphism<DataLhs, DataAk, DataTsk>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        auto_key: &AutomorphismKey<DataAk, FFT64>,
        tensor_key: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataLhs, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataAk, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataTsk, FFT64>: MatZnxDftToRef<FFT64>,
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
                auto_key.rank(),
                "ggsw_in rank: {} != auto_key rank: {}",
                self.rank(),
                auto_key.rank()
            );
            assert_eq!(
                self.rank(),
                tensor_key.rank(),
                "ggsw_in rank: {} != tensor_key rank: {}",
                self.rank(),
                tensor_key.rank()
            );
        };

        let cols: usize = self.rank() + 1;

        let (res_data, scratch1) = scratch.tmp_vec_znx(&module, cols, self.size());
        let mut res: GLWECiphertext<&mut [u8]> = GLWECiphertext::<&mut [u8]> {
            data: res_data,
            basek: self.basek(),
            k: self.k(),
        };

        let (mut ci_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, self.size());

        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            lhs.keyswitch_internal_col0(module, row_i, &mut res, &auto_key.key, scratch2);

            // Isolates DFT(AUTO(a[i]))
            (0..cols).for_each(|col_i| {
                // (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2) -> (-(a0s0 + a1s1 + a2s2) + pi(M[i]), a0, a1, a2)
                module.vec_znx_automorphism_inplace(auto_key.p(), &mut res, col_i);
                module.vec_znx_dft(&mut ci_dft, col_i, &res, col_i);
            });

            self.set_row(module, row_i, 0, &ci_dft);

            // Generates
            //
            // col 1: (-(b0s0 + b1s1 + b2s2)    , b0 + pi(M[i]), b1           , b2           )
            // col 2: (-(c0s0 + c1s1 + c2s2)    , c0           , c1 + pi(M[i]), c2           )
            // col 3: (-(d0s0 + d1s1 + d2s2)    , d0           , d1           , d2 + pi(M[i]))
            (1..cols).for_each(|col_j| {
                self.expand_row(module, col_j, &mut res, &ci_dft, tensor_key, scratch2);

                let (mut res_dft, _) = scratch2.tmp_vec_znx_dft(module, cols, self.size());
                (0..cols).for_each(|i| {
                    module.vec_znx_dft(&mut res_dft, i, &res, i);
                });

                self.set_row(module, row_i, col_j, &res_dft);
            })
        })
    }

    pub fn automorphism_inplace<DataKsk, DataTsk>(
        &mut self,
        module: &Module<FFT64>,
        auto_key: &AutomorphismKey<DataKsk, FFT64>,
        tensor_key: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) where
        MatZnxDft<DataKsk, FFT64>: MatZnxDftToRef<FFT64>,
        MatZnxDft<DataTsk, FFT64>: MatZnxDftToRef<FFT64>,
    {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf, FFT64> = self as *mut GGSWCiphertext<DataSelf, FFT64>;
            self.automorphism(module, &*self_ptr, auto_key, tensor_key, scratch);
        }
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

impl<DataSelf> GGSWCiphertext<DataSelf, FFT64>
where
    MatZnxDft<DataSelf, FFT64>: MatZnxDftToRef<FFT64>,
{
    pub(crate) fn keyswitch_internal_col0<DataRes, DataKsk>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        res: &mut GLWECiphertext<DataRes>,
        ksk: &GLWESwitchingKey<DataKsk, FFT64>,
        scratch: &mut Scratch,
    ) where
        VecZnx<DataRes>: VecZnxToMut + VecZnxToRef,
        MatZnxDft<DataKsk, FFT64>: MatZnxDftToRef<FFT64>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), ksk.rank());
            assert_eq!(res.rank(), ksk.rank());
        }

        let (tmp_dft_in_data, scratch2) = scratch.tmp_vec_znx_dft(module, self.rank() + 1, self.size());
        let mut tmp_dft_in: GLWECiphertextFourier<&mut [u8], FFT64> = GLWECiphertextFourier::<&mut [u8], FFT64> {
            data: tmp_dft_in_data,
            basek: self.basek(),
            k: self.k(),
        };
        self.get_row(module, row_i, 0, &mut tmp_dft_in);
        res.keyswitch_from_fourier(module, &tmp_dft_in, ksk, scratch2);
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
