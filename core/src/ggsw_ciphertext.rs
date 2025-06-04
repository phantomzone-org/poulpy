use backend::{
    Backend, FFT64, MatZnxDft, MatZnxDftAlloc, MatZnxDftOps, MatZnxDftScratch, Module, ScalarZnx, Scratch, VecZnxAlloc,
    VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDft, VecZnxDftAlloc, VecZnxDftOps, VecZnxOps, VecZnxToMut, ZnxInfos,
    ZnxZero,
};
use sampling::source::Source;

use crate::{
    AutomorphismKey, GLWECiphertext, GLWECiphertextFourier, GLWESecret, GLWESwitchingKey, GetRow, Infos, ScratchCore, SetRow,
    TensorKey, div_ceil,
};

pub struct GGSWCiphertext<C, B: Backend> {
    pub(crate) data: MatZnxDft<C, B>,
    pub(crate) basek: usize,
    pub(crate) k: usize,
    pub(crate) digits: usize,
}

impl<B: Backend> GGSWCiphertext<Vec<u8>, B> {
    pub fn alloc(module: &Module<B>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> Self {
        Self {
            data: module.new_mat_znx_dft(div_ceil(rows, digits), rank + 1, rank + 1, div_ceil(basek, k)),
            basek,
            k: k,
            digits,
        }
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rows: usize, digits: usize, rank: usize) -> usize {
        module.bytes_of_mat_znx_dft(div_ceil(rows, digits), rank + 1, rank + 1, div_ceil(basek, k))
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

    pub fn digits(&self) -> usize {
        self.digits
    }
}

impl GGSWCiphertext<Vec<u8>, FFT64> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        let size = div_ceil(basek, k);
        GLWECiphertext::encrypt_sk_scratch_space(module, basek, k)
            + module.bytes_of_vec_znx(rank + 1, size)
            + module.bytes_of_vec_znx(1, size)
            + module.bytes_of_vec_znx_dft(rank + 1, size)
    }

    pub(crate) fn expand_row_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        self_k: usize,
        tsk_k: usize,
        rank: usize,
    ) -> usize {
        let tsk_size: usize = div_ceil(basek, tsk_k);
        let self_size: usize = div_ceil(basek, self_k);
        let tmp_dft_i: usize = module.bytes_of_vec_znx_dft(rank + 1, tsk_size);
        let tmp_dft_col_data: usize = module.bytes_of_vec_znx_dft(1, self_size);
        let vmp: usize = tmp_dft_col_data + module.vmp_apply_tmp_bytes(self_size, self_size, self_size, rank, rank, tsk_size);
        let tmp_idft: usize = module.bytes_of_vec_znx_big(1, tsk_size);
        let norm: usize = module.vec_znx_big_normalize_tmp_bytes();
        tmp_dft_i + ((tmp_dft_col_data + vmp) | (tmp_idft + norm))
    }

    pub(crate) fn keyswitch_internal_col0_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        ksk_k: usize,
        rank: usize,
    ) -> usize {
        GLWECiphertext::keyswitch_from_fourier_scratch_space(module, basek, out_k, rank, in_k, rank, ksk_k)
            + module.bytes_of_vec_znx_dft(rank + 1, div_ceil(basek, in_k))
    }

    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        ksk_k: usize,
        tsk_k: usize,
        rank: usize,
    ) -> usize {
        let out_size: usize = div_ceil(basek, out_k);

        let res_znx: usize = module.bytes_of_vec_znx(rank + 1, out_size);
        let ci_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        let ks: usize = GGSWCiphertext::keyswitch_internal_col0_scratch_space(module, basek, out_k, in_k, ksk_k, rank);
        let expand_rows: usize = GGSWCiphertext::expand_row_scratch_space(module, basek, out_k, tsk_k, rank);
        let res_dft: usize = module.bytes_of_vec_znx_dft(rank + 1, out_size);
        res_znx + ci_dft + (ks | expand_rows | res_dft)
    }

    pub fn keyswitch_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        ksk_k: usize,
        tsk_k: usize,
        rank: usize,
    ) -> usize {
        GGSWCiphertext::keyswitch_scratch_space(module, basek, out_k, out_k, ksk_k, tsk_k, rank)
    }

    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        atk_k: usize,
        tsk_k: usize,
        rank: usize,
    ) -> usize {
        let cols: usize = rank + 1;
        let out_size: usize = div_ceil(basek, out_k);
        let res: usize = module.bytes_of_vec_znx(cols, out_size);
        let res_dft: usize = module.bytes_of_vec_znx_dft(cols, out_size);
        let ci_dft: usize = module.bytes_of_vec_znx_dft(cols, out_size);
        let ks_internal: usize = GGSWCiphertext::keyswitch_internal_col0_scratch_space(module, basek, out_k, in_k, atk_k, rank);
        let expand: usize = GGSWCiphertext::expand_row_scratch_space(module, basek, out_k, tsk_k, rank);
        res + ci_dft + (ks_internal | expand | res_dft)
    }

    pub fn automorphism_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        atk_k: usize,
        tsk_k: usize,
        rank: usize,
    ) -> usize {
        GGSWCiphertext::automorphism_scratch_space(module, basek, out_k, out_k, atk_k, tsk_k, rank)
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        in_k: usize,
        ggsw_k: usize,
        rank: usize,
    ) -> usize {
        let tmp_in: usize = GLWECiphertextFourier::bytes_of(module, basek, in_k, rank);
        let tmp_out: usize = GLWECiphertextFourier::bytes_of(module, basek, out_k, rank);
        let ggsw: usize = GLWECiphertextFourier::external_product_scratch_space(module, basek, out_k, in_k, ggsw_k, rank);
        tmp_in + tmp_out + ggsw
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        basek: usize,
        out_k: usize,
        ggsw_k: usize,
        rank: usize,
    ) -> usize {
        let tmp: usize = GLWECiphertextFourier::bytes_of(module, basek, out_k, rank);
        let ggsw: usize = GLWECiphertextFourier::external_product_inplace_scratch_space(module, basek, out_k, ggsw_k, rank);
        tmp + ggsw
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> GGSWCiphertext<DataSelf, FFT64> {
    pub fn encrypt_sk<DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        pt: &ScalarZnx<DataPt>,
        sk: &GLWESecret<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk.n(), module.n());
        }

        let basek: usize = self.basek();
        let k: usize = self.k();
        let rank: usize = self.rank();
        let digits: usize = self.digits();

        let (mut tmp_pt, scratch1) = scratch.tmp_glwe_pt(module, basek, k);
        let (mut tmp_ct, scratch2) = scratch1.tmp_glwe_ct(module, basek, k, rank);

        (0..self.rows()).for_each(|row_i| {
            tmp_pt.data.zero();

            // Adds the scalar_znx_pt to the i-th limb of the vec_znx_pt
            module.vec_znx_add_scalar_inplace(&mut tmp_pt.data, 0, row_i * digits, pt, 0);
            module.vec_znx_normalize_inplace(basek, &mut tmp_pt.data, 0, scratch2);

            (0..rank + 1).for_each(|col_j| {
                // rlwe encrypt of vec_znx_pt into vec_znx_ct

                tmp_ct.encrypt_sk_private(
                    module,
                    Some((&tmp_pt, col_j)),
                    sk,
                    source_xa,
                    source_xe,
                    sigma,
                    scratch2,
                );

                // Switch vec_znx_ct into DFT domain
                {
                    let (mut tmp_ct_dft, _) = scratch2.tmp_glwe_fourier(module, basek, k, rank);
                    tmp_ct.dft(module, &mut tmp_ct_dft);
                    self.set_row(module, row_i, col_j, &tmp_ct_dft);
                }
            });
        });
    }

    pub(crate) fn expand_row<R, DataCi: AsRef<[u8]>, DataTsk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        col_j: usize,
        res: &mut R,
        ci_dft: &VecZnxDft<DataCi, FFT64>,
        tsk: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) where
        R: VecZnxToMut,
    {
        let cols: usize = self.rank() + 1;

        assert!(
            scratch.available() >= GGSWCiphertext::expand_row_scratch_space(module, self.basek(), self.k(), tsk.k(), self.rank())
        );

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
                        &tsk.at(col_i - 1, col_j - 1).0.data, // Selects Enc(s[i]s[j])
                        scratch2,
                    );
                } else {
                    module.vmp_apply_add(
                        &mut tmp_dft_i,
                        &tmp_dft_col_data,
                        &tsk.at(col_i - 1, col_j - 1).0.data, // Selects Enc(s[i]s[j])
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

    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataKsk: AsRef<[u8]>, DataTsk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        ksk: &GLWESwitchingKey<DataKsk, FFT64>,
        tsk: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) {
        let rank: usize = self.rank();
        let cols: usize = rank + 1;
        let basek: usize = self.basek();

        let (mut tmp_res, scratch1) = scratch.tmp_glwe_ct(module, basek, self.k(), rank);
        let (mut ci_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, lhs.size());

        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0s0' + a1s1' + a2s2') + M[i], a0, a1, a2)
            lhs.keyswitch_internal_col0(module, row_i, &mut tmp_res, ksk, scratch2);

            // Isolates DFT(a[i])
            (0..cols).for_each(|col_i| {
                module.vec_znx_dft(&mut ci_dft, col_i, &tmp_res.data, col_i);
            });

            module.vmp_prepare_row(&mut self.data, row_i, 0, &ci_dft);

            // Generates
            //
            // col 1: (-(b0s0' + b1s1' + b2s2')    , b0 + M[i], b1       , b2       )
            // col 2: (-(c0s0' + c1s1' + c2s2')    , c0       , c1 + M[i], c2       )
            // col 3: (-(d0s0' + d1s1' + d2s2')    , d0       , d1       , d2 + M[i])
            (1..cols).for_each(|col_j| {
                self.expand_row(module, col_j, &mut tmp_res.data, &ci_dft, tsk, scratch2);
                let (mut tmp_res_dft, _) = scratch2.tmp_glwe_fourier(module, basek, self.k(), rank);
                tmp_res.dft(module, &mut tmp_res_dft);
                self.set_row(module, row_i, col_j, &tmp_res_dft);
            });
        })
    }

    pub fn keyswitch_inplace<DataKsk: AsRef<[u8]>, DataTsk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        ksk: &GLWESwitchingKey<DataKsk, FFT64>,
        tsk: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf, FFT64> = self as *mut GGSWCiphertext<DataSelf, FFT64>;
            self.keyswitch(module, &*self_ptr, ksk, tsk, scratch);
        }
    }

    pub fn automorphism<DataLhs: AsRef<[u8]>, DataAk: AsRef<[u8]>, DataTsk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        auto_key: &AutomorphismKey<DataAk, FFT64>,
        tensor_key: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) {
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
            assert!(
                scratch.available()
                    >= GGSWCiphertext::automorphism_scratch_space(
                        module,
                        self.basek(),
                        self.k(),
                        lhs.k(),
                        auto_key.k(),
                        tensor_key.k(),
                        self.rank()
                    )
            )
        };

        let rank: usize = self.rank();
        let cols: usize = rank + 1;
        let basek: usize = self.basek();

        let (mut tmp_res, scratch1) = scratch.tmp_glwe_ct(module, basek, self.k(), rank);
        let (mut ci_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, lhs.size());

        // Keyswitch the j-th row of the col 0
        (0..lhs.rows()).for_each(|row_i| {
            // Key-switch column 0, i.e.
            // col 0: (-(a0s0 + a1s1 + a2s2) + M[i], a0, a1, a2) -> (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2)
            lhs.keyswitch_internal_col0(module, row_i, &mut tmp_res, &auto_key.key, scratch2);

            // Isolates DFT(AUTO(a[i]))
            (0..cols).for_each(|col_i| {
                // (-(a0pi^-1(s0) + a1pi^-1(s1) + a2pi^-1(s2)) + M[i], a0, a1, a2) -> (-(a0s0 + a1s1 + a2s2) + pi(M[i]), a0, a1, a2)
                module.vec_znx_automorphism_inplace(auto_key.p(), &mut tmp_res.data, col_i);
                module.vec_znx_dft(&mut ci_dft, col_i, &tmp_res.data, col_i);
            });

            module.vmp_prepare_row(&mut self.data, row_i, 0, &ci_dft);

            // Generates
            //
            // col 1: (-(b0s0 + b1s1 + b2s2)    , b0 + pi(M[i]), b1           , b2           )
            // col 2: (-(c0s0 + c1s1 + c2s2)    , c0           , c1 + pi(M[i]), c2           )
            // col 3: (-(d0s0 + d1s1 + d2s2)    , d0           , d1           , d2 + pi(M[i]))
            (1..cols).for_each(|col_j| {
                self.expand_row(
                    module,
                    col_j,
                    &mut tmp_res.data,
                    &ci_dft,
                    tensor_key,
                    scratch2,
                );
                let (mut tmp_res_dft, _) = scratch2.tmp_glwe_fourier(module, basek, self.k(), rank);
                tmp_res.dft(module, &mut tmp_res_dft);
                self.set_row(module, row_i, col_j, &tmp_res_dft);
            });
        })
    }

    pub fn automorphism_inplace<DataKsk: AsRef<[u8]>, DataTsk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        auto_key: &AutomorphismKey<DataKsk, FFT64>,
        tensor_key: &TensorKey<DataTsk, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GGSWCiphertext<DataSelf, FFT64> = self as *mut GGSWCiphertext<DataSelf, FFT64>;
            self.automorphism(module, &*self_ptr, auto_key, tensor_key, scratch);
        }
    }

    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GGSWCiphertext<DataLhs, FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
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

        let (mut tmp_ct_in, scratch1) = scratch.tmp_glwe_fourier(module, lhs.basek(), lhs.k(), lhs.rank());
        let (mut tmp_ct_out, scratch2) = scratch1.tmp_glwe_fourier(module, self.basek(), self.k(), self.rank());

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                lhs.get_row(module, row_j, col_i, &mut tmp_ct_in);
                tmp_ct_out.external_product(module, &tmp_ct_in, rhs, scratch2);
                self.set_row(module, row_j, col_i, &tmp_ct_out);
            });
        });

        tmp_ct_out.data.zero();

        (self.rows().min(lhs.rows())..self.rows()).for_each(|row_i| {
            (0..self.rank() + 1).for_each(|col_j| {
                self.set_row(module, row_i, col_j, &tmp_ct_out);
            });
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
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

        let (mut tmp_ct, scratch1) = scratch.tmp_glwe_fourier(module, self.basek(), self.k(), self.rank());

        (0..self.rank() + 1).for_each(|col_i| {
            (0..self.rows()).for_each(|row_j| {
                self.get_row(module, row_j, col_i, &mut tmp_ct);
                tmp_ct.external_product_inplace(module, rhs, scratch1);
                self.set_row(module, row_j, col_i, &tmp_ct);
            });
        });
    }
}

impl<DataSelf: AsRef<[u8]>> GGSWCiphertext<DataSelf, FFT64> {
    pub(crate) fn keyswitch_internal_col0<DataRes: AsMut<[u8]> + AsRef<[u8]>, DataKsk: AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        res: &mut GLWECiphertext<DataRes>,
        ksk: &GLWESwitchingKey<DataKsk, FFT64>,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), ksk.rank());
            assert_eq!(res.rank(), ksk.rank());
            assert!(
                scratch.available()
                    >= GGSWCiphertext::keyswitch_internal_col0_scratch_space(
                        module,
                        self.basek(),
                        res.k(),
                        self.k(),
                        ksk.k(),
                        ksk.rank()
                    )
            )
        }
        let (mut tmp_dft_dft, scratch1) = scratch.tmp_glwe_fourier(module, self.basek(), self.k(), self.rank());
        self.get_row(module, row_i, 0, &mut tmp_dft_dft);
        res.keyswitch_from_fourier(module, &tmp_dft_dft, ksk, scratch1);
    }
}

impl<DataSelf: AsRef<[u8]>> GetRow<FFT64> for GGSWCiphertext<DataSelf, FFT64> {
    fn get_row<R: AsMut<[u8]> + AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        res: &mut GLWECiphertextFourier<R, FFT64>,
    ) {
        module.vmp_extract_row(&mut res.data, &self.data, row_i, col_j);
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetRow<FFT64> for GGSWCiphertext<DataSelf, FFT64> {
    fn set_row<R: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        row_i: usize,
        col_j: usize,
        a: &GLWECiphertextFourier<R, FFT64>,
    ) {
        module.vmp_prepare_row(&mut self.data, row_i, col_j, &a.data);
    }
}
