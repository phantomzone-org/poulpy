use backend::{
    AddNormal, Backend, FFT64, FillUniform, MatZnxDftOps, MatZnxDftScratch, Module, ScalarZnxAlloc, ScalarZnxDftAlloc,
    ScalarZnxDftOps, Scratch, VecZnx, VecZnxAlloc, VecZnxBig, VecZnxBigAlloc, VecZnxBigOps, VecZnxBigScratch, VecZnxDftAlloc,
    VecZnxDftOps, VecZnxOps, VecZnxToMut, VecZnxToRef, ZnxZero,
};
use sampling::source::Source;

use crate::{
    SIX_SIGMA,
    automorphism::AutomorphismKey,
    elem::{Infos, SetMetaData},
    ggsw_ciphertext::GGSWCiphertext,
    glwe_ciphertext_fourier::GLWECiphertextFourier,
    glwe_ops::GLWEOps,
    glwe_plaintext::GLWEPlaintext,
    keys::{GLWEPublicKey, SecretDistribution, SecretKeyFourier},
    keyswitch_key::GLWESwitchingKey,
    utils::derive_size,
};

pub struct GLWECiphertext<C> {
    pub data: VecZnx<C>,
    pub basek: usize,
    pub k: usize,
}

impl GLWECiphertext<Vec<u8>> {
    pub fn alloc<B: Backend>(module: &Module<B>, basek: usize, k: usize, rank: usize) -> Self {
        Self {
            data: module.new_vec_znx(rank + 1, derive_size(basek, k)),
            basek,
            k,
        }
    }

    pub fn bytes_of(module: &Module<FFT64>, basek: usize, k: usize, rank: usize) -> usize {
        module.bytes_of_vec_znx(rank + 1, derive_size(basek, k))
    }
}

impl<T> Infos for GLWECiphertext<T> {
    type Inner = VecZnx<T>;

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

impl<T> GLWECiphertext<T> {
    pub fn rank(&self) -> usize {
        self.cols() - 1
    }
}

impl<C: AsRef<[u8]>> GLWECiphertext<C> {
    #[allow(dead_code)]
    pub(crate) fn dft<R: AsMut<[u8]> + AsRef<[u8]>>(&self, module: &Module<FFT64>, res: &mut GLWECiphertextFourier<R, FFT64>) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), res.rank());
            assert_eq!(self.basek(), res.basek())
        }

        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_dft(&mut res.data, i, &self.data, i);
        })
    }
}

impl GLWECiphertext<Vec<u8>> {
    pub fn encrypt_sk_scratch_space(module: &Module<FFT64>, ct_size: usize) -> usize {
        module.vec_znx_big_normalize_tmp_bytes() + module.bytes_of_vec_znx_dft(1, ct_size) + module.bytes_of_vec_znx(1, ct_size)
    }
    pub fn encrypt_pk_scratch_space(module: &Module<FFT64>, pk_size: usize) -> usize {
        ((module.bytes_of_vec_znx_dft(1, pk_size) + module.bytes_of_vec_znx_big(1, pk_size)) | module.bytes_of_scalar_znx(1))
            + module.bytes_of_scalar_znx_dft(1)
            + module.vec_znx_big_normalize_tmp_bytes()
    }

    pub fn decrypt_scratch_space(module: &Module<FFT64>, ct_size: usize) -> usize {
        (module.vec_znx_big_normalize_tmp_bytes() | module.bytes_of_vec_znx_dft(1, ct_size))
            + module.bytes_of_vec_znx_big(1, ct_size)
    }

    pub fn keyswitch_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        in_size: usize,
        in_rank: usize,
        ksk_size: usize,
    ) -> usize {
        let res_dft: usize = module.bytes_of_vec_znx_dft(out_rank + 1, ksk_size);
        let vmp: usize = module.vmp_apply_tmp_bytes(out_size, in_size, in_size, in_rank, out_rank + 1, ksk_size)
            + module.bytes_of_vec_znx_dft(in_rank, in_size);
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes();

        return res_dft + (vmp | normalize);
    }

    pub fn keyswitch_from_fourier_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        in_size: usize,
        in_rank: usize,
        ksk_size: usize,
    ) -> usize {
        let res_dft = module.bytes_of_vec_znx_dft(out_rank + 1, ksk_size);

        let vmp: usize = module.vmp_apply_tmp_bytes(out_size, in_size, in_size, in_rank, out_rank + 1, ksk_size)
            + module.bytes_of_vec_znx_dft(in_rank, in_size);

        let norm: usize = module.vec_znx_big_normalize_tmp_bytes();

        res_dft + (vmp | norm)
    }

    pub fn keyswitch_inplace_scratch_space(module: &Module<FFT64>, out_size: usize, out_rank: usize, ksk_size: usize) -> usize {
        GLWECiphertext::keyswitch_scratch_space(module, out_size, out_rank, out_size, out_rank, ksk_size)
    }

    pub fn automorphism_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        in_size: usize,
        autokey_size: usize,
    ) -> usize {
        GLWECiphertext::keyswitch_scratch_space(module, out_size, out_rank, in_size, out_rank, autokey_size)
    }

    pub fn automorphism_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        autokey_size: usize,
    ) -> usize {
        GLWECiphertext::keyswitch_scratch_space(module, out_size, out_rank, out_size, out_rank, autokey_size)
    }

    pub fn external_product_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        in_size: usize,
        ggsw_size: usize,
    ) -> usize {
        let res_dft: usize = module.bytes_of_vec_znx_dft(out_rank + 1, ggsw_size);
        let vmp: usize = module.bytes_of_vec_znx_dft(out_rank + 1, in_size)
            + module.vmp_apply_tmp_bytes(
                out_size,
                in_size,
                in_size,      // rows
                out_rank + 1, // cols in
                out_rank + 1, // cols out
                ggsw_size,
            );
        let normalize: usize = module.vec_znx_big_normalize_tmp_bytes();

        res_dft + (vmp | normalize)
    }

    pub fn external_product_inplace_scratch_space(
        module: &Module<FFT64>,
        out_size: usize,
        out_rank: usize,
        ggsw_size: usize,
    ) -> usize {
        GLWECiphertext::external_product_scratch_space(module, out_size, out_rank, out_size, ggsw_size)
    }
}

impl<DataSelf: AsMut<[u8]> + AsRef<[u8]>> SetMetaData for GLWECiphertext<DataSelf> {
    fn set_k(&mut self, k: usize) {
        self.k = k
    }

    fn set_basek(&mut self, basek: usize) {
        self.basek = basek
    }
}

impl<DataSelf: AsRef<[u8]> + AsMut<[u8]>> GLWECiphertext<DataSelf> {
    pub fn encrypt_sk<DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        pt: &GLWEPlaintext<DataPt>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        self.encrypt_sk_private(
            module,
            Some((pt, 0)),
            sk_dft,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_zero_sk<DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        self.encrypt_sk_private(
            module,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            sk_dft,
            source_xa,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_pk<DataPt: AsRef<[u8]>, DataPk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        pt: &GLWEPlaintext<DataPt>,
        pk: &GLWEPublicKey<DataPk, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        self.encrypt_pk_private(
            module,
            Some((pt, 0)),
            pk,
            source_xu,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn encrypt_zero_pk<DataPk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        pk: &GLWEPublicKey<DataPk, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        self.encrypt_pk_private(
            module,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            pk,
            source_xu,
            source_xe,
            sigma,
            scratch,
        );
    }

    pub fn automorphism<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        self.keyswitch(module, lhs, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i);
        })
    }

    pub fn automorphism_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        self.keyswitch_inplace(module, &rhs.key, scratch);
        (0..self.rank() + 1).for_each(|i| {
            module.vec_znx_automorphism_inplace(rhs.p(), &mut self.data, i);
        })
    }

    pub fn automorphism_add<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        Self::keyswitch_private::<_, _, 1>(self, rhs.p(), module, lhs, &rhs.key, scratch);
    }

    pub fn automorphism_add_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            Self::keyswitch_private::<_, _, 1>(self, rhs.p(), module, &*self_ptr, &rhs.key, scratch);
        }
    }

    pub fn automorphism_sub_ab<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        Self::keyswitch_private::<_, _, 2>(self, rhs.p(), module, lhs, &rhs.key, scratch);
    }

    pub fn automorphism_sub_ab_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            Self::keyswitch_private::<_, _, 2>(self, rhs.p(), module, &*self_ptr, &rhs.key, scratch);
        }
    }

    pub fn automorphism_sub_ba<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        Self::keyswitch_private::<_, _, 3>(self, rhs.p(), module, lhs, &rhs.key, scratch);
    }

    pub fn automorphism_sub_ba_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &AutomorphismKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            Self::keyswitch_private::<_, _, 3>(self, rhs.p(), module, &*self_ptr, &rhs.key, scratch);
        }
    }

    pub(crate) fn keyswitch_from_fourier<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertextFourier<DataLhs, FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(lhs.rank(), rhs.rank_in());
            assert_eq!(self.rank(), rhs.rank_out());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::keyswitch_from_fourier_scratch_space(
                        module,
                        self.size(),
                        self.rank(),
                        lhs.size(),
                        lhs.rank(),
                        rhs.size(),
                    )
            );
        }

        let cols_in: usize = rhs.rank_in();
        let cols_out: usize = rhs.rank_out() + 1;

        // Buffer of the result of VMP in DFT
        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols_out, rhs.size()); // Todo optimise

        {
            // Applies VMP
            let (mut ai_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols_in, lhs.size());
            (0..cols_in).for_each(|col_i| {
                module.vec_znx_dft_copy(&mut ai_dft, col_i, &lhs.data, col_i + 1);
            });
            module.vmp_apply(&mut res_dft, &ai_dft, &rhs.0.data, scratch2);
        }

        module.vec_znx_dft_add_inplace(&mut res_dft, 0, &lhs.data, 0);

        // Switches result of VMP outside of DFT
        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume::<&mut [u8]>(res_dft);

        (0..cols_out).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch1);
        });
    }

    pub fn keyswitch<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        Self::keyswitch_private::<_, _, 0>(self, 0, module, lhs, rhs, scratch);
    }

    pub(crate) fn keyswitch_private<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>, const OP: u8>(
        &mut self,
        apply_auto: i64,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(lhs.rank(), rhs.rank_in());
            assert_eq!(self.rank(), rhs.rank_out());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
            assert!(
                scratch.available()
                    >= GLWECiphertext::keyswitch_scratch_space(
                        module,
                        self.size(),
                        self.rank(),
                        lhs.size(),
                        lhs.rank(),
                        rhs.size(),
                    )
            );
        }

        let cols_in: usize = rhs.rank_in();
        let cols_out: usize = rhs.rank_out() + 1;

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols_out, rhs.size()); // Todo optimise

        {
            let (mut ai_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols_in, lhs.size());
            (0..cols_in).for_each(|col_i| {
                module.vec_znx_dft(&mut ai_dft, col_i, &lhs.data, col_i + 1);
            });
            module.vmp_apply(&mut res_dft, &ai_dft, &rhs.0.data, scratch2);
        }

        let mut res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        module.vec_znx_big_add_small_inplace(&mut res_big, 0, &lhs.data, 0);

        (0..cols_out).for_each(|i| {
            if apply_auto != 0 {
                module.vec_znx_big_automorphism_inplace(apply_auto, &mut res_big, i);
            }

            match OP {
                1 => module.vec_znx_big_add_small_inplace(&mut res_big, i, &lhs.data, i),
                2 => module.vec_znx_big_sub_small_a_inplace(&mut res_big, i, &lhs.data, i),
                3 => module.vec_znx_big_sub_small_b_inplace(&mut res_big, i, &lhs.data, i),
                _ => {}
            }
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch1);
        });
    }

    pub fn keyswitch_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GLWESwitchingKey<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.keyswitch(&module, &*self_ptr, rhs, scratch);
        }
    }

    pub fn external_product<DataLhs: AsRef<[u8]>, DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        lhs: &GLWECiphertext<DataLhs>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        let basek: usize = self.basek();

        #[cfg(debug_assertions)]
        {
            assert_eq!(rhs.rank(), lhs.rank());
            assert_eq!(rhs.rank(), self.rank());
            assert_eq!(self.basek(), basek);
            assert_eq!(lhs.basek(), basek);
            assert_eq!(rhs.n(), module.n());
            assert_eq!(self.n(), module.n());
            assert_eq!(lhs.n(), module.n());
        }

        let cols: usize = rhs.rank() + 1;

        let (mut res_dft, scratch1) = scratch.tmp_vec_znx_dft(module, cols, rhs.size()); // Todo optimise

        {
            let (mut a_dft, scratch2) = scratch1.tmp_vec_znx_dft(module, cols, lhs.size());
            (0..cols).for_each(|col_i| {
                module.vec_znx_dft(&mut a_dft, col_i, &lhs.data, col_i);
            });
            module.vmp_apply(&mut res_dft, &a_dft, &rhs.data, scratch2);
        }

        let res_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(res_dft);

        (0..cols).for_each(|i| {
            module.vec_znx_big_normalize(basek, &mut self.data, i, &res_big, i, scratch1);
        });
    }

    pub fn external_product_inplace<DataRhs: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        rhs: &GGSWCiphertext<DataRhs, FFT64>,
        scratch: &mut Scratch,
    ) {
        unsafe {
            let self_ptr: *mut GLWECiphertext<DataSelf> = self as *mut GLWECiphertext<DataSelf>;
            self.external_product(&module, &*self_ptr, rhs, scratch);
        }
    }

    pub(crate) fn encrypt_sk_private<DataPt: AsRef<[u8]>, DataSk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(sk_dft.n(), module.n());
            assert_eq!(self.n(), module.n());
            if let Some((pt, col)) = pt {
                assert_eq!(pt.n(), module.n());
                assert!(col < self.rank() + 1);
            }
            assert!(
                scratch.available() >= GLWECiphertext::encrypt_sk_scratch_space(module, self.size()),
                "scratch.available(): {} < GLWECiphertext::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GLWECiphertext::encrypt_sk_scratch_space(module, self.size())
            )
        }

        let basek: usize = self.basek();
        let k: usize = self.k();
        let size: usize = self.size();
        let cols: usize = self.rank() + 1;

        let (mut c0_big, scratch_1) = scratch.tmp_vec_znx(module, 1, size);
        c0_big.zero();

        {
            // c[i] = uniform
            // c[0] -= c[i] * s[i],
            (1..cols).for_each(|i| {
                let (mut ci_dft, scratch_2) = scratch_1.tmp_vec_znx_dft(module, 1, size);

                // c[i] = uniform
                self.data.fill_uniform(basek, i, size, source_xa);

                // c[i] = norm(IDFT(DFT(c[i]) * DFT(s[i])))
                module.vec_znx_dft(&mut ci_dft, 0, &self.data, i);
                module.svp_apply_inplace(&mut ci_dft, 0, &sk_dft.data, i - 1);
                let ci_big: VecZnxBig<&mut [u8], FFT64> = module.vec_znx_idft_consume(ci_dft);

                // use c[0] as buffer, which is overwritten later by the normalization step
                module.vec_znx_big_normalize(basek, &mut self.data, 0, &ci_big, 0, scratch_2);

                // c0_tmp = -c[i] * s[i] (use c[0] as buffer)
                module.vec_znx_sub_ab_inplace(&mut c0_big, 0, &self.data, 0);

                // c[i] += m if col = i
                if let Some((pt, col)) = pt {
                    if i == col {
                        module.vec_znx_add_inplace(&mut self.data, i, &pt.data, 0);
                        module.vec_znx_normalize_inplace(basek, &mut self.data, i, scratch_2);
                    }
                }
            });
        }

        // c[0] += e
        c0_big.add_normal(basek, 0, k, source_xe, sigma, sigma * SIX_SIGMA);

        // c[0] += m if col = 0
        if let Some((pt, col)) = pt {
            if col == 0 {
                module.vec_znx_add_inplace(&mut c0_big, 0, &pt.data, 0);
            }
        }

        // c[0] = norm(c[0])
        module.vec_znx_normalize(basek, &mut self.data, 0, &c0_big, 0, scratch_1);
    }

    pub(crate) fn encrypt_pk_private<DataPt: AsRef<[u8]>, DataPk: AsRef<[u8]>>(
        &mut self,
        module: &Module<FFT64>,
        pt: Option<(&GLWEPlaintext<DataPt>, usize)>,
        pk: &GLWEPublicKey<DataPk, FFT64>,
        source_xu: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.basek(), pk.basek());
            assert_eq!(self.n(), module.n());
            assert_eq!(pk.n(), module.n());
            assert_eq!(self.rank(), pk.rank());
            if let Some((pt, _)) = pt {
                assert_eq!(pt.basek(), pk.basek());
                assert_eq!(pt.n(), module.n());
            }
        }

        let basek: usize = pk.basek();
        let size_pk: usize = pk.size();
        let cols: usize = self.rank() + 1;

        // Generates u according to the underlying secret distribution.
        let (mut u_dft, scratch_1) = scratch.tmp_scalar_znx_dft(module, 1);

        {
            let (mut u, _) = scratch_1.tmp_scalar_znx(module, 1);
            match pk.dist {
                SecretDistribution::NONE => panic!(
                    "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through \
                     Self::generate"
                ),
                SecretDistribution::TernaryFixed(hw) => u.fill_ternary_hw(0, hw, source_xu),
                SecretDistribution::TernaryProb(prob) => u.fill_ternary_prob(0, prob, source_xu),
                SecretDistribution::ZERO => {}
            }

            module.svp_prepare(&mut u_dft, 0, &u, 0);
        }

        // ct[i] = pk[i] * u + ei (+ m if col = i)
        (0..cols).for_each(|i| {
            let (mut ci_dft, scratch_2) = scratch_1.tmp_vec_znx_dft(module, 1, size_pk);
            // ci_dft = DFT(u) * DFT(pk[i])
            module.svp_apply(&mut ci_dft, 0, &u_dft, 0, &pk.data.data, i);

            // ci_big = u * p[i]
            let mut ci_big = module.vec_znx_idft_consume(ci_dft);

            // ci_big = u * pk[i] + e
            ci_big.add_normal(basek, 0, pk.k(), source_xe, sigma, sigma * SIX_SIGMA);

            // ci_big = u * pk[i] + e + m (if col = i)
            if let Some((pt, col)) = pt {
                if col == i {
                    module.vec_znx_big_add_small_inplace(&mut ci_big, 0, &pt.data, 0);
                }
            }

            // ct[i] = norm(ci_big)
            module.vec_znx_big_normalize(basek, &mut self.data, i, &ci_big, 0, scratch_2);
        });
    }
}

impl<DataSelf: AsRef<[u8]>> GLWECiphertext<DataSelf> {
    pub fn clone(&self) -> GLWECiphertext<Vec<u8>> {
        GLWECiphertext {
            data: self.data.clone(),
            basek: self.basek(),
            k: self.k(),
        }
    }

    pub fn decrypt<DataPt: AsMut<[u8]> + AsRef<[u8]>, DataSk: AsRef<[u8]>>(
        &self,
        module: &Module<FFT64>,
        pt: &mut GLWEPlaintext<DataPt>,
        sk_dft: &SecretKeyFourier<DataSk, FFT64>,
        scratch: &mut Scratch,
    ) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.rank(), sk_dft.rank());
            assert_eq!(self.n(), module.n());
            assert_eq!(pt.n(), module.n());
            assert_eq!(sk_dft.n(), module.n());
        }

        let cols: usize = self.rank() + 1;

        let (mut c0_big, scratch_1) = scratch.tmp_vec_znx_big(module, 1, self.size()); // TODO optimize size when pt << ct
        c0_big.zero();

        {
            (1..cols).for_each(|i| {
                // ci_dft = DFT(a[i]) * DFT(s[i])
                let (mut ci_dft, _) = scratch_1.tmp_vec_znx_dft(module, 1, self.size()); // TODO optimize size when pt << ct
                module.vec_znx_dft(&mut ci_dft, 0, &self.data, i);
                module.svp_apply_inplace(&mut ci_dft, 0, &sk_dft.data, i - 1);
                let ci_big = module.vec_znx_idft_consume(ci_dft);

                // c0_big += a[i] * s[i]
                module.vec_znx_big_add_inplace(&mut c0_big, 0, &ci_big, 0);
            });
        }

        // c0_big = (a * s) + (-a * s + m + e) = BIG(m + e)
        module.vec_znx_big_add_small_inplace(&mut c0_big, 0, &self.data, 0);

        // pt = norm(BIG(m + e))
        module.vec_znx_big_normalize(self.basek(), &mut pt.data, 0, &mut c0_big, 0, scratch_1);

        pt.basek = self.basek();
        pt.k = pt.k().min(self.k());
    }
}

pub trait GLWECiphertextToRef {
    fn to_ref(&self) -> GLWECiphertext<&[u8]>;
}

impl<D: AsRef<[u8]>> GLWECiphertextToRef for GLWECiphertext<D> {
    fn to_ref(&self) -> GLWECiphertext<&[u8]> {
        GLWECiphertext {
            data: self.data.to_ref(),
            basek: self.basek,
            k: self.k,
        }
    }
}

pub trait GLWECiphertextToMut {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]>;
}

impl<D: AsMut<[u8]> + AsRef<[u8]>> GLWECiphertextToMut for GLWECiphertext<D> {
    fn to_mut(&mut self) -> GLWECiphertext<&mut [u8]> {
        GLWECiphertext {
            data: self.data.to_mut(),
            basek: self.basek,
            k: self.k,
        }
    }
}

impl<D> GLWEOps for GLWECiphertext<D>
where
    D: AsRef<[u8]> + AsMut<[u8]>,
    GLWECiphertext<D>: GLWECiphertextToMut + Infos + SetMetaData,
{
}
