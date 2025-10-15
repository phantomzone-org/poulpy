use poulpy_hal::{
    api::{
        ScratchAvailable, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace,
        VecZnxAddNormal, VecZnxBigAddNormal, VecZnxBigAddSmallInplace, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxFillUniform, VecZnxIdftApplyConsume, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxSub, VecZnxSubInplace,
    },
    layouts::{Backend, DataMut, Module, ScalarZnx, Scratch, VecZnx, VecZnxBig, VecZnxToMut, ZnxInfos, ZnxZero},
    source::Source,
};

use crate::{
    dist::Distribution,
    encryption::{SIGMA, SIGMA_BOUND},
    layouts::{
        GLWE, GLWEInfos, GLWEPlaintext, GLWEPlaintextToRef, GLWEToMut, LWEInfos,
        prepared::{GLWEPublicKeyPrepared, GLWEPublicKeyPreparedToRef, GLWESecretPrepared, GLWESecretPreparedToRef},
    },
};

impl GLWE<Vec<u8>> {
    pub fn encrypt_sk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GLWEInfos,
        Module<B>: VecZnxNormalizeTmpBytes + VecZnxDftBytesOf,
    {
        let size: usize = infos.size();
        assert_eq!(module.n() as u32, infos.n());
        module.vec_znx_normalize_tmp_bytes() + 2 * VecZnx::bytes_of(module.n(), 1, size) + module.bytes_of_vec_znx_dft(1, size)
    }
    pub fn encrypt_pk_scratch_space<B: Backend, A>(module: &Module<B>, infos: &A) -> usize
    where
        A: GLWEInfos,
        Module<B>: VecZnxDftBytesOf + SvpPPolBytesOf + VecZnxBigBytesOf + VecZnxNormalizeTmpBytes,
    {
        let size: usize = infos.size();
        assert_eq!(module.n() as u32, infos.n());
        ((module.bytes_of_vec_znx_dft(1, size) + module.bytes_of_vec_znx_big(1, size)) | ScalarZnx::bytes_of(module.n(), 1))
            + module.bytes_of_svp_ppol(1)
            + module.vec_znx_normalize_tmp_bytes()
    }
}

impl<D: DataMut> GLWE<D> {
    pub fn encrypt_sk<R, P, S, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<B>,
        Module<B>: GLWEEncryptSk<B>,
    {
        module.glwe_encrypt_sk(self, pt, sk, source_xa, source_xe, scratch);
    }

    pub fn encrypt_zero_sk<S, B: Backend>(
        &mut self,
        module: &Module<B>,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        S: GLWESecretPreparedToRef<B>,
        Module<B>: GLWEEncryptZeroSk<B>,
    {
        module.glwe_encrypt_zero_sk(self, sk, source_xa, source_xe, scratch);
    }

    pub fn encrypt_pk<P, K, B: Backend>(
        &mut self,
        module: &Module<B>,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        P: GLWEPlaintextToRef,
        K: GLWEPublicKeyPreparedToRef<B>,
        Module<B>: GLWEEncryptPk<B>,
    {
        module.glwe_encrypt_pk(self, pt, pk, source_xu, source_xe, scratch);
    }

    pub fn encrypt_zero_pk<K, B: Backend>(
        &mut self,
        module: &Module<B>,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        K: GLWEPublicKeyPreparedToRef<B>,
        Module<B>: GLWEEncryptZeroPk<B>,
    {
        module.glwe_encrypt_zero_pk(self, pk, source_xu, source_xe, scratch);
    }
}

pub trait GLWEEncryptSk<B: Backend> {
    fn glwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<B>;
}

impl<B: Backend> GLWEEncryptSk<B> for Module<B>
where
    Module<B>: GLWEEncryptSkInternal<B> + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf,
    Scratch<B>: ScratchAvailable,
{
    fn glwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<B>,
    {
        let mut res: GLWE<&mut [u8]> = res.to_mut();
        let pt: GLWEPlaintext<&[u8]> = pt.to_ref();

        #[cfg(debug_assertions)]
        {
            let sk: GLWESecretPrepared<&[u8], B> = sk.to_ref();
            assert_eq!(res.rank(), sk.rank());
            assert_eq!(res.n(), self.n() as u32);
            assert_eq!(sk.n(), self.n() as u32);
            assert_eq!(pt.n(), self.n() as u32);
            assert!(
                scratch.available() >= GLWE::encrypt_sk_scratch_space(self, &res),
                "scratch.available(): {} < GLWECiphertext::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GLWE::encrypt_sk_scratch_space(self, &res)
            )
        }

        let cols: usize = (res.rank() + 1).into();
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            res.k().into(),
            res.data_mut(),
            cols,
            false,
            Some((&pt, 0)),
            sk,
            source_xa,
            source_xe,
            SIGMA,
            scratch,
        );
    }
}

pub trait GLWEEncryptZeroSk<B: Backend> {
    fn glwe_encrypt_zero_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        S: GLWESecretPreparedToRef<B>;
}

impl<B: Backend> GLWEEncryptZeroSk<B> for Module<B>
where
    Module<B>: GLWEEncryptSkInternal<B> + VecZnxNormalizeTmpBytes + VecZnxDftBytesOf,
    Scratch<B>: ScratchAvailable,
{
    fn glwe_encrypt_zero_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        S: GLWESecretPreparedToRef<B>,
    {
        let mut res: GLWE<&mut [u8]> = res.to_mut();

        #[cfg(debug_assertions)]
        {
            let sk: GLWESecretPrepared<&[u8], B> = sk.to_ref();
            assert_eq!(res.rank(), sk.rank());
            assert_eq!(res.n(), self.n() as u32);
            assert_eq!(sk.n(), self.n() as u32);
            assert!(
                scratch.available() >= GLWE::encrypt_sk_scratch_space(self, &res),
                "scratch.available(): {} < GLWECiphertext::encrypt_sk_scratch_space: {}",
                scratch.available(),
                GLWE::encrypt_sk_scratch_space(self, &res)
            )
        }

        let cols: usize = (res.rank() + 1).into();
        self.glwe_encrypt_sk_internal(
            res.base2k().into(),
            res.k().into(),
            res.data_mut(),
            cols,
            false,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            sk,
            source_xa,
            source_xe,
            SIGMA,
            scratch,
        );
    }
}

pub trait GLWEEncryptPk<B: Backend> {
    fn glwe_encrypt_pk<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        K: GLWEPublicKeyPreparedToRef<B>;
}

impl<B: Backend> GLWEEncryptPk<B> for Module<B>
where
    Module<B>: GLWEEncryptPkInternal<B>,
{
    fn glwe_encrypt_pk<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        K: GLWEPublicKeyPreparedToRef<B>,
    {
        self.glwe_encrypt_pk_internal(res, Some((pt, 0)), pk, source_xu, source_xe, scratch);
    }
}

pub trait GLWEEncryptZeroPk<B: Backend> {
    fn glwe_encrypt_zero_pk<R, K>(
        &self,
        res: &mut R,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        K: GLWEPublicKeyPreparedToRef<B>;
}

impl<B: Backend> GLWEEncryptZeroPk<B> for Module<B>
where
    Module<B>: GLWEEncryptPkInternal<B>,
{
    fn glwe_encrypt_zero_pk<R, K>(
        &self,
        res: &mut R,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        K: GLWEPublicKeyPreparedToRef<B>,
    {
        self.glwe_encrypt_pk_internal(
            res,
            None::<(&GLWEPlaintext<Vec<u8>>, usize)>,
            pk,
            source_xu,
            source_xe,
            scratch,
        );
    }
}

pub(crate) trait GLWEEncryptPkInternal<B: Backend> {
    fn glwe_encrypt_pk_internal<R, P, K>(
        &self,
        res: &mut R,
        pt: Option<(&P, usize)>,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        K: GLWEPublicKeyPreparedToRef<B>;
}

impl<B: Backend> GLWEEncryptPkInternal<B> for Module<B>
where
    Module<B>: SvpPrepare<B>
        + SvpApplyDftToDft<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxBigAddNormal<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigNormalize<B>,
    Scratch<B>:,
{
    fn glwe_encrypt_pk_internal<R, P, K>(
        &self,
        res: &mut R,
        pt: Option<(&P, usize)>,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<B>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        K: GLWEPublicKeyPreparedToRef<B>,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let pk: &GLWEPublicKeyPrepared<&[u8], B> = &pk.to_ref();

        #[cfg(debug_assertions)]
        {
            assert_eq!(res.base2k(), pk.base2k());
            assert_eq!(res.n(), pk.n());
            assert_eq!(res.rank(), pk.rank());
            if let Some((pt, _)) = pt {
                assert_eq!(pt.to_ref().base2k(), pk.base2k());
                assert_eq!(pt.to_ref().n(), pk.n());
            }
        }

        let base2k: usize = pk.base2k().into();
        let size_pk: usize = pk.size();
        let cols: usize = (res.rank() + 1).into();

        // Generates u according to the underlying secret distribution.
        let (mut u_dft, scratch_1) = scratch.take_svp_ppol(res.n().into(), 1);

        {
            let (mut u, _) = scratch_1.take_scalar_znx(res.n().into(), 1);
            match pk.dist {
                Distribution::NONE => panic!(
                    "invalid public key: SecretDistribution::NONE, ensure it has been correctly intialized through \
                     Self::generate"
                ),
                Distribution::TernaryFixed(hw) => u.fill_ternary_hw(0, hw, source_xu),
                Distribution::TernaryProb(prob) => u.fill_ternary_prob(0, prob, source_xu),
                Distribution::BinaryFixed(hw) => u.fill_binary_hw(0, hw, source_xu),
                Distribution::BinaryProb(prob) => u.fill_binary_prob(0, prob, source_xu),
                Distribution::BinaryBlock(block_size) => u.fill_binary_block(0, block_size, source_xu),
                Distribution::ZERO => {}
            }

            self.svp_prepare(&mut u_dft, 0, &u, 0);
        }

        // ct[i] = pk[i] * u + ei (+ m if col = i)
        (0..cols).for_each(|i| {
            let (mut ci_dft, scratch_2) = scratch_1.take_vec_znx_dft(res.n().into(), 1, size_pk);
            // ci_dft = DFT(u) * DFT(pk[i])
            self.svp_apply_dft_to_dft(&mut ci_dft, 0, &u_dft, 0, &pk.data, i);

            // ci_big = u * p[i]
            let mut ci_big = self.vec_znx_idft_apply_consume(ci_dft);

            // ci_big = u * pk[i] + e
            self.vec_znx_big_add_normal(
                base2k,
                &mut ci_big,
                0,
                pk.k().into(),
                source_xe,
                SIGMA,
                SIGMA_BOUND,
            );

            // ci_big = u * pk[i] + e + m (if col = i)
            if let Some((pt, col)) = pt
                && col == i
            {
                self.vec_znx_big_add_small_inplace(&mut ci_big, 0, &pt.to_ref().data, 0);
            }

            // ct[i] = norm(ci_big)
            self.vec_znx_big_normalize(base2k, &mut res.data, i, base2k, &ci_big, 0, scratch_2);
        });
    }
}

pub(crate) trait GLWEEncryptSkInternal<B: Backend> {
    fn glwe_encrypt_sk_internal<R, P, S>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        cols: usize,
        compressed: bool,
        pt: Option<(&P, usize)>,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<B>;
}

impl<B: Backend> GLWEEncryptSkInternal<B> for Module<B>
where
    Module<B>: VecZnxDftBytesOf
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxFillUniform
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalizeInplace<B>
        + VecZnxAddNormal
        + VecZnxNormalize<B>
        + VecZnxSub,
    Scratch<B>: ScratchAvailable,
{
    fn glwe_encrypt_sk_internal<R, P, S>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut R,
        cols: usize,
        compressed: bool,
        pt: Option<(&P, usize)>,
        sk: &S,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        scratch: &mut Scratch<B>,
    ) where
        R: VecZnxToMut,
        P: GLWEPlaintextToRef,
        S: GLWESecretPreparedToRef<B>,
    {
        let ct: &mut VecZnx<&mut [u8]> = &mut res.to_mut();
        let sk: GLWESecretPrepared<&[u8], B> = sk.to_ref();

        #[cfg(debug_assertions)]
        {
            if compressed {
                assert_eq!(
                    ct.cols(),
                    1,
                    "invalid ciphertext: compressed tag=true but #cols={} != 1",
                    ct.cols()
                )
            }
        }

        let size: usize = ct.size();

        let (mut c0, scratch_1) = scratch.take_vec_znx(ct.n(), 1, size);
        c0.zero();

        {
            let (mut ci, scratch_2) = scratch_1.take_vec_znx(ct.n(), 1, size);

            // ct[i] = uniform
            // ct[0] -= c[i] * s[i],
            (1..cols).for_each(|i| {
                let col_ct: usize = if compressed { 0 } else { i };

                // ct[i] = uniform (+ pt)
                self.vec_znx_fill_uniform(base2k, ct, col_ct, source_xa);

                let (mut ci_dft, scratch_3) = scratch_2.take_vec_znx_dft(ct.n(), 1, size);

                // ci = ct[i] - pt
                // i.e. we act as we sample ct[i] already as uniform + pt
                // and if there is a pt, then we subtract it before applying DFT
                if let Some((pt, col)) = pt {
                    if i == col {
                        self.vec_znx_sub(&mut ci, 0, ct, col_ct, &pt.to_ref().data, 0);
                        self.vec_znx_normalize_inplace(base2k, &mut ci, 0, scratch_3);
                        self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, &ci, 0);
                    } else {
                        self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, ct, col_ct);
                    }
                } else {
                    self.vec_znx_dft_apply(1, 0, &mut ci_dft, 0, ct, col_ct);
                }

                self.svp_apply_dft_to_dft_inplace(&mut ci_dft, 0, &sk.data, i - 1);
                let ci_big: VecZnxBig<&mut [u8], B> = self.vec_znx_idft_apply_consume(ci_dft);

                // use c[0] as buffer, which is overwritten later by the normalization step
                self.vec_znx_big_normalize(base2k, &mut ci, 0, base2k, &ci_big, 0, scratch_3);

                // c0_tmp = -c[i] * s[i] (use c[0] as buffer)
                self.vec_znx_sub_inplace(&mut c0, 0, &ci, 0);
            });
        }

        // c[0] += e
        self.vec_znx_add_normal(base2k, &mut c0, 0, k, source_xe, sigma, SIGMA_BOUND);

        // c[0] += m if col = 0
        if let Some((pt, col)) = pt
            && col == 0
        {
            self.vec_znx_add_inplace(&mut c0, 0, &pt.to_ref().data, 0);
        }

        // c[0] = norm(c[0])
        self.vec_znx_normalize(base2k, ct, 0, base2k, &c0, 0, scratch_1);
    }
}
