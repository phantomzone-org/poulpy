use backend::hal::{
    api::{TakeMatZnx, TakeScalarZnx, TakeSvpPPol, TakeVecZnx, TakeVecZnxDft, TakeVmpPMat},
    layouts::{Backend, DataRef, Scratch},
    oep::{TakeMatZnxImpl, TakeScalarZnxImpl, TakeSvpPPolImpl, TakeVecZnxDftImpl, TakeVecZnxImpl, TakeVmpPMatImpl},
};

use crate::{
    dist::Distribution,
    layouts::{
        GGLWEAutomorphismKey, GGLWECiphertext, GGLWESwitchingKey, GGLWETensorKey, GGSWCiphertext, GLWECiphertext, GLWEPlaintext,
        GLWEPublicKey, GLWESecret, Infos,
        prepared::{
            GGLWEAutomorphismKeyPrepared, GGLWECiphertextPrepared, GGLWESwitchingKeyPrepared, GGLWETensorKeyPrepared,
            GGSWCiphertextPrepared, GLWEPublicKeyPrepared, GLWESecretPrepared,
        },
    },
};

pub trait TakeLike<'a, B: Backend, T> {
    type Output;
    fn take_like(&'a mut self, template: &T) -> (Self::Output, &'a mut Self);
}

pub trait TakeGLWECt {
    fn take_glwe_ct(&mut self, n: usize, basek: usize, k: usize, rank: usize) -> (GLWECiphertext<&mut [u8]>, &mut Self);
}

pub trait TakeGLWECtSlice {
    fn take_glwe_ct_slice(
        &mut self,
        size: usize,
        n: usize,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (Vec<GLWECiphertext<&mut [u8]>>, &mut Self);
}

pub trait TakeGLWEPt<B: Backend> {
    fn take_glwe_pt(&mut self, n: usize, basek: usize, k: usize) -> (GLWEPlaintext<&mut [u8]>, &mut Self);
}

pub trait TakeGGLWE {
    #[allow(clippy::too_many_arguments)]
    fn take_gglwe(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertext<&mut [u8]>, &mut Self);
}

pub trait TakeGGLWEPrepared<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn take_gglwe_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertextPrepared<&mut [u8], B>, &mut Self);
}

pub trait TakeGGSW {
    fn take_ggsw(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGSWCiphertext<&mut [u8]>, &mut Self);
}

pub trait TakeGGSWPrepared<B: Backend> {
    fn take_ggsw_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGSWCiphertextPrepared<&mut [u8], B>, &mut Self);
}

pub trait TakeGLWESecret {
    fn take_glwe_secret(&mut self, n: usize, rank: usize) -> (GLWESecret<&mut [u8]>, &mut Self);
}

pub trait TakeGLWESecretPrepared<B: Backend> {
    fn take_glwe_secret_prepared(&mut self, n: usize, rank: usize) -> (GLWESecretPrepared<&mut [u8], B>, &mut Self);
}

pub trait TakeGLWEPk {
    fn take_glwe_pk(&mut self, n: usize, basek: usize, k: usize, rank: usize) -> (GLWEPublicKey<&mut [u8]>, &mut Self);
}

pub trait TakeGLWEPkPrepared<B: Backend> {
    fn take_glwe_pk_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWEPublicKeyPrepared<&mut [u8], B>, &mut Self);
}

pub trait TakeGLWESwitchingKey {
    #[allow(clippy::too_many_arguments)]
    fn take_glwe_switching_key(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWESwitchingKey<&mut [u8]>, &mut Self);
}

pub trait TakeGLWESwitchingKeyPrepared<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn take_glwe_switching_key_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWESwitchingKeyPrepared<&mut [u8], B>, &mut Self);
}

pub trait TakeTensorKey {
    fn take_tensor_key(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGLWETensorKey<&mut [u8]>, &mut Self);
}

pub trait TakeTensorKeyPrepared<B: Backend> {
    fn take_tensor_key_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGLWETensorKeyPrepared<&mut [u8], B>, &mut Self);
}

pub trait TakeAutomorphismKey {
    fn take_automorphism_key(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGLWEAutomorphismKey<&mut [u8]>, &mut Self);
}

pub trait TakeAutomorphismKeyPrepared<B: Backend> {
    fn take_automorphism_key_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGLWEAutomorphismKeyPrepared<&mut [u8], B>, &mut Self);
}

impl<B: Backend> TakeGLWECt for Scratch<B>
where
    Scratch<B>: TakeVecZnx,
{
    fn take_glwe_ct(&mut self, n: usize, basek: usize, k: usize, rank: usize) -> (GLWECiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_vec_znx(n, rank + 1, k.div_ceil(basek));
        (GLWECiphertext { data, basek, k }, scratch)
    }
}

impl<'a, B, D> TakeLike<'a, B, GLWECiphertext<D>> for Scratch<B>
where
    B: Backend + TakeVecZnxImpl<B>,
    D: DataRef,
{
    type Output = GLWECiphertext<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GLWECiphertext<D>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_vec_znx_impl(self, template.n(), template.cols(), template.size());
        (
            GLWECiphertext {
                data,
                basek: template.basek(),
                k: template.k(),
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWECtSlice for Scratch<B>
where
    Scratch<B>: TakeVecZnx,
{
    fn take_glwe_ct_slice(
        &mut self,
        size: usize,
        n: usize,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (Vec<GLWECiphertext<&mut [u8]>>, &mut Self) {
        let mut scratch: &mut Scratch<B> = self;
        let mut cts: Vec<GLWECiphertext<&mut [u8]>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_glwe_ct(n, basek, k, rank);
            scratch = new_scratch;
            cts.push(ct);
        }
        (cts, scratch)
    }
}

impl<B: Backend> TakeGLWEPt<B> for Scratch<B>
where
    Scratch<B>: TakeVecZnx,
{
    fn take_glwe_pt(&mut self, n: usize, basek: usize, k: usize) -> (GLWEPlaintext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_vec_znx(n, 1, k.div_ceil(basek));
        (GLWEPlaintext { data, basek, k }, scratch)
    }
}

impl<'a, B, D> TakeLike<'a, B, GLWEPlaintext<D>> for Scratch<B>
where
    B: Backend + TakeVecZnxImpl<B>,
    D: DataRef,
{
    type Output = GLWEPlaintext<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GLWEPlaintext<D>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_vec_znx_impl(self, template.n(), template.cols(), template.size());
        (
            GLWEPlaintext {
                data,
                basek: template.basek(),
                k: template.k(),
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGGLWE for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_gglwe(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_mat_znx(
            n,
            rows.div_ceil(digits),
            rank_in,
            rank_out + 1,
            k.div_ceil(basek),
        );
        (
            GGLWECiphertext {
                data,
                basek,
                k,
                digits,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GGLWECiphertext<D>> for Scratch<B>
where
    B: Backend + TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = GGLWECiphertext<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GGLWECiphertext<D>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_mat_znx_impl(
            self,
            template.n(),
            template.rows(),
            template.data.cols_in(),
            template.data.cols_out(),
            template.size(),
        );
        (
            GGLWECiphertext {
                data,
                basek: template.basek(),
                k: template.k(),
                digits: template.digits(),
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGGLWEPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeVmpPMat<B>,
{
    fn take_gglwe_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWECiphertextPrepared<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_vmp_pmat(
            n,
            rows.div_ceil(digits),
            rank_in,
            rank_out + 1,
            k.div_ceil(basek),
        );
        (
            GGLWECiphertextPrepared {
                data,
                basek,
                k,
                digits,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GGLWECiphertextPrepared<D, B>> for Scratch<B>
where
    B: Backend + TakeVmpPMatImpl<B>,
    D: DataRef,
{
    type Output = GGLWECiphertextPrepared<&'a mut [u8], B>;

    fn take_like(&'a mut self, template: &GGLWECiphertextPrepared<D, B>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_vmp_pmat_impl(
            self,
            template.n(),
            template.rows(),
            template.data.cols_in(),
            template.data.cols_out(),
            template.size(),
        );
        (
            GGLWECiphertextPrepared {
                data,
                basek: template.basek(),
                k: template.k(),
                digits: template.digits(),
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGGSW for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_ggsw(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGSWCiphertext<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_mat_znx(
            n,
            rows.div_ceil(digits),
            rank + 1,
            rank + 1,
            k.div_ceil(basek),
        );
        (
            GGSWCiphertext {
                data,
                basek,
                k,
                digits,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GGSWCiphertext<D>> for Scratch<B>
where
    B: Backend + TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = GGSWCiphertext<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GGSWCiphertext<D>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_mat_znx_impl(
            self,
            template.n(),
            template.rows(),
            template.data.cols_in(),
            template.data.cols_out(),
            template.size(),
        );
        (
            GGSWCiphertext {
                data,
                basek: template.basek(),
                k: template.k(),
                digits: template.digits(),
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGGSWPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeVmpPMat<B>,
{
    fn take_ggsw_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGSWCiphertextPrepared<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_vmp_pmat(
            n,
            rows.div_ceil(digits),
            rank + 1,
            rank + 1,
            k.div_ceil(basek),
        );
        (
            GGSWCiphertextPrepared {
                data,
                basek,
                k,
                digits,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GGSWCiphertextPrepared<D, B>> for Scratch<B>
where
    B: Backend + TakeVmpPMatImpl<B>,
    D: DataRef,
{
    type Output = GGSWCiphertextPrepared<&'a mut [u8], B>;

    fn take_like(&'a mut self, template: &GGSWCiphertextPrepared<D, B>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_vmp_pmat_impl(
            self,
            template.n(),
            template.rows(),
            template.data.cols_in(),
            template.data.cols_out(),
            template.size(),
        );
        (
            GGSWCiphertextPrepared {
                data,
                basek: template.basek(),
                k: template.k(),
                digits: template.digits(),
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWEPk for Scratch<B>
where
    Scratch<B>: TakeVecZnx,
{
    fn take_glwe_pk(&mut self, n: usize, basek: usize, k: usize, rank: usize) -> (GLWEPublicKey<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_vec_znx(n, rank + 1, k.div_ceil(basek));
        (
            GLWEPublicKey {
                data,
                k,
                basek,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GLWEPublicKey<D>> for Scratch<B>
where
    B: Backend + TakeVecZnxImpl<B>,
    D: DataRef,
{
    type Output = GLWEPublicKey<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GLWEPublicKey<D>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_vec_znx_impl(self, template.n(), template.cols(), template.size());
        (
            GLWEPublicKey {
                data,
                basek: template.basek(),
                k: template.k(),
                dist: template.dist,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWEPkPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeVecZnxDft<B>,
{
    fn take_glwe_pk_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rank: usize,
    ) -> (GLWEPublicKeyPrepared<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_vec_znx_dft(n, rank + 1, k.div_ceil(basek));
        (
            GLWEPublicKeyPrepared {
                data,
                k,
                basek,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GLWEPublicKeyPrepared<D, B>> for Scratch<B>
where
    B: Backend + TakeVecZnxDftImpl<B>,
    D: DataRef,
{
    type Output = GLWEPublicKeyPrepared<&'a mut [u8], B>;

    fn take_like(&'a mut self, template: &GLWEPublicKeyPrepared<D, B>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_vec_znx_dft_impl(self, template.n(), template.cols(), template.size());
        (
            GLWEPublicKeyPrepared {
                data,
                basek: template.basek(),
                k: template.k(),
                dist: template.dist,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESecret for Scratch<B>
where
    Scratch<B>: TakeScalarZnx,
{
    fn take_glwe_secret(&mut self, n: usize, rank: usize) -> (GLWESecret<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_scalar_znx(n, rank);
        (
            GLWESecret {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GLWESecret<D>> for Scratch<B>
where
    B: Backend + TakeScalarZnxImpl<B>,
    D: DataRef,
{
    type Output = GLWESecret<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GLWESecret<D>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_scalar_znx_impl(self, template.n(), template.rank());
        (
            GLWESecret {
                data,
                dist: template.dist,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESecretPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeSvpPPol<B>,
{
    fn take_glwe_secret_prepared(&mut self, n: usize, rank: usize) -> (GLWESecretPrepared<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_svp_ppol(n, rank);
        (
            GLWESecretPrepared {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GLWESecretPrepared<D, B>> for Scratch<B>
where
    B: Backend + TakeSvpPPolImpl<B>,
    D: DataRef,
{
    type Output = GLWESecretPrepared<&'a mut [u8], B>;

    fn take_like(&'a mut self, template: &GLWESecretPrepared<D, B>) -> (Self::Output, &'a mut Self) {
        let (data, scratch) = B::take_svp_ppol_impl(self, template.n(), template.rank());
        (
            GLWESecretPrepared {
                data,
                dist: template.dist,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESwitchingKey for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_glwe_switching_key(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWESwitchingKey<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_gglwe(n, basek, k, rows, digits, rank_in, rank_out);
        (
            GGLWESwitchingKey {
                key: data,
                sk_in_n: 0,
                sk_out_n: 0,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GGLWESwitchingKey<D>> for Scratch<B>
where
    Scratch<B>: TakeLike<'a, B, GGLWECiphertext<D>, Output = GGLWECiphertext<&'a mut [u8]>>,
    B: Backend + TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = GGLWESwitchingKey<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GGLWESwitchingKey<D>) -> (Self::Output, &'a mut Self) {
        let (key, scratch) = self.take_like(&template.key);
        (
            GGLWESwitchingKey {
                key,
                sk_in_n: template.sk_in_n,
                sk_out_n: template.sk_out_n,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESwitchingKeyPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeGGLWEPrepared<B>,
{
    fn take_glwe_switching_key_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank_in: usize,
        rank_out: usize,
    ) -> (GGLWESwitchingKeyPrepared<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_gglwe_prepared(n, basek, k, rows, digits, rank_in, rank_out);
        (
            GGLWESwitchingKeyPrepared {
                key: data,
                sk_in_n: 0,
                sk_out_n: 0,
            },
            scratch,
        )
    }
}

impl<'a, B, D> TakeLike<'a, B, GGLWESwitchingKeyPrepared<D, B>> for Scratch<B>
where
    Scratch<B>: TakeLike<'a, B, GGLWECiphertextPrepared<D, B>, Output = GGLWECiphertextPrepared<&'a mut [u8], B>>,
    B: Backend + TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = GGLWESwitchingKeyPrepared<&'a mut [u8], B>;

    fn take_like(&'a mut self, template: &GGLWESwitchingKeyPrepared<D, B>) -> (Self::Output, &'a mut Self) {
        let (key, scratch) = self.take_like(&template.key);
        (
            GGLWESwitchingKeyPrepared {
                key,
                sk_in_n: template.sk_in_n,
                sk_out_n: template.sk_out_n,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeAutomorphismKey for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_automorphism_key(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGLWEAutomorphismKey<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_glwe_switching_key(n, basek, k, rows, digits, rank, rank);
        (GGLWEAutomorphismKey { key: data, p: 0 }, scratch)
    }
}

impl<'a, B, D> TakeLike<'a, B, GGLWEAutomorphismKey<D>> for Scratch<B>
where
    Scratch<B>: TakeLike<'a, B, GGLWESwitchingKey<D>, Output = GGLWESwitchingKey<&'a mut [u8]>>,
    B: Backend + TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = GGLWEAutomorphismKey<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GGLWEAutomorphismKey<D>) -> (Self::Output, &'a mut Self) {
        let (key, scratch) = self.take_like(&template.key);
        (GGLWEAutomorphismKey { key, p: template.p }, scratch)
    }
}

impl<B: Backend> TakeAutomorphismKeyPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeGLWESwitchingKeyPrepared<B>,
{
    fn take_automorphism_key_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGLWEAutomorphismKeyPrepared<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_glwe_switching_key_prepared(n, basek, k, rows, digits, rank, rank);
        (GGLWEAutomorphismKeyPrepared { key: data, p: 0 }, scratch)
    }
}

impl<'a, B, D> TakeLike<'a, B, GGLWEAutomorphismKeyPrepared<D, B>> for Scratch<B>
where
    Scratch<B>: TakeLike<'a, B, GGLWESwitchingKeyPrepared<D, B>, Output = GGLWESwitchingKeyPrepared<&'a mut [u8], B>>,
    B: Backend + TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = GGLWEAutomorphismKeyPrepared<&'a mut [u8], B>;

    fn take_like(&'a mut self, template: &GGLWEAutomorphismKeyPrepared<D, B>) -> (Self::Output, &'a mut Self) {
        let (key, scratch) = self.take_like(&template.key);
        (GGLWEAutomorphismKeyPrepared { key, p: template.p }, scratch)
    }
}

impl<B: Backend> TakeTensorKey for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_tensor_key(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGLWETensorKey<&mut [u8]>, &mut Self) {
        let mut keys: Vec<GGLWESwitchingKey<&mut [u8]>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);

        let mut scratch: &mut Scratch<B> = self;

        if pairs != 0 {
            let (gglwe, s) = scratch.take_glwe_switching_key(n, basek, k, rows, digits, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_glwe_switching_key(n, basek, k, rows, digits, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        (GGLWETensorKey { keys }, scratch)
    }
}

impl<'a, B, D> TakeLike<'a, B, GGLWETensorKey<D>> for Scratch<B>
where
    Scratch<B>: TakeLike<'a, B, GGLWESwitchingKey<D>, Output = GGLWESwitchingKey<&'a mut [u8]>>,
    B: Backend + TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = GGLWETensorKey<&'a mut [u8]>;

    fn take_like(&'a mut self, template: &GGLWETensorKey<D>) -> (Self::Output, &'a mut Self) {
        let mut keys: Vec<GGLWESwitchingKey<&mut [u8]>> = Vec::new();
        let pairs: usize = template.keys.len();

        let mut scratch: &mut Scratch<B> = self;

        if pairs != 0 {
            let (gglwe, s) = scratch.take_like(template.at(0, 0));
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_like(template.at(0, 0));
            scratch = s;
            keys.push(gglwe);
        }

        (GGLWETensorKey { keys }, scratch)
    }
}

impl<B: Backend> TakeTensorKeyPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeVmpPMat<B>,
{
    fn take_tensor_key_prepared(
        &mut self,
        n: usize,
        basek: usize,
        k: usize,
        rows: usize,
        digits: usize,
        rank: usize,
    ) -> (GGLWETensorKeyPrepared<&mut [u8], B>, &mut Self) {
        let mut keys: Vec<GGLWESwitchingKeyPrepared<&mut [u8], B>> = Vec::new();
        let pairs: usize = (((rank + 1) * rank) >> 1).max(1);

        let mut scratch: &mut Scratch<B> = self;

        if pairs != 0 {
            let (gglwe, s) = scratch.take_glwe_switching_key_prepared(n, basek, k, rows, digits, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_glwe_switching_key_prepared(n, basek, k, rows, digits, 1, rank);
            scratch = s;
            keys.push(gglwe);
        }
        (GGLWETensorKeyPrepared { keys }, scratch)
    }
}

impl<'a, B, D> TakeLike<'a, B, GGLWETensorKeyPrepared<D, B>> for Scratch<B>
where
    Scratch<B>: TakeLike<'a, B, GGLWESwitchingKeyPrepared<D, B>, Output = GGLWESwitchingKeyPrepared<&'a mut [u8], B>>,
    B: Backend + TakeMatZnxImpl<B>,
    D: DataRef,
{
    type Output = GGLWETensorKeyPrepared<&'a mut [u8], B>;

    fn take_like(&'a mut self, template: &GGLWETensorKeyPrepared<D, B>) -> (Self::Output, &'a mut Self) {
        let mut keys: Vec<GGLWESwitchingKeyPrepared<&mut [u8], B>> = Vec::new();
        let pairs: usize = template.keys.len();

        let mut scratch: &mut Scratch<B> = self;

        if pairs != 0 {
            let (gglwe, s) = scratch.take_like(template.at(0, 0));
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_like(template.at(0, 0));
            scratch = s;
            keys.push(gglwe);
        }

        (GGLWETensorKeyPrepared { keys }, scratch)
    }
}
