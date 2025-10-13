use poulpy_hal::{
    api::{TakeMatZnx, TakeScalarZnx, TakeSvpPPol, TakeVecZnx, TakeVecZnxDft, TakeVmpPMat},
    layouts::{Backend, Scratch},
};

use crate::{
    dist::Distribution,
    layouts::{
        AutomorphismKey, Degree, GGLWE, GGLWEInfos, GGSW, GGSWInfos, GLWECiphertext, GLWEInfos, GLWEPlaintext, GLWEPublicKey,
        GLWESecret, GLWESwitchingKey, Rank, TensorKey,
        prepared::{
            AutomorphismKeyPrepared, GGLWEPrepared, GGSWPrepared, GLWEPublicKeyPrepared, GLWESecretPrepared,
            GLWESwitchingKeyPrepared, TensorKeyPrepared,
        },
    },
};

pub trait TakeGLWECt {
    fn take_glwe_ct<A>(&mut self, infos: &A) -> (GLWECiphertext<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos;
}

pub trait TakeGLWECtSlice {
    fn take_glwe_ct_slice<A>(&mut self, size: usize, infos: &A) -> (Vec<GLWECiphertext<&mut [u8]>>, &mut Self)
    where
        A: GLWEInfos;
}

pub trait TakeGLWEPt<B: Backend> {
    fn take_glwe_pt<A>(&mut self, infos: &A) -> (GLWEPlaintext<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos;
}

pub trait TakeGGLWE {
    fn take_gglwe<A>(&mut self, infos: &A) -> (GGLWE<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos;
}

pub trait TakeGGLWEPrepared<B: Backend> {
    fn take_gglwe_prepared<A>(&mut self, infos: &A) -> (GGLWEPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos;
}

pub trait TakeGGSW {
    fn take_ggsw<A>(&mut self, infos: &A) -> (GGSW<&mut [u8]>, &mut Self)
    where
        A: GGSWInfos;
}

pub trait TakeGGSWPrepared<B: Backend> {
    fn take_ggsw_prepared<A>(&mut self, infos: &A) -> (GGSWPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGSWInfos;
}

pub trait TakeGGSWPreparedSlice<B: Backend> {
    fn take_ggsw_prepared_slice<A>(&mut self, size: usize, infos: &A) -> (Vec<GGSWPrepared<&mut [u8], B>>, &mut Self)
    where
        A: GGSWInfos;
}

pub trait TakeGLWESecret {
    fn take_glwe_secret(&mut self, n: Degree, rank: Rank) -> (GLWESecret<&mut [u8]>, &mut Self);
}

pub trait TakeGLWESecretPrepared<B: Backend> {
    fn take_glwe_secret_prepared(&mut self, n: Degree, rank: Rank) -> (GLWESecretPrepared<&mut [u8], B>, &mut Self);
}

pub trait TakeGLWEPk {
    fn take_glwe_pk<A>(&mut self, infos: &A) -> (GLWEPublicKey<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos;
}

pub trait TakeGLWEPkPrepared<B: Backend> {
    fn take_glwe_pk_prepared<A>(&mut self, infos: &A) -> (GLWEPublicKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GLWEInfos;
}

pub trait TakeGLWESwitchingKey {
    fn take_glwe_switching_key<A>(&mut self, infos: &A) -> (GLWESwitchingKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos;
}

pub trait TakeGGLWESwitchingKeyPrepared<B: Backend> {
    fn take_gglwe_switching_key_prepared<A>(&mut self, infos: &A) -> (GLWESwitchingKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos;
}

pub trait TakeTensorKey {
    fn take_tensor_key<A>(&mut self, infos: &A) -> (TensorKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos;
}

pub trait TakeGGLWETensorKeyPrepared<B: Backend> {
    fn take_gglwe_tensor_key_prepared<A>(&mut self, infos: &A) -> (TensorKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos;
}

pub trait TakeGGLWEAutomorphismKey {
    fn take_gglwe_automorphism_key<A>(&mut self, infos: &A) -> (AutomorphismKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos;
}

pub trait TakeGGLWEAutomorphismKeyPrepared<B: Backend> {
    fn take_gglwe_automorphism_key_prepared<A>(&mut self, infos: &A) -> (AutomorphismKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos;
}

impl<B: Backend> TakeGLWECt for Scratch<B>
where
    Scratch<B>: TakeVecZnx,
{
    fn take_glwe_ct<A>(&mut self, infos: &A) -> (GLWECiphertext<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(infos.n().into(), (infos.rank() + 1).into(), infos.size());
        (
            GLWECiphertext::builder()
                .base2k(infos.base2k())
                .k(infos.k())
                .data(data)
                .build()
                .unwrap(),
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWECtSlice for Scratch<B>
where
    Scratch<B>: TakeVecZnx,
{
    fn take_glwe_ct_slice<A>(&mut self, size: usize, infos: &A) -> (Vec<GLWECiphertext<&mut [u8]>>, &mut Self)
    where
        A: GLWEInfos,
    {
        let mut scratch: &mut Scratch<B> = self;
        let mut cts: Vec<GLWECiphertext<&mut [u8]>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_glwe_ct(infos);
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
    fn take_glwe_pt<A>(&mut self, infos: &A) -> (GLWEPlaintext<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(infos.n().into(), 1, infos.size());
        (
            GLWEPlaintext::builder()
                .base2k(infos.base2k())
                .k(infos.k())
                .data(data)
                .build()
                .unwrap(),
            scratch,
        )
    }
}

impl<B: Backend> TakeGGLWE for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_gglwe<A>(&mut self, infos: &A) -> (GGLWE<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_mat_znx(
            infos.n().into(),
            infos.dnum().0.div_ceil(infos.dsize().0) as usize,
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        );
        (
            GGLWE::builder()
                .base2k(infos.base2k())
                .k(infos.k())
                .dsize(infos.dsize())
                .data(data)
                .build()
                .unwrap(),
            scratch,
        )
    }
}

impl<B: Backend> TakeGGLWEPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeVmpPMat<B>,
{
    fn take_gglwe_prepared<A>(&mut self, infos: &A) -> (GGLWEPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_vmp_pmat(
            infos.n().into(),
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        );
        (
            GGLWEPrepared::builder()
                .base2k(infos.base2k())
                .dsize(infos.dsize())
                .k(infos.k())
                .data(data)
                .build()
                .unwrap(),
            scratch,
        )
    }
}

impl<B: Backend> TakeGGSW for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_ggsw<A>(&mut self, infos: &A) -> (GGSW<&mut [u8]>, &mut Self)
    where
        A: GGSWInfos,
    {
        let (data, scratch) = self.take_mat_znx(
            infos.n().into(),
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        (
            GGSW::builder()
                .base2k(infos.base2k())
                .dsize(infos.dsize())
                .k(infos.k())
                .data(data)
                .build()
                .unwrap(),
            scratch,
        )
    }
}

impl<B: Backend> TakeGGSWPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeVmpPMat<B>,
{
    fn take_ggsw_prepared<A>(&mut self, infos: &A) -> (GGSWPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGSWInfos,
    {
        let (data, scratch) = self.take_vmp_pmat(
            infos.n().into(),
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        (
            GGSWPrepared::builder()
                .base2k(infos.base2k())
                .dsize(infos.dsize())
                .k(infos.k())
                .data(data)
                .build()
                .unwrap(),
            scratch,
        )
    }
}

impl<B: Backend> TakeGGSWPreparedSlice<B> for Scratch<B>
where
    Scratch<B>: TakeGGSWPrepared<B>,
{
    fn take_ggsw_prepared_slice<A>(&mut self, size: usize, infos: &A) -> (Vec<GGSWPrepared<&mut [u8], B>>, &mut Self)
    where
        A: GGSWInfos,
    {
        let mut scratch: &mut Scratch<B> = self;
        let mut cts: Vec<GGSWPrepared<&mut [u8], B>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_ggsw_prepared(infos);
            scratch = new_scratch;
            cts.push(ct)
        }
        (cts, scratch)
    }
}

impl<B: Backend> TakeGLWEPk for Scratch<B>
where
    Scratch<B>: TakeVecZnx,
{
    fn take_glwe_pk<A>(&mut self, infos: &A) -> (GLWEPublicKey<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(infos.n().into(), (infos.rank() + 1).into(), infos.size());
        (
            GLWEPublicKey::builder()
                .base2k(infos.base2k())
                .k(infos.k())
                .base2k(infos.base2k())
                .data(data)
                .build()
                .unwrap(),
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWEPkPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeVecZnxDft<B>,
{
    fn take_glwe_pk_prepared<A>(&mut self, infos: &A) -> (GLWEPublicKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx_dft(infos.n().into(), (infos.rank() + 1).into(), infos.size());
        (
            GLWEPublicKeyPrepared::builder()
                .base2k(infos.base2k())
                .k(infos.k())
                .data(data)
                .build()
                .unwrap(),
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESecret for Scratch<B>
where
    Scratch<B>: TakeScalarZnx,
{
    fn take_glwe_secret(&mut self, n: Degree, rank: Rank) -> (GLWESecret<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_scalar_znx(n.into(), rank.into());
        (
            GLWESecret {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESecretPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeSvpPPol<B>,
{
    fn take_glwe_secret_prepared(&mut self, n: Degree, rank: Rank) -> (GLWESecretPrepared<&mut [u8], B>, &mut Self) {
        let (data, scratch) = self.take_svp_ppol(n.into(), rank.into());
        (
            GLWESecretPrepared {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGLWESwitchingKey for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_glwe_switching_key<A>(&mut self, infos: &A) -> (GLWESwitchingKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_gglwe(infos);
        (
            GLWESwitchingKey {
                key: data,
                sk_in_n: 0,
                sk_out_n: 0,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGGLWESwitchingKeyPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeGGLWEPrepared<B>,
{
    fn take_gglwe_switching_key_prepared<A>(&mut self, infos: &A) -> (GLWESwitchingKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_gglwe_prepared(infos);
        (
            GLWESwitchingKeyPrepared {
                key: data,
                sk_in_n: 0,
                sk_out_n: 0,
            },
            scratch,
        )
    }
}

impl<B: Backend> TakeGGLWEAutomorphismKey for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_gglwe_automorphism_key<A>(&mut self, infos: &A) -> (AutomorphismKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_glwe_switching_key(infos);
        (AutomorphismKey { key: data, p: 0 }, scratch)
    }
}

impl<B: Backend> TakeGGLWEAutomorphismKeyPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeGGLWESwitchingKeyPrepared<B>,
{
    fn take_gglwe_automorphism_key_prepared<A>(&mut self, infos: &A) -> (AutomorphismKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_gglwe_switching_key_prepared(infos);
        (AutomorphismKeyPrepared { key: data, p: 0 }, scratch)
    }
}

impl<B: Backend> TakeTensorKey for Scratch<B>
where
    Scratch<B>: TakeMatZnx,
{
    fn take_tensor_key<A>(&mut self, infos: &A) -> (TensorKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKey"
        );
        let mut keys: Vec<GLWESwitchingKey<&mut [u8]>> = Vec::new();
        let pairs: usize = (((infos.rank_out().0 + 1) * infos.rank_out().0) >> 1).max(1) as usize;

        let mut scratch: &mut Scratch<B> = self;

        let mut ksk_infos: crate::layouts::GGLWECiphertextLayout = infos.layout();
        ksk_infos.rank_in = Rank(1);

        if pairs != 0 {
            let (gglwe, s) = scratch.take_glwe_switching_key(&ksk_infos);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_glwe_switching_key(&ksk_infos);
            scratch = s;
            keys.push(gglwe);
        }
        (TensorKey { keys }, scratch)
    }
}

impl<B: Backend> TakeGGLWETensorKeyPrepared<B> for Scratch<B>
where
    Scratch<B>: TakeVmpPMat<B>,
{
    fn take_gglwe_tensor_key_prepared<A>(&mut self, infos: &A) -> (TensorKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyPrepared"
        );

        let mut keys: Vec<GLWESwitchingKeyPrepared<&mut [u8], B>> = Vec::new();
        let pairs: usize = (((infos.rank_out().0 + 1) * infos.rank_out().0) >> 1).max(1) as usize;

        let mut scratch: &mut Scratch<B> = self;

        let mut ksk_infos: crate::layouts::GGLWECiphertextLayout = infos.layout();
        ksk_infos.rank_in = Rank(1);

        if pairs != 0 {
            let (gglwe, s) = scratch.take_gglwe_switching_key_prepared(&ksk_infos);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_gglwe_switching_key_prepared(&ksk_infos);
            scratch = s;
            keys.push(gglwe);
        }
        (TensorKeyPrepared { keys }, scratch)
    }
}
