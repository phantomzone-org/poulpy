use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchTakeBasic, SvpPPolBytesOf, VecZnxDftBytesOf, VmpPMatBytesOf},
    layouts::{Backend, Scratch},
};

use crate::{
    dist::Distribution,
    layouts::{
        AutomorphismKey, GGLWE, GGLWEInfos, GGSW, GGSWInfos, GLWE, GLWEInfos, GLWEPlaintext, GLWEPublicKey, GLWESecret,
        GLWESwitchingKey, Rank, TensorKey,
        prepared::{
            AutomorphismKeyPrepared, GGLWEPrepared, GGSWPrepared, GLWEPublicKeyPrepared, GLWESecretPrepared,
            GLWESwitchingKeyPrepared, TensorKeyPrepared,
        },
    },
};

pub trait ScratchTakeCore<B: Backend>
where
    Self: ScratchTakeBasic + ScratchAvailable,
{
    fn take_glwe_ct<A, M>(&mut self, module: &M, infos: &A) -> (GLWE<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
        M: ModuleN,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vec_znx(module, (infos.rank() + 1).into(), infos.size());
        (
            GLWE {
                k: infos.k(),
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    fn take_glwe_ct_slice<A, M>(&mut self, module: &M, size: usize, infos: &A) -> (Vec<GLWE<&mut [u8]>>, &mut Self)
    where
        A: GLWEInfos,
        M: ModuleN,
    {
        let mut scratch: &mut Self = self;
        let mut cts: Vec<GLWE<&mut [u8]>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_glwe_ct(module, infos);
            scratch = new_scratch;
            cts.push(ct);
        }
        (cts, scratch)
    }

    fn take_glwe_pt<A, M>(&mut self, module: &M, infos: &A) -> (GLWEPlaintext<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
        M: ModuleN,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vec_znx(module, 1, infos.size());
        (
            GLWEPlaintext {
                k: infos.k(),
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    fn take_gglwe<A, M>(&mut self, module: &M, infos: &A) -> (GGLWE<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_mat_znx(
            module,
            infos.dnum().0.div_ceil(infos.dsize().0) as usize,
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        );
        (
            GGLWE {
                k: infos.k(),
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data,
            },
            scratch,
        )
    }

    fn take_gglwe_prepared<A, M>(&mut self, module: &M, infos: &A) -> (GGLWEPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vmp_pmat(
            module,
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            infos.size(),
        );
        (
            GGLWEPrepared {
                k: infos.k(),
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data,
            },
            scratch,
        )
    }

    fn take_ggsw<A, M>(&mut self, module: &M, infos: &A) -> (GGSW<&mut [u8]>, &mut Self)
    where
        A: GGSWInfos,
        M: ModuleN,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_mat_znx(
            module,
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        (
            GGSW {
                k: infos.k(),
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data,
            },
            scratch,
        )
    }

    fn take_ggsw_prepared<A, M>(&mut self, module: &M, infos: &A) -> (GGSWPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGSWInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vmp_pmat(
            module,
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            infos.size(),
        );
        (
            GGSWPrepared {
                k: infos.k(),
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data,
            },
            scratch,
        )
    }

    fn take_ggsw_prepared_slice<A, M>(
        &mut self,
        module: &M,
        size: usize,
        infos: &A,
    ) -> (Vec<GGSWPrepared<&mut [u8], B>>, &mut Self)
    where
        A: GGSWInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        let mut scratch: &mut Self = self;
        let mut cts: Vec<GGSWPrepared<&mut [u8], B>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_ggsw_prepared(module, infos);
            scratch = new_scratch;
            cts.push(ct)
        }
        (cts, scratch)
    }

    fn take_glwe_pk<A, M>(&mut self, module: &M, infos: &A) -> (GLWEPublicKey<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
        M: ModuleN,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vec_znx(module, (infos.rank() + 1).into(), infos.size());
        (
            GLWEPublicKey {
                k: infos.k(),
                dist: Distribution::NONE,
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    fn take_glwe_pk_prepared<A, M>(&mut self, module: &M, infos: &A) -> (GLWEPublicKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GLWEInfos,
        M: ModuleN + VecZnxDftBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vec_znx_dft(module, (infos.rank() + 1).into(), infos.size());
        (
            GLWEPublicKeyPrepared {
                k: infos.k(),
                dist: Distribution::NONE,
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    fn take_glwe_secret<M>(&mut self, module: &M, rank: Rank) -> (GLWESecret<&mut [u8]>, &mut Self)
    where
        M: ModuleN,
    {
        let (data, scratch) = self.take_scalar_znx(module, rank.into());
        (
            GLWESecret {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }

    fn take_glwe_secret_prepared<M>(&mut self, module: &M, rank: Rank) -> (GLWESecretPrepared<&mut [u8], B>, &mut Self)
    where
        M: ModuleN + SvpPPolBytesOf,
    {
        let (data, scratch) = self.take_svp_ppol(module, rank.into());
        (
            GLWESecretPrepared {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }

    fn take_glwe_switching_key<A, M>(&mut self, module: &M, infos: &A) -> (GLWESwitchingKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_gglwe(module, infos);
        (
            GLWESwitchingKey {
                key: data,
                sk_in_n: 0,
                sk_out_n: 0,
            },
            scratch,
        )
    }

    fn take_gglwe_switching_key_prepared<A, M>(
        &mut self,
        module: &M,
        infos: &A,
    ) -> (GLWESwitchingKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_gglwe_prepared(module, infos);
        (
            GLWESwitchingKeyPrepared {
                key: data,
                sk_in_n: 0,
                sk_out_n: 0,
            },
            scratch,
        )
    }

    fn take_gglwe_automorphism_key<A, M>(&mut self, module: &M, infos: &A) -> (AutomorphismKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_glwe_switching_key(module, infos);
        (AutomorphismKey { key: data, p: 0 }, scratch)
    }

    fn take_gglwe_automorphism_key_prepared<A, M>(
        &mut self,
        module: &M,
        infos: &A,
    ) -> (AutomorphismKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_gglwe_switching_key_prepared(module, infos);
        (AutomorphismKeyPrepared { key: data, p: 0 }, scratch)
    }

    fn take_tensor_key<A, M>(&mut self, module: &M, infos: &A) -> (TensorKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN,
    {
        assert_eq!(module.n() as u32, infos.n());
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKey"
        );
        let mut keys: Vec<GLWESwitchingKey<&mut [u8]>> = Vec::new();
        let pairs: usize = (((infos.rank_out().0 + 1) * infos.rank_out().0) >> 1).max(1) as usize;

        let mut scratch: &mut Self = self;

        let mut ksk_infos: crate::layouts::GGLWECiphertextLayout = infos.gglwe_layout();
        ksk_infos.rank_in = Rank(1);

        if pairs != 0 {
            let (gglwe, s) = scratch.take_glwe_switching_key(module, &ksk_infos);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_glwe_switching_key(module, &ksk_infos);
            scratch = s;
            keys.push(gglwe);
        }
        (TensorKey { keys }, scratch)
    }

    fn take_gglwe_tensor_key_prepared<A, M>(&mut self, module: &M, infos: &A) -> (TensorKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWETensorKeyPrepared"
        );

        let mut keys: Vec<GLWESwitchingKeyPrepared<&mut [u8], B>> = Vec::new();
        let pairs: usize = (((infos.rank_out().0 + 1) * infos.rank_out().0) >> 1).max(1) as usize;

        let mut scratch: &mut Self = self;

        let mut ksk_infos: crate::layouts::GGLWECiphertextLayout = infos.gglwe_layout();
        ksk_infos.rank_in = Rank(1);

        if pairs != 0 {
            let (gglwe, s) = scratch.take_gglwe_switching_key_prepared(module, &ksk_infos);
            scratch = s;
            keys.push(gglwe);
        }
        for _ in 1..pairs {
            let (gglwe, s) = scratch.take_gglwe_switching_key_prepared(module, &ksk_infos);
            scratch = s;
            keys.push(gglwe);
        }
        (TensorKeyPrepared { keys }, scratch)
    }
}

impl<B: Backend> ScratchTakeCore<B> for Scratch<B> where Self: ScratchTakeBasic + ScratchAvailable {}
