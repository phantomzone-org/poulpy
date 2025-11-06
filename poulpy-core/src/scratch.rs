use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchTakeBasic, SvpPPolBytesOf, VecZnxDftBytesOf, VmpPMatBytesOf},
    layouts::{Backend, Scratch},
};

use crate::{
    dist::Distribution,
    layouts::{
        Degree, GGLWE, GGLWEInfos, GGLWELayout, GGSW, GGSWInfos, GLWE, GLWEAutomorphismKey, GLWEInfos, GLWEPlaintext,
        GLWEPrepared, GLWEPublicKey, GLWESecret, GLWESecretTensor, GLWESwitchingKey, GLWETensorKey, Rank,
        prepared::{
            GGLWEPrepared, GGSWPrepared, GLWEAutomorphismKeyPrepared, GLWEPublicKeyPrepared, GLWESecretPrepared,
            GLWESwitchingKeyPrepared, GLWETensorKeyPrepared,
        },
    },
};

pub trait ScratchTakeCore<B: Backend>
where
    Self: ScratchTakeBasic + ScratchAvailable,
{
    fn take_glwe<A>(&mut self, infos: &A) -> (GLWE<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(infos.n().into(), (infos.rank() + 1).into(), infos.size());
        (
            GLWE {
                k: infos.k(),
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    fn take_glwe_slice<A>(&mut self, size: usize, infos: &A) -> (Vec<GLWE<&mut [u8]>>, &mut Self)
    where
        A: GLWEInfos,
    {
        let mut scratch: &mut Self = self;
        let mut cts: Vec<GLWE<&mut [u8]>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_glwe(infos);
            scratch = new_scratch;
            cts.push(ct);
        }
        (cts, scratch)
    }

    fn take_glwe_plaintext<A>(&mut self, infos: &A) -> (GLWEPlaintext<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(infos.n().into(), 1, infos.size());
        (
            GLWEPlaintext {
                k: infos.k(),
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

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

    fn take_ggsw_slice<A>(
        &mut self,
        size: usize,
        infos: &A,
    ) -> (Vec<GGSW<&mut [u8]>>, &mut Self)
    where
        A: GGSWInfos,
    {
        let mut scratch: &mut Self = self;
        let mut cts: Vec<GGSW<&mut [u8]>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_ggsw(infos);
            scratch = new_scratch;
            cts.push(ct)
        }
        (cts, scratch)
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

    fn take_glwe_public_key<A>(&mut self, infos: &A) -> (GLWEPublicKey<&mut [u8]>, &mut Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_glwe(infos);
        (
            GLWEPublicKey {
                dist: Distribution::NONE,
                key: data,
            },
            scratch,
        )
    }

    fn take_glwe_public_key_prepared<A, M>(&mut self, module: &M, infos: &A) -> (GLWEPublicKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GLWEInfos,
        M: ModuleN + VecZnxDftBytesOf,
    {
        let (data, scratch) = self.take_glwe_prepared(module, infos);
        (
            GLWEPublicKeyPrepared {
                dist: Distribution::NONE,
                key: data,
            },
            scratch,
        )
    }

    fn take_glwe_prepared<A, M>(&mut self, module: &M, infos: &A) -> (GLWEPrepared<&mut [u8], B>, &mut Self)
    where
        A: GLWEInfos,
        M: ModuleN + VecZnxDftBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_vec_znx_dft(module, (infos.rank() + 1).into(), infos.size());
        (
            GLWEPrepared {
                k: infos.k(),
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

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

    fn take_glwe_secret_tensor(&mut self, n: Degree, rank: Rank) -> (GLWESecretTensor<&mut [u8]>, &mut Self) {
        let (data, scratch) = self.take_scalar_znx(n.into(), GLWESecretTensor::pairs(rank.into()));
        (
            GLWESecretTensor {
                data,
                rank,
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

    fn take_glwe_switching_key<A>(&mut self, infos: &A) -> (GLWESwitchingKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_gglwe(infos);
        (
            GLWESwitchingKey {
                key: data,
                input_degree: Degree(0),
                output_degree: Degree(0),
            },
            scratch,
        )
    }

    fn take_glwe_switching_key_prepared<A, M>(
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
                input_degree: Degree(0),
                output_degree: Degree(0),
            },
            scratch,
        )
    }

    fn take_glwe_automorphism_key<A>(&mut self, infos: &A) -> (GLWEAutomorphismKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
    {
        let (data, scratch) = self.take_gglwe(infos);
        (GLWEAutomorphismKey { key: data, p: 0 }, scratch)
    }

    fn take_glwe_automorphism_key_prepared<A, M>(
        &mut self,
        module: &M,
        infos: &A,
    ) -> (GLWEAutomorphismKeyPrepared<&mut [u8], B>, &mut Self)
    where
        A: GGLWEInfos,
        M: ModuleN + VmpPMatBytesOf,
    {
        assert_eq!(module.n() as u32, infos.n());
        let (data, scratch) = self.take_gglwe_prepared(module, infos);
        (GLWEAutomorphismKeyPrepared { key: data, p: 0 }, scratch)
    }

    fn take_glwe_tensor_key<A, M>(&mut self, infos: &A) -> (GLWETensorKey<&mut [u8]>, &mut Self)
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GLWETensorKey"
        );

        let pairs: u32 = (((infos.rank_out().0 + 1) * infos.rank_out().0) >> 1).max(1);
        let mut ksk_infos: GGLWELayout = infos.gglwe_layout();
        ksk_infos.rank_in = Rank(pairs);
        let (data, scratch) = self.take_gglwe(infos);
        (GLWETensorKey(data), scratch)
    }

    fn take_glwe_tensor_key_prepared<A, M>(&mut self, module: &M, infos: &A) -> (GLWETensorKeyPrepared<&mut [u8], B>, &mut Self)
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

        let pairs: u32 = (((infos.rank_out().0 + 1) * infos.rank_out().0) >> 1).max(1);
        let mut ksk_infos: GGLWELayout = infos.gglwe_layout();
        ksk_infos.rank_in = Rank(pairs);
        let (data, scratch) = self.take_gglwe_prepared(module, infos);
        (GLWETensorKeyPrepared(data), scratch)
    }
}

impl<B: Backend> ScratchTakeCore<B> for Scratch<B> where Self: ScratchTakeBasic + ScratchAvailable {}
