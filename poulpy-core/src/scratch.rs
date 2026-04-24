use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeBasic, SvpPPolBytesOf, VmpPMatBytesOf},
    layouts::{Backend, ScratchArena},
};

use crate::{
    dist::Distribution,
    layouts::{
        Degree, GGLWE, GGLWEInfos, GGSW, GGSWInfos, GLWE, GLWEInfos, GLWEPlaintext, GLWESecret, GLWESecretTensor, GLWETensor,
        LWE, LWEInfos, LWEPlaintext, Rank,
        prepared::{GGLWEPrepared, GGSWPrepared, GLWESecretPrepared},
    },
};

/// Backend-native arena allocation for core ciphertext/key layouts.
///
/// Returns backend-native borrows (`B::BufMut<'a>`) carved from a [`ScratchArena`].
pub trait ScratchArenaTakeCore<'a, B: Backend>: ScratchArenaTakeBasic<'a, B> + Sized {
    /// Allocates an [`LWE`] ciphertext from scratch space.
    fn take_lwe<A>(self, infos: &A) -> (LWE<B::BufMut<'a>>, Self)
    where
        A: LWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(infos.n().into(), 1, infos.size());
        (
            LWE {
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    /// Allocates an [`LWEPlaintext`] from scratch space.
    fn take_lwe_plaintext<A>(self, infos: &A) -> (LWEPlaintext<B::BufMut<'a>>, Self)
    where
        A: LWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(1, 1, infos.size());
        (
            LWEPlaintext {
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    /// Allocates a [`GLWE`] ciphertext from scratch space.
    fn take_glwe<A>(self, infos: &A) -> (GLWE<B::BufMut<'a>>, Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(infos.n().into(), (infos.rank() + 1).into(), infos.size());
        (
            GLWE {
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    /// Allocates a `Vec` of `size` [`GLWE`] ciphertexts from scratch space.
    fn take_glwe_slice<A>(self, size: usize, infos: &A) -> (Vec<GLWE<B::BufMut<'a>>>, Self)
    where
        A: GLWEInfos,
    {
        let mut scratch: Self = self;
        let mut cts: Vec<GLWE<B::BufMut<'a>>> = Vec::with_capacity(size);
        for _ in 0..size {
            let (ct, new_scratch) = scratch.take_glwe(infos);
            scratch = new_scratch;
            cts.push(ct);
        }
        (cts, scratch)
    }

    /// Allocates a [`GLWETensor`] from scratch space.
    fn take_glwe_tensor<A>(self, infos: &A) -> (GLWETensor<B::BufMut<'a>>, Self)
    where
        A: GLWEInfos,
    {
        let cols: usize = infos.rank().as_usize() + 1;
        let pairs: usize = (((cols + 1) * cols) >> 1).max(1);
        let (data, scratch) = self.take_vec_znx(infos.n().into(), pairs, infos.size());
        (
            GLWETensor {
                base2k: infos.base2k(),
                rank: infos.rank(),
                data,
            },
            scratch,
        )
    }

    /// Allocates a [`GLWEPlaintext`] from scratch space.
    fn take_glwe_plaintext<A>(self, infos: &A) -> (GLWEPlaintext<B::BufMut<'a>>, Self)
    where
        A: GLWEInfos,
    {
        let (data, scratch) = self.take_vec_znx(infos.n().into(), 1, infos.size());
        (
            GLWEPlaintext {
                base2k: infos.base2k(),
                data,
            },
            scratch,
        )
    }

    /// Allocates a [`GLWESecretPrepared`] (DFT-domain secret key) from scratch space.
    fn take_glwe_secret_prepared<M>(self, module: &M, rank: Rank) -> (GLWESecretPrepared<B::BufMut<'a>, B>, Self)
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

    /// Allocates a [`GLWESecret`] from scratch space.
    fn take_glwe_secret(self, n: Degree, rank: Rank) -> (GLWESecret<B::BufMut<'a>>, Self) {
        let (data, scratch) = self.take_scalar_znx(n.into(), rank.into());
        (
            GLWESecret {
                data,
                dist: Distribution::NONE,
            },
            scratch,
        )
    }

    /// Allocates a [`GLWESecretTensor`] from scratch space.
    fn take_glwe_secret_tensor(self, n: Degree, rank: Rank) -> (GLWESecretTensor<B::BufMut<'a>>, Self) {
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

    /// Allocates a [`GGLWE`] ciphertext from scratch space.
    fn take_gglwe<A>(self, infos: &A) -> (GGLWE<B::BufMut<'a>>, Self)
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
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data,
            },
            scratch,
        )
    }

    /// Allocates a [`GGLWEPrepared`] (DFT-domain GGLWE) from scratch space.
    fn take_gglwe_prepared<A, M>(self, module: &M, infos: &A) -> (GGLWEPrepared<B::BufMut<'a>, B>, Self)
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
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data,
            },
            scratch,
        )
    }

    /// Allocates a [`GGSW`] ciphertext from scratch space.
    fn take_ggsw<A>(self, infos: &A) -> (GGSW<B::BufMut<'a>>, Self)
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
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data,
            },
            scratch,
        )
    }

    /// Allocates a [`GGSWPrepared`] (DFT-domain GGSW) from scratch space.
    fn take_ggsw_prepared<A, M>(self, module: &M, infos: &A) -> (GGSWPrepared<B::BufMut<'a>, B>, Self)
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
                base2k: infos.base2k(),
                dsize: infos.dsize(),
                data,
            },
            scratch,
        )
    }
}

impl<'a, B: Backend> ScratchArenaTakeCore<'a, B> for ScratchArena<'a, B> {}
