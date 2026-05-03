use std::ops::{Deref, DerefMut};

use poulpy_hal::layouts::{
    Backend, ScalarZnx, SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef, VmpPMatReborrowBackendMut,
    VmpPMatReborrowBackendRef, mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut, vec_znx_backend_mut_from_mut,
    vec_znx_backend_ref_from_mut,
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, GGLWE, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEPrepared, GGLWEPreparedBackendMut,
        GGLWEPreparedBackendRef, GGLWEPreparedToBackendMut, GGLWEPreparedToBackendRef, GGLWEToBackendMut, GGLWEToBackendRef,
        GGSW, GGSWBackendMut, GGSWBackendRef, GGSWInfos, GGSWPrepared, GGSWPreparedBackendMut, GGSWPreparedBackendRef,
        GGSWPreparedToBackendMut, GGSWPreparedToBackendRef, GGSWToBackendMut, GGSWToBackendRef, GLWE, GLWEBackendMut,
        GLWEBackendRef, GLWEInfos, GLWEPlaintext, GLWESecret, GLWESecretBackendMut, GLWESecretBackendRef, GLWESecretPrepared,
        GLWESecretPreparedBackendMut, GLWESecretPreparedBackendRef, GLWESecretPreparedToBackendMut,
        GLWESecretPreparedToBackendRef, GLWESecretTensor, GLWESecretToBackendMut, GLWESecretToBackendRef, GLWETensor,
        GLWEToBackendMut, GLWEToBackendRef, LWE, LWEBackendMut, LWEBackendRef, LWEInfos, LWEPlaintext, LWEPlaintextBackendMut,
        LWEPlaintextBackendRef, LWEPlaintextToBackendMut, LWEPlaintextToBackendRef, LWEToBackendMut, LWEToBackendRef, Rank,
        SetGGLWEInfos, SetLWEInfos,
    },
};

macro_rules! scratch_view {
    ($name:ident, $inner:ty) => {
        pub struct $name<'a, BE: Backend + 'a> {
            inner: $inner,
        }

        impl<'a, BE: Backend + 'a> $name<'a, BE> {
            pub fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn into_inner(self) -> $inner {
                self.inner
            }
        }

        impl<'a, BE: Backend + 'a> Deref for $name<'a, BE> {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl<'a, BE: Backend + 'a> DerefMut for $name<'a, BE> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }

        impl<'a, BE: Backend + 'a> LWEInfos for $name<'a, BE> {
            fn base2k(&self) -> Base2K {
                self.inner.base2k()
            }

            fn n(&self) -> crate::layouts::Degree {
                self.inner.n()
            }

            fn size(&self) -> usize {
                self.inner.size()
            }
        }
    };
}

scratch_view!(LWEScratchMut, LWE<BE::BufMut<'a>>);
scratch_view!(LWEPlaintextScratchMut, LWEPlaintext<BE::BufMut<'a>>);
scratch_view!(GLWEScratchMut, GLWE<BE::BufMut<'a>>);
scratch_view!(GLWEPlaintextScratchMut, GLWEPlaintext<BE::BufMut<'a>>);
scratch_view!(GLWETensorScratchMut, GLWETensor<BE::BufMut<'a>>);
scratch_view!(GLWESecretScratchMut, GLWESecret<BE::BufMut<'a>>);
scratch_view!(GLWESecretTensorScratchMut, GLWESecretTensor<BE::BufMut<'a>>);
scratch_view!(GLWESecretPreparedScratchMut, GLWESecretPrepared<BE::BufMut<'a>, BE>);
scratch_view!(GGLWEScratchMut, GGLWE<BE::BufMut<'a>>);
scratch_view!(GGLWEPreparedScratchMut, GGLWEPrepared<BE::BufMut<'a>, BE>);
scratch_view!(GGSWScratchMut, GGSW<BE::BufMut<'a>>);
scratch_view!(GGSWPreparedScratchMut, GGSWPrepared<BE::BufMut<'a>, BE>);

macro_rules! impl_set_lwe_infos {
    ($name:ident) => {
        impl<'a, BE: Backend + 'a> SetLWEInfos for $name<'a, BE> {
            fn set_base2k(&mut self, base2k: Base2K) {
                self.inner.set_base2k(base2k);
            }
        }
    };
}

impl_set_lwe_infos!(LWEScratchMut);
impl_set_lwe_infos!(GLWEScratchMut);
impl_set_lwe_infos!(GLWEPlaintextScratchMut);

impl<'a, BE: Backend + 'a> SetLWEInfos for LWEPlaintextScratchMut<'a, BE> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.inner.base2k = base2k;
    }
}

macro_rules! impl_glwe_infos {
    ($name:ident) => {
        impl<'a, BE: Backend + 'a> GLWEInfos for $name<'a, BE> {
            fn rank(&self) -> Rank {
                self.inner.rank()
            }
        }
    };
}

impl_glwe_infos!(GLWEScratchMut);
impl_glwe_infos!(GLWEPlaintextScratchMut);
impl_glwe_infos!(GLWETensorScratchMut);
impl_glwe_infos!(GLWESecretScratchMut);
impl_glwe_infos!(GLWESecretTensorScratchMut);
impl_glwe_infos!(GLWESecretPreparedScratchMut);
impl_glwe_infos!(GGLWEScratchMut);
impl_glwe_infos!(GGLWEPreparedScratchMut);
impl_glwe_infos!(GGSWScratchMut);
impl_glwe_infos!(GGSWPreparedScratchMut);

macro_rules! impl_dist {
    ($name:ident) => {
        impl<'a, BE: Backend + 'a> GetDistribution for $name<'a, BE> {
            fn dist(&self) -> &Distribution {
                self.inner.dist()
            }
        }

        impl<'a, BE: Backend + 'a> GetDistributionMut for $name<'a, BE> {
            fn dist_mut(&mut self) -> &mut Distribution {
                self.inner.dist_mut()
            }
        }
    };
}

impl_dist!(GLWESecretTensorScratchMut);
impl_dist!(GLWESecretPreparedScratchMut);

impl<'a, BE: Backend + 'a> GetDistribution for GLWESecretScratchMut<'a, BE> {
    fn dist(&self) -> &Distribution {
        self.inner.dist()
    }
}

impl<'a, BE: Backend + 'a> GGLWEInfos for GGLWEScratchMut<'a, BE> {
    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.inner.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.inner.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.inner.rank_out()
    }
}

impl<'a, BE: Backend + 'a> GGLWEInfos for GGLWEPreparedScratchMut<'a, BE> {
    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.inner.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.inner.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.inner.rank_out()
    }
}

impl<'a, BE: Backend + 'a> SetGGLWEInfos for GGLWEScratchMut<'a, BE> {
    fn set_dsize(&mut self, dsize: usize) {
        self.inner.dsize = dsize.into();
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWScratchMut<'a, BE> {
    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.inner.dsize()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWPreparedScratchMut<'a, BE> {
    fn dnum(&self) -> crate::layouts::Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> crate::layouts::Dsize {
        self.inner.dsize()
    }
}

impl<'a, BE: Backend + 'a> LWEToBackendRef<BE> for LWEScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> LWEBackendRef<'_, BE> {
        LWE {
            base2k: self.inner.base2k,
            data: vec_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        }
    }
}

impl<'a, BE: Backend + 'a> LWEToBackendMut<BE> for LWEScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> LWEBackendMut<'_, BE> {
        LWE {
            base2k: self.inner.base2k,
            data: vec_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        }
    }
}

impl<'a, BE: Backend + 'a> LWEPlaintextToBackendRef<BE> for LWEPlaintextScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE> {
        LWEPlaintext {
            base2k: self.inner.base2k,
            data: vec_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        }
    }
}

impl<'a, BE: Backend + 'a> LWEPlaintextToBackendMut<BE> for LWEPlaintextScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE> {
        LWEPlaintext {
            base2k: self.inner.base2k,
            data: vec_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        }
    }
}

macro_rules! impl_glwe_to_backend {
    ($name:ident) => {
        impl<'a, BE: Backend + 'a> GLWEToBackendRef<BE> for $name<'a, BE> {
            fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
                GLWE {
                    base2k: self.inner.base2k,
                    data: vec_znx_backend_ref_from_mut::<BE>(&self.inner.data),
                }
            }
        }

        impl<'a, BE: Backend + 'a> GLWEToBackendMut<BE> for $name<'a, BE> {
            fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
                GLWE {
                    base2k: self.inner.base2k,
                    data: vec_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
                }
            }
        }
    };
}

impl_glwe_to_backend!(GLWEScratchMut);
impl_glwe_to_backend!(GLWEPlaintextScratchMut);
impl_glwe_to_backend!(GLWETensorScratchMut);

impl<'a, BE: Backend + 'a> GLWESecretToBackendRef<BE> for GLWESecretScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            dist: self.inner.dist,
            data: ScalarZnx::from_data(
                BE::view_ref_mut(&self.inner.data.data),
                self.inner.data.n(),
                self.inner.data.cols(),
            ),
        }
    }
}

impl<'a, BE: Backend + 'a> GLWESecretToBackendMut<BE> for GLWESecretScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE> {
        let n = self.inner.data.n();
        let cols = self.inner.data.cols();
        GLWESecret {
            dist: self.inner.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.inner.data.data), n, cols),
        }
    }
}

impl<'a, BE: Backend + 'a> GLWESecretToBackendRef<BE> for GLWESecretTensorScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            dist: self.inner.dist,
            data: ScalarZnx::from_data(
                BE::view_ref_mut(&self.inner.data.data),
                self.inner.data.n(),
                self.inner.data.cols(),
            ),
        }
    }
}

impl<'a, BE: Backend + 'a> GLWESecretToBackendMut<BE> for GLWESecretTensorScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE> {
        let n = self.inner.data.n();
        let cols = self.inner.data.cols();
        GLWESecret {
            dist: self.inner.dist,
            data: ScalarZnx::from_data(BE::view_mut_ref(&mut self.inner.data.data), n, cols),
        }
    }
}

impl<'a, BE: Backend + 'a> GLWESecretPreparedToBackendRef<BE> for GLWESecretPreparedScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> GLWESecretPreparedBackendRef<'_, BE> {
        GLWESecretPrepared {
            dist: self.inner.dist,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<'a, BE: Backend + 'a> GLWESecretPreparedToBackendMut<BE> for GLWESecretPreparedScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GLWESecretPreparedBackendMut<'_, BE> {
        GLWESecretPrepared {
            dist: self.inner.dist,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}

impl<'a, BE: Backend + 'a> GGLWEToBackendRef<BE> for GGLWEScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWE {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        }
    }
}

impl<'a, BE: Backend + 'a> GGLWEToBackendMut<BE> for GGLWEScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWE {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        }
    }
}

impl<'a, BE: Backend + 'a> GGLWEPreparedToBackendRef<BE> for GGLWEPreparedScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> GGLWEPreparedBackendRef<'_, BE> {
        GGLWEPrepared {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<'a, BE: Backend + 'a> GGLWEPreparedToBackendMut<BE> for GGLWEPreparedScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGLWEPreparedBackendMut<'_, BE> {
        GGLWEPrepared {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendRef<BE> for GGSWScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSW {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        }
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendMut<BE> for GGSWScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        GGSW {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        }
    }
}

impl<'a, BE: Backend + 'a> GGSWPreparedToBackendRef<BE> for GGSWPreparedScratchMut<'a, BE> {
    fn to_backend_ref(&self) -> GGSWPreparedBackendRef<'_, BE> {
        GGSWPrepared {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: self.inner.data.reborrow_backend_ref(),
        }
    }
}

impl<'a, BE: Backend + 'a> GGSWPreparedToBackendMut<BE> for GGSWPreparedScratchMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGSWPreparedBackendMut<'_, BE> {
        GGSWPrepared {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: self.inner.data.reborrow_backend_mut(),
        }
    }
}
