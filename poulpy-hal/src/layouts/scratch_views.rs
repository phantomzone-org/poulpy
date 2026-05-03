use std::ops::{Deref, DerefMut};

use crate::layouts::{
    Backend, CnvPVecL, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecLReborrowBackendMut, CnvPVecLReborrowBackendRef,
    CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecR, CnvPVecRBackendMut, CnvPVecRBackendRef, CnvPVecRReborrowBackendMut,
    CnvPVecRReborrowBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, MatZnx, MatZnxBackendMut, MatZnxBackendRef,
    MatZnxToBackendMut, MatZnxToBackendRef, ScalarZnx, ScalarZnxBackendMut, ScalarZnxBackendRef, ScalarZnxToBackendMut,
    ScalarZnxToBackendRef, SvpPPol, SvpPPolBackendMut, SvpPPolBackendRef, SvpPPolReborrowBackendMut, SvpPPolReborrowBackendRef,
    SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut,
    VecZnxBigBackendRef, VecZnxBigReborrowBackendMut, VecZnxBigReborrowBackendRef, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
    VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftReborrowBackendMut, VecZnxDftReborrowBackendRef,
    VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef, VecZnxToBackendMut,
    VecZnxToBackendRef, VmpPMat, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatReborrowBackendMut, VmpPMatReborrowBackendRef,
    VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxInfos, mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut,
};

macro_rules! scratch_view {
    ($name:ident, $inner:ty) => {
        pub struct $name<'a, B: Backend + 'a> {
            inner: $inner,
        }

        impl<'a, B: Backend + 'a> $name<'a, B> {
            pub fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn into_inner(self) -> $inner {
                self.inner
            }
        }

        impl<'a, B: Backend + 'a> Deref for $name<'a, B> {
            type Target = $inner;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl<'a, B: Backend + 'a> DerefMut for $name<'a, B> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.inner
            }
        }

        impl<'a, B: Backend + 'a> ZnxInfos for $name<'a, B> {
            fn cols(&self) -> usize {
                self.inner.cols()
            }

            fn rows(&self) -> usize {
                self.inner.rows()
            }

            fn n(&self) -> usize {
                self.inner.n()
            }

            fn size(&self) -> usize {
                self.inner.size()
            }

            fn poly_count(&self) -> usize {
                self.inner.poly_count()
            }
        }
    };
}

scratch_view!(CnvPVecLScratchMut, CnvPVecL<B::BufMut<'a>, B>);
scratch_view!(CnvPVecRScratchMut, CnvPVecR<B::BufMut<'a>, B>);
scratch_view!(MatZnxScratchMut, MatZnx<B::BufMut<'a>>);
scratch_view!(ScalarZnxScratchMut, ScalarZnx<B::BufMut<'a>>);
scratch_view!(SvpPPolScratchMut, SvpPPol<B::BufMut<'a>, B>);
scratch_view!(VecZnxScratchMut, VecZnx<B::BufMut<'a>>);
scratch_view!(VecZnxBigScratchMut, VecZnxBig<B::BufMut<'a>, B>);
scratch_view!(VecZnxDftScratchMut, VecZnxDft<B::BufMut<'a>, B>);
scratch_view!(VmpPMatScratchMut, VmpPMat<B::BufMut<'a>, B>);

impl<'a, B: Backend + 'a> CnvPVecLToBackendRef<B> for CnvPVecLScratchMut<'a, B> {
    fn to_backend_ref(&self) -> CnvPVecLBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> CnvPVecLToBackendMut<B> for CnvPVecLScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> CnvPVecLBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> CnvPVecRToBackendRef<B> for CnvPVecRScratchMut<'a, B> {
    fn to_backend_ref(&self) -> CnvPVecRBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> CnvPVecRToBackendMut<B> for CnvPVecRScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> CnvPVecRBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> MatZnxToBackendRef<B> for MatZnxScratchMut<'a, B> {
    fn to_backend_ref(&self) -> MatZnxBackendRef<'_, B> {
        mat_znx_backend_ref_from_mut::<B>(&self.inner)
    }
}

impl<'a, B: Backend + 'a> MatZnxToBackendMut<B> for MatZnxScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> MatZnxBackendMut<'_, B> {
        mat_znx_backend_mut_from_mut::<B>(&mut self.inner)
    }
}

impl<'a, B: Backend + 'a> ScalarZnxToBackendRef<B> for ScalarZnxScratchMut<'a, B> {
    fn to_backend_ref(&self) -> ScalarZnxBackendRef<'_, B> {
        ScalarZnx::from_data(B::view_ref_mut(&self.inner.data), self.inner.n(), self.inner.cols())
    }
}

impl<'a, B: Backend + 'a> ScalarZnxToBackendMut<B> for ScalarZnxScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> ScalarZnxBackendMut<'_, B> {
        let n = self.inner.n();
        let cols = self.inner.cols();
        ScalarZnx::from_data(B::view_mut_ref(&mut self.inner.data), n, cols)
    }
}

impl<'a, B: Backend + 'a> SvpPPolToBackendRef<B> for SvpPPolScratchMut<'a, B> {
    fn to_backend_ref(&self) -> SvpPPolBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> SvpPPolToBackendMut<B> for SvpPPolScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> SvpPPolBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> VecZnxToBackendRef<B> for VecZnxScratchMut<'a, B> {
    fn to_backend_ref(&self) -> VecZnxBackendRef<'_, B> {
        <VecZnx<B::BufMut<'a>> as VecZnxReborrowBackendRef<B>>::reborrow_backend_ref(&self.inner)
    }
}

impl<'a, B: Backend + 'a> VecZnxToBackendMut<B> for VecZnxScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> VecZnxBackendMut<'_, B> {
        <VecZnx<B::BufMut<'a>> as VecZnxReborrowBackendMut<B>>::reborrow_backend_mut(&mut self.inner)
    }
}

impl<'a, B: Backend + 'a> VecZnxBigToBackendRef<B> for VecZnxBigScratchMut<'a, B> {
    fn to_backend_ref(&self) -> VecZnxBigBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> VecZnxBigToBackendMut<B> for VecZnxBigScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> VecZnxBigBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> VecZnxDftToBackendRef<B> for VecZnxDftScratchMut<'a, B> {
    fn to_backend_ref(&self) -> VecZnxDftBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> VecZnxDftToBackendMut<B> for VecZnxDftScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> VecZnxDftBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}

impl<'a, B: Backend + 'a> VmpPMatToBackendRef<B> for VmpPMatScratchMut<'a, B> {
    fn to_backend_ref(&self) -> VmpPMatBackendRef<'_, B> {
        self.inner.reborrow_backend_ref()
    }
}

impl<'a, B: Backend + 'a> VmpPMatToBackendMut<B> for VmpPMatScratchMut<'a, B> {
    fn to_backend_mut(&mut self) -> VmpPMatBackendMut<'_, B> {
        self.inner.reborrow_backend_mut()
    }
}
