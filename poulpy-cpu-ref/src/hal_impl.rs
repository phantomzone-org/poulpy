use crate::{
    FFT64Ref, NTT120Ref,
    hal_defaults::{
        FFT64ConvolutionDefaults, FFT64ModuleDefaults, FFT64SvpDefaults, FFT64VecZnxBigDefaults, FFT64VecZnxDftDefaults,
        FFT64VmpDefaults, HalVecZnxDefaults, NTT120ConvolutionDefaults, NTT120ModuleDefaults, NTT120SvpDefaults,
        NTT120VecZnxBigDefaults, NTT120VecZnxDftDefaults, NTT120VmpDefaults,
    },
};
use poulpy_hal::{
    api::{VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, CnvPVecLToMut, CnvPVecLToRef, CnvPVecRToMut, CnvPVecRToRef, Data, Module, NoiseInfos,
        VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, ZnxInfos,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

#[macro_use]
mod vec_znx;
#[macro_use]
mod module;
#[macro_use]
mod vmp;
#[macro_use]
mod convolution;
#[macro_use]
mod vec_znx_big;
#[macro_use]
mod svp;
#[macro_use]
mod vec_znx_dft;
#[cfg(test)]
pub(crate) mod delegating_backend;

unsafe impl HalVecZnxImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<FFT64Ref> for FFT64Ref {
    hal_impl_module!(FFT64ModuleDefaults);
}

unsafe impl HalVmpImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vmp!(FFT64VmpDefaults);
}

unsafe impl HalConvolutionImpl<FFT64Ref> for FFT64Ref {
    hal_impl_convolution!(FFT64ConvolutionDefaults);
}

unsafe impl HalVecZnxBigImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx_big!(FFT64VecZnxBigDefaults);
}

unsafe impl HalSvpImpl<FFT64Ref> for FFT64Ref {
    hal_impl_svp!(FFT64SvpDefaults);
}

unsafe impl HalVecZnxDftImpl<FFT64Ref> for FFT64Ref {
    hal_impl_vec_znx_dft!(FFT64VecZnxDftDefaults);
}

unsafe impl HalVecZnxImpl<NTT120Ref> for NTT120Ref {
    hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<NTT120Ref> for NTT120Ref {
    hal_impl_module!(NTT120ModuleDefaults);
}

unsafe impl HalVmpImpl<NTT120Ref> for NTT120Ref {
    hal_impl_vmp!(NTT120VmpDefaults);
}

unsafe impl HalConvolutionImpl<NTT120Ref> for NTT120Ref {
    hal_impl_convolution!(NTT120ConvolutionDefaults);
}

unsafe impl HalVecZnxBigImpl<NTT120Ref> for NTT120Ref {
    hal_impl_vec_znx_big!(NTT120VecZnxBigDefaults);
}

unsafe impl HalSvpImpl<NTT120Ref> for NTT120Ref {
    hal_impl_svp!(NTT120SvpDefaults);
}

unsafe impl HalVecZnxDftImpl<NTT120Ref> for NTT120Ref {
    hal_impl_vec_znx_dft!(NTT120VecZnxDftDefaults);
}
