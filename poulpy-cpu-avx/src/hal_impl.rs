use crate::{FFT64Avx, NTT120Avx};
use poulpy_cpu_ref::hal_defaults::{
    FFT64ConvolutionDefaults, FFT64ModuleDefaults, FFT64SvpDefaults, FFT64VecZnxBigDefaults, FFT64VecZnxDftDefaults,
    FFT64VmpDefaults, HalVecZnxDefaults, NTT120ConvolutionDefaults, NTT120ModuleDefaults, NTT120SvpDefaults,
    NTT120VecZnxBigDefaults, NTT120VecZnxDftDefaults, NTT120VmpDefaults,
};
use poulpy_hal::{
    api::{VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{Backend, Module, NoiseInfos, VecZnxBackendMut, VecZnxBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, ZnxInfos},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

unsafe impl HalVecZnxImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_module!(FFT64ModuleDefaults);
}

unsafe impl HalVmpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vmp!(FFT64VmpDefaults);
}

unsafe impl HalConvolutionImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_convolution!(FFT64ConvolutionDefaults);
}

unsafe impl HalVecZnxBigImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(FFT64VecZnxBigDefaults);
}

unsafe impl HalSvpImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_svp!(FFT64SvpDefaults);
}

unsafe impl HalVecZnxDftImpl<FFT64Avx> for FFT64Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefaults);
}

unsafe impl HalVecZnxImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_module!(NTT120ModuleDefaults);
}

unsafe impl HalVmpImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vmp!(NTT120VmpDefaults);
}

unsafe impl HalConvolutionImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_convolution!(NTT120ConvolutionDefaults);
}

unsafe impl HalVecZnxBigImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_big!(NTT120VecZnxBigDefaults);
}

unsafe impl HalSvpImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_svp!(NTT120SvpDefaults);
}

unsafe impl HalVecZnxDftImpl<NTT120Avx> for NTT120Avx {
    poulpy_cpu_ref::hal_impl_vec_znx_dft!(NTT120VecZnxDftDefaults);
}
