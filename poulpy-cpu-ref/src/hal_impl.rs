use crate::{
    FFT64Ref, NTT120Ref, NTTIfmaRef,
    hal_defaults::{
        FFT64ConvolutionDefaults, FFT64ModuleDefaults, FFT64SvpDefaults, FFT64VecZnxBigDefaults, FFT64VecZnxDftDefaults,
        FFT64VmpDefaults, HalScratchDefaults, HalVecZnxDefaults, NTT120ConvolutionDefaults, NTT120IfmaConvolutionDefaults,
        NTT120IfmaModuleDefaults, NTT120IfmaSvpDefaults, NTT120IfmaVecZnxBigDefaults, NTT120IfmaVecZnxDftDefaults,
        NTT120IfmaVmpDefaults, NTT120ModuleDefaults, NTT120SvpDefaults, NTT120VecZnxBigDefaults, NTT120VecZnxDftDefaults,
        NTT120VmpDefaults,
    },
};
use poulpy_hal::{
    api::{ScratchTakeBasic, VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, CnvPVecLToMut, CnvPVecLToRef, CnvPVecRToMut, CnvPVecRToRef, Data, MatZnxToRef, Module, NoiseInfos,
        ScalarZnxToRef, Scratch, ScratchOwned, SvpPPolToMut, SvpPPolToRef, VecZnx, VecZnxBig, VecZnxBigToMut, VecZnxBigToRef,
        VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToMut, VecZnxToRef, VmpPMat, VmpPMatToMut, VmpPMatToRef, ZnxInfos,
    },
    oep::HalImpl,
    source::Source,
};

#[macro_use]
mod scratch;
#[macro_use]
mod vec_znx;
#[macro_use]
mod family_common;
#[macro_use]
mod module_fft64;
#[macro_use]
mod module_ntt120;
#[macro_use]
mod module_ntt_ifma;
#[macro_use]
mod vmp_fft64;
#[macro_use]
mod vmp_ntt120;
#[macro_use]
mod vmp_ntt_ifma;
#[macro_use]
mod convolution_fft64;
#[macro_use]
mod convolution_ntt120;
#[macro_use]
mod convolution_ntt_ifma;
#[macro_use]
mod vec_znx_big_fft64;
#[macro_use]
mod vec_znx_big_ntt120;
#[macro_use]
mod vec_znx_big_ntt_ifma;
#[macro_use]
mod svp_fft64;
#[macro_use]
mod svp_ntt120;
#[macro_use]
mod svp_ntt_ifma;
#[macro_use]
mod vec_znx_dft_fft64;
#[macro_use]
mod vec_znx_dft_ntt120;
#[macro_use]
mod vec_znx_dft_ntt_ifma;

unsafe impl HalImpl<FFT64Ref> for FFT64Ref {
    hal_impl_scratch!();
    hal_impl_vec_znx!();
    hal_impl_family_common!();
    hal_impl_module_fft64!();
    hal_impl_vmp_fft64!();
    hal_impl_convolution_fft64!();
    hal_impl_vec_znx_big_fft64!();
    hal_impl_svp_fft64!();
    hal_impl_vec_znx_dft_fft64!();
}

unsafe impl HalImpl<NTT120Ref> for NTT120Ref {
    hal_impl_scratch!();
    hal_impl_vec_znx!();
    hal_impl_family_common!();
    hal_impl_module_ntt120!();
    hal_impl_vmp_ntt120!();
    hal_impl_convolution_ntt120!();
    hal_impl_vec_znx_big_ntt120!();
    hal_impl_svp_ntt120!();
    hal_impl_vec_znx_dft_ntt120!();
}

unsafe impl HalImpl<NTTIfmaRef> for NTTIfmaRef {
    hal_impl_scratch!();
    hal_impl_vec_znx!();
    hal_impl_family_common!();
    hal_impl_module_ntt_ifma!();
    hal_impl_vmp_ntt_ifma!();
    hal_impl_convolution_ntt_ifma!();
    hal_impl_vec_znx_big_ntt_ifma!();
    hal_impl_svp_ntt_ifma!();
    hal_impl_vec_znx_dft_ntt_ifma!();
}
