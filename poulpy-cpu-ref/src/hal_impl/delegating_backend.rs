use poulpy_core::{
    api::{GLWERotate, ModuleTransfer},
    layouts::{GLWE, GLWEBackendMut, GLWEBackendRef, GLWEInfos, LWEInfos},
    oep::{GLWEMulXpMinusOneImpl, GLWERotateImpl},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftApply, VecZnxDftZero, VmpApplyDftToDft},
    layouts::{
        Backend, CnvPVecLToMut, CnvPVecLToRef, CnvPVecRToMut, CnvPVecRToRef, Data, MatZnxToRef, Module, NoiseInfos, ScratchArena,
        ScratchOwned, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut,
        VecZnxDftToRef, VecZnxToMut, VecZnxToRef, ZnxInfos,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use crate::{
    FFT64Ref,
    hal_defaults::{
        FFT64ConvolutionDefaults, FFT64ModuleDefaults, FFT64SvpDefaults, FFT64VecZnxBigDefaults, FFT64VecZnxDftDefaults,
        FFT64VmpDefaults, HalVecZnxDefaults,
    },
    reference::{
        fft64::{
            convolution::I64Ops,
            reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable},
            reim4::{Reim4BlkMatVec, Reim4Convolution},
        },
        vec_znx::{vec_znx_mul_xp_minus_one, vec_znx_mul_xp_minus_one_inplace},
        znx::{
            ZnxAdd, ZnxAddInplace, ZnxAutomorphism, ZnxCopy, ZnxExtractDigitAddMul, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo,
            ZnxMulPowerOfTwoInplace, ZnxNegate, ZnxNegateInplace, ZnxNormalizeDigit, ZnxNormalizeFinalStep,
            ZnxNormalizeFinalStepInplace, ZnxNormalizeFinalStepSub, ZnxNormalizeFirstStep, ZnxNormalizeFirstStepCarryOnly,
            ZnxNormalizeFirstStepInplace, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepCarryOnly, ZnxNormalizeMiddleStepInplace,
            ZnxNormalizeMiddleStepSub, ZnxRef, ZnxRotate, ZnxSub, ZnxSubInplace, ZnxSubNegateInplace, ZnxSwitchRing, ZnxZero,
        },
    },
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DelegatingFFT64Ref;

poulpy_hal::impl_backend_from!(DelegatingFFT64Ref, FFT64Ref);

macro_rules! impl_forward_znx_trait {
    ($trait_name:ident, $method:ident($($arg:ident : $ty:ty),*)) => {
        impl $trait_name for DelegatingFFT64Ref {
            #[inline(always)]
            fn $method($($arg : $ty),*) {
                <FFT64Ref as $trait_name>::$method($($arg),*)
            }
        }
    };
}

macro_rules! impl_forward_znx_const_trait {
    ($trait_name:ident, $method:ident($($arg:ident : $ty:ty),*)) => {
        impl $trait_name for DelegatingFFT64Ref {
            #[inline(always)]
            fn $method<const OVERWRITE: bool>($($arg : $ty),*) {
                <FFT64Ref as $trait_name>::$method::<OVERWRITE>($($arg),*)
            }
        }
    };
}

impl_forward_znx_trait!(ZnxAdd, znx_add(res: &mut [i64], a: &[i64], b: &[i64]));
impl_forward_znx_trait!(ZnxAddInplace, znx_add_inplace(res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxSub, znx_sub(res: &mut [i64], a: &[i64], b: &[i64]));
impl_forward_znx_trait!(ZnxSubInplace, znx_sub_inplace(res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxSubNegateInplace, znx_sub_negate_inplace(res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxMulAddPowerOfTwo, znx_muladd_power_of_two(k: i64, res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxMulPowerOfTwo, znx_mul_power_of_two(k: i64, res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxMulPowerOfTwoInplace, znx_mul_power_of_two_inplace(k: i64, res: &mut [i64]));
impl_forward_znx_trait!(ZnxAutomorphism, znx_automorphism(p: i64, res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxCopy, znx_copy(res: &mut [i64], a: &[i64]));
impl_forward_znx_trait!(ZnxNegate, znx_negate(res: &mut [i64], src: &[i64]));
impl_forward_znx_trait!(ZnxNegateInplace, znx_negate_inplace(res: &mut [i64]));
impl_forward_znx_trait!(ZnxRotate, znx_rotate(p: i64, res: &mut [i64], src: &[i64]));
impl_forward_znx_trait!(ZnxZero, znx_zero(res: &mut [i64]));
impl_forward_znx_trait!(ZnxSwitchRing, znx_switch_ring(res: &mut [i64], a: &[i64]));
impl_forward_znx_const_trait!(
    ZnxNormalizeFirstStep,
    znx_normalize_first_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_const_trait!(
    ZnxNormalizeMiddleStep,
    znx_normalize_middle_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_const_trait!(
    ZnxNormalizeFinalStep,
    znx_normalize_final_step(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeFirstStepCarryOnly,
    znx_normalize_first_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeFirstStepInplace,
    znx_normalize_first_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeMiddleStepCarryOnly,
    znx_normalize_middle_step_carry_only(base2k: usize, lsh: usize, x: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeMiddleStepInplace,
    znx_normalize_middle_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeMiddleStepSub,
    znx_normalize_middle_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeFinalStepSub,
    znx_normalize_final_step_sub(base2k: usize, lsh: usize, x: &mut [i64], a: &[i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxNormalizeFinalStepInplace,
    znx_normalize_final_step_inplace(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64])
);
impl_forward_znx_trait!(
    ZnxExtractDigitAddMul,
    znx_extract_digit_addmul(base2k: usize, lsh: usize, res: &mut [i64], src: &mut [i64])
);
impl_forward_znx_trait!(ZnxNormalizeDigit, znx_normalize_digit(base2k: usize, res: &mut [i64], src: &mut [i64]));

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for DelegatingFFT64Ref {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        <FFT64Ref as ReimFFTExecute<ReimFFTTable<f64>, f64>>::reim_dft_execute(table, data)
    }
}

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for DelegatingFFT64Ref {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        <FFT64Ref as ReimFFTExecute<ReimIFFTTable<f64>, f64>>::reim_dft_execute(table, data)
    }
}

impl ReimArith for DelegatingFFT64Ref {}
impl Reim4BlkMatVec for DelegatingFFT64Ref {}
impl Reim4Convolution for DelegatingFFT64Ref {}
impl I64Ops for DelegatingFFT64Ref {}

unsafe impl HalVecZnxImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_vec_znx!();
}

unsafe impl HalModuleImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_module!(FFT64ModuleDefaults);
}

unsafe impl HalVmpImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_vmp!(FFT64VmpDefaults);
}

unsafe impl HalConvolutionImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_convolution!(FFT64ConvolutionDefaults);
}

unsafe impl HalVecZnxBigImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_vec_znx_big!(FFT64VecZnxBigDefaults);
}

unsafe impl HalSvpImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_svp!(FFT64SvpDefaults);
}

unsafe impl HalVecZnxDftImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    crate::hal_impl_vec_znx_dft!(FFT64VecZnxDftDefaults);
}

unsafe impl GLWEMulXpMinusOneImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    fn glwe_mul_xp_minus_one<R, A>(module: &Module<DelegatingFFT64Ref>, k: i64, res: &mut R, a: &A)
    where
        R: poulpy_core::layouts::GLWEToMut,
        A: poulpy_core::layouts::GLWEToRef,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();
        let a: &GLWE<&[u8]> = &a.to_ref();

        assert_eq!(res.n(), module.n() as u32);
        assert_eq!(a.n(), module.n() as u32);
        assert_eq!(res.rank(), a.rank());

        for i in 0..res.rank().as_usize() + 1 {
            vec_znx_mul_xp_minus_one::<_, _, ZnxRef>(k, res.data_mut(), i, a.data(), i);
        }
    }

    fn glwe_mul_xp_minus_one_inplace<'s, R>(
        module: &Module<DelegatingFFT64Ref>,
        k: i64,
        res: &mut R,
        _scratch: &mut ScratchArena<'s, DelegatingFFT64Ref>,
    ) where
        R: poulpy_core::layouts::GLWEToMut,
    {
        let res: &mut GLWE<&mut [u8]> = &mut res.to_mut();

        assert_eq!(res.n(), module.n() as u32);

        let mut tmp = vec![0i64; module.n()];
        for i in 0..res.rank().as_usize() + 1 {
            vec_znx_mul_xp_minus_one_inplace::<_, ZnxRef>(k, res.data_mut(), i, &mut tmp);
        }
    }
}

unsafe impl GLWERotateImpl<DelegatingFFT64Ref> for DelegatingFFT64Ref {
    fn glwe_rotate_tmp_bytes(module: &Module<DelegatingFFT64Ref>) -> usize {
        let delegate: Module<FFT64Ref> = <Module<FFT64Ref> as ModuleNew<FFT64Ref>>::new(module.n() as u64);
        <Module<FFT64Ref> as GLWERotate<FFT64Ref>>::glwe_rotate_tmp_bytes(&delegate)
    }

    fn glwe_rotate<'r, 'a>(
        module: &Module<DelegatingFFT64Ref>,
        k: i64,
        res: &mut GLWEBackendMut<'r, DelegatingFFT64Ref>,
        a: &GLWEBackendRef<'a, DelegatingFFT64Ref>,
    ) {
        let delegate: Module<FFT64Ref> = <Module<FFT64Ref> as ModuleNew<FFT64Ref>>::new(module.n() as u64);
        <Module<FFT64Ref> as GLWERotate<FFT64Ref>>::glwe_rotate(&delegate, k, res, a);
    }

    fn glwe_rotate_inplace<'s, 'r>(
        module: &Module<DelegatingFFT64Ref>,
        k: i64,
        res: &mut GLWEBackendMut<'r, DelegatingFFT64Ref>,
        _scratch: &mut ScratchArena<'s, DelegatingFFT64Ref>,
    ) {
        let delegate: Module<FFT64Ref> = <Module<FFT64Ref> as ModuleNew<FFT64Ref>>::new(module.n() as u64);

        let res_host: GLWE<Vec<u8>> = poulpy_hal::layouts::ToOwnedDeep::to_owned_deep(res);
        let res_src: GLWE<<DelegatingFFT64Ref as Backend>::OwnedBuf> = res_host.reinterpret::<DelegatingFFT64Ref>();
        let mut res_delegate =
            <Module<FFT64Ref> as ModuleTransfer<FFT64Ref>>::upload_glwe::<DelegatingFFT64Ref>(&delegate, &res_src);

        let mut scratch_owned: ScratchOwned<FFT64Ref> = <ScratchOwned<FFT64Ref> as ScratchOwnedAlloc<FFT64Ref>>::alloc(
            <Module<FFT64Ref> as GLWERotate<FFT64Ref>>::glwe_rotate_tmp_bytes(&delegate),
        );
        let mut scratch_delegate = <ScratchOwned<FFT64Ref> as ScratchOwnedBorrow<FFT64Ref>>::borrow(&mut scratch_owned);
        let mut res_delegate_backend: GLWEBackendMut<'_, FFT64Ref> =
            <GLWE<<FFT64Ref as Backend>::OwnedBuf> as poulpy_core::layouts::GLWEToBackendMut<FFT64Ref>>::to_backend_mut(
                &mut res_delegate,
            );

        <Module<FFT64Ref> as GLWERotate<FFT64Ref>>::glwe_rotate_inplace(
            &delegate,
            k,
            &mut res_delegate_backend,
            &mut scratch_delegate,
        );

        let res_back: GLWE<<DelegatingFFT64Ref as Backend>::OwnedBuf> =
            <Module<FFT64Ref> as ModuleTransfer<FFT64Ref>>::download_glwe::<FFT64Ref>(&delegate, &res_delegate)
                .reinterpret::<DelegatingFFT64Ref>();
        let res_back_ref = poulpy_core::layouts::GLWEToRef::to_ref(&res_back);

        let mut bytes = Vec::new();
        poulpy_hal::layouts::WriterTo::write_to(&res_back_ref, &mut bytes)
            .expect("failed to serialize delegated GLWE rotate inplace result");

        let mut cursor = std::io::Cursor::new(bytes);
        poulpy_hal::layouts::ReaderFrom::read_from(res, &mut cursor)
            .expect("failed to write delegated GLWE rotate inplace result back");
    }
}
