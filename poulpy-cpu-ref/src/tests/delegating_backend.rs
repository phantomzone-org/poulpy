use poulpy_core::{
    api::{GLWEMulXpMinusOne, GLWERotate},
    layouts::{
        Base2K, Degree, GLWE, GLWEBackendMut, GLWEBackendRef, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, Rank,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::ModuleNew,
    layouts::{FillUniform, Module},
    source::Source,
};

use crate::{FFT64Ref, hal_impl::delegating_backend::DelegatingFFT64Ref};

fn sample_glwe() -> GLWE<Vec<u8>> {
    let layout = GLWELayout {
        n: Degree(256),
        base2k: Base2K(17),
        k: TorusPrecision(50),
        rank: Rank(2),
    };
    let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&layout);
    let mut source = Source::new([7u8; 32]);
    ct.fill_uniform(40, &mut source);
    ct
}

#[test]
fn delegating_backend_manual_family_matches_fft64_ref() {
    let module_delegating: Module<DelegatingFFT64Ref> = Module::new(256);
    let module_ref: Module<FFT64Ref> = Module::new(256);

    let input = sample_glwe();
    let mut delegating_out = GLWE::alloc_from_infos(&input);
    let mut ref_out = GLWE::alloc_from_infos(&input);

    module_delegating.glwe_mul_xp_minus_one(-7, &mut delegating_out, &input);
    module_ref.glwe_mul_xp_minus_one(-7, &mut ref_out, &input);

    assert_eq!(delegating_out, ref_out);
}

#[test]
fn delegating_backend_delegated_family_matches_fft64_ref() {
    let module_delegating: Module<DelegatingFFT64Ref> = Module::new(256);
    let module_ref: Module<FFT64Ref> = Module::new(256);

    let input = sample_glwe();
    let mut delegating_out = GLWE::alloc_from_infos(&input);
    let mut ref_out = GLWE::alloc_from_infos(&input);

    let input_delegating: GLWEBackendRef<'_, DelegatingFFT64Ref> =
        <GLWE<Vec<u8>> as GLWEToBackendRef<DelegatingFFT64Ref>>::to_backend_ref(&input);
    let mut delegating_out_backend: GLWEBackendMut<'_, DelegatingFFT64Ref> =
        <GLWE<Vec<u8>> as GLWEToBackendMut<DelegatingFFT64Ref>>::to_backend_mut(&mut delegating_out);
    module_delegating.glwe_rotate(11, &mut delegating_out_backend, &input_delegating);

    let input_ref: GLWEBackendRef<'_, FFT64Ref> = <GLWE<Vec<u8>> as GLWEToBackendRef<FFT64Ref>>::to_backend_ref(&input);
    let mut ref_out_backend: GLWEBackendMut<'_, FFT64Ref> =
        <GLWE<Vec<u8>> as GLWEToBackendMut<FFT64Ref>>::to_backend_mut(&mut ref_out);
    module_ref.glwe_rotate(11, &mut ref_out_backend, &input_ref);

    assert_eq!(delegating_out, ref_out);
}
