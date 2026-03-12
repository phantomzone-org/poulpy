use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_core::{
    GLWEDecrypt, LWEEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Dnum, GLWE, GLWELayout, GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, LWE, LWEInfos, LWELayout,
        LWESecret, TorusPrecision,
    },
};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, FillUniform, Module, Scratch, ScratchOwned},
    source::Source,
};
use poulpy_schemes::bin_fhe::blind_rotation::{
    BlindRotationAlgo, BlindRotationExecute, BlindRotationKey, BlindRotationKeyEncryptSk, BlindRotationKeyFactory,
    BlindRotationKeyLayout, BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, CGGI, LookUpTableLayout, LookupTable,
    LookupTableFactory,
};

pub fn benc_blind_rotate<BE: Backend, BRA: BlindRotationAlgo>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + BlindRotationExecute<BRA, BE>
        + LookupTableFactory
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let group_name: String = format!("blind_rotate::{label}");

    let mut group = c.benchmark_group(group_name);

    let n_glwe: usize = 512;
    let n_lwe: usize = 687;
    let rank: usize = 3;
    let block_size: usize = 3;
    let extension_factor: usize = 1;

    let log_message_modulus: usize = 2;
    let message_modulus: usize = 1 << log_message_modulus;

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 24);

    let module: Module<BE> = Module::<BE>::new(n_glwe as u64);

    let mut source_xs: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);

    let brk_infos: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: n_glwe.into(),
        n_lwe: n_lwe.into(),
        base2k: Base2K(18),
        k: TorusPrecision(36),
        dnum: Dnum(1),
        rank: rank.into(),
    };

    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe.into(),
        base2k: Base2K(18),
        k: TorusPrecision(18),
        rank: rank.into(),
    };

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe.into(),
        k: TorusPrecision(18),
        base2k: Base2K(18),
    };

    let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_glwe_dft: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(&module, &glwe_infos);
    sk_glwe_dft.prepare(&module, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut brk: BlindRotationKey<Vec<u8>, BRA> = BlindRotationKey::<Vec<u8>, BRA>::alloc(&brk_infos);

    brk.encrypt_sk(
        &module,
        &sk_glwe_dft,
        &sk_lwe,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut brk_prepared: BlindRotationKeyPrepared<Vec<u8>, BRA, BE> = BlindRotationKeyPrepared::alloc(&module, &brk);
    brk_prepared.prepare(&module, &brk, scratch.borrow());

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_infos);
    res.data_mut().fill_uniform(glwe_infos.base2k().as_usize(), &mut source_xa);
    let mut lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_infos);
    lwe.data_mut().fill_uniform(lwe_infos.base2k().as_usize(), &mut source_xa);

    let f = |x: i64| -> i64 { 2 * x + 1 };

    let mut f_vec: Vec<i64> = vec![0i64; message_modulus];
    f_vec.iter_mut().enumerate().for_each(|(i, x)| *x = f(i as i64));

    let lut_infos = LookUpTableLayout {
        n: module.n().into(),
        extension_factor,
        k: TorusPrecision(2),
        base2k: Base2K(17),
    };

    let mut lut: LookupTable = LookupTable::alloc(&lut_infos);
    lut.set(&module, &f_vec, log_message_modulus + 1);

    let id: BenchmarkId = BenchmarkId::from_parameter(format!("{} - {}", n_glwe, n_lwe));

    group.bench_with_input(id, &(), |b, _| {
        b.iter(|| {
            brk_prepared.execute(&module, &mut res, &lwe, &lut, scratch.borrow());
            black_box(())
        })
    });
    group.finish();
}

fn bench_blind_rotate_fft64(c: &mut Criterion) {
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    let label = "fft64_avx";
    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    let label = "fft64_ref";
    benc_blind_rotate::<BackendImpl, CGGI>(c, label);
}

criterion_group!(benches, bench_blind_rotate_fft64);
criterion_main!(benches);
