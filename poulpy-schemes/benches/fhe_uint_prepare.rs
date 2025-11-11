use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_backend::{FFT64Avx, FFT64Ref};
use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWENoise, ScratchTakeCore,
    layouts::{GGSWLayout, GLWELayout, GLWESecretPreparedFactory, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use rand::RngCore;

use poulpy_schemes::tfhe::{
    bdd_arithmetic::{
        BDDKeyEncryptSk, BDDKeyPrepared, BDDKeyPreparedFactory, ExecuteBDDCircuit2WTo1W, FheUint, FheUintPrepare,
        FheUintPrepareDebug, FheUintPrepared, FheUintPreparedEncryptSk, FheUintPreparedFactory,
        tests::test_suite::TestContext,
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory, CGGI},
};

pub fn benc_bdd_prepare<BRA: BlindRotationAlgo, BE: Backend>(
    c: &mut Criterion,
    label: &str,
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWENoise<BE>
        + FheUintPreparedFactory<u32, BE>
        + FheUintPreparedEncryptSk<u32, BE>
        + FheUintPrepareDebug<BRA, u32, BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + GGSWNoise<BE>
        + FheUintPrepare<BRA, BE>
        + ExecuteBDDCircuit2WTo1W<u32, BE>
        + GLWEEncryptSk<BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let group_name: String = format!("bdd_prepare::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE: Backend, BRA: BlindRotationAlgo>(test_context: &TestContext<BRA, BE>) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE>
            + GLWESecretPreparedFactory<BE>
            + GLWEDecrypt<BE>
            + GLWENoise<BE>
            + FheUintPreparedFactory<u32, BE>
            + FheUintPreparedEncryptSk<u32, BE>
            + FheUintPrepareDebug<BRA, u32, BE>
            + BDDKeyEncryptSk<BRA, BE>
            + BDDKeyPreparedFactory<BRA, BE>
            + GGSWNoise<BE>
            + FheUintPrepare<BRA, BE>
            + ExecuteBDDCircuit2WTo1W<u32, BE>
            + GLWEEncryptSk<BE>,
        BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        let glwe_infos: GLWELayout = test_context.glwe_infos();
        let ggsw_infos: GGSWLayout = test_context.ggsw_infos();

        let module: &Module<BE> = &test_context.module;
        let sk_glwe_prep: &GLWESecretPrepared<Vec<u8>, BE> = &test_context.sk_glwe;
        let bdd_key_prepared: &BDDKeyPrepared<Vec<u8>, BRA, BE> = &test_context.bdd_key;

        let mut source: Source = Source::new([6u8; 32]);

        let mut source_xa: Source = Source::new([2u8; 32]);
        let mut source_xe: Source = Source::new([3u8; 32]);

        let threads = 1;

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc((1 << 22) * threads);

        // GLWE(value)
        let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&glwe_infos);
        let value: u32 = source.next_u32();
        c_enc.encrypt_sk(
            module,
            value,
            sk_glwe_prep,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        // GGSW(0)
        let mut c_enc_prep: FheUintPrepared<Vec<u8>, u32, BE> =
            FheUintPrepared::<Vec<u8>, u32, BE>::alloc_from_infos(module, &ggsw_infos);

        // GGSW(value)
        move || {
            c_enc_prep.prepare_custom_multi_thread(threads, module, &c_enc, 0, 32, bdd_key_prepared, scratch.borrow());
            black_box(());
        }
    }

    let id: BenchmarkId = BenchmarkId::from_parameter(format!("n_glwe: {}", test_context.module.n()));
    let mut runner = runner::<BE, BRA>(test_context);
    group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));

    group.finish();
}

fn bench_bdd_prepare_cpu_ref_fft64(c: &mut Criterion) {
    benc_bdd_prepare::<CGGI, FFT64Avx>(
        c,
        "bdd_prepare_fft64_ref",
        &TestContext::<CGGI, FFT64Avx>::new(),
    );
}

criterion_group!(benches, bench_bdd_prepare_cpu_ref_fft64,);

criterion_main!(benches);
