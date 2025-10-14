use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_backend::{FFT64Avx, FFT64Ref, FFT64Spqlios};
use poulpy_core::layouts::{
    AutomorphismKeyLayout, Dsize, GGSW, GGSWCiphertextLayout, GLWESecret, LWE, LWECiphertextLayout, LWESecret, TensorKeyLayout,
    prepared::PrepareAlloc,
};
use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc,
        SvpPPolBytesOf, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism,
        VecZnxAutomorphismInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigAutomorphismInplace,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallNegateInplace, VecZnxCopy,
        VecZnxDftAddInplace, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxRshInplace, VecZnxSub,
        VecZnxSubInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc,
        VmpPrepare, ZnAddNormal, ZnFillUniform, ZnNormalizeInplace,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeMatZnxImpl, TakeScalarZnxImpl, TakeSliceImpl,
        TakeSvpPPolImpl, TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxDftSliceImpl, TakeVecZnxImpl, TakeVecZnxSliceImpl,
    },
    source::Source,
};
use poulpy_schemes::tfhe::{
    blind_rotation::{
        BlincRotationExecute, BlindRotationAlgo, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk,
        BlindRotationKeyInfos, BlindRotationKeyLayout, BlindRotationKeyPrepared, CGGI,
    },
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute,
    },
};

pub fn benc_circuit_bootstrapping<B: Backend, BRA: BlindRotationAlgo>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B>
        + VecZnxFillUniform
        + VecZnxAddNormal
        + VecZnxNormalizeInplace<B>
        + VecZnxDftBytesOf
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubInplace
        + VecZnxAddInplace
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxAddScalarInplace
        + VecZnxAutomorphism
        + VecZnxSwitchRing
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpA<B>
        + SvpApplyDftToDft<B>
        + VecZnxBigAddInplace<B>
        + VecZnxBigAddSmallInplace<B>
        + VecZnxBigAlloc<B>
        + VecZnxDftAlloc<B>
        + VecZnxBigNormalizeTmpBytes
        + VmpPMatAlloc<B>
        + VmpPrepare<B>
        + SvpPrepare<B>
        + SvpPPolAlloc<B>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<B>
        + VmpApplyDftToDftAdd<B>
        + SvpPPolBytesOf
        + VecZnxRotateInplace<B>
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallNegateInplace<B>
        + VecZnxRotateInplaceTmpBytes
        + VecZnxBigBytesOf
        + VecZnxDftAddInplace<B>
        + VecZnxRotate
        + ZnFillUniform
        + ZnAddNormal
        + ZnNormalizeInplace<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B: Backend
        + ScratchOwnedAllocImpl<B>
        + ScratchOwnedBorrowImpl<B>
        + TakeVecZnxDftImpl<B>
        + ScratchAvailableImpl<B>
        + TakeVecZnxImpl<B>
        + TakeScalarZnxImpl<B>
        + TakeSvpPPolImpl<B>
        + TakeVecZnxBigImpl<B>
        + TakeVecZnxDftSliceImpl<B>
        + TakeMatZnxImpl<B>
        + TakeVecZnxSliceImpl<B>
        + TakeSliceImpl<B>,
    BlindRotationKey<Vec<u8>, BRA>: PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>,
    BlindRotationKeyPrepared<Vec<u8>, BRA, B>: BlincRotationExecute<B>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<B>,
{
    let group_name: String = format!("circuit_bootstrapping::{label}");

    let mut group = c.benchmark_group(group_name);

    struct Params {
        name: String,
        extension_factor: usize,
        k_pt: usize,
        block_size: usize,
        lwe_infos: LWECiphertextLayout,
        ggsw_infos: GGSWCiphertextLayout,
        cbt_infos: CircuitBootstrappingKeyLayout,
    }

    fn runner<B: Backend, BRA: BlindRotationAlgo>(params: &Params) -> impl FnMut()
    where
        Module<B>: ModuleNew<B>
            + VecZnxFillUniform
            + VecZnxAddNormal
            + VecZnxNormalizeInplace<B>
            + VecZnxDftBytesOf
            + VecZnxBigNormalize<B>
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxSubInplace
            + VecZnxAddInplace
            + VecZnxNormalize<B>
            + VecZnxSub
            + VecZnxAddScalarInplace
            + VecZnxAutomorphism
            + VecZnxSwitchRing
            + VecZnxBigBytesOf
            + VecZnxIdftApplyTmpA<B>
            + SvpApplyDftToDft<B>
            + VecZnxBigAddInplace<B>
            + VecZnxBigAddSmallInplace<B>
            + VecZnxBigAlloc<B>
            + VecZnxDftAlloc<B>
            + VecZnxBigNormalizeTmpBytes
            + VmpPMatAlloc<B>
            + VmpPrepare<B>
            + SvpPrepare<B>
            + SvpPPolAlloc<B>
            + VmpApplyDftToDftTmpBytes
            + VmpApplyDftToDft<B>
            + VmpApplyDftToDftAdd<B>
            + SvpPPolBytesOf
            + VecZnxRotateInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace<B>
            + VecZnxDftCopy<B>
            + VecZnxNegateInplace
            + VecZnxCopy
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigSubSmallNegateInplace<B>
            + VecZnxRotateInplaceTmpBytes
            + VecZnxBigBytesOf
            + VecZnxDftAddInplace<B>
            + VecZnxRotate
            + ZnFillUniform
            + ZnAddNormal
            + ZnNormalizeInplace<B>,
        B: Backend
            + ScratchOwnedAllocImpl<B>
            + ScratchOwnedBorrowImpl<B>
            + TakeVecZnxDftImpl<B>
            + ScratchAvailableImpl<B>
            + TakeVecZnxImpl<B>
            + TakeScalarZnxImpl<B>
            + TakeSvpPPolImpl<B>
            + TakeVecZnxBigImpl<B>
            + TakeVecZnxDftSliceImpl<B>
            + TakeMatZnxImpl<B>
            + TakeVecZnxSliceImpl<B>
            + TakeSliceImpl<B>,
        BlindRotationKey<Vec<u8>, BRA>: PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>,
        BlindRotationKeyPrepared<Vec<u8>, BRA, B>: BlincRotationExecute<B>,
        BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<B>,
    {
        // Scratch space (4MB)
        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 22);

        let n_glwe: poulpy_core::layouts::Degree = params.cbt_infos.layout_brk.n_glwe();
        let n_lwe: poulpy_core::layouts::Degree = params.cbt_infos.layout_brk.n_lwe();
        let rank: poulpy_core::layouts::Rank = params.cbt_infos.layout_brk.rank;

        let module: Module<B> = Module::<B>::new(n_glwe.as_u32() as u64);

        let mut source_xs: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([1u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);

        let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
        sk_lwe.fill_binary_block(params.block_size, &mut source_xs);
        sk_lwe.fill_zero();

        let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n_glwe, rank);
        sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

        let ct_lwe: LWE<Vec<u8>> = LWE::alloc_from_infos(&params.lwe_infos);

        // Circuit bootstrapping evaluation key
        let cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::encrypt_sk(
            &module,
            &sk_lwe,
            &sk_glwe,
            &params.cbt_infos,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut res: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&params.ggsw_infos);
        let cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B> = cbt_key.prepare_alloc(&module, scratch.borrow());

        move || {
            cbt_prepared.execute_to_constant(
                &module,
                &mut res,
                &ct_lwe,
                params.k_pt,
                params.extension_factor,
                scratch.borrow(),
            );
            black_box(());
        }
    }

    for params in [Params {
        name: String::from("1-bit"),
        extension_factor: 1,
        k_pt: 1,
        lwe_infos: LWECiphertextLayout {
            n: 574_u32.into(),
            k: 13_u32.into(),
            base2k: 13_u32.into(),
        },
        block_size: 7,
        ggsw_infos: GGSWCiphertextLayout {
            n: 1024_u32.into(),
            base2k: 13_u32.into(),
            k: 26_u32.into(),
            dnum: 2_u32.into(),
            dsize: 1_u32.into(),
            rank: 2_u32.into(),
        },
        cbt_infos: CircuitBootstrappingKeyLayout {
            layout_brk: BlindRotationKeyLayout {
                n_glwe: 1024_u32.into(),
                n_lwe: 574_u32.into(),
                base2k: 13_u32.into(),
                k: 52_u32.into(),
                dnum: 3_u32.into(),
                rank: 2_u32.into(),
            },
            layout_atk: AutomorphismKeyLayout {
                n: 1024_u32.into(),
                base2k: 13_u32.into(),
                k: 52_u32.into(),
                dnum: 3_u32.into(),
                dsize: Dsize(1),
                rank: 2_u32.into(),
            },
            layout_tsk: TensorKeyLayout {
                n: 1024_u32.into(),
                base2k: 13_u32.into(),
                k: 52_u32.into(),
                dnum: 3_u32.into(),
                dsize: Dsize(1),
                rank: 2_u32.into(),
            },
        },
    }] {
        let id: BenchmarkId = BenchmarkId::from_parameter(params.name.clone());
        let mut runner = runner::<B, BRA>(&params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

fn bench_circuit_bootstrapping_cpu_ref_fft64(c: &mut Criterion) {
    benc_circuit_bootstrapping::<FFT64Ref, CGGI>(c, "fft64_ref");
}

fn bench_circuit_bootstrapping_cpu_avx_fft64(c: &mut Criterion) {
    benc_circuit_bootstrapping::<FFT64Avx, CGGI>(c, "fft64_avx");
}

fn bench_circuit_bootstrapping_cpu_spqlios_fft64(c: &mut Criterion) {
    benc_circuit_bootstrapping::<FFT64Spqlios, CGGI>(c, "fft64_spqlios");
}

criterion_group!(
    benches,
    bench_circuit_bootstrapping_cpu_ref_fft64,
    bench_circuit_bootstrapping_cpu_avx_fft64,
    bench_circuit_bootstrapping_cpu_spqlios_fft64,
);

criterion_main!(benches);
