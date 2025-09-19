use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_backend::{FFT64Avx, FFT64Ref, FFT64Spqlios};
use poulpy_core::layouts::{GGSWCiphertext, GLWESecret, LWECiphertext, LWESecret, prepared::PrepareAlloc};
use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpApplyDftToDftInplace, SvpPPolAlloc,
        SvpPPolAllocBytes, SvpPrepare, VecZnxAddInplace, VecZnxAddNormal, VecZnxAddScalarInplace, VecZnxAutomorphism,
        VecZnxAutomorphismInplace, VecZnxBigAddInplace, VecZnxBigAddSmallInplace, VecZnxBigAlloc, VecZnxBigAllocBytes,
        VecZnxBigAutomorphismInplace, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallBInplace, VecZnxCopy,
        VecZnxDftAddInplace, VecZnxDftAlloc, VecZnxDftAllocBytes, VecZnxDftApply, VecZnxDftCopy, VecZnxFillUniform,
        VecZnxIdftApplyConsume, VecZnxIdftApplyTmpA, VecZnxNegateInplace, VecZnxNormalize, VecZnxNormalizeInplace,
        VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateInplace, VecZnxRotateInplaceTmpBytes, VecZnxRshInplace, VecZnxSub,
        VecZnxSubABInplace, VecZnxSwitchRing, VmpApplyDftToDft, VmpApplyDftToDftAdd, VmpApplyDftToDftTmpBytes, VmpPMatAlloc,
        VmpPrepare, ZnAddNormal, ZnFillUniform, ZnNormalizeInplace,
    },
    layouts::{Backend, Module, ScratchOwned},
    oep::{
        ScratchAvailableImpl, ScratchOwnedAllocImpl, ScratchOwnedBorrowImpl, TakeMatZnxImpl, TakeScalarZnxImpl, TakeSvpPPolImpl,
        TakeVecZnxBigImpl, TakeVecZnxDftImpl, TakeVecZnxDftSliceImpl, TakeVecZnxImpl, TakeVecZnxSliceImpl,
    },
    source::Source,
};
use poulpy_schemes::tfhe::{
    blind_rotation::{
        BlincRotationExecute, BlindRotationAlgo, BlindRotationKey, BlindRotationKeyAlloc, BlindRotationKeyEncryptSk,
        BlindRotationKeyPrepared, CGGI,
    },
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyPrepared, CirtuitBootstrappingExecute,
    },
};

pub fn benc_circuit_bootstrapping<B: Backend, BRA: BlindRotationAlgo>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B>
        + VecZnxFillUniform
        + VecZnxAddNormal
        + VecZnxNormalizeInplace<B>
        + VecZnxDftAllocBytes
        + VecZnxBigNormalize<B>
        + VecZnxDftApply<B>
        + SvpApplyDftToDftInplace<B>
        + VecZnxIdftApplyConsume<B>
        + VecZnxNormalizeTmpBytes
        + VecZnxSubABInplace
        + VecZnxAddInplace
        + VecZnxNormalize<B>
        + VecZnxSub
        + VecZnxAddScalarInplace
        + VecZnxAutomorphism
        + VecZnxSwitchRing
        + VecZnxBigAllocBytes
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
        + SvpPPolAllocBytes
        + VecZnxRotateInplace<B>
        + VecZnxBigAutomorphismInplace<B>
        + VecZnxRshInplace<B>
        + VecZnxDftCopy<B>
        + VecZnxNegateInplace
        + VecZnxCopy
        + VecZnxAutomorphismInplace<B>
        + VecZnxBigSubSmallBInplace<B>
        + VecZnxRotateInplaceTmpBytes
        + VecZnxBigAllocBytes
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
        + TakeVecZnxSliceImpl<B>,
    BlindRotationKey<Vec<u8>, BRA>: PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>,
    BlindRotationKeyPrepared<Vec<u8>, BRA, B>: BlincRotationExecute<B>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<B>,
{
    let group_name: String = format!("circuit_bootstrapping::{label}");

    let mut group = c.benchmark_group(group_name);

    struct Params {
        name: String,
        n_glwe: usize,
        n_lwe: usize,
        rank: usize,
        basek: usize,
        extension_factor: usize,
        block_size: usize,
        rows_ggsw: usize,
        rows_brk: usize,
        rows_trace: usize,
        rows_tsk: usize,
        k_pt: usize,
        k_ggsw: usize,
        k_lwe: usize,
        k_brk: usize,
        k_trace: usize,
        k_tsk: usize,
    }

    fn runner<B: Backend, BRA: BlindRotationAlgo>(params: &Params) -> impl FnMut()
    where
        Module<B>: ModuleNew<B>
            + VecZnxFillUniform
            + VecZnxAddNormal
            + VecZnxNormalizeInplace<B>
            + VecZnxDftAllocBytes
            + VecZnxBigNormalize<B>
            + VecZnxDftApply<B>
            + SvpApplyDftToDftInplace<B>
            + VecZnxIdftApplyConsume<B>
            + VecZnxNormalizeTmpBytes
            + VecZnxSubABInplace
            + VecZnxAddInplace
            + VecZnxNormalize<B>
            + VecZnxSub
            + VecZnxAddScalarInplace
            + VecZnxAutomorphism
            + VecZnxSwitchRing
            + VecZnxBigAllocBytes
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
            + SvpPPolAllocBytes
            + VecZnxRotateInplace<B>
            + VecZnxBigAutomorphismInplace<B>
            + VecZnxRshInplace<B>
            + VecZnxDftCopy<B>
            + VecZnxNegateInplace
            + VecZnxCopy
            + VecZnxAutomorphismInplace<B>
            + VecZnxBigSubSmallBInplace<B>
            + VecZnxRotateInplaceTmpBytes
            + VecZnxBigAllocBytes
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
            + TakeVecZnxSliceImpl<B>,
        BlindRotationKey<Vec<u8>, BRA>: PrepareAlloc<B, BlindRotationKeyPrepared<Vec<u8>, BRA, B>>,
        BlindRotationKeyPrepared<Vec<u8>, BRA, B>: BlincRotationExecute<B>,
        BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyAlloc + BlindRotationKeyEncryptSk<B>,
    {
        let n_glwe: usize = params.n_glwe;
        let basek: usize = params.basek;
        let extension_factor: usize = params.extension_factor;
        let rank: usize = params.rank;
        let n_lwe: usize = params.n_lwe;
        let k_pt: usize = params.k_pt;
        let k_lwe: usize = params.k_lwe;
        let block_size: usize = params.block_size;
        let rows_ggsw: usize = params.rows_ggsw;
        let k_ggsw: usize = params.k_ggsw;
        let rows_brk: usize = params.rows_brk;
        let k_brk: usize = params.k_brk;
        let rows_trace: usize = params.rows_trace;
        let k_trace: usize = params.k_trace;
        let rows_tsk: usize = params.rows_tsk;
        let k_tsk: usize = params.k_tsk;

        // Scratch space (4MB)
        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(1 << 22);

        let module: Module<B> = Module::<B>::new(n_glwe as u64);

        let mut source_xs: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([1u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);

        let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
        sk_lwe.fill_binary_block(block_size, &mut source_xs);
        sk_lwe.fill_zero();

        let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n_glwe, rank);
        sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

        let ct_lwe: LWECiphertext<Vec<u8>> = LWECiphertext::alloc(n_lwe, basek, k_lwe);

        // Circuit bootstrapping evaluation key
        let cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::encrypt_sk(
            &module,
            basek,
            &sk_lwe,
            &sk_glwe,
            k_brk,
            rows_brk,
            k_trace,
            rows_trace,
            k_tsk,
            rows_tsk,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut res: GGSWCiphertext<Vec<u8>> = GGSWCiphertext::alloc(n_glwe, basek, k_ggsw, rows_ggsw, 1, rank);
        let cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, B> = cbt_key.prepare_alloc(&module, scratch.borrow());

        move || {
            cbt_prepared.execute_to_constant(
                &module,
                &mut res,
                &ct_lwe,
                k_pt,
                extension_factor,
                scratch.borrow(),
            );
            black_box(());
        }
    }

    for params in [Params {
        name: String::from("1-bit"),
        n_glwe: 1024,
        basek: 13,
        extension_factor: 1,
        rank: 2,
        n_lwe: 574,
        k_pt: 1,
        k_lwe: 13,
        block_size: 7,
        rows_ggsw: 2,
        k_ggsw: 39,
        rows_brk: 3,
        k_brk: 52,
        rows_trace: 3,
        k_trace: 52,
        rows_tsk: 3,
        k_tsk: 52,
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
