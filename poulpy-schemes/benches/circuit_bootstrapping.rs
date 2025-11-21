use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk, ScratchTakeCore,
    layouts::{
        Dsize, GGLWEToGGSWKeyLayout, GGSW, GGSWLayout, GGSWPreparedFactory, GLWEAutomorphismKeyLayout, GLWESecret,
        GLWESecretPreparedFactory, LWE, LWELayout, LWESecret,
    },
};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateInplace},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use poulpy_schemes::bin_fhe::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory, BlindRotationKeyInfos, BlindRotationKeyLayout, CGGI,
    },
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared, CircuitBootstrappingKeyPreparedFactory, CirtuitBootstrappingExecute,
    },
};

pub fn benc_circuit_bootstrapping<BE: Backend, BRA: BlindRotationAlgo>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
        + CirtuitBootstrappingExecute<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GGSWNoise<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateInplace<BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>, // TODO find a way to remove this bound or move it to CBT KEY
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let group_name: String = format!("circuit_bootstrapping::{label}");

    let mut group = c.benchmark_group(group_name);

    struct Params {
        name: String,
        extension_factor: usize,
        k_pt: usize,
        block_size: usize,
        lwe_infos: LWELayout,
        ggsw_infos: GGSWLayout,
        cbt_infos: CircuitBootstrappingKeyLayout,
    }

    fn runner<BE: Backend, BRA: BlindRotationAlgo>(params: &Params) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE>
            + ModuleN
            + GLWESecretPreparedFactory<BE>
            + GLWEExternalProduct<BE>
            + GLWEDecrypt<BE>
            + LWEEncryptSk<BE>
            + CircuitBootstrappingKeyEncryptSk<BRA, BE>
            + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
            + CirtuitBootstrappingExecute<BRA, BE>
            + GGSWPreparedFactory<BE>
            + GGSWNoise<BE>
            + GLWEEncryptSk<BE>
            + VecZnxRotateInplace<BE>,
        BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>, /* TODO find a way to remove this bound or move it to CBT KEY */
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // Scratch space (4MB)
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

        let n_glwe: poulpy_core::layouts::Degree = params.cbt_infos.layout_brk.n_glwe();
        let n_lwe: poulpy_core::layouts::Degree = params.cbt_infos.layout_brk.n_lwe();
        let rank: poulpy_core::layouts::Rank = params.cbt_infos.layout_brk.rank;

        let module: Module<BE> = Module::<BE>::new(n_glwe.as_u32() as u64);

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
        let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::alloc_from_infos(&params.cbt_infos);
        cbt_key.encrypt_sk(
            &module,
            &sk_lwe,
            &sk_glwe,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut res: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&params.ggsw_infos);
        let mut cbt_prepared: CircuitBootstrappingKeyPrepared<Vec<u8>, BRA, BE> =
            CircuitBootstrappingKeyPrepared::alloc_from_infos(&module, &params.cbt_infos);
        cbt_prepared.prepare(&module, &cbt_key, scratch.borrow());
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

    let params: Params = Params {
        name: String::from("1-bit"),
        extension_factor: 1,
        k_pt: 1,
        lwe_infos: LWELayout {
            n: 574_u32.into(),
            k: 13_u32.into(),
            base2k: 13_u32.into(),
        },
        block_size: 7,
        ggsw_infos: GGSWLayout {
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
            layout_atk: GLWEAutomorphismKeyLayout {
                n: 1024_u32.into(),
                base2k: 13_u32.into(),
                k: 52_u32.into(),
                dnum: 3_u32.into(),
                dsize: Dsize(1),
                rank: 2_u32.into(),
            },
            layout_tsk: GGLWEToGGSWKeyLayout {
                n: 1024_u32.into(),
                base2k: 13_u32.into(),
                k: 52_u32.into(),
                dnum: 3_u32.into(),
                dsize: Dsize(1),
                rank: 2_u32.into(),
            },
        },
    };

    let id: BenchmarkId = BenchmarkId::from_parameter(params.name.clone());
    let mut runner = runner::<BE, BRA>(&params);
    group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));

    group.finish();
}

fn bench_circuit_bootstrapping_fft64(c: &mut Criterion) {
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    let label = "fft64_avx";
    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    let label = "fft64_ref";
    benc_circuit_bootstrapping::<BackendImpl, CGGI>(c, label);
}

criterion_group!(benches, bench_circuit_bootstrapping_fft64);
criterion_main!(benches);
