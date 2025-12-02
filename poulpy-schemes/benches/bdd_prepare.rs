use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_core::{
    GLWEDecrypt, GLWEEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEToGGSWKeyLayout, GGSWLayout, GGSWPreparedFactory, GLWEAutomorphismKeyLayout,
        GLWELayout, GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESwitchingKeyLayout, GLWEToLWEKeyLayout,
        LWESecret, Rank, TorusPrecision,
    },
};

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
pub use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
pub use poulpy_cpu_ref::FFT64Ref as BackendImpl;

use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};
use poulpy_schemes::bin_fhe::{
    bdd_arithmetic::{
        BDDKey, BDDKeyEncryptSk, BDDKeyLayout, BDDKeyPrepared, BDDKeyPreparedFactory, FheUint, FheUintPrepare, FheUintPrepared,
    },
    blind_rotation::{
        BlindRotationAlgo, BlindRotationKey, BlindRotationKeyFactory, BlindRotationKeyInfos, BlindRotationKeyLayout, CGGI,
    },
    circuit_bootstrapping::{CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout},
};

pub fn benc_bdd_prepare<BE: Backend, BRA: BlindRotationAlgo>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + FheUintPrepare<BRA, BE>,
    BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>, // TODO find a way to remove this bound or move it to CBT KEY
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let group_name: String = format!("bdd_arithmetic::{label}");

    let mut group = c.benchmark_group(group_name);

    struct Params {
        name: String,
        block_size: usize,
        glwe_layout: GLWELayout,
        ggsw_layout: GGSWLayout,
        bdd_layout: BDDKeyLayout,
    }

    fn runner<BE: Backend, BRA: BlindRotationAlgo>(params: &Params) -> impl FnMut()
    where
        Module<BE>: ModuleNew<BE>
            + ModuleN
            + GLWESecretPreparedFactory<BE>
            + GLWEDecrypt<BE>
            + CircuitBootstrappingKeyEncryptSk<BRA, BE>
            + GGSWPreparedFactory<BE>
            + GLWEEncryptSk<BE>
            + BDDKeyEncryptSk<BRA, BE>
            + BDDKeyPreparedFactory<BRA, BE>
            + FheUintPrepare<BRA, BE>,
        BlindRotationKey<Vec<u8>, BRA>: BlindRotationKeyFactory<BRA>, /* TODO find a way to remove this bound or move it to CBT KEY */
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        // Scratch space (4MB)
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 25);

        let n_glwe: poulpy_core::layouts::Degree = params.bdd_layout.cbt_layout.brk_layout.n_glwe();
        let n_lwe: poulpy_core::layouts::Degree = params.bdd_layout.cbt_layout.brk_layout.n_lwe();
        let rank: poulpy_core::layouts::Rank = params.bdd_layout.cbt_layout.brk_layout.rank;

        let module: Module<BE> = Module::<BE>::new(n_glwe.as_u32() as u64);

        let mut source_xs: Source = Source::new([1u8; 32]);
        let mut source_xa: Source = Source::new([1u8; 32]);
        let mut source_xe: Source = Source::new([1u8; 32]);

        let mut sk_lwe: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe);
        sk_lwe.fill_binary_block(params.block_size, &mut source_xs);

        let mut sk_glwe: GLWESecret<Vec<u8>> = GLWESecret::alloc(n_glwe, rank);
        sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_glwe_prepared = GLWESecretPrepared::alloc_from_infos(&module, &params.glwe_layout);
        sk_glwe_prepared.prepare(&module, &sk_glwe);

        let mut bdd_key: BDDKey<Vec<u8>, BRA> = BDDKey::alloc_from_infos(&params.bdd_layout);
        bdd_key.encrypt_sk(
            &module,
            &sk_lwe,
            &sk_glwe,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let input_a = 255_u32;

        let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&params.glwe_layout);
        a_enc.encrypt_sk(
            &module,
            input_a,
            &sk_glwe_prepared,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        let mut bdd_key_prepared: BDDKeyPrepared<Vec<u8>, BRA, BE> =
            BDDKeyPrepared::alloc_from_infos(&module, &params.bdd_layout);
        bdd_key_prepared.prepare(&module, &bdd_key, scratch.borrow());

        let mut a_enc_prepared: FheUintPrepared<Vec<u8>, u32, BE> =
            FheUintPrepared::alloc_from_infos(&module, &params.ggsw_layout);

        move || {
            a_enc_prepared.prepare(&module, &a_enc, &bdd_key_prepared, scratch.borrow());
            black_box(());
        }
    }

    const N_GLWE: u32 = 1024;
    const N_LWE: u32 = 679;
    const BINARY_BLOCK_SIZE: u32 = 7;
    const BASE2K: u32 = 15;
    const RANK: u32 = 2;

    let params: Params = Params {
        name: format!("n_glwe={N_GLWE}"),
        block_size: BINARY_BLOCK_SIZE as usize,
        glwe_layout: GLWELayout {
            n: Degree(N_GLWE),
            base2k: Base2K(BASE2K),
            k: TorusPrecision(2 * BASE2K),
            rank: Rank(RANK),
        },
        ggsw_layout: GGSWLayout {
            n: Degree(N_GLWE),
            base2k: Base2K(BASE2K),
            k: TorusPrecision(3 * BASE2K),
            rank: Rank(RANK),
            dnum: Dnum(3),
            dsize: Dsize(1),
        },
        bdd_layout: BDDKeyLayout {
            cbt_layout: CircuitBootstrappingKeyLayout {
                brk_layout: BlindRotationKeyLayout {
                    n_glwe: Degree(N_GLWE),
                    n_lwe: Degree(N_LWE),
                    base2k: Base2K(BASE2K),
                    k: TorusPrecision(4 * BASE2K),
                    dnum: Dnum(4),
                    rank: Rank(RANK),
                },
                atk_layout: GLWEAutomorphismKeyLayout {
                    n: Degree(N_GLWE),
                    base2k: Base2K(BASE2K),
                    k: TorusPrecision(4 * BASE2K),
                    dnum: Dnum(4),
                    dsize: Dsize(1),
                    rank: Rank(RANK),
                },
                tsk_layout: GGLWEToGGSWKeyLayout {
                    n: Degree(N_GLWE),
                    base2k: Base2K(BASE2K),
                    k: TorusPrecision(4 * BASE2K),
                    dnum: Dnum(4),
                    dsize: Dsize(1),
                    rank: Rank(RANK),
                },
            },
            ks_glwe_layout: Some(GLWESwitchingKeyLayout {
                n: Degree(N_GLWE),
                base2k: Base2K(BASE2K),
                k: TorusPrecision(4 * BASE2K),
                dnum: Dnum(4),
                dsize: Dsize(1),
                rank_in: Rank(RANK),
                rank_out: Rank(1),
            }),
            ks_lwe_layout: GLWEToLWEKeyLayout {
                n: Degree(N_GLWE),
                base2k: Base2K(BASE2K),
                k: TorusPrecision(4 * BASE2K),
                rank_in: Rank(1),
                dnum: Dnum(4),
            },
        },
    };

    let id: BenchmarkId = BenchmarkId::from_parameter(params.name.clone());
    let mut runner = runner::<BE, BRA>(&params);
    group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));

    group.finish();
}

fn bench_bdd_prepare_fft64(c: &mut Criterion) {
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    let label = "fft64_avx";
    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    let label = "fft64_ref";
    benc_bdd_prepare::<BackendImpl, CGGI>(c, label);
}

criterion_group!(benches, bench_bdd_prepare_fft64);
criterion_main!(benches);
