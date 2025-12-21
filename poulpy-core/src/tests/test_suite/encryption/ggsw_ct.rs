use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GGSWCompressedEncryptSk, GGSWEncryptSk, GGSWNoise, ScratchTakeCore,
    encryption::SIGMA,
    layouts::{
        GGSW, GGSWDecompress, GGSWInfos, GGSWLayout, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        compressed::GGSWCompressed, prepared::GLWESecretPrepared,
    },
};

pub fn test_ggsw_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
    Module<BE>: GGSWEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GGSWNoise<BE>,
{
    let base2k: usize = 12;
    let k: usize = 54;
    let dsize: usize = k / base2k;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k - di * base2k) / (di * base2k);

            let ggsw_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut ct: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GGSW::encrypt_sk_tmp_bytes(module, &ggsw_infos));

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ggsw_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_prepared.prepare(module, &sk);

            ct.encrypt_sk(
                module,
                &pt_scalar,
                &sk_prepared,
                &mut source_xa,
                &mut source_xe,
                scratch.borrow(),
            );

            let noise_f = |_col_i: usize| -(k as f64) + SIGMA.log2() + 0.5;

            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    assert!(
                        ct.noise(module, row, col, &pt_scalar, &sk_prepared, scratch.borrow())
                            .std()
                            .log2()
                            <= noise_f(col)
                    )
                }
            }
        }
    }
}

pub fn test_ggsw_compressed_encrypt_sk<BE: Backend>(module: &Module<BE>)
where
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
    Module<BE>: GGSWCompressedEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GGSWNoise<BE> + GGSWDecompress,
{
    let base2k: usize = 12;
    let k: usize = 54;
    let dsize: usize = k / base2k;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k - di * base2k) / (di * base2k);

            let ggsw_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                k: k.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            };

            let mut ct_compressed: GGSWCompressed<Vec<u8>> = GGSWCompressed::alloc_from_infos(&ggsw_infos);

            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GGSWCompressed::encrypt_sk_tmp_bytes(module, &ggsw_infos));

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&ggsw_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc(module, rank.into());
            sk_prepared.prepare(module, &sk);

            let seed_xa: [u8; 32] = [1u8; 32];

            ct_compressed.encrypt_sk(module, &pt_scalar, &sk_prepared, seed_xa, &mut source_xe, scratch.borrow());

            let noise_f = |_col_i: usize| -(k as f64) + SIGMA.log2() + 0.5;

            let mut ct: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);
            ct.decompress(module, &ct_compressed);

            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    assert!(
                        ct.noise(module, row, col, &pt_scalar, &sk_prepared, scratch.borrow())
                            .std()
                            .log2()
                            <= noise_f(col)
                    )
                }
            }
        }
    }
}
