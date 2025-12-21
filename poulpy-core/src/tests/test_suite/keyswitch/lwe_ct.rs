use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize},
    layouts::{Backend, Module, Scratch, ScratchOwned, ZnxView},
    source::Source,
};

use crate::{
    LWEDecrypt, LWEEncryptSk, LWEKeySwitch, LWESwitchingKeyEncrypt, ScratchTakeCore,
    layouts::{
        LWE, LWELayout, LWEPlaintext, LWESecret, LWESwitchingKey, LWESwitchingKeyLayout, LWESwitchingKeyPreparedFactory,
        prepared::LWESwitchingKeyPrepared,
    },
};

pub fn test_lwe_keyswitch<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: LWEKeySwitch<BE>
        + LWESwitchingKeyEncrypt<BE>
        + LWEEncryptSk<BE>
        + LWESwitchingKeyPreparedFactory<BE>
        + LWEDecrypt<BE>
        + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let n: usize = module.n();
    let in_base2k: usize = 17;
    let out_base2k: usize = 15;
    let key_base2k: usize = 13;

    let n_lwe_in: usize = module.n() >> 1;
    let n_lwe_out: usize = module.n() >> 1;
    let k_lwe_ct: usize = 102;
    let k_lwe_pt: usize = 8;

    let k_ksk: usize = k_lwe_ct + key_base2k;
    let dnum: usize = k_lwe_ct.div_ceil(key_base2k);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);

    let key_apply_infos: LWESwitchingKeyLayout = LWESwitchingKeyLayout {
        n: n.into(),
        base2k: key_base2k.into(),
        k: k_ksk.into(),
        dnum: dnum.into(),
    };

    let lwe_in_infos: LWELayout = LWELayout {
        n: n_lwe_in.into(),
        base2k: in_base2k.into(),
        k: k_lwe_ct.into(),
    };

    let lwe_out_infos: LWELayout = LWELayout {
        n: n_lwe_out.into(),
        k: k_lwe_ct.into(),
        base2k: out_base2k.into(),
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        LWESwitchingKey::encrypt_sk_tmp_bytes(module, &key_apply_infos)
            | LWE::keyswitch_tmp_bytes(module, &lwe_out_infos, &lwe_in_infos, &key_apply_infos),
    );

    let mut sk_lwe_in: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe_in.into());
    sk_lwe_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_lwe_out: LWESecret<Vec<u8>> = LWESecret::alloc(n_lwe_out.into());
    sk_lwe_out.fill_ternary_prob(0.5, &mut source_xs);

    let data: i64 = 17;

    let mut lwe_pt_in: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc(in_base2k.into(), k_lwe_pt.into());
    lwe_pt_in.encode_i64(data, k_lwe_pt.into());

    let mut lwe_ct_in: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_in_infos);
    lwe_ct_in.encrypt_sk(
        module,
        &lwe_pt_in,
        &sk_lwe_in,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut ksk: LWESwitchingKey<Vec<u8>> = LWESwitchingKey::alloc_from_infos(&key_apply_infos);

    ksk.encrypt_sk(
        module,
        &sk_lwe_in,
        &sk_lwe_out,
        &mut source_xa,
        &mut source_xe,
        scratch.borrow(),
    );

    let mut lwe_ct_out: LWE<Vec<u8>> = LWE::alloc_from_infos(&lwe_out_infos);

    let mut ksk_prepared: LWESwitchingKeyPrepared<Vec<u8>, BE> = LWESwitchingKeyPrepared::alloc_from_infos(module, &ksk);
    ksk_prepared.prepare(module, &ksk, scratch.borrow());

    lwe_ct_out.keyswitch(module, &lwe_ct_in, &ksk_prepared, scratch.borrow());

    let mut lwe_pt_out: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_out_infos);
    lwe_ct_out.decrypt(module, &mut lwe_pt_out, &sk_lwe_out, scratch.borrow());

    let mut lwe_pt_want: LWEPlaintext<Vec<u8>> = LWEPlaintext::alloc_from_infos(&lwe_out_infos);
    module.vec_znx_normalize(
        lwe_pt_want.data_mut(),
        out_base2k,
        0,
        0,
        lwe_pt_in.data(),
        in_base2k,
        0,
        scratch.borrow(),
    );

    assert_eq!(lwe_pt_want.data.at(0, 0)[0], lwe_pt_out.data.at(0, 0)[0]);
}
