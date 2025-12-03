use std::collections::HashMap;

use itertools::Itertools;
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

use crate::{
    GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWENoise, GLWEPacking, GLWERotate, GLWESub, ScratchTakeCore,
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
};

pub fn test_glwe_packing<BE: Backend>(module: &Module<BE>)
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEPacking<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub
        + GLWEDecrypt<BE>
        + GLWERotate<BE>
        + GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let n: usize = module.n();
    let base2k_out: usize = 15;
    let base2k_key: usize = 10;
    let k_ct: usize = 36;
    let pt_k: usize = base2k_out;
    let rank: usize = 3;
    let dsize: usize = 1;
    let k_ksk: usize = k_ct + base2k_key * dsize;

    let dnum: usize = k_ct.div_ceil(base2k_key * dsize);

    let glwe_out_infos: GLWELayout = GLWELayout {
        n: n.into(),
        base2k: base2k_out.into(),
        k: k_ct.into(),
        rank: rank.into(),
    };

    let key_infos: GLWEAutomorphismKeyLayout = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k_key.into(),
        k: k_ksk.into(),
        rank: rank.into(),
        dsize: dsize.into(),
        dnum: dnum.into(),
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        GLWE::encrypt_sk_tmp_bytes(module, &glwe_out_infos)
            .max(GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &key_infos))
            .max(module.glwe_pack_tmp_bytes(&glwe_out_infos, &key_infos)),
    );

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_out_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prep: GLWESecretPrepared<Vec<u8>, BE> = GLWESecretPrepared::alloc_from_infos(module, &sk);
    sk_prep.prepare(module, &sk);

    let mut pt: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = i as i64;
    });

    pt.encode_vec_i64(&data, pt_k.into());

    let gal_els: Vec<i64> = module.glwe_pack_galois_elements();

    let mut auto_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<Vec<u8>, BE>> = HashMap::new();
    let mut tmp: GLWEAutomorphismKey<Vec<u8>> = GLWEAutomorphismKey::alloc_from_infos(&key_infos);
    gal_els.iter().for_each(|gal_el| {
        tmp.encrypt_sk(module, *gal_el, &sk, &mut source_xa, &mut source_xe, scratch.borrow());
        let mut atk_prepared: GLWEAutomorphismKeyPrepared<Vec<u8>, BE> =
            GLWEAutomorphismKeyPrepared::alloc_from_infos(module, &tmp);
        atk_prepared.prepare(module, &tmp, scratch.borrow());
        auto_keys.insert(*gal_el, atk_prepared);
    });

    let mut cts = (0..n)
        .step_by(5)
        .map(|_| {
            let mut ct = GLWE::alloc_from_infos(&glwe_out_infos);
            ct.encrypt_sk(module, &pt, &sk_prep, &mut source_xa, &mut source_xe, scratch.borrow());
            module.glwe_rotate_inplace(-5, &mut pt, scratch.borrow()); // X^-batch * pt
            ct
        })
        .collect_vec();

    let mut cts_map: HashMap<usize, &mut GLWE<Vec<u8>>> = HashMap::new();

    for (i, ct) in cts.iter_mut().enumerate() {
        cts_map.insert(5 * i, ct);
    }

    let mut res: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);

    module.glwe_pack(&mut res, cts_map, 0, &auto_keys, scratch.borrow());

    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        if i.is_multiple_of(5) {
            *x = i as i64;
        }
    });

    pt_want.encode_vec_i64(&data, pt_k.into());

    assert!(res.noise(module, &pt_want, &sk_prep, scratch.borrow()).std().log2() <= ((k_ct - base2k_out) as f64));
}
