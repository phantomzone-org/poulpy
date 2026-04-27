use std::collections::HashMap;

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWEEncryptSk, GLWEPacker, GLWEPackerOps, GLWEPacking,
    GLWERotate, GLWESub, ScratchArenaTakeCore,
    glwe_packer::{glwe_packer_add, glwe_packer_flush, glwe_packer_galois_elements, glwe_packer_tmp_bytes},
    layouts::{
        GLWE, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWEPlaintext,
        GLWESecret, GLWESecretPreparedFactory, GLWEToBackendMut, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
};

pub fn test_glwe_packer<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEPackerOps<BE>
        + GLWEPacking<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESub<BE>
        + GLWEDecrypt<BE>
        + GLWERotate<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([0u8; 32]);
    let mut source_xa: Source = Source::new([0u8; 32]);

    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_ct: usize = 4 * out_base2k + 1;
    let pt_k: usize = 2 * out_base2k + 1;
    let rank: usize = 3;
    let dsize: usize = 1;
    let k_ksk: usize = k_ct + key_base2k * dsize;

    let dnum: usize = k_ct.div_ceil(key_base2k * dsize);

    let glwe_out_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n.into(),
        base2k: out_base2k.into(),
        k: k_ct.into(),
        rank: rank.into(),
    })
    .unwrap();

    let key_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: key_base2k.into(),
        k: k_ksk.into(),
        rank: rank.into(),
        dsize: dsize.into(),
        dnum: dnum.into(),
    })
    .unwrap();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        (module).glwe_encrypt_sk_tmp_bytes(&glwe_out_infos)
            | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&key_infos)
            | glwe_packer_tmp_bytes(module, &glwe_out_infos, &key_infos)
            | module.glwe_pack_tmp_bytes(&glwe_out_infos, &key_infos),
    );

    let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_out_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
    module.glwe_secret_prepare(&mut sk_dft, &sk);

    let mut pt: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
    let mut data: Vec<i64> = vec![0i64; n];
    data.iter_mut().enumerate().for_each(|(i, x)| {
        *x = i as i64;
    });

    pt.encode_vec_i64(&data, pt_k.into());

    let gal_els: Vec<i64> = glwe_packer_galois_elements(module);

    let mut auto_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let mut tmp: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&key_infos);
    gal_els.iter().for_each(|gal_el| {
        module.glwe_automorphism_key_encrypt_sk(
            &mut tmp,
            *gal_el,
            &sk,
            &key_infos,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::scratch_host_arena(&mut scratch),
        );
        let mut atk_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&tmp);
        module.glwe_automorphism_key_prepare(&mut atk_prepared, &tmp, &mut scratch.borrow());
        auto_keys.insert(*gal_el, atk_prepared);
    });

    let log_batch: usize = 0;

    let mut packer: GLWEPacker<BE::OwnedBuf> = GLWEPacker::alloc(module, &glwe_out_infos, log_batch);
    let mut cts_oracle: Vec<(usize, GLWE<Vec<u8>>)> = Vec::new();

    let mut ct: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);

    module.glwe_encrypt_sk(
        &mut ct,
        &pt,
        &sk_dft,
        &glwe_out_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let log_n: usize = module.log_n();

    (0..n >> log_batch).for_each(|i| {
        module.glwe_encrypt_sk(
            &mut ct,
            &pt,
            &sk_dft,
            &glwe_out_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        {
            let mut pt_backend = <GLWEPlaintext<Vec<u8>> as GLWEToBackendMut<BE>>::to_backend_mut(&mut pt);
            module.glwe_rotate_inplace(-(1 << log_batch), &mut pt_backend, &mut scratch.borrow()); // X^-batch * pt
        }

        if reverse_bits_msb(i, log_n as u32).is_multiple_of(5) {
            cts_oracle.push((reverse_bits_msb(i, log_n as u32), ct.clone()));
            glwe_packer_add(module, &mut packer, Some(&ct), &auto_keys, &mut scratch.borrow());
        } else {
            glwe_packer_add(module, &mut packer, None::<&GLWE<Vec<u8>>>, &auto_keys, &mut scratch.borrow())
        }
    });

    let mut res: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
    glwe_packer_flush(module, &mut packer, &mut res, &mut scratch.borrow());

    let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);
    let mut res_oracle: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
    let mut cts_oracle_map: std::collections::HashMap<usize, &mut GLWE<Vec<u8>>> = std::collections::HashMap::new();
    for (idx, ct) in cts_oracle.iter_mut() {
        cts_oracle_map.insert(*idx, ct);
    }
    module.glwe_pack(&mut res_oracle, cts_oracle_map, 0, &auto_keys, &mut scratch.borrow());
    module.glwe_decrypt(&res_oracle, &mut pt_want, &sk_dft, &mut scratch.borrow());

    module.glwe_decrypt(&res, &mut pt, &sk_dft, &mut scratch.borrow());

    module.glwe_sub_assign(&mut pt, &pt_want);

    let noise_have: f64 = pt.stats().std().log2();

    assert!(noise_have < -((k_ct - out_base2k) as f64), "noise: {noise_have}");
}

#[inline(always)]
fn reverse_bits_msb(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
}
