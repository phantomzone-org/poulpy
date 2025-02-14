use base2k::{
    Encoding, FFT64, Infos, Sampling, Scalar, SvpPPolOps, VecZnx, VecZnxApi, VecZnxBig, VecZnxDft,
    VecZnxOps,
};
use rlwe::{
    ciphertext::{Ciphertext, GadgetCiphertext},
    decryptor::{Decryptor, decrypt_rlwe_thread_safe, decrypt_rlwe_thread_safe_tmp_byte},
    elem::Elem,
    encryptor::{
        EncryptorSk, encrypt_grlwe_sk_thread_safe, encrypt_grlwe_sk_tmp_bytes,
        encrypt_rlwe_sk_tmp_bytes,
    },
    evaluator::{gadget_product_inplace_thread_safe, gadget_product_tmp_bytes},
    key_generator::{gen_switching_key_thread_safe, gen_switching_key_thread_safe_tmp_bytes},
    keys::{SecretKey, SwitchingKey},
    parameters::{Parameters, ParametersLiteral},
    plaintext::Plaintext,
};
use sampling::source::{Source, new_seed};

fn main() {
    let params_lit: ParametersLiteral = ParametersLiteral {
        log_n: 4,
        log_q: 68,
        log_p: 17,
        log_base2k: 17,
        log_scale: 20,
        xe: 3.2,
        xs: 8,
    };

    let params: Parameters = Parameters::new::<FFT64>(&params_lit);

    let mut tmp_bytes: Vec<u8> = vec![
        0u8;
        params.decrypt_rlwe_thread_safe_tmp_byte(params.log_q())
            | params.encrypt_rlwe_sk_tmp_bytes(params.log_q())
            | gen_switching_key_thread_safe_tmp_bytes(
                params.module(),
                params.log_base2k(),
                params.limbs_q(),
                params.log_q()
            )
            | gadget_product_tmp_bytes(
                params.module(),
                params.log_base2k(),
                params.log_q(),
                params.log_q(),
                params.limbs_q(),
                params.log_qp()
            )
            | encrypt_grlwe_sk_tmp_bytes(
                params.module(),
                params.log_base2k(),
                params.limbs_qp(),
                params.log_qp()
            )
    ];

    let mut source: Source = Source::new([3; 32]);

    let mut sk0: SecretKey = SecretKey::new(params.module());
    let mut sk1: SecretKey = SecretKey::new(params.module());

    sk0.fill_ternary_hw(params.xs(), &mut source);
    sk1.fill_ternary_hw(params.xs(), &mut source);

    let mut want = vec![i64::default(); params.n()];

    want.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    let log_base2k = params.log_base2k();

    let log_k: usize = params.log_q() - 2 * log_base2k;

    let mut source_xe: Source = Source::new([4; 32]);
    let mut source_xa: Source = Source::new([5; 32]);

    let mut sk0_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
    params.module().svp_prepare(&mut sk0_svp_ppol, &sk0.0);

    let mut sk1_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
    params.module().svp_prepare(&mut sk1_svp_ppol, &sk1.0);

    let mut gadget_ct: GadgetCiphertext = GadgetCiphertext::new(
        params.module(),
        log_base2k,
        params.limbs_q(),
        params.log_qp(),
    );

    encrypt_grlwe_sk_thread_safe(
        params.module(),
        &mut gadget_ct,
        &sk0.0,
        &sk1_svp_ppol,
        &mut source_xa,
        &mut source_xe,
        params.xe(),
        &mut tmp_bytes,
    );

    println!("DONE?");

    let mut pt: Plaintext<VecZnx> = Plaintext::<VecZnx>::new(
        params.module(),
        params.log_base2k(),
        params.log_q(),
        params.log_scale(),
    );

    let mut want = vec![i64::default(); params.n()];
    want.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
    pt.0.value[0].encode_vec_i64(log_base2k, log_k, &want, 32);
    pt.0.value[0].normalize(log_base2k, &mut tmp_bytes);

    let mut ct: Ciphertext = params.new_ciphertext(params.log_q());

    params.encrypt_rlwe_sk_thread_safe(
        &mut ct,
        Some(&pt),
        &sk0_svp_ppol,
        &mut source_xa,
        &mut source_xe,
        &mut tmp_bytes,
    );

    gadget_product_inplace_thread_safe::<true, _>(
        params.module(),
        &mut ct.0,
        &gadget_ct,
        &mut tmp_bytes,
    );

    println!("ct.limbs()={}", ct.limbs());
    println!("gadget_ct.rows()={}", gadget_ct.rows());
    println!("gadget_ct.cols()={}", gadget_ct.cols());
    println!("res.limbs()={}", ct.limbs());
    println!();

    decrypt_rlwe_thread_safe(
        params.module(),
        &mut pt.0,
        &ct.0,
        &sk1_svp_ppol,
        &mut tmp_bytes,
    );

    pt.0.value[0].print_limbs(pt.limbs(), 16);

    let mut have: Vec<i64> = vec![i64::default(); params.n()];

    println!("pt: {}", log_k);
    pt.0.value[0].decode_vec_i64(pt.log_base2k(), log_k, &mut have);

    println!("want: {:?}", &want[..16]);
    println!("have: {:?}", &have[..16]);
}
