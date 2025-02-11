use base2k::{Encoding, FFT64, SvpPPolOps};
use rlwe::{
    ciphertext::Ciphertext,
    decryptor::{Decryptor, decrypt_rlwe_thread_safe_tmp_byte},
    encryptor::{EncryptorSk, encrypt_rlwe_sk_tmp_bytes},
    keys::SecretKey,
    parameters::{Parameters, ParametersLiteral},
    plaintext::Plaintext,
};
use sampling::source::{Source, new_seed};

fn main() {
    let params_lit: ParametersLiteral = ParametersLiteral {
        log_n: 10,
        log_q: 54,
        log_p: 0,
        log_base2k: 17,
        log_scale: 20,
        xe: 3.2,
        xs: 128,
    };

    let params: Parameters = Parameters::new::<FFT64>(&params_lit);

    let mut tmp_bytes: Vec<u8> = vec![
        0u8;
        params.decrypt_rlwe_thread_safe_tmp_byte(params.log_q())
            | params.encrypt_rlwe_sk_tmp_bytes(params.log_q())
    ];

    let sk: SecretKey = SecretKey::new(params.module());

    let mut want = vec![i64::default(); params.n()];

    want.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    let mut pt: Plaintext = params.new_plaintext(params.log_q());

    let log_base2k = pt.log_base2k();

    let log_k: usize = params.log_q() - 20;

    pt.0.value[0].encode_vec_i64(log_base2k, log_k, &want, 32);
    pt.0.value[0].normalize(log_base2k, &mut tmp_bytes);

    println!("log_k: {}", log_k);
    pt.0.value[0].print_limbs(pt.limbs(), 16);

    let mut ct: Ciphertext = params.new_ciphertext(params.log_q());

    let mut source_xe: Source = Source::new(new_seed());
    let mut source_xa: Source = Source::new(new_seed());

    let mut sk_svp_ppol: base2k::SvpPPol = params.module().svp_new_ppol();
    params.module().svp_prepare(&mut sk_svp_ppol, &sk.0);

    params.encrypt_rlwe_sk_thread_safe(
        &mut ct,
        Some(&pt),
        &sk_svp_ppol,
        &mut source_xa,
        &mut source_xe,
        &mut tmp_bytes,
    );

    params.decrypt_rlwe_thread_safe(&mut pt, &ct, &sk_svp_ppol, &mut tmp_bytes);

    pt.0.value[0].print_limbs(pt.limbs(), 16);

    let mut have = vec![i64::default(); params.n()];

    println!("pt: {}", log_k);
    pt.0.value[0].decode_vec_i64(pt.log_base2k(), log_k, &mut have);

    println!("want: {:?}", &want[..16]);
    println!("have: {:?}", &have[..16]);
}
