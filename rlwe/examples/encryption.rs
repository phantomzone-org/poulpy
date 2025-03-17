use base2k::{Encoding, SvpPPolOps, VecZnx, alloc_aligned};
use rlwe::{
    ciphertext::Ciphertext,
    elem::ElemCommon,
    keys::SecretKey,
    parameters::{Parameters, ParametersLiteral},
    plaintext::Plaintext,
};
use sampling::source::Source;

fn main() {
    let params_lit: ParametersLiteral = ParametersLiteral {
        backend: base2k::MODULETYPE::FFT64,
        log_n: 10,
        log_q: 54,
        log_p: 0,
        log_base2k: 17,
        log_scale: 20,
        xe: 3.2,
        xs: 128,
    };

    let params: Parameters = Parameters::new(&params_lit);

    let mut tmp_bytes: Vec<u8> = alloc_aligned(
        params.decrypt_rlwe_tmp_byte(params.log_q())
            | params.encrypt_rlwe_sk_tmp_bytes(params.log_q()),
    );

    let mut source: Source = Source::new([0; 32]);
    let mut sk: SecretKey = SecretKey::new(params.module());
    sk.fill_ternary_hw(params.xs(), &mut source);

    let mut want = vec![i64::default(); params.n()];

    want.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    let mut pt: Plaintext = params.new_plaintext(params.log_q());

    let log_base2k = pt.log_base2k();

    let log_k: usize = params.log_q() - 20;

    pt.0.value[0].encode_vec_i64(log_base2k, log_k, &want, 32);
    pt.0.value[0].normalize(log_base2k, &mut tmp_bytes);

    println!("log_k: {}", log_k);
    pt.0.value[0].print(pt.cols(), 16);
    println!();

    let mut ct: Ciphertext<VecZnx> = params.new_ciphertext(params.log_q());

    let mut source_xe: Source = Source::new([1; 32]);
    let mut source_xa: Source = Source::new([2; 32]);

    let mut sk_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
    params.module().svp_prepare(&mut sk_svp_ppol, &sk.0);

    params.encrypt_rlwe_sk(
        &mut ct,
        Some(&pt),
        &sk_svp_ppol,
        &mut source_xa,
        &mut source_xe,
        &mut tmp_bytes,
    );

    params.decrypt_rlwe(&mut pt, &ct, &sk_svp_ppol, &mut tmp_bytes);
    pt.0.value[0].print(pt.cols(), 16);

    let mut have = vec![i64::default(); params.n()];

    println!("pt: {}", log_k);
    pt.0.value[0].decode_vec_i64(pt.log_base2k(), log_k, &mut have);

    println!("want: {:?}", &want[..16]);
    println!("have: {:?}", &have[..16]);
}
