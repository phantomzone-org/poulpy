use base2k::{FFT64, Infos, Sampling, Scalar, SvpPPolOps, VecZnx, VecZnxBig, VecZnxDft, VecZnxOps};
use rlwe::{
    ciphertext::{Ciphertext, GadgetCiphertext},
    decryptor::{Decryptor, decrypt_rlwe_thread_safe, decrypt_rlwe_thread_safe_tmp_byte},
    elem::Elem,
    encryptor::{
        EncryptorSk, encrypt_grlwe_sk_thread_safe, encrypt_grlwe_sk_tmp_bytes,
        encrypt_rlwe_sk_tmp_bytes,
    },
    evaluator::{gadget_product_thread_safe, gadget_product_tmp_bytes},
    key_generator::{gen_switching_key_thread_safe, gen_switching_key_thread_safe_tmp_bytes},
    keys::{SecretKey, SwitchingKey},
    parameters::{Parameters, ParametersLiteral},
    plaintext::Plaintext,
};
use sampling::source::{Source, new_seed};

fn main() {
    let params_lit: ParametersLiteral = ParametersLiteral {
        log_n: 10,
        log_q: 68,
        log_p: 17,
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

    let mut sk: SecretKey = SecretKey::new(params.module());

    sk.fill_ternary_hw(params.xs(), &mut source);

    let mut want = vec![i64::default(); params.n()];

    want.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    let log_base2k = params.log_base2k();

    let log_k: usize = params.log_q() - 2 * log_base2k;

    let mut source_xe: Source = Source::new([4; 32]);
    let mut source_xa: Source = Source::new([5; 32]);

    let mut sk_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
    params.module().svp_prepare(&mut sk_svp_ppol, &sk.0);

    let mut gadget_ct: GadgetCiphertext = GadgetCiphertext::new(
        params.module(),
        log_base2k,
        params.limbs_q(),
        params.log_qp(),
    );

    let mut m: Scalar = Scalar::new(params.n());
    m.fill_ternary_prob(0.5, &mut source_xa);

    encrypt_grlwe_sk_thread_safe(
        params.module(),
        &mut gadget_ct,
        &m,
        &sk_svp_ppol,
        &mut source_xa,
        &mut source_xe,
        params.xe(),
        &mut tmp_bytes,
    );

    let mut res: Elem = Elem::new(params.module(), log_base2k, params.log_q(), 1, 0);
    let mut a: VecZnx = VecZnx::new(params.module().n(), params.limbs_q());
    a.fill_uniform(params.log_base2k(), a.limbs(), &mut source_xa);
    gadget_product_thread_safe(params.module(), &mut res, &a, &gadget_ct, &mut tmp_bytes);

    println!("a.limbs()={}", a.limbs());
    println!("gadget_ct.rows()={}", gadget_ct.rows());
    println!("gadget_ct.cols()={}", gadget_ct.cols());
    println!("res.limbs()={}", res.limbs());
    println!();

    println!("a:");
    a.print_limbs(a.limbs(), 16);
    println!();

    println!("m:");
    println!("{:?}", &m.0[..16]);
    println!();

    let mut a_res: Elem = Elem::new(params.module(), params.log_base2k(), params.log_q(), 0, 0);

    decrypt_rlwe_thread_safe(
        params.module(),
        &mut a_res,
        &res,
        &sk_svp_ppol,
        &mut tmp_bytes,
    );

    let mut m_svp_ppol = params.module().new_svp_ppol();
    params.module().svp_prepare(&mut m_svp_ppol, &m);

    let mut a_dft: VecZnxDft = params.module().new_vec_znx_dft(a.limbs());
    let mut a_big: VecZnxBig = a_dft.as_vec_znx_big();

    params
        .module()
        .svp_apply_dft(&mut a_dft, &m_svp_ppol, &a, a.limbs());
    params
        .module()
        .vec_znx_idft_tmp_a(&mut a_big, &mut a_dft, a.limbs());
    params
        .module()
        .vec_znx_big_normalize(params.log_base2k(), &mut a, &a_big, &mut tmp_bytes);

    params.module().vec_znx_sub_inplace(&mut a, &a_res.value[0]);

    println!("a*m - dec(a * GRLWE(m))");
    a.print_limbs(a.limbs(), 16);
}
