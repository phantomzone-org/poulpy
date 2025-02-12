use base2k::{Encoding, FFT64, SvpPPolOps};
use rlwe::{
    ciphertext::Ciphertext,
    decryptor::{decrypt_rlwe_thread_safe_tmp_byte, Decryptor},
    encryptor::{encrypt_grlwe_sk_tmp_bytes, encrypt_rlwe_sk_tmp_bytes, EncryptorSk},
    evaluator::{gadget_product_thread_safe, gadget_product_inplace_thread_safe, gadget_product_tmp_bytes},
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
            ) | encrypt_grlwe_sk_tmp_bytes(params.module(), params.log_base2k(), params.limbs_qp(), params.log_qp())
    ];

    println!("limbsQP: {}", params.limbs_qp());

    let mut source: Source = Source::new([3; 32]);

    let mut sk0: SecretKey = SecretKey::new(params.module());
    let mut sk1: SecretKey = SecretKey::new(params.module());

    sk0.fill_ternary_hw(params.xs(), &mut source);
    sk1.fill_ternary_hw(params.xs(), &mut source);


    let mut want = vec![i64::default(); params.n()];

    want.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

    let log_base2k = params.log_base2k();

    let log_k: usize = params.log_q() - 2*log_base2k;
    
    let mut ct: Ciphertext = params.new_ciphertext(params.log_qp());

    let mut source_xe: Source = Source::new([4; 32]);
    let mut source_xa: Source = Source::new([5; 32]);

    let mut sk0_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
    params.module().svp_prepare(&mut sk0_svp_ppol, &sk0.0);

    let mut sk1_svp_ppol: base2k::SvpPPol = params.module().new_svp_ppol();
    params.module().svp_prepare(&mut sk1_svp_ppol, &sk1.0);

    let mut pt_out: Plaintext = Plaintext::new(params.module(), params.log_base2k(), params.log_qp(), ct.log_scale());

    //pt_out.0.value[0].encode_vec_i64(log_base2k, log_k, &want, 32);
    //pt_out.0.value[0].normalize(log_base2k, &mut tmp_bytes);
        
    params.encrypt_rlwe_sk_thread_safe(
        &mut ct,
        Some(&pt_out),
        &sk0_svp_ppol,
        &mut source_xa,
        &mut source_xe,
        &mut tmp_bytes,
    );

    params.decrypt_rlwe_thread_safe(&mut pt_out, &ct, &sk0_svp_ppol, &mut tmp_bytes);

    println!("DECRYPT");
    pt_out.0.value[0].print_limbs(pt_out.limbs(), 16);


    let mut swk: SwitchingKey = SwitchingKey::new(
        params.module(),
        params.log_base2k(),
        params.limbs_qp(),
        params.log_qp(),
    );

    gen_switching_key_thread_safe(
        params.module(),
        &mut swk,
        &sk0,
        &sk1_svp_ppol,
        &mut source_xa,
        &mut source_xe,
        params.xe(),
        &mut tmp_bytes,
    );

    println!("{}", swk.cols());

    let mut ct_out: Ciphertext = Ciphertext::new(params.module(), params.log_base2k(), params.log_q()+17, 1, ct.log_scale());

    gadget_product_thread_safe(params.module(), &mut ct_out, &ct, &swk.0, &mut tmp_bytes);

    pt_out.zero();

    params.decrypt_rlwe_thread_safe(&mut pt_out, &ct_out, &sk1_svp_ppol, &mut tmp_bytes);

    pt_out.0.value[0].print_limbs(pt_out.limbs(), 16);

    let mut have = vec![i64::default(); params.n()];

    //println!("pt_out: {}", log_k);
    //pt_out.0.value[0].decode_vec_i64(pt_out.log_base2k(), log_k, &mut have);

    //println!("want: {:?}", &want[..16]);
    //println!("have: {:?}", &have[..16]);
 
}


pub fn cast_mut_u64_to_mut_u8_slice(data: &mut [u64]) -> &mut [u8] {
    let ptr: *mut u8 = data.as_mut_ptr() as *mut u8;
    let len: usize = data.len() * std::mem::size_of::<u64>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}