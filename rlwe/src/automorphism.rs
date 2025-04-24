use crate::{
    ciphertext::{Ciphertext, new_gadget_ciphertext},
    elem::ElemCommon,
    encryptor::{encrypt_grlwe_sk, encrypt_grlwe_sk_tmp_bytes},
    key_switching::{key_switch_rlwe, key_switch_rlwe_inplace, key_switch_tmp_bytes},
    keys::SecretKey,
    parameters::Parameters,
};
use base2k::{
    Module, Scalar, ScalarOps, SvpPPol, SvpPPolOps, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxOps, VmpPMat,
    VmpPMatOps, assert_alignement,
};
use sampling::source::Source;
use std::{cmp::min, collections::HashMap};

/// Stores DFT([-A*AUTO(s, -p) + 2^{-K*i}*s + E, A]) where AUTO(X, p): X^{i} -> X^{i*p}
pub struct AutomorphismKey {
    pub value: Ciphertext<VmpPMat>,
    pub p: i64,
}

pub fn automorphis_key_new_tmp_bytes(module: &Module, log_base2k: usize, rows: usize, log_q: usize) -> usize {
    module.bytes_of_scalar() + module.bytes_of_svp_ppol() + encrypt_grlwe_sk_tmp_bytes(module, log_base2k, rows, log_q)
}

impl Parameters {
    pub fn automorphism_key_new_tmp_bytes(&self, rows: usize, log_q: usize) -> usize {
        automorphis_key_new_tmp_bytes(self.module(), self.log_base2k(), rows, log_q)
    }

    pub fn automorphism_tmp_bytes(&self, res_logq: usize, in_logq: usize, gct_logq: usize) -> usize {
        automorphism_tmp_bytes(
            self.module(),
            self.log_base2k(),
            res_logq,
            in_logq,
            gct_logq,
        )
    }
}

impl AutomorphismKey {
    pub fn new(
        module: &Module,
        p: i64,
        sk: &SecretKey,
        log_base2k: usize,
        rows: usize,
        log_q: usize,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        tmp_bytes: &mut [u8],
    ) -> Self {
        Self::new_many_core(
            module,
            &vec![p],
            sk,
            log_base2k,
            rows,
            log_q,
            source_xa,
            source_xe,
            sigma,
            tmp_bytes,
        )
        .into_iter()
        .next()
        .unwrap()
    }

    pub fn new_many(
        module: &Module,
        p: &Vec<i64>,
        sk: &SecretKey,
        log_base2k: usize,
        rows: usize,
        log_q: usize,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        tmp_bytes: &mut [u8],
    ) -> HashMap<i64, AutomorphismKey> {
        Self::new_many_core(
            module, p, sk, log_base2k, rows, log_q, source_xa, source_xe, sigma, tmp_bytes,
        )
        .into_iter()
        .zip(p.iter().cloned())
        .map(|(key, pi)| (pi, key))
        .collect()
    }

    fn new_many_core(
        module: &Module,
        p: &Vec<i64>,
        sk: &SecretKey,
        log_base2k: usize,
        rows: usize,
        log_q: usize,
        source_xa: &mut Source,
        source_xe: &mut Source,
        sigma: f64,
        tmp_bytes: &mut [u8],
    ) -> Vec<Self> {
        let (sk_auto_bytes, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_scalar());
        let (sk_out_bytes, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_svp_ppol());

        let sk_auto: Scalar = module.new_scalar_from_bytes_borrow(sk_auto_bytes);
        let mut sk_out: SvpPPol = module.new_svp_ppol_from_bytes_borrow(sk_out_bytes);

        let mut keys: Vec<AutomorphismKey> = Vec::new();

        p.iter().for_each(|pi| {
            let mut value: Ciphertext<VmpPMat> = new_gadget_ciphertext(module, log_base2k, rows, log_q);

            let p_inv: i64 = module.galois_element_inv(*pi);

            module.vec_znx_automorphism(p_inv, &mut sk_auto.as_vec_znx(), &sk.0.as_vec_znx());
            module.svp_prepare(&mut sk_out, &sk_auto);
            encrypt_grlwe_sk(
                module, &mut value, &sk.0, &sk_out, source_xa, source_xe, sigma, tmp_bytes,
            );

            keys.push(Self {
                value: value,
                p: *pi,
            })
        });

        keys
    }
}

pub fn automorphism_tmp_bytes(module: &Module, log_base2k: usize, res_logq: usize, in_logq: usize, gct_logq: usize) -> usize {
    key_switch_tmp_bytes(module, log_base2k, res_logq, in_logq, gct_logq)
}

pub fn automorphism(
    module: &Module,
    c: &mut Ciphertext<VecZnx>,
    a: &Ciphertext<VecZnx>,
    b: &AutomorphismKey,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    key_switch_rlwe(module, c, a, &b.value, b_cols, tmp_bytes);
    // c[0] = AUTO([-b*AUTO(s, -p) + m + e], p) = [-AUTO(b, p)*s + AUTO(m, p) + AUTO(b, e)]
    module.vec_znx_automorphism_inplace(b.p, c.at_mut(0));
    // c[1] = AUTO(b, p)
    module.vec_znx_automorphism_inplace(b.p, c.at_mut(1));
}

pub fn automorphism_inplace_tmp_bytes(module: &Module, c_cols: usize, a_cols: usize, b_rows: usize, b_cols: usize) -> usize {
    return module.vmp_apply_dft_to_dft_tmp_bytes(c_cols, a_cols, b_rows, b_cols)
        + 2 * module.bytes_of_vec_znx_dft(std::cmp::min(c_cols, a_cols));
}

pub fn automorphism_inplace(
    module: &Module,
    a: &mut Ciphertext<VecZnx>,
    b: &AutomorphismKey,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    key_switch_rlwe_inplace(module, a, &b.value, b_cols, tmp_bytes);
    // a[0] = AUTO([-b*AUTO(s, -p) + m + e], p) = [-AUTO(b, p)*s + AUTO(m, p) + AUTO(b, e)]
    module.vec_znx_automorphism_inplace(b.p, a.at_mut(0));
    // a[1] = AUTO(b, p)
    module.vec_znx_automorphism_inplace(b.p, a.at_mut(1));
}

pub fn automorphism_big(
    module: &Module,
    c: &mut Ciphertext<VecZnxBig>,
    a: &Ciphertext<VecZnx>,
    b: &AutomorphismKey,
    tmp_bytes: &mut [u8],
) {
    let cols = std::cmp::min(c.cols(), a.cols());

    #[cfg(debug_assertions)]
    {
        assert!(tmp_bytes.len() >= automorphism_tmp_bytes(module, c.cols(), a.cols(), b.value.rows(), b.value.cols()));
        assert_alignement(tmp_bytes.as_ptr());
    }

    let (tmp_bytes_b1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));
    let (tmp_bytes_res_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let mut a1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_b1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_res_dft);

    // a1_dft = DFT(a[1])
    module.vec_znx_dft(&mut a1_dft, a.at(1));

    // res_dft = IDFT(<DFT(a), DFT([-A*AUTO(s, -p) + 2^{-K*i}*s + E])>) = [-b*AUTO(s, -p) + a * s + e]
    module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.value.at(0), tmp_bytes);
    module.vec_znx_idft_tmp_a(c.at_mut(0), &mut res_dft);

    // res_dft = [-b*AUTO(s, -p) + a * s + e] + [-a * s + m + e] = [-b*AUTO(s, -p) + m + e]
    module.vec_znx_big_add_small_inplace(c.at_mut(0), a.at(0));

    // c[0] = AUTO([-b*AUTO(s, -p) + m + e], p) = [-AUTO(b, p)*s + AUTO(m, p) + AUTO(b, e)]
    module.vec_znx_big_automorphism_inplace(b.p, c.at_mut(0));

    // res_dft = IDFT(<DFT(a), DFT([A])>) = [b]
    module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.value.at(1), tmp_bytes);
    module.vec_znx_idft_tmp_a(c.at_mut(1), &mut res_dft);

    // c[1] = AUTO(b, p)
    module.vec_znx_big_automorphism_inplace(b.p, c.at_mut(1));
}

#[cfg(test)]
mod test {
    use super::{AutomorphismKey, automorphism};
    use crate::{
        ciphertext::Ciphertext,
        decryptor::decrypt_rlwe,
        elem::ElemCommon,
        encryptor::encrypt_rlwe_sk,
        keys::SecretKey,
        parameters::{Parameters, ParametersLiteral},
        plaintext::Plaintext,
    };
    use base2k::{BACKEND, Encoding, Module, SvpPPol, SvpPPolOps, VecZnx, VecZnxOps, alloc_aligned};
    use sampling::source::{Source, new_seed};

    #[test]
    fn test_automorphism() {
        let log_base2k: usize = 10;
        let log_q: usize = 50;
        let log_p: usize = 15;

        // Basic parameters with enough limbs to test edge cases
        let params_lit: ParametersLiteral = ParametersLiteral {
            backend: BACKEND::FFT64,
            log_n: 12,
            log_q: log_q,
            log_p: log_p,
            log_base2k: log_base2k,
            log_scale: 20,
            xe: 3.2,
            xs: 1 << 11,
        };

        let params: Parameters = Parameters::new(&params_lit);

        let module: &Module = params.module();
        let log_q: usize = params.log_q();
        let log_qp: usize = params.log_qp();
        let gct_rows: usize = params.cols_q();
        let gct_cols: usize = params.cols_qp();

        // scratch space
        let mut tmp_bytes: Vec<u8> = alloc_aligned(
            params.decrypt_rlwe_tmp_byte(log_q)
                | params.encrypt_rlwe_sk_tmp_bytes(log_q)
                | params.automorphism_key_new_tmp_bytes(gct_rows, log_qp)
                | params.automorphism_tmp_bytes(log_q, log_q, log_qp),
        );

        // Samplers for public and private randomness
        let mut source_xe: Source = Source::new(new_seed());
        let mut source_xa: Source = Source::new(new_seed());
        let mut source_xs: Source = Source::new(new_seed());

        let mut sk: SecretKey = SecretKey::new(module);
        sk.fill_ternary_hw(params.xs(), &mut source_xs);
        let mut sk_svp_ppol: SvpPPol = module.new_svp_ppol();
        module.svp_prepare(&mut sk_svp_ppol, &sk.0);

        let p: i64 = -5;

        let auto_key: AutomorphismKey = AutomorphismKey::new(
            module,
            p,
            &sk,
            log_base2k,
            gct_rows,
            log_qp,
            &mut source_xa,
            &mut source_xe,
            params.xe(),
            &mut tmp_bytes,
        );

        let mut data: Vec<i64> = vec![0i64; params.n()];

        data.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);

        let log_k: usize = 2 * log_base2k;

        let mut ct: Ciphertext<VecZnx> = params.new_ciphertext(log_q);
        let mut pt: Plaintext = params.new_plaintext(log_q);
        let mut pt_auto: Plaintext = params.new_plaintext(log_q);

        pt.at_mut(0).encode_vec_i64(log_base2k, log_k, &data, 32);
        module.vec_znx_automorphism(p, pt_auto.at_mut(0), pt.at(0));

        encrypt_rlwe_sk(
            module,
            &mut ct.elem_mut(),
            Some(pt.at(0)),
            &sk_svp_ppol,
            &mut source_xa,
            &mut source_xe,
            params.xe(),
            &mut tmp_bytes,
        );

        let mut ct_auto: Ciphertext<VecZnx> = params.new_ciphertext(log_q);

        // ct <- AUTO(ct)
        automorphism(
            module,
            &mut ct_auto,
            &ct,
            &auto_key,
            gct_cols,
            &mut tmp_bytes,
        );

        // pt = dec(auto(ct)) - auto(pt)
        decrypt_rlwe(
            module,
            pt.elem_mut(),
            ct_auto.elem(),
            &sk_svp_ppol,
            &mut tmp_bytes,
        );

        module.vec_znx_sub_ba_inplace(pt.at_mut(0), pt_auto.at(0));

        // pt.at(0).print(pt.cols(), 16);

        let noise_have: f64 = pt.at(0).std(log_base2k).log2();

        let var_msg: f64 = (params.xs() as f64) / params.n() as f64;
        let var_a_err: f64 = 1f64 / 12f64;

        let noise_pred: f64 = params.noise_grlwe_product(var_msg, var_a_err, ct_auto.log_q(), auto_key.value.log_q());

        println!("noise_pred: {}", noise_pred);
        println!("noise_have: {}", noise_have);

        assert!(noise_have <= noise_pred + 1.0);
    }
}
