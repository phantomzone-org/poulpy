use crate::{
    ciphertext::{Ciphertext, new_gadget_ciphertext},
    elem::ElemCommon,
    encryptor::{encrypt_grlwe_sk, encrypt_grlwe_sk_tmp_bytes},
    keys::SecretKey,
    parameters::Parameters,
};
use base2k::{
    Module, Scalar, ScalarOps, SvpPPol, SvpPPolOps, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft,
    VecZnxDftOps, VecZnxOps, VmpPMat, VmpPMatOps, assert_alignement,
};
use sampling::source::Source;
use std::cmp::min;

/// Stores DFT([-A*AUTO(s, -p) + 2^{-K*i}*s + E, A]) where AUTO(X, p): X^{i} -> X^{i*p}
pub struct AutomorphismKey {
    pub value: Ciphertext<VmpPMat>,
    pub p: i64,
}

pub fn automorphis_key_new_tmp_bytes(
    module: &Module,
    log_base2k: usize,
    rows: usize,
    log_q: usize,
) -> usize {
    module.bytes_of_scalar()
        + module.bytes_of_svp_ppol()
        + encrypt_grlwe_sk_tmp_bytes(module, log_base2k, rows, log_q)
}

impl Parameters {
    pub fn automorphism_key_new_tmp_bytes(&self, rows: usize, log_q: usize) -> usize {
        automorphis_key_new_tmp_bytes(self.module(), self.log_base2k(), rows, log_q)
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
        let (sk_auto_bytes, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_scalar());
        let (sk_out_bytes, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_svp_ppol());

        let sk_auto: Scalar = module.new_scalar_from_bytes_borrow(sk_auto_bytes);
        let mut sk_out: SvpPPol = module.new_svp_ppol_from_bytes_borrow(sk_out_bytes);
        let mut value: Ciphertext<VmpPMat> = new_gadget_ciphertext(module, log_base2k, rows, log_q);

        let p_inv: i64 = module.galois_element_inv(p);

        module.vec_znx_automorphism(p_inv, &mut sk_auto.as_vec_znx(), &sk.0.as_vec_znx());
        module.svp_prepare(&mut sk_out, &sk_auto);
        encrypt_grlwe_sk(
            module, &mut value, &sk.0, &sk_out, source_xa, source_xe, sigma, tmp_bytes,
        );

        Self { value: value, p: p }
    }
}

pub fn automorphism_tmp_bytes(
    module: &Module,
    c_cols: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
) -> usize {
    return module.vmp_apply_dft_to_dft_tmp_bytes(c_cols, a_cols, b_rows, b_cols)
        + 2 * module.bytes_of_vec_znx_dft(std::cmp::min(c_cols, a_cols));
}

pub fn automorphism(
    module: &Module,
    c: &mut Ciphertext<VecZnx>,
    a: &Ciphertext<VecZnx>,
    b: &AutomorphismKey,
    tmp_bytes: &mut [u8],
) {
    let cols: usize = min(min(c.cols(), a.cols()), b.value.rows());

    #[cfg(debug_assertions)]
    {
        assert!(
            tmp_bytes.len()
                >= automorphism_tmp_bytes(
                    module,
                    c.cols(),
                    a.cols(),
                    b.value.rows(),
                    b.value.cols()
                )
        );
        assert_alignement(tmp_bytes.as_ptr());
    }

    let (tmp_bytes_b1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));
    let (tmp_bytes_res_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let mut a1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_b1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_res_dft);
    let mut res_big: VecZnxBig = res_dft.as_vec_znx_big();

    // a1_dft = DFT(a[1])
    module.vec_znx_dft(&mut a1_dft, a.at(1));

    // res_dft = IDFT(<DFT(a), DFT([-A*AUTO(s, -p) + 2^{-K*i}*s + E])>) = [-b*AUTO(s, -p) + a * s + e]
    module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.value.at(0), tmp_bytes);
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);

    // res_dft = [-b*AUTO(s, -p) + a * s + e] + [-a * s + m + e] = [-b*AUTO(s, -p) + m + e]
    module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0));

    // c[0] = NORMALIZE([-b*AUTO(s, -p) + m + e])
    module.vec_znx_big_normalize(c.log_base2k(), c.at_mut(0), &mut res_big, tmp_bytes);

    // c[0] = AUTO([-b*AUTO(s, -p) + m + e], p) = [-AUTO(b, p)*s + AUTO(m, p) + AUTO(b, e)]
    module.vec_znx_automorphism_inplace(b.p, c.at_mut(0));

    // res_dft = IDFT(<DFT(a), DFT([A])>) = [b]
    module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.value.at(1), tmp_bytes);
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);

    // c[1] = b
    module.vec_znx_big_normalize(c.log_base2k(), c.at_mut(1), &mut res_big, tmp_bytes);

    // c[1] = AUTO(b, p)
    module.vec_znx_automorphism_inplace(b.p, c.at_mut(1));
}

pub fn automorphism_inplace_tmp_bytes(
    module: &Module,
    c_cols: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
) -> usize {
    return module.vmp_apply_dft_to_dft_tmp_bytes(c_cols, a_cols, b_rows, b_cols)
        + 2 * module.bytes_of_vec_znx_dft(std::cmp::min(c_cols, a_cols));
}

pub fn automorphism_inplace(
    module: &Module,
    a: &mut Ciphertext<VecZnx>,
    b: &AutomorphismKey,
    tmp_bytes: &mut [u8],
) {
    let cols: usize = min(a.cols(), b.value.rows());

    #[cfg(debug_assertions)]
    {
        assert!(
            tmp_bytes.len()
                >= automorphism_inplace_tmp_bytes(
                    module,
                    a.cols(),
                    a.cols(),
                    b.value.rows(),
                    b.value.cols()
                )
        );
        assert_alignement(tmp_bytes.as_ptr());
    }

    let (tmp_bytes_b1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));
    let (tmp_bytes_res_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));

    let mut a1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_b1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_res_dft);
    let mut res_big: VecZnxBig = res_dft.as_vec_znx_big();

    // a1_dft = DFT(a[1])
    module.vec_znx_dft(&mut a1_dft, a.at(1));

    // res_dft = IDFT(<DFT(a), DFT([-A*AUTO(s, -p) + 2^{-K*i}*s + E])>) = [-b*AUTO(s, -p) + a * s + e]
    module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.value.at(0), tmp_bytes);
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);

    // res_dft = [-b*AUTO(s, -p) + a * s + e] + [-a * s + m + e] = [-b*AUTO(s, -p) + m + e]
    module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0));

    // a[0] = NORMALIZE([-b*AUTO(s, -p) + m + e])
    module.vec_znx_big_normalize(a.log_base2k(), a.at_mut(0), &mut res_big, tmp_bytes);

    // a[0] = AUTO([-b*AUTO(s, -p) + m + e], p) = [-AUTO(b, p)*s + AUTO(m, p) + AUTO(b, e)]
    module.vec_znx_automorphism_inplace(b.p, a.at_mut(0));

    // res_dft = IDFT(<DFT(a), DFT([A])>) = [b]
    module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, b.value.at(1), tmp_bytes);
    module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);

    // a[1] = b
    module.vec_znx_big_normalize(a.log_base2k(), a.at_mut(1), &mut res_big, tmp_bytes);

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
        assert!(
            tmp_bytes.len()
                >= automorphism_tmp_bytes(
                    module,
                    c.cols(),
                    a.cols(),
                    b.value.rows(),
                    b.value.cols()
                )
        );
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
    use crate::{
        ciphertext::{new_gadget_ciphertext, Ciphertext}, decryptor::decrypt_rlwe, elem::{Elem, ElemCommon, ElemVecZnx}, encryptor::encrypt_rlwe_sk, keys::SecretKey, parameters::{Parameters, ParametersLiteral}, plaintext::Plaintext
    };
    use base2k::{
        alloc_aligned, Encoding, Infos, Sampling, SvpPPolOps, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VecZnxOps, VmpPMat, BACKEND
    };
    use sampling::source::{Source, new_seed};

    use super::{automorphis_key_new_tmp_bytes, automorphism, AutomorphismKey};

    #[test]
    fn test_automorphism() {
        let log_base2k: usize = 10;
        let q_cols: usize = 4;
        let p_cols: usize = 1;

        // Basic parameters with enough limbs to test edge cases
        let params_lit: ParametersLiteral = ParametersLiteral {
            backend: BACKEND::FFT64,
            log_n: 12,
            log_q: q_cols * log_base2k,
            log_p: p_cols * log_base2k,
            log_base2k: log_base2k,
            log_scale: 20,
            xe: 3.2,
            xs: 1 << 11,
        };

        let params: Parameters = Parameters::new(&params_lit);

        let module: &base2k::Module = params.module();
        let log_q: usize = params.log_q();
        let log_qp: usize = params.log_qp();
        let rows: usize = params.cols_q();

        // scratch space
        let mut tmp_bytes: Vec<u8> = alloc_aligned(
            params.decrypt_rlwe_tmp_byte(log_q)
                | params.encrypt_rlwe_sk_tmp_bytes(log_q)
                | params.gadget_product_tmp_bytes(
                    log_qp,
                    log_qp,
                    rows,
                    log_qp,
                )
                | params.encrypt_grlwe_sk_tmp_bytes(rows, log_qp)
                | params.automorphism_key_new_tmp_bytes(rows, log_qp),
        );

        // Samplers for public and private randomness
        let mut source_xe: Source = Source::new(new_seed());
        let mut source_xa: Source = Source::new(new_seed());
        let mut source_xs: Source = Source::new(new_seed());

        // Two secret keys
        let mut sk: SecretKey = SecretKey::new(module);
        sk.fill_ternary_hw(params.xs(), &mut source_xs);
        let mut sk_svp_ppol: base2k::SvpPPol = module.new_svp_ppol();
        module.svp_prepare(&mut sk_svp_ppol, &sk.0);

        let p: i64 = -5;

        let auto_key: AutomorphismKey = AutomorphismKey::new(
            module,
            p,
            &sk,
            log_base2k,
            rows,
            log_qp,
            &mut source_xa,
            &mut source_xe,
            params.xe(),
            &mut tmp_bytes,
        );

        let mut data: Vec<i64> = vec![0i64; params.n()];

        data.iter_mut().enumerate().for_each(|(i, x)|{
            *x = i as i64
        });

        let log_k: usize = 2*log_base2k;

        let mut ct: Ciphertext<VecZnx> = Ciphertext::new(module, log_base2k, log_q, 2);
        let mut pt: Plaintext = Plaintext::new(module, log_base2k, log_q);

        pt.at_mut(0).encode_vec_i64(log_base2k, log_k, &data, 32);

        encrypt_rlwe_sk(module, &mut ct.elem_mut(), Some(pt.elem()), &sk_svp_ppol, &mut source_xa, &mut source_xe, params.xe(), &mut tmp_bytes);

        module.vec_znx_automorphism_inplace(p, pt.at_mut(0));

        let mut ct_auto: Ciphertext<VecZnx> = Ciphertext::new(module, log_base2k, log_q, 2);

        automorphism(module, &mut ct_auto, &ct, &auto_key, &mut tmp_bytes);

        module.vec_znx_sub_inplace(ct_auto.at_mut(0), pt.at(0));
        ct_auto.at_mut(0).normalize(log_base2k, &mut tmp_bytes);

        decrypt_rlwe(module, pt.elem_mut(), ct_auto.elem(), &sk_svp_ppol, &mut tmp_bytes);
  
        let noise_have: f64 = pt.at(0).std(log_base2k).log2();

        let var_a_err: f64;
        if ct_auto.cols() < ct.cols() {
            var_a_err = 1f64 / 12f64;
        } else {
            var_a_err = 0f64;
        }

        let var_msg: f64 = (params.xs() as f64) / params.n() as f64;

        let noise_pred: f64 =
            params.noise_grlwe_product(var_msg, var_a_err, ct_auto.log_q(), auto_key.value.log_q());

            println!("noise_pred: {}", noise_have);
            println!("noise_have: {}", noise_pred);
            
        assert!(noise_have <= noise_pred + 1.0);

        

    }
}
