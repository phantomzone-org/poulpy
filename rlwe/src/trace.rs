use crate::{automorphism::AutomorphismKey, ciphertext::Ciphertext, elem::ElemCommon, parameters::Parameters};
use base2k::{Module, VecZnx, VecZnxBig, VecZnxBigOps, VecZnxDft, VecZnxDftOps, VmpPMatOps, assert_alignement};
use std::collections::HashMap;

pub fn trace_galois_elements(module: &Module) -> Vec<i64> {
    let mut gal_els: Vec<i64> = Vec::new();
    (0..module.log_n()).for_each(|i| {
        if i == 0 {
            gal_els.push(-1);
        } else {
            gal_els.push(module.galois_element(1 << (i - 1)));
        }
    });
    gal_els
}

impl Parameters {
    pub fn trace_tmp_bytes(&self, res_logq: usize, in_logq: usize, gct_logq: usize) -> usize {
        self.automorphism_tmp_bytes(res_logq, in_logq, gct_logq)
    }
}

pub fn trace_tmp_bytes(module: &Module, c_cols: usize, a_cols: usize, b_rows: usize, b_cols: usize) -> usize {
    return module.vmp_apply_dft_to_dft_tmp_bytes(c_cols, a_cols, b_rows, b_cols)
        + 2 * module.bytes_of_vec_znx_dft(std::cmp::min(c_cols, a_cols));
}

pub fn trace_inplace(
    module: &Module,
    a: &mut Ciphertext<VecZnx>,
    start: usize,
    end: usize,
    b: &HashMap<i64, AutomorphismKey>,
    b_cols: usize,
    tmp_bytes: &mut [u8],
) {
    let cols: usize = a.cols();

    let b_rows: usize;

    if let Some((_, key)) = b.iter().next() {
        b_rows = key.value.rows();
        #[cfg(debug_assertions)]
        {
            println!("{} {}", b_cols, key.value.cols());
            assert!(b_cols <= key.value.cols())
        }
    } else {
        panic!("b: HashMap<i64, AutomorphismKey>, is empty")
    }

    #[cfg(debug_assertions)]
    {
        assert!(start <= end);
        assert!(end <= module.n());
        assert!(tmp_bytes.len() >= trace_tmp_bytes(module, cols, cols, b_rows, b_cols));
        assert_alignement(tmp_bytes.as_ptr());
    }

    let cols: usize = std::cmp::min(b_cols, a.cols());

    let (tmp_bytes_b1_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(cols));
    let (tmp_bytes_res_dft, tmp_bytes) = tmp_bytes.split_at_mut(module.bytes_of_vec_znx_dft(b_cols));

    let mut a1_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(cols, tmp_bytes_b1_dft);
    let mut res_dft: VecZnxDft = module.new_vec_znx_dft_from_bytes_borrow(b_cols, tmp_bytes_res_dft);
    let mut res_big: VecZnxBig = res_dft.as_vec_znx_big();

    let log_base2k: usize = a.log_base2k();

    (start..end).for_each(|i| {
        a.at_mut(0).rsh(log_base2k, 1, tmp_bytes);
        a.at_mut(1).rsh(log_base2k, 1, tmp_bytes);

        let p: i64;
        if i == 0 {
            p = -1;
        } else {
            p = module.galois_element(1 << (i - 1));
        }

        if let Some(key) = b.get(&p) {
            module.vec_znx_dft(&mut a1_dft, a.at(1));

            // a[0] = NORMALIZE(a[0] + AUTO(a[0] + IDFT(<DFT(a[1]), key[0]>)))
            module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, key.value.at(0), tmp_bytes);
            module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);
            module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0));
            module.vec_znx_big_automorphism_inplace(p, &mut res_big);
            module.vec_znx_big_add_small_inplace(&mut res_big, a.at(0));
            module.vec_znx_big_normalize(a.log_base2k(), a.at_mut(0), &mut res_big, tmp_bytes);

            // a[1] = NORMALIZE(a[1] + AUTO(IDFT(<DFT(a[1]), key[1]>)))
            module.vmp_apply_dft_to_dft(&mut res_dft, &a1_dft, key.value.at(1), tmp_bytes);
            module.vec_znx_idft_tmp_a(&mut res_big, &mut res_dft);
            module.vec_znx_big_automorphism_inplace(p, &mut res_big);
            module.vec_znx_big_add_small_inplace(&mut res_big, a.at(1));
            module.vec_znx_big_normalize(a.log_base2k(), a.at_mut(1), &mut res_big, tmp_bytes);
        } else {
            panic!("b[{}] is empty", p)
        }
    })
}

#[cfg(test)]
mod test {
    use super::{trace_galois_elements, trace_inplace};
    use crate::{
        automorphism::AutomorphismKey,
        ciphertext::Ciphertext,
        decryptor::decrypt_rlwe,
        elem::ElemCommon,
        encryptor::encrypt_rlwe_sk,
        keys::SecretKey,
        parameters::{DEFAULT_SIGMA, Parameters, ParametersLiteral},
        plaintext::Plaintext,
    };
    use base2k::{BACKEND, Encoding, Module, SvpPPol, SvpPPolOps, VecZnx, alloc_aligned};
    use sampling::source::{Source, new_seed};
    use std::collections::HashMap;

    #[test]
    fn test_trace_inplace() {
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

        let gal_els: Vec<i64> = trace_galois_elements(module);

        let auto_keys: HashMap<i64, AutomorphismKey> = AutomorphismKey::new_many(
            module,
            &gal_els,
            &sk,
            log_base2k,
            gct_rows,
            log_qp,
            &mut source_xa,
            &mut source_xe,
            DEFAULT_SIGMA,
            &mut tmp_bytes,
        );

        let mut data: Vec<i64> = vec![0i64; params.n()];

        data.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = 1 + i as i64);

        let log_k: usize = 2 * log_base2k;

        let mut ct: Ciphertext<VecZnx> = params.new_ciphertext(log_q);
        let mut pt: Plaintext = params.new_plaintext(log_q);

        pt.at_mut(0).encode_vec_i64(log_base2k, log_k, &data, 32);
        pt.at_mut(0).normalize(log_base2k, &mut tmp_bytes);

        pt.at(0).decode_vec_i64(log_base2k, log_k, &mut data);

        pt.at(0).print(pt.cols(), 16);

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

        trace_inplace(module, &mut ct, 0, 4, &auto_keys, gct_cols, &mut tmp_bytes);
        trace_inplace(
            module,
            &mut ct,
            4,
            module.log_n(),
            &auto_keys,
            gct_cols,
            &mut tmp_bytes,
        );

        // pt = dec(auto(ct)) - auto(pt)
        decrypt_rlwe(
            module,
            pt.elem_mut(),
            ct.elem(),
            &sk_svp_ppol,
            &mut tmp_bytes,
        );

        pt.at(0).print(pt.cols(), 16);

        pt.at(0).decode_vec_i64(log_base2k, log_k, &mut data);

        println!("trace: {:?}", &data[..16]);
    }
}
