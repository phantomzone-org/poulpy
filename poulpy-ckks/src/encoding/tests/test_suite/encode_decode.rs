use crate::{
    encoding::classical::{decode, encode},
    layouts::plaintext::CKKSPlaintext,
};
use poulpy_core::layouts::{Base2K, Degree, TorusPrecision};

pub fn test_encode_decode(n: usize) {
    const BASE2K: u32 = 52;

    let m = n / 2;
    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(29 * BASE2K);

    for log_delta in [30, 40, 50] {
        let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
        let im_in: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();

        let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, log_delta);
        encode(&mut pt, &re_in, &im_in);
        let (re_out, im_out) = decode(&pt);

        let mut max_err: f64 = 0.0;
        for j in 0..m {
            max_err = max_err.max((re_out[j] - re_in[j]).abs());
            max_err = max_err.max((im_out[j] - im_in[j]).abs());
        }

        let precision = -(max_err.log2());

        eprintln!("encode_decode: n={n} base2k={BASE2K} log_delta={log_delta}: precision={precision:.1} bits",);

        assert!(
            precision >= log_delta as f64 - 12.0,
            "precision {precision:.1} bits is too low"
        );
    }
}
