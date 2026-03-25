use crate::{
    encoding::classical::{decode, encode, encode_tmp_bytes},
    layouts::plaintext::CKKSPlaintext,
};
use poulpy_core::layouts::{Base2K, Degree, TorusPrecision};
use poulpy_hal::{
    api::{
        ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxIdftApplyConsume,
    },
    layouts::{Backend, Module, ScratchOwned},
};

pub fn test_encode_decode<BE: Backend<ScalarPrep = f64, ScalarBig = i64>>(module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyConsume<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    const BASE2K: u32 = 52;

    let n = module.n();
    let m = n / 2;
    let base2k = Base2K(BASE2K);
    let k = TorusPrecision(17 * BASE2K);

    for log_delta in [20, 30, 40, 50] {
        let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64) - 0.5).collect();
        let im_in: Vec<f64> = (0..m).map(|i| 0.1 * (i as f64) / (m as f64)).collect();

        let mut pt = CKKSPlaintext::alloc(Degree(n as u32), base2k, k, log_delta);
        let mut scratch = ScratchOwned::<BE>::alloc(encode_tmp_bytes(module));
        encode(module, &mut pt, &re_in, &im_in, scratch.borrow());

        let (re_out, im_out) = decode(module, &pt);

        let tol = 8.0 * (n as f64).sqrt() / (1u64 << log_delta) as f64;
        for j in 0..m {
            assert!(
                (re_out[j] - re_in[j]).abs() < tol,
                "log_delta={log_delta}, re[{j}]: got {}, expected {} (ratio: {})",
                re_out[j],
                re_in[j],
                re_out[j] / re_in[j]
            );
            assert!(
                (im_out[j] - im_in[j]).abs() < tol,
                "log_delta={log_delta}, im[{j}]: got {}, expected {} (ratio: {})",
                im_out[j],
                im_in[j],
                im_out[j] / im_in[j]
            );
        }
    }
}
