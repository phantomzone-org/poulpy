use crate::{
    layouts::{VecZnx, ZnxInfos, ZnxViewMut},
    source::Source,
};

pub fn test_vec_znx_encode_vec_i64_lo_norm() {
    let n: usize = 32;
    let basek: usize = 17;
    let size: usize = 5;
    let k: usize = size * basek - 5;
    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, 2, size);
    let mut source: Source = Source::new([0u8; 32]);
    let raw: &mut [i64] = a.raw_mut();
    raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
    (0..a.cols()).for_each(|col_i| {
        let mut have: Vec<i64> = vec![i64::default(); n];
        have.iter_mut()
            .for_each(|x| *x = (source.next_i64() << 56) >> 56);
        a.encode_vec_i64(basek, col_i, k, &have, 10);
        let mut want: Vec<i64> = vec![i64::default(); n];
        a.decode_vec_i64(basek, col_i, k, &mut want);
        assert_eq!(have, want, "{:?} != {:?}", &have, &want);
    });
}

pub fn test_vec_znx_encode_vec_i64_hi_norm() {
    let n: usize = 32;
    let basek: usize = 17;
    let size: usize = 5;
    for k in [1, basek / 2, size * basek - 5] {
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, 2, size);
        let mut source = Source::new([0u8; 32]);
        let raw: &mut [i64] = a.raw_mut();
        raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        (0..a.cols()).for_each(|col_i| {
            let mut have: Vec<i64> = vec![i64::default(); n];
            have.iter_mut().for_each(|x| {
                if k < 64 {
                    *x = source.next_u64n(1 << k, (1 << k) - 1) as i64;
                } else {
                    *x = source.next_i64();
                }
            });
            a.encode_vec_i64(basek, col_i, k, &have, 63);
            let mut want: Vec<i64> = vec![i64::default(); n];
            a.decode_vec_i64(basek, col_i, k, &mut want);
            assert_eq!(have, want, "{:?} != {:?}", &have, &want);
        })
    }
}
