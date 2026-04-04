use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};
use rand::Rng;

use crate::{
    GLWEAlign, GLWECopy, ScratchTakeCore,
    layouts::{Base2K, Degree, GLWE, GLWELayout, Rank, TorusPrecision},
};

fn assert_glwe_decodes_to(glwe: &GLWE<Vec<u8>>, coeffs_by_col: &[Vec<i64>], k: TorusPrecision, label: &str) {
    for (col, want) in coeffs_by_col.iter().enumerate() {
        let mut have = vec![0i64; want.len()];
        glwe.data()
            .decode_vec_i64(glwe.base2k.as_usize(), col, k.as_usize(), &mut have);
        assert_eq!(have, *want, "{label}: decoded column {col} mismatch");
    }
}

pub fn test_glwe_align<BE: Backend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWEAlign<BE> + GLWECopy,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let base2k = params.base2k as u32;
    let prefix_src = 4 * base2k + 3;
    let offset_src = 2 * base2k + 1;
    let delta = base2k / 2 + 1;
    let payload_bits = prefix_src - offset_src;

    let src_layout = GLWELayout {
        n: Degree(module.n() as u32),
        base2k: Base2K(base2k),
        k: TorusPrecision(prefix_src),
        rank: Rank(2),
    };

    let up_layout = GLWELayout {
        k: TorusPrecision(offset_src + delta + payload_bits),
        ..src_layout
    };
    let down_layout = GLWELayout {
        k: TorusPrecision(offset_src - delta + payload_bits),
        ..src_layout
    };
    let trunc_layout = GLWELayout {
        k: TorusPrecision(prefix_src - delta),
        ..src_layout
    };

    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_align_tmp_bytes(&up_layout, &src_layout)
            .max(module.glwe_align_tmp_bytes(&down_layout, &src_layout))
            .max(module.glwe_align_tmp_bytes(&trunc_layout, &src_layout)),
    );

    let mut source = Source::new([7u8; 32]);
    let mut src = GLWE::alloc_from_infos(&src_layout);
    let mut coeffs_by_col = Vec::with_capacity((src_layout.rank + 1).as_usize());
    for col in 0..(src_layout.rank + 1).as_usize() {
        let mut coeffs = vec![0i64; module.n()];
        for x in &mut coeffs {
            let r = source.next_u64() & ((1u64 << 20) - 1);
            *x = ((r << (64 - 20)) as i64) >> (64 - 20);
        }
        src.data_mut()
            .encode_vec_i64(src_layout.base2k.as_usize(), col, src_layout.k.as_usize(), &coeffs);
        coeffs_by_col.push(coeffs);
    }
    assert_glwe_decodes_to(&src, &coeffs_by_col, src_layout.k, "source");

    let mut aligned_up = GLWE::alloc_from_infos(&up_layout);
    module.glwe_align(&mut aligned_up, offset_src + delta, &src, offset_src, scratch.borrow());
    assert_glwe_decodes_to(&aligned_up, &coeffs_by_col, up_layout.k, "aligned_up");

    let mut roundtrip_up = GLWE::alloc_from_infos(&src_layout);
    module.glwe_align(
        &mut roundtrip_up,
        offset_src,
        &aligned_up,
        offset_src + delta,
        scratch.borrow(),
    );
    assert_eq!(
        roundtrip_up, src,
        "glwe_align up/down roundtrip must preserve the GLWE exactly"
    );
    assert_glwe_decodes_to(&roundtrip_up, &coeffs_by_col, src_layout.k, "roundtrip_up");

    let mut aligned_down = GLWE::alloc_from_infos(&down_layout);
    module.glwe_align(&mut aligned_down, offset_src - delta, &src, offset_src, scratch.borrow());
    assert_glwe_decodes_to(&aligned_down, &coeffs_by_col, down_layout.k, "aligned_down");

    let mut roundtrip_down = GLWE::alloc_from_infos(&src_layout);
    module.glwe_align(
        &mut roundtrip_down,
        offset_src,
        &aligned_down,
        offset_src - delta,
        scratch.borrow(),
    );
    assert_glwe_decodes_to(&roundtrip_down, &coeffs_by_col, src_layout.k, "roundtrip_down");

    let mut trunc_have = GLWE::alloc_from_infos(&trunc_layout);
    module.glwe_align(&mut trunc_have, offset_src, &src, offset_src, scratch.borrow());

    let mut trunc_want = GLWE::alloc_from_infos(&trunc_layout);
    module.glwe_copy(&mut trunc_want, &src);
    assert_eq!(trunc_have, trunc_want, "same-offset glwe_align must reduce to glwe_copy");
}
