use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{ZnxCopy, ZnxZero},
};

pub fn vec_znx_copy<'r, 'a, BE>(res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, a: &VecZnxBackendRef<'a, BE>, a_col: usize)
where
    BE: Backend + ZnxCopy + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size = res.size();
    let a_size = a.size();

    let min_size = res_size.min(a_size);

    for j in 0..min_size {
        BE::znx_copy(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_copy_range<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    res_limb: usize,
    res_offset: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    a_limb: usize,
    a_offset: usize,
    len: usize,
) where
    BE: Backend + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    assert!(res_offset + len <= res.n());
    assert!(a_offset + len <= a.n());

    BE::znx_copy(
        &mut res.at_mut(res_col, res_limb)[res_offset..res_offset + len],
        &a.at(a_col, a_limb)[a_offset..a_offset + len],
    );
}
