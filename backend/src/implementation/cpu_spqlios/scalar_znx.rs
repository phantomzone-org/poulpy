use crate::{
    hal::{
        api::{ZnxInfos, ZnxSliceSize, ZnxView, ZnxViewMut},
        layouts::{Backend, Module, ScalarZnx, ScalarZnxOwned, ScalarZnxToMut, ScalarZnxToRef},
        oep::{
            ScalarZnxAllocBytesImpl, ScalarZnxAllocImpl, ScalarZnxAutomorphismImpl, ScalarZnxAutomorphismInplaceIml,
            ScalarZnxFromBytesImpl,
        },
    },
    implementation::cpu_spqlios::{
        CPUAVX,
        ffi::{module::module_info_t, vec_znx},
    },
};

unsafe impl<B: Backend> ScalarZnxAllocBytesImpl<B> for B
where
    B: CPUAVX,
{
    fn scalar_znx_alloc_bytes_impl(n: usize, cols: usize) -> usize {
        ScalarZnxOwned::bytes_of(n, cols)
    }
}

unsafe impl<B: Backend> ScalarZnxAllocImpl<B> for B
where
    B: CPUAVX,
{
    fn scalar_znx_alloc_impl(n: usize, cols: usize) -> ScalarZnxOwned {
        ScalarZnxOwned::alloc(n, cols)
    }
}

unsafe impl<B: Backend> ScalarZnxFromBytesImpl<B> for B
where
    B: CPUAVX,
{
    fn scalar_znx_from_bytes_impl(n: usize, cols: usize, bytes: Vec<u8>) -> ScalarZnxOwned {
        ScalarZnxOwned::from_bytes(n, cols, bytes)
    }
}

unsafe impl<B: Backend> ScalarZnxAutomorphismImpl<B> for B
where
    B: CPUAVX,
{
    fn scalar_znx_automorphism_impl<R, A>(module: &Module<B>, k: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: ScalarZnxToMut,
        A: ScalarZnxToRef,
    {
        let a: ScalarZnx<&[u8]> = a.to_ref();
        let mut res: ScalarZnx<&mut [u8]> = res.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
            assert_eq!(res.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                module.ptr() as *const module_info_t,
                k,
                res.at_mut_ptr(res_col, 0),
                res.size() as u64,
                res.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}

unsafe impl<B: Backend> ScalarZnxAutomorphismInplaceIml<B> for B
where
    B: CPUAVX,
{
    fn scalar_znx_automorphism_inplace_impl<A>(module: &Module<B>, k: i64, a: &mut A, a_col: usize)
    where
        A: ScalarZnxToMut,
    {
        let mut a: ScalarZnx<&mut [u8]> = a.to_mut();
        #[cfg(debug_assertions)]
        {
            assert_eq!(a.n(), module.n());
        }
        unsafe {
            vec_znx::vec_znx_automorphism(
                module.ptr() as *const module_info_t,
                k,
                a.at_mut_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
                a.at_ptr(a_col, 0),
                a.size() as u64,
                a.sl() as u64,
            )
        }
    }
}
