use poulpy_hal::{
    api::SvpPrepare,
    layouts::{Backend, DataMut, Module, ScalarZnx, SvpPPolOwned, SvpPPolToBackendMut, ZnxInfos, ZnxViewMut},
};

pub(crate) fn set_xai_plus_y<C, B: Backend>(
    module: &Module<B>,
    ai: usize,
    y: i64,
    res: &mut SvpPPolOwned<B>,
    buf: &mut ScalarZnx<C>,
) where
    C: DataMut,
    Module<B>: SvpPrepare<B>,
{
    let n: usize = res.n();

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);
        if ai < n {
            raw[ai] = 1;
        } else {
            raw[(ai - n) & (n - 1)] = -1;
        }
        raw[0] += y;
    }

    let mut res_backend = res.to_backend_mut();
    module.svp_prepare(&mut res_backend, 0, buf, 0);

    {
        let raw: &mut [i64] = buf.at_mut(0, 0);

        if ai < n {
            raw[ai] = 0;
        } else {
            raw[(ai - n) & (n - 1)] = 0;
        }
        raw[0] = 0;
    }
}
