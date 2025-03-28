#![feature(bigint_helper_methods)]
#![feature(test)]

pub mod automorphism;
pub mod dft;
pub mod modulus;
pub mod num_bigint;
pub mod poly;
pub mod ring;
pub mod scalar;

pub const CHUNK: usize = 8;
pub const GALOISGENERATOR: usize = 5;

pub mod macros {

    #[macro_export]
    macro_rules! apply_v {
        ($self:expr, $f:expr, $a:expr, $CHUNK:expr) => {
            match CHUNK {
                8 => {
                    $a.chunks_exact_mut(8).for_each(|a| {
                        $f(&$self, &mut a[0]);
                        $f(&$self, &mut a[1]);
                        $f(&$self, &mut a[2]);
                        $f(&$self, &mut a[3]);
                        $f(&$self, &mut a[4]);
                        $f(&$self, &mut a[5]);
                        $f(&$self, &mut a[6]);
                        $f(&$self, &mut a[7]);
                    });

                    let n: usize = $a.len();
                    let m = n - (n & (CHUNK - 1));
                    $a[m..].iter_mut().for_each(|a| {
                        $f(&$self, a);
                    });
                }
                _ => {
                    $a.iter_mut().for_each(|a| {
                        $f(&$self, a);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $CHUNK:expr) => {
            let n: usize = $a.len();
            debug_assert!(
                $b.len() == n,
                "invalid argument b: b.len() = {} != a.len() = {}",
                $b.len(),
                n
            );
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!($a.chunks_exact(8), $b.chunks_exact_mut(8)).for_each(|(a, b)| {
                        $f(&$self, &a[0], &mut b[0]);
                        $f(&$self, &a[1], &mut b[1]);
                        $f(&$self, &a[2], &mut b[2]);
                        $f(&$self, &a[3], &mut b[3]);
                        $f(&$self, &a[4], &mut b[4]);
                        $f(&$self, &a[5], &mut b[5]);
                        $f(&$self, &a[6], &mut b[6]);
                        $f(&$self, &a[7], &mut b[7]);
                    });

                    let m = n - (n & (CHUNK - 1));
                    izip!($a[m..].iter(), $b[m..].iter_mut()).for_each(|(a, b)| {
                        $f(&$self, a, b);
                    });
                }
                _ => {
                    izip!($a.iter(), $b.iter_mut()).for_each(|(a, b)| {
                        $f(&$self, a, b);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vvv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $CHUNK:expr) => {
            let n: usize = $a.len();
            debug_assert!(
                $b.len() == n,
                "invalid argument b: b.len() = {} != a.len() = {}",
                $b.len(),
                n
            );
            debug_assert!(
                $c.len() == n,
                "invalid argument c: b.len() = {} != a.len() = {}",
                $c.len(),
                n
            );
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!(
                        $a.chunks_exact(8),
                        $b.chunks_exact(8),
                        $c.chunks_exact_mut(8)
                    )
                    .for_each(|(a, b, c)| {
                        $f(&$self, &a[0], &b[0], &mut c[0]);
                        $f(&$self, &a[1], &b[1], &mut c[1]);
                        $f(&$self, &a[2], &b[2], &mut c[2]);
                        $f(&$self, &a[3], &b[3], &mut c[3]);
                        $f(&$self, &a[4], &b[4], &mut c[4]);
                        $f(&$self, &a[5], &b[5], &mut c[5]);
                        $f(&$self, &a[6], &b[6], &mut c[6]);
                        $f(&$self, &a[7], &b[7], &mut c[7]);
                    });

                    let m = n - (n & 7);
                    izip!($a[m..].iter(), $b[m..].iter(), $c[m..].iter_mut()).for_each(
                        |(a, b, c)| {
                            $f(&$self, a, b, c);
                        },
                    );
                }
                _ => {
                    izip!($a.iter(), $b.iter(), $c.iter_mut()).for_each(|(a, b, c)| {
                        $f(&$self, a, b, c);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_sv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $CHUNK:expr) => {
            let n: usize = $b.len();
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!($b.chunks_exact_mut(8)).for_each(|b| {
                        $f(&$self, $a, &mut b[0]);
                        $f(&$self, $a, &mut b[1]);
                        $f(&$self, $a, &mut b[2]);
                        $f(&$self, $a, &mut b[3]);
                        $f(&$self, $a, &mut b[4]);
                        $f(&$self, $a, &mut b[5]);
                        $f(&$self, $a, &mut b[6]);
                        $f(&$self, $a, &mut b[7]);
                    });

                    let m = n - (n & 7);
                    izip!($b[m..].iter_mut()).for_each(|b| {
                        $f(&$self, $a, b);
                    });
                }
                _ => {
                    izip!($b.iter_mut()).for_each(|b| {
                        $f(&$self, $a, b);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_svv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $CHUNK:expr) => {
            let n: usize = $b.len();
            debug_assert!(
                $c.len() == n,
                "invalid argument c: c.len() = {} != b.len() = {}",
                $c.len(),
                n
            );
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!($b.chunks_exact(8), $c.chunks_exact_mut(8)).for_each(|(b, c)| {
                        $f(&$self, $a, &b[0], &mut c[0]);
                        $f(&$self, $a, &b[1], &mut c[1]);
                        $f(&$self, $a, &b[2], &mut c[2]);
                        $f(&$self, $a, &b[3], &mut c[3]);
                        $f(&$self, $a, &b[4], &mut c[4]);
                        $f(&$self, $a, &b[5], &mut c[5]);
                        $f(&$self, $a, &b[6], &mut c[6]);
                        $f(&$self, $a, &b[7], &mut c[7]);
                    });

                    let m = n - (n & 7);
                    izip!($b[m..].iter(), $c[m..].iter_mut()).for_each(|(b, c)| {
                        $f(&$self, $a, b, c);
                    });
                }
                _ => {
                    izip!($b.iter(), $c.iter_mut()).for_each(|(b, c)| {
                        $f(&$self, $a, b, c);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vvsv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $d:expr, $CHUNK:expr) => {
            let n: usize = $a.len();
            debug_assert!(
                $b.len() == n,
                "invalid argument b: b.len() = {} != a.len() = {}",
                $b.len(),
                n
            );
            debug_assert!(
                $d.len() == n,
                "invalid argument d: d.len() = {} != a.len() = {}",
                $d.len(),
                n
            );
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!(
                        $a.chunks_exact(8),
                        $b.chunks_exact(8),
                        $d.chunks_exact_mut(8)
                    )
                    .for_each(|(a, b, d)| {
                        $f(&$self, &a[0], &b[0], $c, &mut d[0]);
                        $f(&$self, &a[1], &b[1], $c, &mut d[1]);
                        $f(&$self, &a[2], &b[2], $c, &mut d[2]);
                        $f(&$self, &a[3], &b[3], $c, &mut d[3]);
                        $f(&$self, &a[4], &b[4], $c, &mut d[4]);
                        $f(&$self, &a[5], &b[5], $c, &mut d[5]);
                        $f(&$self, &a[6], &b[6], $c, &mut d[6]);
                        $f(&$self, &a[7], &b[7], $c, &mut d[7]);
                    });

                    let m = n - (n & 7);
                    izip!($a[m..].iter(), $b[m..].iter(), $d[m..].iter_mut()).for_each(
                        |(a, b, d)| {
                            $f(&$self, a, b, $c, d);
                        },
                    );
                }
                _ => {
                    izip!($a.iter(), $b.iter(), $d.iter_mut()).for_each(|(a, b, d)| {
                        $f(&$self, a, b, $c, d);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vsv {
        ($self:expr, $f:expr, $a:expr, $c:expr, $b:expr, $CHUNK:expr) => {
            let n: usize = $a.len();
            debug_assert!(
                $b.len() == n,
                "invalid argument b: b.len() = {} != a.len() = {}",
                $b.len(),
                n
            );
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!($a.chunks_exact(8), $b.chunks_exact_mut(8)).for_each(|(a, b)| {
                        $f(&$self, &a[0], $c, &mut b[0]);
                        $f(&$self, &a[1], $c, &mut b[1]);
                        $f(&$self, &a[2], $c, &mut b[2]);
                        $f(&$self, &a[3], $c, &mut b[3]);
                        $f(&$self, &a[4], $c, &mut b[4]);
                        $f(&$self, &a[5], $c, &mut b[5]);
                        $f(&$self, &a[6], $c, &mut b[6]);
                        $f(&$self, &a[7], $c, &mut b[7]);
                    });

                    let m = n - (n & 7);
                    izip!($a[m..].iter(), $b[m..].iter_mut()).for_each(|(a, b)| {
                        $f(&$self, a, $c, b);
                    });
                }
                _ => {
                    izip!($a.iter(), $b.iter_mut()).for_each(|(a, b)| {
                        $f(&$self, a, $c, b);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vssv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $d:expr, $CHUNK:expr) => {
            let n: usize = $a.len();
            debug_assert!(
                $d.len() == n,
                "invalid argument d: d.len() = {} != a.len() = {}",
                $d.len(),
                n
            );
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!($a.chunks_exact(8), $d.chunks_exact_mut(8)).for_each(|(a, d)| {
                        $f(&$self, &a[0], $b, $c, &mut d[0]);
                        $f(&$self, &a[1], $b, $c, &mut d[1]);
                        $f(&$self, &a[2], $b, $c, &mut d[2]);
                        $f(&$self, &a[3], $b, $c, &mut d[3]);
                        $f(&$self, &a[4], $b, $c, &mut d[4]);
                        $f(&$self, &a[5], $b, $c, &mut d[5]);
                        $f(&$self, &a[6], $b, $c, &mut d[6]);
                        $f(&$self, &a[7], $b, $c, &mut d[7]);
                    });

                    let m = n - (n & 7);
                    izip!($a[m..].iter(), $d[m..].iter_mut()).for_each(|(a, d)| {
                        $f(&$self, a, $b, $c, d);
                    });
                }
                _ => {
                    izip!($a.iter(), $d.iter_mut()).for_each(|(a, d)| {
                        $f(&$self, a, $b, $c, d);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_ssv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $CHUNK:expr) => {
            let n: usize = $c.len();
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!($c.chunks_exact_mut(8)).for_each(|c| {
                        $f(&$self, $a, $b, &mut c[0]);
                        $f(&$self, $a, $b, &mut c[1]);
                        $f(&$self, $a, $b, &mut c[2]);
                        $f(&$self, $a, $b, &mut c[3]);
                        $f(&$self, $a, $b, &mut c[4]);
                        $f(&$self, $a, $b, &mut c[5]);
                        $f(&$self, $a, $b, &mut c[6]);
                        $f(&$self, $a, $b, &mut c[7]);
                    });

                    let m = n - (n & 7);
                    izip!($c[m..].iter_mut()).for_each(|c| {
                        $f(&$self, $a, $b, c);
                    });
                }
                _ => {
                    izip!($c.iter_mut()).for_each(|c| {
                        $f(&$self, $a, $b, c);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vvssv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $CHUNK:expr) => {
            let n: usize = $a.len();
            debug_assert!(
                $b.len() == n,
                "invalid argument b: b.len() = {} != a.len() = {}",
                $b.len(),
                n
            );
            debug_assert!(
                $e.len() == n,
                "invalid argument e: e.len() = {} != a.len() = {}",
                $e.len(),
                n
            );
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!(
                        $a.chunks_exact(8),
                        $b.chunks_exact(8),
                        $e.chunks_exact_mut(8)
                    )
                    .for_each(|(a, b, e)| {
                        $f(&$self, &a[0], &b[0], $c, $d, &mut e[0]);
                        $f(&$self, &a[1], &b[1], $c, $d, &mut e[1]);
                        $f(&$self, &a[2], &b[2], $c, $d, &mut e[2]);
                        $f(&$self, &a[3], &b[3], $c, $d, &mut e[3]);
                        $f(&$self, &a[4], &b[4], $c, $d, &mut e[4]);
                        $f(&$self, &a[5], &b[5], $c, $d, &mut e[5]);
                        $f(&$self, &a[6], &b[6], $c, $d, &mut e[6]);
                        $f(&$self, &a[7], &b[7], $c, $d, &mut e[7]);
                    });

                    let m = n - (n & 7);
                    izip!($a[m..].iter(), $b[m..].iter(), $e[m..].iter_mut()).for_each(
                        |(a, b, e)| {
                            $f(&$self, a, b, $c, $d, e);
                        },
                    );
                }
                _ => {
                    izip!($a.iter(), $b.iter(), $e.iter_mut()).for_each(|(a, b, e)| {
                        $f(&$self, a, b, $c, $d, e);
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! apply_vsssvv {
        ($self:expr, $f:expr, $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $g:expr, $CHUNK:expr) => {
            let n: usize = $a.len();
            debug_assert!(
                $e.len() == n,
                "invalid argument b: e.len() = {} != a.len() = {}",
                $e.len(),
                n
            );
            debug_assert!(
                $g.len() == n,
                "invalid argument g: g.len() = {} != a.len() = {}",
                $g.len(),
                n
            );
            debug_assert!(
                CHUNK & (CHUNK - 1) == 0,
                "invalid CHUNK const: not a power of two"
            );

            match CHUNK {
                8 => {
                    izip!(
                        $a.chunks_exact(8),
                        $e.chunks_exact_mut(8),
                        $g.chunks_exact_mut(8)
                    )
                    .for_each(|(a, e, g)| {
                        $f(&$self, &a[0], $b, $c, $d, &mut e[0], &mut g[0]);
                        $f(&$self, &a[1], $b, $c, $d, &mut e[1], &mut g[1]);
                        $f(&$self, &a[2], $b, $c, $d, &mut e[2], &mut g[2]);
                        $f(&$self, &a[3], $b, $c, $d, &mut e[3], &mut g[3]);
                        $f(&$self, &a[4], $b, $c, $d, &mut e[4], &mut g[4]);
                        $f(&$self, &a[5], $b, $c, $d, &mut e[5], &mut g[5]);
                        $f(&$self, &a[6], $b, $c, $d, &mut e[6], &mut g[6]);
                        $f(&$self, &a[7], $b, $c, $d, &mut e[7], &mut g[7]);
                    });

                    let m = n - (n & 7);
                    izip!($a[m..].iter(), $e[m..].iter_mut(), $g[m..].iter_mut()).for_each(
                        |(a, e, g)| {
                            $f(&$self, a, $b, $c, $d, e, g);
                        },
                    );
                }
                _ => {
                    izip!($a.iter(), $e.iter_mut(), $g.iter_mut()).for_each(|(a, e, g)| {
                        $f(&$self, a, $b, $c, $d, e, g);
                    });
                }
            }
        };
    }
}
