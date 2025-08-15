# 🐙 Poulpy

<p  align="center">
<img  src="poulpy.png"  />
</p>

[![CI](https://github.com/phantomzone-org/poulpy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/phantomzone-org/poulpy/actions/workflows/ci.yml)

**Poulpy** is a fast & modular FHE library that implements Ring-Learning-With-Errors based homomorphic encryption. It adopts the bivariate polynomial representation proposed in [Revisiting Key Decomposition Techniques for FHE: Simpler, Faster and More Generic](https://eprint.iacr.org/2023/771). In addition to simpler and more efficient arithmetic than the residue number system (RNS), this representation provides a common plaintext space for all schemes and allows easy switching between any two schemes. Poulpy also decouples the schemes implementations from the polynomial arithmetic backend by being built around a hardware abstraction layer (HAL). This enables user to easily provide or use a custom backend.

### Bivariate Polynomial Representation

Existing FHE implementations (such as [Lattigo](https://github.com/tuneinsight/lattigo) or [OpenFHE](https://github.com/openfheorg/openfhe-development)) use the [residue-number-system](https://en.wikipedia.org/wiki/Residue_number_system) (RNS) to represent large integers. Although the parallelism and carry-less arithmetic provided by the RNS representation provides a very efficient modular arithmetic over large-integers, it suffers from various drawbacks when used in the context of FHE. The main idea behind the bivariate representation is to decouple the cyclotomic arithmetic from the large number arithmetic. Instead of using the RNS representation for large integer, integers are decomposed in base $2^{-K}$ over the Torus $\mathbb{T}_{N}[X]$. 

This provides the following benefits:

- **Intuitive, efficient and reusable parameterization & instances:** Only the bit-size of the modulus is required from the user (i.e. Torus precision). As such, parameterization is natural and generic, and instances can be reused for any circuit consuming the same homomorphic capacity, without loss of efficiency. With the RNS representation, individual NTT friendly primes needs to be specified for each level, making the parameterization not user friendly and circuit-specific.

- **Optimal and granular rescaling:** Ciphertext rescaling is carried out with bit-shifting, enabling a bit-level granular rescaling and optimal noise/homomorphic capacity management. In the RNS representation, ciphertext division can only be done by one of the primes composing the modulus, leading to difficult scaling management and frequent inefficient noise/homomorphic capacity management.

- **Linear number of DFT in the half external product:** The bivariate representation of the coefficients implicitly provides the digit decomposition, as such the number of DFT is linear in the number of limbs, contrary to the RNS representation where it is quadratic due to the RNS basis conversion. This enables a much more efficient key-switching, which is the **most used and expensive** FHE operation. 

- **Unified plaintext space:** The bivariate polynomial representation is by essence a high precision discretized representation of the Torus $\mathbb{T}_{N}[X]$. Using the Torus as the common plaintext space for all schemes achieves the vision of [CHIMERA: Combining Ring-LWE-based Fully Homomorphic Encryption Schemes](https://eprint.iacr.org/2018/758) which is to unify all RLWE-based FHE schemes (TFHE, FHEW, BGV, BFV, CLPX, GBFV, CKKS, ...) under a single scheme with different encodings, enabling native and efficient scheme-switching functionalities.

- **Simpler implementation**: Since the cyclotomic arithmetic is decoupled from the coefficient representation, the same pipeline (including DFT) can be reused for all limbs (unlike in the RNS representation), making this representation a prime target for hardware acceleration.

- **Deterministic computation**: Although being defined on the Torus, bivariate arithmetic remains integer polynomial arithmetic, ensuring all computations are deterministic, the contract being that output should be reproducible and identical, regardless of the backend or hardware.

### Hardware Abstraction Layer 

In addition to providing a general purpose FHE library over a unified plaintext space, Poulpy is also designed from the ground up around a **hardware abstraction layer** that closely matches the API of [spqlios-arithmetic](https://github.com/tfhe/spqlios-arithmetic). The bivariate representation is by itself hardware friendly as it uses flat, aligned & vectorized memory layout. Finally, generic opaque write only structs (prepared versions) are provided, making it easy for developers to provide hardware focused/optimized operations. This makes possible for anyone to provide or use a custom backend.

## Library Overview

- **`backend/hal`**: hardware abstraction layer. This layer targets users that want to provide their own backend or use a third party backend.
 
  - **`api`**: fixed public low-level polynomial level arithmetic API closely matching spqlios-arithmetic. The goal is to eventually freeze this API, in order to decouple it from the OEP traits, ensuring that changes to implementations do not affect the front end API.

	```rust
	pub trait SvpPrepare<B: Backend> {
	    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
	    where
	        R: SvpPPolToMut<B>,
	        A: ScalarZnxToRef;
	}
	````

  - **`delegates`**: link between the user facing API and implementation OEP. Each trait of `api` is implemented by calling its corresponding trait on the `oep`.

	```rust
	impl<B> SvpPrepare<B> for Module<B>
	where
	    B: Backend + SvpPrepareImpl<B>,
	{
	    fn svp_prepare<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
	    where
	        R: SvpPPolToMut<B>,
	        A: ScalarZnxToRef,
	    {
	        B::svp_prepare_impl(self, res, res_col, a, a_col);
	    }
	}
	```

  - **`layouts`**: defines the layouts of the front-end algebraic structs matching spqlios-arithmetic definitions, such as `ScalarZnx`, `VecZnx` or opaque backend prepared struct such as `SvpPPol` and `VmpPMat`.

	```rust
	pub struct SvpPPol<D: Data, B: Backend> {
	    data: D,
	    n: usize,
	    cols: usize,
	    _phantom: PhantomData<B>,
	}
	```

  - **`oep`**: open extension points, which can be implemented by the user to provide a custom backend.

	```rust
	pub unsafe trait SvpPrepareImpl<B: Backend> {
	    fn svp_prepare_impl<R, A>(module: &Module<B>, res: &mut R, res_col: usize, a: &A, a_col: usize)
	    where
	        R: SvpPPolToMut<B>,
	        A: ScalarZnxToRef;
	}
	```

  - **`tests`**: exported generic tests for the OEP/structs. Their goal is to enable a user to automatically be able to test its backend implementation, without having to re-implement any tests.
  
- **`backend/implementation`**:
  - **`cpu_spqlios`**: concrete cpu implementation of the hal through the oep using bindings on spqlios-arithmetic. This implementation currently supports the `FFT64` backend and will be extended to support the `NTT120` backend once it is available in spqlios-arithmetic.

	```rust
	unsafe impl SvpPrepareImpl<Self> for FFT64 {
	    fn svp_prepare_impl<R, A>(module: &Module<Self>, res: &mut R, res_col: usize, a: &A, a_col: usize)
	    where
	        R: SvpPPolToMut<Self>,
	        A: ScalarZnxToRef,
	    {
	        unsafe {
	            svp::svp_prepare(
	                module.ptr(),
	                res.to_mut().at_mut_ptr(res_col, 0) as *mut svp::svp_ppol_t,
	                a.to_ref().at_ptr(a_col, 0),
	            )
	        }
	    }
	}
	```

- **`core`**: core of the FHE library, implementing scheme agnostic RLWE arithmetic for LWE, GLWE, GGLWE and GGSW ciphertexts. It notably includes all possible cross-ciphertext operations, for example applying an external product on a GGLWE or an automorphism on a GGSW, as well as blind rotation. This crate is entirely implemented using the hardware abstraction layer API, and is thus solely defined over generic and traits (including tests). As such it will work over any backend, as long as it implements the necessary traits defined in the OEP.

	```rust
	pub struct GLWESecret<D: Data> {
	    pub(crate) data: ScalarZnx<D>,
	    pub(crate) dist: Distribution,
	}

	pub struct GLWESecrecPrepared<D: Data, B: Backend> {
	    pub(crate) data: SvpPPol<D, B>,
	    pub(crate) dist: Distribution,
	}
 
	impl<D: DataMut, B: Backend> GLWESecretPrepared<D, B> {
	    pub fn prepare<O>(&mut self, module: &Module<B>, sk: &GLWESecret<O>)
	    where
	        O: DataRef,
	        Module<B>: SvpPrepare<B>,
	    {
	        (0..self.rank()).for_each(|i| {
	            module.svp_prepare(&mut self.data, i, &sk.data, i);
	        });
	        self.dist = sk.dist
	    }
	}
	```

## Installation

TBD — currently not published on crates.io. Clone the repository and use via path-based dependencies.

## Documentation

* Full `cargo doc` documentation is coming soon.
* Architecture diagrams and design notes will be added in the [`/doc`](./doc) folder.

## Contributing

We welcome external contributions, please see [CONTRIBUTING](./CONTRIBUTING.md).

## Security

Please see [SECURITY](./SECURITY.md).

## License

Poulpy is licensed under the Apache 2.0 License. See [NOTICE](./NOTICE) & [LICENSE](./LICENSE).

## Acknowledgement

**Poulpy** is inspired by the modular architecture of [Lattigo](https://github.com/tuneinsight/lattigo) and [TFHE-go](https://github.com/sp301415/tfhe-go), and its development is lead by Lattigo’s co-author and main contributor [@Pro7ech](https://github.com/Pro7ech). Poulpy reflects the experience gained from over five years of designing and maintaining Lattigo, and represents the next evolution in architecture, performance, and backend philosophy.

## Citing
Please use the following BibTex entry for citing Lattigo

    @misc{poulpy,
	    title = {Poulpy v0.1.0},
	    howpublished = {Online: \url{https://github.com/phantomzone-org/poulpy}},
	    month = Aug,
	    year = 2025,
	    note = {Phantom Zone}
    }
