use std::env;

fn nvcc_available() -> bool {
    std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() {
    println!("cargo:rerun-if-changed=cuda/ntt120_ntt.cu");
    println!("cargo:rerun-if-changed=cuda/ntt120_crt.cu");
    println!("cargo:rerun-if-changed=cuda/ntt120_cnv.cu");
    println!("cargo:rerun-if-changed=cuda/ntt120_vec_znx.cu");
    println!("cargo:rerun-if-changed=cuda/ntt120_big.cu");
    println!("cargo:rerun-if-changed=cuda/ntt120_shift.cu");
    println!("cargo:rerun-if-changed=cuda/ntt120_internal.cuh");

    if !nvcc_available() {
        println!("cargo:warning=nvcc not found; skipping NTT120 CUDA kernel compilation");
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let mut build = cc::Build::new();
    build.cuda(true).flag("-std=c++17").flag("--expt-relaxed-constexpr");

    match env::var("CUDA_ARCH") {
        Ok(arch) => {
            // Accept "sm_89", "89", etc.  Strip optional "sm_" prefix.
            let cc = arch.trim().trim_start_matches("sm_");
            // Generate SASS for the target arch and PTX for forward compatibility.
            build.flag(format!("-gencode=arch=compute_{cc},code=sm_{cc}"));
            build.flag(format!("-gencode=arch=compute_{cc},code=compute_{cc}"));
        }
        Err(_) => {
            // Default: SASS for the build-host GPU.
            build.flag("-arch=native");
        }
    }

    build
        .file("cuda/ntt120_ntt.cu")
        .file("cuda/ntt120_crt.cu")
        .file("cuda/ntt120_cnv.cu")
        .file("cuda/ntt120_vec_znx.cu")
        .file("cuda/ntt120_big.cu")
        .file("cuda/ntt120_shift.cu")
        .compile("ntt120_kernels");

    // Link the CUDA runtime (needed for cudaStream_t / kernel launches).
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
