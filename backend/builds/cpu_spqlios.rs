use std::env;
use std::path::PathBuf;

pub fn build() {
    let manifest_dir: PathBuf = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let spqlios_dir: PathBuf = manifest_dir.join("src/implementation/cpu_spqlios/spqlios-arithmetic/build/spqlios");

    // println!("cargo:warning=>> CARGO_MANIFEST_DIR: {}", manifest_dir.display());
    // println!("cargo:warning=>> Linking from: {}", spqlios_dir.display());

    println!("cargo:rustc-link-search=native={}", spqlios_dir.display());
    println!("cargo:rustc-link-lib=static=spqlios");
    // println!("cargo:rustc-link-lib=dylib=spqlios")
}
