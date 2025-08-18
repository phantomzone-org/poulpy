use std::path::PathBuf;

pub fn build() {
    let dst: PathBuf = cmake::Config::new("src/cpu_spqlios/spqlios-arithmetic")
        .define("ENABLE_TESTING", "FALSE")
        .build();

    let lib_dir: PathBuf = dst.join("lib");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=spqlios");
}
