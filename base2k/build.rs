use std::path::absolute;

fn main() {
    println!(
        "cargo:rustc-link-search=native={}",
        absolute("./spqlios-arithmetic/build/spqlios")
            .unwrap()
            .to_str()
            .unwrap()
    );
    println!("cargo:rustc-link-lib=static=spqlios"); //"cargo:rustc-link-lib=dylib=spqlios"
}
