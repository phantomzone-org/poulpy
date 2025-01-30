//use bindgen;
//use std::env;
//use std::fs;
use std::path::absolute;
//use std::path::PathBuf;
//use std::time::SystemTime;

fn main() {
    /*

    [build-dependencies]
    bindgen ="0.71.1"

    // Path to the C header file
    let header_paths: [&str; 2] = [
        "spqlios-arithmetic/spqlios/coeffs/coeffs_arithmetic.h",
        "spqlios-arithmetic/spqlios/arithmetic/vec_znx_arithmetic.h",
    ];

    let out_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_file = out_path.join("bindings.rs");

    let regenerate: bool = header_paths.iter().any(|header| {
        let header_metadata: SystemTime = fs::metadata(header)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let bindings_metadata: SystemTime = fs::metadata(&bindings_file)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        header_metadata > bindings_metadata
    });

    if regenerate {
        // Generate the Rust bindings
        let mut builder: bindgen::Builder = bindgen::Builder::default();
        for header in header_paths {
            builder = builder.header(header);
        }

        let bindings = builder
            .generate_comments(false) // Optional: includes comments in bindings
            .generate_inline_functions(true) // Optional: includes inline functions
            .generate()
            .expect("Unable to generate bindings");

        // Write the bindings to the OUT_DIR
        bindings
            .write_to_file(&bindings_file)
            .expect("Couldn't write bindings!");
    }
    */

    println!(
        "cargo:rustc-link-search=native={}",
        absolute("./spqlios-arithmetic/build/spqlios")
            .unwrap()
            .to_str()
            .unwrap()
    );
    println!("cargo:rustc-link-lib=static=spqlios"); //"cargo:rustc-link-lib=dylib=spqlios"
}
