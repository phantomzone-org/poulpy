/*
[build-dependencies]
bindgen ="0.71.1"

//use bindgen;
//use std::env;
//use std::fs;
//use std::path::PathBuf;
//use std::time::SystemTime;

// Path to the C header file
let header_paths: [&str; 2] = [
    "spqlios-arithmetic/spqlios/coeffs/coeffs_arithmetic.h",
    "spqlios-arithmetic/spqlios/arithmetic/vec_znx_arithmetic.h",
];

let out_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap());
let bindings_file = out_path.join("bindings.rs");

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

*/