use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // --- bindgen: 编译时从 cviruntime.h 生成 CVIRuntime FFI 绑定 ---
    let header = crate_dir.join("include/cviruntime.h");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", crate_dir.join("include/cvitpu_debug.h").display());

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy().into_owned())
        .clang_arg(format!("-I{}", crate_dir.join("include").display()))
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
        .allowlist_function("CVI_NN_.*")
        .allowlist_type("CVI_.*")
        .allowlist_var("CVI_.*")
        .generate()
        .expect("Unable to generate CVI bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write CVI bindings!");

    // --- 链接 CVI 运行时库 ---
    let lib_dir = crate_dir.join("../starry-apps/act-infer-tpu/lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cviruntime");
    println!("cargo:rustc-link-lib=dylib=cvikernel");
    println!("cargo:rustc-link-lib=dylib=cvimath");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=gcc_s");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
}
