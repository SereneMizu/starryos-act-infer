use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // --- bindgen: 编译时从 rknn_api.h 生成 RKNN FFI 绑定 ---
    let header = crate_dir.join("include/rknn_api.h");
    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy().into_owned())
        .clang_arg(format!("-I{}", crate_dir.join("include").display()))
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false })
        .allowlist_function("rknn_.*")
        .allowlist_type("rknn_.*|_rknn_.*")
        .allowlist_var("RKNN_.*")
        .generate()
        .expect("Unable to generate RKNN bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write RKNN bindings!");

    // --- 链接 RKNN 运行时库 (aarch64 glibc) ---
    // 动态链接 librknnrt.so；其 C++ 依赖(libstdc++.so.6 等)由 Debian rootfs 在运行时解析，
    // 链接期无需 -lstdc++（ld 默认 --allow-shlib-undefined，不递归校验 .so 的 NEEDED）。
    let lib_dir = crate_dir.join("../starry-apps/act-infer-rknn/lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=rknnrt");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
}
