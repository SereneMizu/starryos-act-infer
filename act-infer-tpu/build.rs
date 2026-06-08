use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_dir = PathBuf::from(&manifest_dir).join("../sg2002-libs");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    println!("cargo:rustc-link-lib=static=cviruntime-static");
    println!("cargo:rustc-link-lib=static=cvikernel-static");
    println!("cargo:rustc-link-lib=static=cvimath-static");

    println!("cargo:rustc-link-lib=static=stdc++");
    println!("cargo:rustc-link-lib=gcc_s");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");

    println!("cargo:rerun-if-changed=../sg2002-libs/libcviruntime-static.a");
    println!("cargo:rerun-if-changed=../sg2002-libs/libcvikernel-static.a");
    println!("cargo:rerun-if-changed=../sg2002-libs/libcvimath-static.a");
}
