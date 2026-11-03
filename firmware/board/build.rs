//! Put `memory.x` on the linker search path. Library link-search paths propagate to the
//! final binaries, so every firmware crate depending on `board` gets the right layout.
fn main() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(out.join("memory.x"), include_bytes!("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
}
