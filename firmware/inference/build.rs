//! MicroFlow embeds the model at compile time but does not tell Cargo about it:
//! rebuild the firmware whenever the exported model changes.
fn main() {
    println!("cargo:rerun-if-changed=models/tap.tflite");
}
