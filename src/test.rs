#![allow(unused_variables)]
use crate::Cifar10;

#[cfg(feature = "download")]
fn test_download_extract_build() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(true)
        .download_and_extract(true)
        .build()
        // or .build_as_flat_f32()
        .expect("Failed to build CIFAR-10 data");
}

#[cfg(feature = "download")]
fn test_download_extract_build_f32() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(true)
        .download_and_extract(true)
        .build_as_flat_f32()
        .expect("Failed to build CIFAR-10 data");
}
