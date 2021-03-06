#![allow(unused_variables)]
#![allow(unused_imports)]
use crate::Cifar10;

#[cfg(feature = "display")]
use crate::display::*;

#[cfg(feature = "download")]
#[test]
fn test_download_extract_build() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(false)
        .download_and_extract(true)
        .build_u8()
        .expect("Failed to download, extract, and build CIFAR-10 data");

    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(false)
        .download_and_extract(true)
        .normalize(true)
        .build_f32()
        .expect("Failed to download, extract, and build CIFAR-10 data");
}

#[cfg(feature = "download")]
#[test]
fn test_normalize_over_u8() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(false)
        .download_and_extract(true)
        .normalize(true)
        .build_u8()
        .expect("Failed to download, extract, and build CIFAR-10 data");
}

#[cfg(feature = "download")]
#[cfg(feature = "display")]
#[test]
fn test_download_extract_build_show() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(true)
        .download_and_extract(true)
        .build()
        // or .build_as_flat_f32()
        .expect("Failed to build CIFAR-10 data");
}
