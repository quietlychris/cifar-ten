#![allow(unused_variables)]
#![allow(unused_imports)]
use crate::*;

#[cfg(not(feature = "download"))]
#[test]
fn test_build() {
    let result = Cifar10::default().build().unwrap();
}

#[cfg(all(not(feature = "download"), feature = "to_ndarray_016"))]
#[cfg(feature = "to_ndarray_016")]
#[test]
fn test_build_to_ndarray_f32() {
    let result = Cifar10::default().build().unwrap().to_ndarray::<f32>();
}

#[cfg(all(feature = "download", feature = "to_ndarray_016"))]
#[test]
fn test_download_extract_build_u8() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .download_and_extract(true)
        .download_url("https://cmoran.xyz/data/cifar/cifar-10-binary.tar.gz")
        .build()
        .unwrap()
        .to_ndarray::<u8>()
        .unwrap();
}

#[cfg(all(feature = "download", feature = "to_ndarray_016"))]
#[test]
fn test_download_extract_build_f32() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .download_and_extract(true)
        .download_url("https://cmoran.xyz/data/cifar/cifar-10-binary.tar.gz")
        .build()
        .unwrap()
        .to_ndarray::<f32>()
        .unwrap();
}
