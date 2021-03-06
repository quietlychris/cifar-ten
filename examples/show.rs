#![allow(unused_variables)]
use cifar_ten::*;

fn main() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(true)
        .download_and_extract(true)
        .build_u8()
        .expect("Failed to build CIFAR-10 data");
}
