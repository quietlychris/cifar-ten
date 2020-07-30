use crate::*;

#[test]
fn test_run() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(true)
        .build()
        .expect("Failed to build CIFAR-10 data");
}
