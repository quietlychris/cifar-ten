[![crates.io](https://img.shields.io/crates/v/cifar-ten.svg)](https://crates.io/crates/cifar-ten) [![Documentation](https://docs.rs/cifar-ten/badge.svg)](https://docs.rs/cifar-ten) ![CI](https://github.com/quietlychris/bissel/actions/workflows/rust.yml/badge.svg)

## cifar-ten

This library parses the binary files of the CIFAR-10 data set and returns them as a tuple struct
- `CifarResult`: `(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)` which is organized as `(train_data, train_labels, test_data, test_labels)`

Convenience methods for converting these to the Rust `ndarray` numeric arrays are provided using the `to_ndarray` feature flag, as
well as for automatically downloading binary training data from a remote url.  
```rust
// $ cargo build --features=download,to_ndarray
use cifar_ten::*;

fn main() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .download_and_extract(true)
        .encode_one_hot(true)
        .build()
        .unwrap()
        .to_ndarray::<f32>()
        .expect("Failed to build CIFAR-10 data");
}
```
 
A `tar.gz` file with the original binaries can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). The crate's author also 
provides several ML data mirrors [here](https://cmoran.xyz/data/) which are used for running tests on this library. Please feel free to use,
but should you expect to make heavy use of these files, please consider creating your own mirror.   
 
If you'd like to verify that the correct images and labels are being provided, the `examples/preview_images.rs` file using `show-image` to
preview a RGB representation of a given image with the corresponding one-hot formatted label. 

Note: Early commits included the dataset, which will make the download size large. For development, it's suggested to clone using

```sh
$ git clone --depth=1 https://github.com/quietlychris/cifar-ten.git
```
