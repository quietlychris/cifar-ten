[![crates.io](https://img.shields.io/crates/v/cifar-ten.svg)](https://crates.io/crates/cifar-ten)
[![Documentation](https://docs.rs/cifar-ten/badge.svg)](https://docs.rs/cifar-ten)

## cifar-ten

Parses the binary files of the CIFAR-10 data set and returns them as a pair of tuples `(data, labels)` with of type and dimension:
- Training data:  `Array4<u8/f32> [50_000, 3, 32, 32]` and `Array2<u8/f32> [50_000, 10]` 
- Testing data:  `Array4<u8/f32> [10_000, 3, 32, 32]` and `Array2<u8/f32> [10_000, 10]` 

A random image from each dataset and the associated label can be displayed upon parsing. A `tar.gz` file with the original binaries can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). This can be downloaded manually, or automatically using the `download` feature. 

```rust
use cifar_ten::*;

fn main() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(false) // won't display a window with the image
        .download_and_extract(true) // must enable the "download" feature
        .normalize(true) // floating point values will be normalized across [0, 1.0] 
        .build_f32() 
        // or .build_u8() if data in standard RGB<u8> format is desired
        .expect("Failed to build CIFAR-10 data");
}
```
To add as a dependency, please use:

```toml
[dependencies]
cifar-ten = "0.2"
# or 
cifar-ten = {git = "https://github.com/quietlychris/cifar-ten", branch = "master"}
```
Note: Previous commits have included the dataset, which will make the download size large. For development, it's suggested to 
```
$ git clone --depth=1 https://github.com/quietlychris/cifar-ten.git
```
