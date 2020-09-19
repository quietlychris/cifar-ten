### CIFAR-10 Parser

Parses the binary files of the CIFAR-10 data set and returns them as a pair of tuples `(data, labels)` with of type and dimension:
- Training data:  `Array4<u8> [50_000, 3, 32, 32]` and `Array2<u8> [50_000, 10]` 
- Testing data:  `Array4<u8> [10_000, 3, 32, 32]` and `Array2<u8> [10_000, 10]` 

**OR** 

- as a set of flattened `Array2<f32>` structures in the same arrangement. 

A random image from each dataset and the associated label can be displayed upon parsing. A `tar.gz` file with the original binaries can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). 

```rust
use cifar::*;

fn main() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(true)
        .build()
        // or .build_as_flat_f32()
        .expect("Failed to build CIFAR-10 data");
}
```
Note: At the moment, this also downloads the CIFAR-10 dataset included in this repostiory, which you'll also need to have an independent copy of in order to run. The uncompressed filesize is slightly under 200 MB. As a result, it may make more sense to clone the repository itself locally, then add its local path to your dependency list instead. 

```toml
[dependencies]
cifar-10 = {git = "https://github.com/quietlychris/cifar-10", branch="master"}
```

#### Dependencies
The crate's `show` feature uses the [`minifb`](https://github.com/emoon/rust_minifb) library to display sample images, which means you may need to add its dependencies via 
```
sudo apt install libxkbcommon-dev libwayland-cursor0 libwayland-dev
```
