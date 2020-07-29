### CIFAR-10 Parser

Parses the binary files of the CIFAR-10 data set and returns them as a pair of tuples `(data, labels)` with of type and dimension:
- Training data:  `Array4<u8> [50_000, 3, 32, 32]` and `Array2<u8> [50_000, 10]` 
- Testing data:  `Array4<u8> [10_000, 3, 32, 32]` and `Array2<u8> [10_000, 10]` 

A random image from each dataset and the associated label is displayed upon parsing. A `tar.gz` file with the original binaries can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

#### Dependencies
This depends on [`minifb`](https://github.com/emoon/rust_minifb) to display sample images, which means you may need to add it's dependencies via 
```
sudo apt install libxkbcommon-dev libwayland-cursor0 libwayland-dev
```
