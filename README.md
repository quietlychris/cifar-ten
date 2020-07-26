### CIFAR-10 Parser
__Not ready for use__

Parses the binary files of the CIFAR-10 data set and returns them as a pair of tuples `(data, labels)` with of type and dimension:
- Training data:  `Array4<u8> [50_000, 3, 32, 32]` and `Array2<u8> [50_000, 10]` 
- Testing data:  `Array4<u8> [10_000, 3, 32, 32]` and `Array2<u8> [10_000, 10]` 

This still runs really slowly, largely because of the repeated ndarray stack calls for adding new images to the test and train data structures. 

TO_DO:
- Pre-allocating allocate memory for the test/train data structures
- Make the `get_train_data()` and `get_test_data()` functions variations on the same code--there's a lot of repeated code in each