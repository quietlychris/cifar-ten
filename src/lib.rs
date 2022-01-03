#![allow(dead_code)]

//! This library parses the binary files of the CIFAR-10 data set and returns them as a tuple struct
//! - `CifarResult`: `(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)` which is organized as `(train_data, train_labels, test_data, test_labels)`
//! 
//! Convenience methods for converting these to the Rust `ndarray` numeric arrays are provided using the `to_ndarray` feature flag, as
//! well as for automatically downloading binary training data from a remote url.  
//!
//! ```rust
//! // $ cargo build --features=download,to_ndarray
//! use cifar_ten::*;
//!
//! fn main() {
//!     let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
//!         .download_and_extract(true)
//!         .encode_one_hot(true)
//!         .build()
//!         .unwrap()
//!         .to_ndarray::<f32>()
//!         .expect("Failed to build CIFAR-10 data");
//! }
//! ```
//! 
//! A `tar.gz` file with the original binaries can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). The crate's author also 
//! provides several ML data mirrors [here](https://cmoran.xyz/data/) which are used for running tests on this library. Please feel free to use,
//! but should you expect to make heavy use of these files, please consider creating your own mirror.   
//! 
//! If you'd like to verify that the correct images and labels are being provided, the `examples/preview_images.rs` file using `show-image` to
//! preview a RGB representation of a given image with the corresponding one-hot formatted label. 


mod test;
#[macro_use]
extern crate serial_test;

#[cfg(feature = "to_ndarray")]
use ndarray::prelude::*;

use std::error::Error;
use std::io::Read;
use std::path::Path;

#[cfg(feature = "download")]
mod download;
// Dependencies for download feature
#[cfg(feature = "download")]
use crate::download::download_and_extract;
#[cfg(feature = "download")]
use std::fs::File;
#[cfg(feature = "download")]
use tar::Archive;

/// Primary data return, wrapper around tuple `(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)`
pub struct CifarResult(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

/// Data structure used to specify where/how the CIFAR-10 binary data is parsed
#[derive(Debug)]
pub struct Cifar10 {
    base_path: String,
    cifar_data_path: String,
    encode_one_hot: bool,
    training_bin_paths: Vec<String>,
    testing_bin_paths: Vec<String>,
    num_records_train: usize,
    num_records_test: usize,
    as_f32: bool,
    normalize: bool,
    download_and_extract: bool,
    download_url: String,
}

impl Cifar10 {
    /// Returns the default struct, looking in the "./data/" directory with default binary names
    pub fn default() -> Self {
        Cifar10 {
            base_path: "data/".into(),
            cifar_data_path: "cifar-10-batches-bin/".into(),
            encode_one_hot: true,
            training_bin_paths: vec![
                "data_batch_1.bin".into(),
                "data_batch_2.bin".into(),
                "data_batch_3.bin".into(),
                "data_batch_4.bin".into(),
                "data_batch_5.bin".into(),
            ],
            testing_bin_paths: vec!["test_batch.bin".into()],
            num_records_train: 50_000,
            num_records_test: 10_000,
            as_f32: false,
            normalize: false,
            download_and_extract: false,
            download_url: "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz".to_string(),
        }
    }

    /// Manually set the base path
    pub fn base_path(mut self, base_path: impl Into<String>) -> Self {
        self.base_path = base_path.into();
        self
    }

    /// Manually set the path for the CIFAR-10 data
    pub fn cifar_data_path(mut self, cifar_data_path: impl Into<String>) -> Self {
        self.cifar_data_path = cifar_data_path.into();
        self
    }

    /// Download CIFAR-10 dataset and extract from compressed tarball
    pub fn download_and_extract(mut self, download_and_extract: bool) -> Self {
        self.download_and_extract = download_and_extract;
        self
    }

    /// Choose a custom url from which to download the CIFAR-10 dataset
    pub fn download_url(mut self, download_url: impl Into<String>) -> Self {
        self.download_url = download_url.into();
        self
    }

    /// Choose if the `labels` return is in one-hot format or not (default yes)
    pub fn encode_one_hot(mut self, encode_one_hot: bool) -> Self {
        self.encode_one_hot = encode_one_hot;
        self
    }

    /// Manually set the path to the training data binaries
    pub fn training_bin_paths(mut self, training_bin_paths: Vec<String>) -> Self {
        self.training_bin_paths = training_bin_paths;
        self
    }

    /// Manually set the path to the testing data binaries
    pub fn testing_bin_paths(mut self, testing_bin_paths: Vec<String>) -> Self {
        self.testing_bin_paths = testing_bin_paths;
        self
    }

    /// Set the number of records in the training set (default 50_000)
    pub fn num_records_train(mut self, num_records_train: usize) -> Self {
        self.num_records_train = num_records_train;
        self
    }

    /// Set the number of records in the training set (default 10_000)
    pub fn num_records_test(mut self, num_records_test: usize) -> Self {
        self.num_records_test = num_records_test;
        self
    }

    /// Returns the array tuple using the specified options in Array4<T> form
    pub fn build(self) -> Result<CifarResult, Box<dyn Error>> {
        #[cfg(feature = "download")]
        match self.download_and_extract {
            false => (),
            true => {
                download_and_extract(self.download_url.clone(), self.base_path.clone())?;
            }
        }

        let (train_data, train_labels) = get_data(&self, "train")?;
        let (test_data, test_labels) = get_data(&self, "test")?;
        Ok(CifarResult(
            train_data,
            train_labels,
            test_data,
            test_labels,
        ))
    }
}

fn get_data(config: &Cifar10, dataset: &str) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let mut buffer: Vec<u8> = Vec::new();

    let (bin_paths, num_records) = match dataset {
        "train" => (config.training_bin_paths.clone(), config.num_records_train),
        "test" => (config.testing_bin_paths.clone(), config.num_records_test),
        _ => panic!("An unexpected value was passed for which dataset should be parsed"),
    };

    for bin in &bin_paths {
        // let full_cifar_path = [config.base_path, config.cifar_data_path, bin.into()].join("");
        let full_cifar_path = Path::new(&config.base_path)
            .join(&config.cifar_data_path)
            .join(bin);
        // println!("{}", full_cifar_path);

        let mut f = std::fs::File::open(full_cifar_path)?;

        // read the whole file
        let mut temp_buffer: Vec<u8> = Vec::new();
        f.read_to_end(&mut temp_buffer)?;
        buffer.extend(&temp_buffer);
        //println!(
        //    "{}",
        //    format!("- Done parsing binary file {} to Vec<u8>", bin).as_str()
        //);
    }

    let mut labels: Vec<u8> = match config.encode_one_hot {
        false => vec![0; num_records],
        true => vec![0; num_records * 10],
    };
    let mut data: Vec<u8> = Vec::with_capacity(num_records * 3072);

    for num in 0..num_records {
        // println!("Through image #{}/{}", num, num_records);
        let base = num * (3073);

        let label = buffer[base];
        if label > 9 {
            panic!(
                "Label is {}, which is inconsistent with the CIFAR-10 scheme",
                label
            );
        }

        data.extend(&buffer[base + 1..=base + 3072]);

        match config.encode_one_hot {
            false => labels[num] = label as u8,
            true => labels[(num * 10) + (label as usize)] = 1u8,
        };
    }

    Ok((data, labels))
}

impl CifarResult {
    #[cfg(feature = "to_ndarray")]
    pub fn to_ndarray<T: std::convert::From<u8>>(
        self,
    ) -> Result<(Array4<T>, Array2<T>, Array4<T>, Array2<T>), Box<dyn Error>> {
        let train_data: Array4<T> =
            Array::from_shape_vec((50_000, 3, 32, 32), self.0)?.mapv(|x| x.into());
        let train_labels: Array2<T> =
            Array::from_shape_vec((50_000, 10), self.1)?.mapv(|x| x.into());
        let test_data: Array4<T> =
            Array::from_shape_vec((10_000, 3, 32, 32), self.2)?.mapv(|x| x.into());
        let test_labels: Array2<T> =
            Array::from_shape_vec((10_000, 10), self.3)?.mapv(|x| x.into());

        Ok((train_data, train_labels, test_data, test_labels))
    }
}

#[cfg(feature = "to_ndarray")]
pub fn return_label_from_one_hot(one_hot: Array1<u8>) -> String {
    if one_hot == array![1, 0, 0, 0, 0, 0, 0, 0, 0, 0] {
        "airplane".to_string()
    } else if one_hot == array![0, 1, 0, 0, 0, 0, 0, 0, 0, 0] {
        "automobile".to_string()
    } else if one_hot == array![0, 0, 1, 0, 0, 0, 0, 0, 0, 0] {
        "bird".to_string()
    } else if one_hot == array![0, 0, 0, 1, 0, 0, 0, 0, 0, 0] {
        "cat".to_string()
    } else if one_hot == array![0, 0, 0, 0, 1, 0, 0, 0, 0, 0] {
        "deer".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 1, 0, 0, 0, 0] {
        "dog".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 0, 1, 0, 0, 0] {
        "frog".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 0, 0, 1, 0, 0] {
        "horse".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 0, 0, 0, 1, 0] {
        "ship".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 0, 0, 0, 0, 1] {
        "truck".to_string()
    } else {
        format!("Error: no valid label could be assigned to {}", one_hot)
    }
}
