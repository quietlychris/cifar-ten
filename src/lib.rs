#![allow(dead_code)]

mod test;


#[cfg(feature = "show")]
use image::*;
#[cfg(feature = "show")]
use minifb::{Key, ScaleMode, Window, WindowOptions};
use ndarray::prelude::*;
#[cfg(feature = "show")]
use rand::prelude::*;

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

#[derive(Debug)]
pub struct Cifar10<'a> {
    base_path: &'a str,
    cifar_data_path: &'a str,
    show_images: bool,
    encode_one_hot: bool,
    training_bin_paths: Vec<&'a str>,
    testing_bin_paths: Vec<&'a str>,
    num_records_train: usize,
    num_records_test: usize,
}

impl<'a> Cifar10<'a> {
    pub fn default() -> Self {
        Cifar10 {
            base_path: "data/",
            cifar_data_path: "cifar-10-batches-bin/",
            show_images: false,
            encode_one_hot: true,
            training_bin_paths: vec![
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin",
            ],
            testing_bin_paths: vec!["test_batch.bin"],
            num_records_train: 50_000,
            num_records_test: 10_000,
        }
    }

    pub fn base_path(mut self, base_path: &'a str) -> Self {
        self.base_path = base_path;
        self
    }

    pub fn cifar_data_path(mut self, cifar_data_path: &'a str) -> Self {
        self.cifar_data_path = cifar_data_path;
        self
    }

    pub fn show_images(mut self, show_images: bool) -> Self {
        self.show_images = show_images;
        self
    }

    pub fn encode_one_hot(mut self, encode_one_hot: bool) -> Self {
        self.encode_one_hot = encode_one_hot;
        self
    }

    pub fn training_bin_paths(mut self, training_bin_paths: Vec<&'a str>) -> Self {
        self.training_bin_paths = training_bin_paths;
        self
    }

    pub fn testing_bin_paths(mut self, testing_bin_paths: Vec<&'a str>) -> Self {
        self.testing_bin_paths = testing_bin_paths;
        self
    }

    pub fn num_records_train(mut self, num_records_train: usize) -> Self {
        self.num_records_train = num_records_train;
        self
    }

    pub fn num_records_test(mut self, num_records_test: usize) -> Self {
        self.num_records_test = num_records_test;
        self
    }


    pub fn build(self) -> Result<(Array4<u8>, Array2<u8>, Array4<u8>, Array2<u8>), Box<dyn Error>> {
        let (train_data, train_labels) = get_data(&self, "train")?;
        let (test_data, test_labels) = get_data(&self, "test")?;

        Ok((train_data, train_labels, test_data, test_labels))

    }

    pub fn build_as_flat_f32(self) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn Error>> {
        
        let (train_data, train_labels) = get_data(&self, "train")?;
        let (test_data, test_labels) = get_data(&self, "test")?;
        
        let train_labels = train_labels.mapv(|x| x as f32);
        let train_data = train_data
            .into_shape((self.num_records_train, 32 * 32 * 3))?
            .mapv(|x| x as f32 / 256.);
        let test_labels = test_labels.mapv(|x| x as f32);
        let test_data = test_data
            .into_shape((self.num_records_test, 32 * 32 * 3))?
            .mapv(|x| x as f32 / 256.);

        Ok((train_data, train_labels, test_data, test_labels))
    }
    
}

#[cfg(feature = "show")]
#[inline]
#[allow(clippy::many_single_char_names)]
fn convert_to_image( array: Array3<u8>) -> RgbImage {
    // println!("- Converting to image!");
    let mut img: RgbImage = ImageBuffer::new(32, 32);
    let (_d, w, h) = (array.shape()[0], array.shape()[1], array.shape()[2]);
    // println!("(d,w,h) = ({},{},{})",d,w,h);
    for y in 0..h {
        for x in 0..w {
            let r = array[[2, x, y]];
            let g = array[[1, x, y]];
            let b = array[[0, x, y]];
            img.put_pixel(y as u32, x as u32, Rgb([b, g, r]));
        }
    }

    img
}

fn get_data(config: &Cifar10, dataset: &str) -> Result<(Array4<u8>, Array2<u8>), Box<dyn Error>> {
    let mut buffer: Vec<u8> = Vec::new();

    let (bin_paths, num_records) = match dataset {
        "train" => (config.training_bin_paths.clone(), config.num_records_train),
        "test" => (config.testing_bin_paths.clone(), config.num_records_test),
        _ => panic!("An unexpected value was passed for which dataset should be parsed"),
    };

    for bin in &bin_paths {
        let full_cifar_path = [config.base_path, config.cifar_data_path, bin].join("");
        // println!("{}", full_cifar_path);

        let mut f = File::open(full_cifar_path)?;

        // read the whole file
        let mut temp_buffer: Vec<u8> = Vec::new();
        f.read_to_end(&mut temp_buffer)?;
        buffer.extend(&temp_buffer);
        //println!(
        //    "{}",
        //    format!("- Done parsing binary file {} to Vec<u8>", bin).as_str()
        //);
    }

    //println!("- Done parsing binary files to Vec<u8>");
    let mut labels: Array2<u8> = Array2::zeros((num_records, 10));
    labels[[0, buffer[0] as usize]] = 1;
    let mut data: Vec<u8> = Vec::with_capacity(num_records * 3072);

    for num in 0..num_records {
        // println!("Through image #{}/{}", num, num_records);
        let base = num * (3073);
        let label = buffer[base];
        if label > 9 {
            panic!(format!(
                "Label is {}, which is inconsistent with the CIFAR-10 scheme",
                label
            ));
        }
        labels[[num, label as usize]] = 1;
        data.extend(&buffer[base + 1..=base + 3072]);
    }
    let data: Array4<u8> = Array::from_shape_vec((num_records, 3, 32, 32), data)?;

    if config.show_images {
        #[cfg(feature = "show")]
        {
            let mut rng = rand::thread_rng();
            let num: usize = rng.gen_range(0, num_records);
            // Displaying in minifb window instead of saving as a .png
            let img_arr = data.slice(s!(num, .., .., ..));
            let mut img_vec: Vec<u32> = Vec::with_capacity(32 * 32);
            let (w, h) = (32, 32);
            for y in 0..h {
                for x in 0..w {
                    let temp: [u8; 4] = [
                        img_arr[[2, y, x]],
                        img_arr[[1, y, x]],
                        img_arr[[0, y, x]],
                        255u8,
                    ];
                    // println!("temp: {:?}", temp);
                    img_vec.push(u32::from_le_bytes(temp));
                }
            }
            println!(
                "Data label: {}",
                return_label_from_one_hot(labels.slice(s![num, ..]).to_owned())
            );
            display_img(img_vec);
        }
        #[cfg(not(feature = "show"))]
        {
            println!("WARNING: Showing images disabled.");
            println!("Please use the crate's 'show' feature to enable it.");
        }
    }

    Ok((data, labels))
}

#[cfg(feature = "show")]
fn display_img(buffer: Vec<u32>) {
    let (window_width, window_height) = (600, 600);
    let mut window = Window::new(
        "Test - ESC to exit",
        window_width,
        window_height,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer, 32, 32).unwrap();
    }
}

fn return_label_from_one_hot(one_hot: Array1<u8>) -> String {
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
