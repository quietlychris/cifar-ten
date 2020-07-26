use ndarray::{prelude::*,stack};
use image::*;

use std::io;
use std::io::prelude::*;
use std::fs::File;
use std::error::Error;
use std::path::Path;

const SAVE_IMAGES: bool = false;

fn main() -> std::result::Result<(),Box<dyn Error>> {

    let (train_data, train_labels) = get_train_data()?;
    let (test_data, test_labels) = get_test_data()?;
 
    Ok(())
}

#[inline]
fn convert_to_image(array: Array3<u8>) -> RgbImage {
    println!("- Converting to image!");
    let mut img: RgbImage = ImageBuffer::new(32,32);
    let (d,w,h) = (array.shape()[0], array.shape()[1], array.shape()[2]);
    println!("(d,w,h) = ({},{},{})",d,w,h);
    for y in 0..h {
        for x in 0..w {
            let r = array[[2,x,y]];
            let g = array[[1,x,y]];
            let b = array[[0,x,y]];
            img.put_pixel(y as u32,x as u32,Rgb([b, g, r]));
        }
    }

    img
}

fn get_train_data() -> Result<(Array4<u8>, Array2<u8>), Box<dyn Error>> {
        // First, we're going to entirely read the training data set into an Array3<u8> structure
    // of [50_000, 3, 32, 32]
    let base_path = "data/";
    let cifar_data_path = "cifar-10-batches-bin/";
    let cifar_bin_path = vec!["data_batch_1.bin","data_batch_2.bin","data_batch_3.bin","data_batch_4.bin","data_batch_5.bin",];

    let images_path = "images/";
    if Path::new(&[base_path,images_path].join("")).exists() == true {
        std::fs::remove_dir_all("data/images/")?;
    } 
    else {
        std::fs::create_dir("data/images/")?;
    }

    let mut buffer: Vec<u8> = Vec::new();
    for bin in &cifar_bin_path {
        let full_cifar_path = [base_path,cifar_data_path,bin].join("");
        println!("{}",full_cifar_path);
    
        let mut f = File::open(full_cifar_path)?;
    
        // read the whole file
        let mut temp_buffer: Vec<u8> = Vec::new();
        f.read_to_end(&mut temp_buffer)?;
        buffer.extend(&temp_buffer);
    }

    println!("- Done parsing binary files to Vec<u8>");
    let mut train_labels: Array2<u8> = Array2::zeros((50_000,10));
    train_labels[[0,buffer[0] as usize]] = 1;
    //let mut train_data: Array4<u8> = Array::zeros((50_000,3,32,32));
    let mut train_data: Array4<u8> = Array::from_vec(buffer[1..=3072].to_vec()).into_shape((1,3,32,32))?;
    
    for num in 1..50_000 {
        println!("Through training image #{}",num);
        let base = num*(3073);
        let label = buffer[base];
        if label < 0 || label > 9 {
            panic!(format!("Label is {}, which is inconsistent with the CIFAR-10 scheme",label));
        }
        // println!("Label is: {}",label);
        train_labels[[num,label as usize]] = 1;

        // train_data.slice_mut(s![num, .., .., ..]) = Array::from_shape_vec((3,32,32), buffer[base+1..=base+3072].to_vec())?.view_mut(); // Doesn't compile
        train_data = stack(Axis(0),&[train_data.view(),Array::from_vec(buffer[base+1..=base+3072].to_vec()).into_shape((1,3,32,32))?.view()])?;
        // println!("array shape: {:?}",train_data.dim());
        if SAVE_IMAGES == true {
            let img = convert_to_image(train_data.slice(s!(num, .., .., ..)).to_owned());
            let image_name = ["data/images/image_",num.to_string().as_str(),".png"].join("");
            img.save(image_name)?;
        }
    }

    Ok((train_data, train_labels))
}

fn get_test_data() -> Result<(Array4<u8>, Array2<u8>), Box<dyn Error>> {
    // First, we're going to entirely read the training data set into an Array3<u8> structure
    // of [10_000, 3, 32, 32]
    let base_path = "data/";
    let cifar_data_path = "cifar-10-batches-bin/";
    let bin = "test_batch.bin";

    let images_path = "images/";
    if Path::new(&[base_path,images_path].join("")).exists() == true {
        std::fs::remove_dir_all("data/images/")?;
    } 
    else {
        std::fs::create_dir("data/images/")?;
    }

    let mut buffer: Vec<u8> = Vec::new();
    let full_cifar_path = [base_path,cifar_data_path,bin].join("");
    println!("{}",full_cifar_path);
    let mut f = File::open(full_cifar_path)?;

    f.read_to_end(&mut buffer)?;
    println!("- Done parsing binary files to Vec<u8>");
    let mut labels: Array2<u8> = Array2::zeros((10_000,10));
    labels[[0,buffer[0] as usize]] = 1;
    let mut data: Array4<u8> = Array::from_vec(buffer[1..=3072].to_vec()).into_shape((1,3,32,32))?;

    for num in 1..10_000 {
        let base = num*(3073);
        let label = buffer[base];
        if label < 0 || label > 9 {
            panic!(format!("Label is {}, which is inconsistent with the CIFAR-10 scheme",label));
        }
        // println!("Label is: {}",label);
        labels[[num,label as usize]] = 1;

        data = stack(Axis(0),&[data.view(),Array::from_vec(buffer[base+1..=base+3072].to_vec()).into_shape((1,3,32,32))?.view()])?;
        // println!("array shape: {:?}",data.dim());
        if SAVE_IMAGES == true {
            let img = convert_to_image(data.slice(s!(num, .., .., ..)).to_owned());
            let image_name = ["data/images/image_",num.to_string().as_str(),".png"].join("");
            img.save(image_name)?;
        }
    }

    Ok((data, labels))
}