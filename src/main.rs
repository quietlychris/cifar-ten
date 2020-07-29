use ndarray::{prelude::*};
use image::*;
use rand::prelude::*;
use minifb::{Key, Window, WindowOptions, ScaleMode};

use std::io;
use std::io::prelude::*;
use std::fs::File;
use std::error::Error;
use std::path::Path;

const SHOW_IMAGES: bool = true;
const BASE_PATH: &str = "data/";
const CIFAR_DATA_PATH: &str = "cifar-10-batches-bin/";
const IMAGES_PATH: &str = "images/";

fn main() -> std::result::Result<(),Box<dyn Error>> {

    let training_bin_paths = vec!["data_batch_1.bin","data_batch_2.bin","data_batch_3.bin","data_batch_4.bin","data_batch_5.bin"];
    let (data, labels) = get_data(training_bin_paths, 50_000)?;
    let test_bin_paths = vec!["test_batch.bin"];
    let (test_data, test_labels) = get_data(test_bin_paths,10_000)?;
 
    Ok(())
}

#[inline]
fn convert_to_image(array: Array3<u8>) -> RgbImage {
    // println!("- Converting to image!");
    let mut img: RgbImage = ImageBuffer::new(32,32);
    let (d,w,h) = (array.shape()[0], array.shape()[1], array.shape()[2]);
    // println!("(d,w,h) = ({},{},{})",d,w,h);
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

fn get_data(bin_paths: Vec<&str>,num_records: usize) -> Result<(Array4<u8>, Array2<u8>), Box<dyn Error>> {

    if Path::new(&[BASE_PATH,IMAGES_PATH].join("")).exists() == true {
        std::fs::remove_dir_all("data/images/")?;
    } 
    else {
        std::fs::create_dir("data/images/")?;
    }

    let mut buffer: Vec<u8> = Vec::new();
    for bin in &bin_paths {
        let full_cifar_path = [BASE_PATH,CIFAR_DATA_PATH,bin].join("");
        println!("{}",full_cifar_path);
    
        let mut f = File::open(full_cifar_path)?;
    
        // read the whole file
        let mut temp_buffer: Vec<u8> = Vec::new();
        f.read_to_end(&mut temp_buffer)?;
        buffer.extend(&temp_buffer);
    }

    println!("- Done parsing binary files to Vec<u8>");
    let mut labels: Array2<u8> = Array2::zeros((num_records,10));
    labels[[0,buffer[0] as usize]] = 1;
    let mut data: Vec<u8> = Vec::with_capacity(num_records*3072);

    for num in 0..num_records {
        println!("Through image #{}/{}",num,num_records);
        let base = num*(3073);
        let label = buffer[base];
        if label < 0 || label > 9 {
            panic!(format!("Label is {}, which is inconsistent with the CIFAR-10 scheme",label));
        }
        labels[[num,label as usize]] = 1;
        data.extend(&buffer[base+1..=base+3072]);
    }
    println!("data.len = {}, / 3072 = {}, remainder from /3072 = {}",data.len(), data.len() / 3072, data.len() % 3072);
    let mut data: Array4<u8> = Array::from_vec(data).into_shape((num_records,3,32,32))?;

    let mut rng = rand::thread_rng();
    let mut num: usize= rng.gen_range(0,num_records);
    if SHOW_IMAGES == true {
        // Displaying in minifb window instead of saving as a .png
        let img_arr = data.slice(s!(num, .., .., ..));
        let mut img_vec: Vec<u32> = Vec::with_capacity(32 * 32);
        let (w,h) = (32,32);
        for y in 0..h {
            for x in 0..w {
                let temp: [u8; 4] = [img_arr[[2,y,x]], img_arr[[1,y,x]], img_arr[[0,y,x]], 255u8];
                println!("temp: {:?}",temp);
                img_vec.push(u32::from_le_bytes(temp));
            }
        }
        println!( "Data label: {}",return_label_from_one_hot( labels.slice(s![num, ..]).to_owned() ) );
        display_img(img_vec);

    }

    Ok((data, labels))
}

fn display_img(mut buffer: Vec<u32>) {
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
        window
            .update_with_buffer(&buffer, 32, 32)
            .unwrap();
    }
}

fn return_label_from_one_hot(one_hot: Array1<u8>) -> String {

    if one_hot == array![1,0,0,0,0,0,0,0,0,0] {
        "airplane".to_string()
    } else if one_hot == array![0,1,0,0,0,0,0,0,0,0] {
        "automobile".to_string()
    } else if one_hot == array![0,0,1,0,0,0,0,0,0,0] {
        "bird".to_string()
    } else if one_hot == array![0,0,0,1,0,0,0,0,0,0] {
        "cat".to_string()
    } else if one_hot == array![0,0,0,0,1,0,0,0,0,0] {
        "deer".to_string()
    } else if one_hot == array![0,0,0,0,0,1,0,0,0,0] {
        "dog".to_string()
    } else if one_hot == array![0,0,0,0,0,0,1,0,0,0] {
        "frog".to_string()
    } else if one_hot == array![0,0,0,0,0,0,0,1,0,0] {
        "horse".to_string()
    } else if one_hot == array![0,0,0,0,0,0,0,0,1,0] {
        "ship".to_string()
    } else if one_hot == array![0,0,0,0,0,0,0,0,0,1] {
        "truck".to_string()
    } else {
        format!("Error: no valid label could be assigned to {}",one_hot).to_string()
    }

}