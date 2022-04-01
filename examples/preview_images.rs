use cifar_ten::*;
#[cfg(feature = "to_ndarray_013")]
use ndarray_013 as ndarray;
#[cfg(feature = "to_ndarray_014")]
use ndarray_014 as ndarray;
#[cfg(feature = "to_ndarray_015")]
use ndarray_015 as ndarray;

#[cfg(any(
    feature = "to_ndarray_015",
    feature = "to_ndarray_014",
    feature = "to_ndarray_013"
))]
use ndarray::prelude::*;

use image::*;
use show_image::{make_window_full, Event, WindowOptions};
use std::error::Error;

fn main() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .download_and_extract(true)
        .base_path("data")
        .download_url("https://cmoran.xyz/data/cifar/cifar-10-binary.tar.gz")
        // .download_url("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
        .encode_one_hot(true)
        .build()
        .unwrap()
        .to_ndarray::<u8>()
        .unwrap();

    let num: usize = 30;
    let img: Array3<u8> = train_data
        .slice(s![num, .., .., ..])
        .to_owned()
        .into_shape((3, 32, 32))
        .unwrap();
    let label: Array1<u8> = train_labels
        .slice(s![num, ..])
        .to_owned()
        .into_shape(10)
        .unwrap();
    println!("The image is of a: {}", return_label_from_one_hot(label));
    display_img(&img);
}

pub fn display_img(img_arr: &Array3<u8>) -> Result<(), Box<dyn Error>> {
    let test_result_img = convert_to_image(img_arr);

    let window_options = WindowOptions {
        name: "image".to_string(),
        size: [100, 100],
        resizable: true,
        preserve_aspect_ratio: true,
    };
    println!("\nPlease hit [ ESC ] to quit window:");
    let window = make_window_full(window_options)?;
    window.set_image(test_result_img, "test_result").unwrap();

    for event in window.events() {
        if let Event::KeyboardEvent(event) = event {
            if event.key == show_image::KeyCode::Escape {
                break;
            }
        }
    }

    show_image::stop()?;
    Ok(())
}

fn convert_to_image(array: &Array3<u8>) -> RgbImage {
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
