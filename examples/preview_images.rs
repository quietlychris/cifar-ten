use cifar_ten::*;
#[cfg(feature = "to_ndarray_013")]
use ndarray_013 as ndarray;
#[cfg(feature = "to_ndarray_014")]
use ndarray_014 as ndarray;
#[cfg(feature = "to_ndarray_015")]
use ndarray_015 as ndarray;
#[cfg(feature = "to_ndarray_016")]
use ndarray_016 as ndarray;

#[cfg(any(
    feature = "to_ndarray_016",
    feature = "to_ndarray_015",
    feature = "to_ndarray_014",
    feature = "to_ndarray_013"
))]
use ndarray::prelude::*;

use image::*;
use show_image::{
    create_window,
    event::{WindowEvent, WindowKeyboardInputEvent},
    glam::UVec2,
    BoxImage, ImageInfo, WindowOptions,
};
use std::error::Error;

#[show_image::main]
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

    // let boxed_image = BoxImage::new(
    //     ImageInfo {
    //         pixel_format: show_image::PixelFormat::Rgb8,
    //         size: UVec2::new(32, 32),
    //         strides: UVec2::new(3, 3 * 32),
    //     },
    //     img_arr,
    // );

    let window_options = WindowOptions {
        resizable: true,
        preserve_aspect_ratio: true,
        ..WindowOptions::default()
    };
    println!("\nPlease hit [ ESC ] to quit window:");
    let window = create_window("cifar-10", Default::default())?;
    window.set_image("test_result", test_result_img).unwrap();

    for event in window.event_channel()? {
        if let WindowEvent::KeyboardInput(WindowKeyboardInputEvent { input, .. }) = event {
            if input.key_code == Some(show_image::event::VirtualKeyCode::Escape) {
                break;
            }
        }
    }

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
