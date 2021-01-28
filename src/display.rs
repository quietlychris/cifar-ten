#[cfg(feature = "display")]
use image::*;
use ndarray::prelude::*;
use show_image::{make_window_full, Event, WindowOptions};
use std::error::Error;

/// Display an image in a stand-alone window
#[cfg(feature = "display")]
pub fn display_img(img_arr: &Array3<u8>) -> Result<(), Box<dyn Error>> {
    let test_result_img = convert_to_image(img_arr);

    let window_options = WindowOptions {
        name: "image".to_string(),
        size: [100, 100],
        resizable: true,
        preserve_aspect_ratio: true,
    };
    println!("\nPlease hit [ ESC ] to quit window:");
    let window = make_window_full(window_options).unwrap();
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

/// Helper function for transition from an normalized NdArray3<f32> structure to an `Image::RgbImage`
#[cfg(feature = "display")]
#[inline]
#[allow(clippy::many_single_char_names)]
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
