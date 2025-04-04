use curl::easy::Easy;
use dir_lock::DirLock;
use filesize::PathExt;
use pbr::ProgressBar;
use std::convert::TryInto;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;
use tar::Archive;

const ARCHIVE: &str = "cifar-10-binary.tar.gz";
const ARCHIVE_DOWNLOAD_SIZE: usize = 170052171;

pub(super) fn download_and_extract(
    download_url: String,
    base_path: impl Into<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let download_dir = base_path.into();
    if !download_dir.exists() {
        println!(
            "Download directory {} does not exists. Creating....",
            download_dir.display()
        );
        fs::create_dir_all(&download_dir)?;
    }
    let _dir_lock = DirLock::new(&download_dir);
    println!("Attempting to download and extract {}...", ARCHIVE);
    download(download_url, &download_dir)?;
    extract(&ARCHIVE, &download_dir)?;

    Ok(())
}

fn download(url: String, download_dir: impl Into<PathBuf>) -> Result<(), Box<dyn Error>> {
    let mut easy = Easy::new();

    let file_name = download_dir.into().join(ARCHIVE); //.clone();
    if Path::new(&file_name).exists() {
        println!(
            "  File {:?} already exists, skipping downloading.",
            file_name
        );
    } else {
        println!(
            "- Downloading from file from {} and saving to file as: {}",
            url,
            file_name.display()
        );

        let mut file = File::create(file_name.clone()).unwrap();

        let full_size = ARCHIVE_DOWNLOAD_SIZE;

        let pb_thread = thread::spawn(move || {
            let mut pb = ProgressBar::new(full_size.try_into().unwrap());
            pb.format("╢=> ╟");

            let mut current_size = 0;
            while current_size < full_size {
                current_size = file_name
                    .size_on_disk()
                    .expect(&format!("Couldn't get metadata on {:?}", file_name))
                    as usize;
                pb.set(current_size.try_into().unwrap());
                thread::sleep(Duration::from_millis(10));
            }
            pb.finish_println(" ");
        });

        easy.url(&url).unwrap();
        easy.write_function(move |data| {
            file.write_all(data).unwrap();
            Ok(data.len())
        })
        .unwrap();
        easy.perform().unwrap();

        pb_thread.join().unwrap();
    }

    Ok(())
}

fn extract(archive_name: &str, download_dir: &Path) -> Result<(), Box<dyn Error>> {
    // And extract the contents
    let archive = download_dir.to_owned().join(archive_name);

    let extract_to = download_dir.to_owned().join("cifar-10-batches-bin");
    if Path::new(&extract_to).exists() {
        println!(
            "  Extracted file {:?} already exists, skipping extraction.",
            extract_to
        );
    } else {
        println!("Beginning extraction of {:?} to {:?}", archive, extract_to);
        use flate2::read::GzDecoder;
        let tar_gz = File::open(archive)?;
        let tar = GzDecoder::new(tar_gz);
        let mut archive = Archive::new(tar);
        archive.unpack(download_dir)?;
    }
    Ok(())
}
