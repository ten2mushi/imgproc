#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

use imgproc_core::{
    error::{Error, Result},
    image::Image,
    pixel::{Gray8, Rgb8},
};
use std::path::Path;
use rayon::prelude::*;

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// PNG format
    Png,
    /// JPEG format
    Jpeg,
    /// BMP format
    Bmp,
    /// TIFF format
    Tiff,
    /// WebP format
    WebP,
    /// GIF format (read-only)
    Gif,
}

impl ImageFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "bmp" => Some(Self::Bmp),
            "tiff" | "tif" => Some(Self::Tiff),
            "webp" => Some(Self::WebP),
            "gif" => Some(Self::Gif),
            _ => None,
        }
    }
}

/// Read an image from file
pub fn read_image<P: AsRef<Path>>(path: P) -> Result<Image<Rgb8>> {
    let path = path.as_ref();

    let img = image::open(path).map_err(|e| Error::OperationFailed {
        message: format!("Failed to open image: {}", e),
    })?;

    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let pixels: Vec<Rgb8> = rgb_img
        .pixels()
        .map(|p| Rgb8::new(p[0], p[1], p[2]))
        .collect();

    Image::from_raw(width, height, pixels)
}

/// Read a grayscale image from file
pub fn read_image_gray<P: AsRef<Path>>(path: P) -> Result<Image<Gray8>> {
    let path = path.as_ref();

    let img = image::open(path).map_err(|e| Error::OperationFailed {
        message: format!("Failed to open image: {}", e),
    })?;

    let gray_img = img.to_luma8();
    let (width, height) = gray_img.dimensions();

    let pixels: Vec<Gray8> = gray_img
        .pixels()
        .map(|p| Gray8::new(p[0]))
        .collect();

    Image::from_raw(width, height, pixels)
}

/// Write an RGB image to file
pub fn write_image<P: AsRef<Path>>(image: &Image<Rgb8>, path: P) -> Result<()> {
    let path = path.as_ref();
    let (width, height) = image.dimensions();

    let mut img_buffer = image::RgbImage::new(width, height);

    for (x, y, pixel) in image.enumerate_pixels() {
        img_buffer.put_pixel(x, y, image::Rgb([pixel.r, pixel.g, pixel.b]));
    }

    img_buffer.save(path).map_err(|e| Error::OperationFailed {
        message: format!("Failed to save image: {}", e),
    })
}

/// Alias for read_image (RGB)
pub fn read_image_rgb<P: AsRef<Path>>(path: P) -> Result<Image<Rgb8>> {
    read_image(path)
}

/// Alias for write_image (RGB)
pub fn write_image_rgb<P: AsRef<Path>>(image: &Image<Rgb8>, path: P) -> Result<()> {
    write_image(image, path)
}

/// Write a grayscale image to file
pub fn write_image_gray<P: AsRef<Path>>(image: &Image<Gray8>, path: P) -> Result<()> {
    let path = path.as_ref();
    let (width, height) = image.dimensions();

    let mut img_buffer = image::GrayImage::new(width, height);

    for (x, y, pixel) in image.enumerate_pixels() {
        img_buffer.put_pixel(x, y, image::Luma([pixel.value]));
    }

    img_buffer.save(path).map_err(|e| Error::OperationFailed {
        message: format!("Failed to save image: {}", e),
    })
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Input directory
    pub input_dir: String,
    /// Output directory
    pub output_dir: String,
    /// File pattern (e.g., "*.jpg")
    pub pattern: Option<String>,
    /// Whether to process recursively
    pub recursive: bool,
    /// Number of parallel threads
    pub threads: Option<usize>,
}

impl BatchConfig {
    /// Create a new batch configuration
    #[must_use]
    pub fn new(input_dir: String, output_dir: String) -> Self {
        Self {
            input_dir,
            output_dir,
            pattern: None,
            recursive: false,
            threads: None,
        }
    }
}

/// Process images in batch
pub fn process_batch<F>(config: &BatchConfig, processor: F) -> Result<Vec<Result<()>>>
where
    F: Fn(&Path, &Path) -> Result<()> + Send + Sync,
{
    use std::fs;

    // Create output directory if it doesn't exist
    fs::create_dir_all(&config.output_dir).map_err(|e| Error::OperationFailed {
        message: format!("Failed to create output directory: {}", e),
    })?;

    // Collect input files
    let mut input_files = Vec::new();
    collect_files(&config.input_dir, &mut input_files, config.recursive)?;

    // Filter by pattern if specified
    if let Some(pattern) = &config.pattern {
        // Simple pattern matching (could be improved with glob)
        let extension = pattern.trim_start_matches("*.");
        input_files.retain(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == extension)
                .unwrap_or(false)
        });
    }

    // Set up thread pool
    if let Some(threads) = config.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| Error::OperationFailed {
                message: format!("Failed to set thread pool: {}", e),
            })?;
    }

    // Process files in parallel
    let results: Vec<Result<()>> = input_files
        .par_iter()
        .map(|input_path| {
            let file_name = input_path
                .file_name()
                .ok_or_else(|| Error::OperationFailed {
                    message: "Invalid file name".to_string(),
                })?;

            let output_path = Path::new(&config.output_dir).join(file_name);
            processor(input_path, &output_path)
        })
        .collect();

    Ok(results)
}

fn collect_files(dir: &str, files: &mut Vec<std::path::PathBuf>, recursive: bool) -> Result<()> {
    use std::fs;

    let entries = fs::read_dir(dir).map_err(|e| Error::OperationFailed {
        message: format!("Failed to read directory: {}", e),
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| Error::OperationFailed {
            message: format!("Failed to read directory entry: {}", e),
        })?;

        let path = entry.path();

        if path.is_file() {
            files.push(path);
        } else if path.is_dir() && recursive {
            if let Ok(dir_str) = path.to_str().ok_or_else(|| Error::OperationFailed {
                message: "Invalid path".to_string(),
            }) {
                collect_files(dir_str, files, recursive)?;
            }
        }
    }

    Ok(())
}
