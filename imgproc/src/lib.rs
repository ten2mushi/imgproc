//! # imgproc - image processing library

//! use imgproc::prelude::*;
//!
//! # fn main() -> imgproc::core::Result<()> {
//! // Load an image
//! let img = imgproc::io::read_image_gray("input.jpg")?;
//!
//! // Apply Gaussian blur
//! let blur = imgproc::ops::filters::GaussianBlur::new(5, 1.5)?;
//! let blurred = blur.apply_gray(&img)?;
//!
//! // Detect edges with Canny
//! let canny = imgproc::ops::edges::Canny::new(50.0, 150.0)?;
//! let edges = canny.apply_gray(&blurred)?;
//!
//! // Save result
//! imgproc::io::write_image_gray(&edges, "output.png")?;
//! # Ok(())
//! # }
//! ```
//!
//! organized into multiple crates:
//!
//! - `imgproc-core`: image types, pixel traits, and base traits
//! - `imgproc-ops`: operation implementations (filters, transforms, etc.)
//! - `imgproc-io`: image I/O and batch processing
//! - `imgproc-pipeline`: YAML-based pipeline execution
//! - `imgproc-cli`: CLI
//! - `imgproc`: main facade crate (this crate)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

pub use imgproc_core as core;
pub use imgproc_ops as ops;
pub use imgproc_io as io;

pub mod prelude {
    pub use crate::core::{
        ColorSpace, Error, Image, Pixel, Rect, Result,
    };
    pub use crate::core::pixel::{Gray8, Rgb8, Rgba8};

    pub use crate::ops::{
        ColorConvert,
        GaussianBlur, MedianFilter, BilateralFilter,
        BoxFilter, Laplacian, LaplacianKernel, Scharr, UnsharpMask,
        GaussianDerivative, DerivativeOrder, DerivativeDirection,
        AnisotropicDiffusion, DiffusionOption,
        Resize, Crop, Rotate,
        Erode, Dilate, MorphOpen, MorphClose, StructuringElement,
        MorphGradient, GradientType,
        TopHat, TopHatVariant,
        Skeletonize, SkeletonMethod,
        Thinning,
        HitOrMiss,
        Canny, Sobel, Prewitt,
        Threshold, OtsuThreshold, AdaptiveThreshold,
        ZeroCrossing, ZeroCrossingMethod,
        LaplacianOfGaussian, DifferenceOfGaussians,
    };

    // for convenience
    pub use crate::ops::MorphOpen as Opening;
    pub use crate::ops::MorphClose as Closing;

    pub use crate::io::{
        read_image, read_image_gray, read_image_rgb,
        write_image, write_image_gray, write_image_rgb,
        BatchConfig, process_batch,
    };
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_image_creation() {
        let img = Image::<Gray8>::new(100, 100);
        assert!(img.is_ok());
        let img = img.unwrap();
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn test_pixel_access() {
        let mut img = Image::<Gray8>::new(10, 10).unwrap();
        let pixel = Gray8::new(128);
        assert!(img.set_pixel(5, 5, pixel).is_ok());
        let retrieved = img.get_pixel(5, 5).unwrap();
        assert_eq!(retrieved.value, 128);
    }
}
