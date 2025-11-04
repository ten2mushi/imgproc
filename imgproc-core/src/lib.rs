//! - image data structures
//! - pixel types and color space abstractions
//! - traits for operations and transformations
//! - errror types

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

pub mod color;
pub mod error;
pub mod image;
pub mod pixel;
pub mod traits;

pub use color::{ColorSpace, ColorType};
pub use error::{Error, Result};
pub use image::{Image, ImageView, ImageViewMut};
pub use pixel::{Gray, Pixel, Rgb, Rgba};
pub use traits::{Operation, InPlaceOperation};

/// Image dimensions type alias
pub type Size = (u32, u32);

/// Rectangle for crop operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// X coordinate of top-left corner
    pub x: u32,
    /// Y coordinate of top-left corner
    pub y: u32,
    /// Width of the rectangle
    pub width: u32,
    /// Height of the rectangle
    pub height: u32,
}

impl Rect {
    /// Create a new rectangle
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// Check if rectangle is valid (non-zero dimensions)
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Calculate area
    #[must_use]
    pub const fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}
