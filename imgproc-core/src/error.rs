use thiserror::Error;

/// Core error type for image processing operations
#[derive(Error, Debug)]
pub enum Error {
    /// Invalid image dimensions
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Image width
        width: u32,
        /// Image height
        height: u32
    },

    /// Invalid parameter value
    #[error("Invalid parameter {name}: {message}")]
    InvalidParameter {
        /// Parameter name
        name: String,
        /// Error message
        message: String,
    },

    /// Operation not supported for given pixel type
    #[error("Operation not supported for pixel type: {pixel_type}")]
    UnsupportedPixelType {
        /// Pixel type name
        pixel_type: String,
    },

    /// Buffer size mismatch
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected size
        expected: usize,
        /// Actual size
        actual: usize,
    },

    /// Out of bounds access
    #[error("Out of bounds: ({x}, {y}) is outside image bounds ({width}x{height})")]
    OutOfBounds {
        /// X coordinate
        x: u32,
        /// Y coordinate
        y: u32,
        /// Image width
        width: u32,
        /// Image height
        height: u32,
    },

    /// Memory allocation failure
    #[error("Memory allocation failed: {size} bytes")]
    AllocationFailed {
        /// Requested size
        size: usize,
    },

    /// Invalid crop rectangle
    #[error("Invalid crop rectangle: ({x}, {y}, {width}, {height}) exceeds image bounds")]
    InvalidCrop {
        /// X coordinate
        x: u32,
        /// Y coordinate
        y: u32,
        /// Crop width
        width: u32,
        /// Crop height
        height: u32,
    },

    /// Color conversion error
    #[error("Color conversion failed: {message}")]
    ColorConversion {
        /// Error message
        message: String,
    },

    /// Generic operation error
    #[error("Operation failed: {message}")]
    OperationFailed {
        /// Error message
        message: String,
    },
}

/// Result type alias for image processing operations
pub type Result<T> = std::result::Result<T, Error>;