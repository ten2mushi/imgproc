//! - Color space conversions
//! - Geometric transformations
//! - Filtering operations
//! - Morphological operations
//! - Edge detection
//! - Segmentation

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

pub mod color;
pub mod filters;
pub mod geometry;
pub mod morphology;
pub mod edges;
pub mod segment;

// Phase 3 modules
pub mod histogram;
pub mod contours;

// Re-export main operations
pub use color::ColorConvert;
pub use filters::{
    GaussianBlur, MedianFilter, BilateralFilter,
    BoxFilter, Laplacian, LaplacianKernel, Scharr, UnsharpMask,
    GaussianDerivative, DerivativeOrder, DerivativeDirection,
    AnisotropicDiffusion, DiffusionOption,
};
pub use geometry::{Resize, Crop, Rotate};
pub use morphology::{
    Erode, Dilate, MorphOpen, MorphClose, StructuringElement,
    MorphGradient, GradientType,
    TopHat, TopHatVariant,
    Skeletonize, SkeletonMethod,
    Thinning,
    HitOrMiss,
};
pub use edges::{
    Canny, Sobel, Prewitt,
    LaplacianOfGaussian, DifferenceOfGaussians,
    ZeroCrossing, ZeroCrossingMethod,
};
pub use segment::{Threshold, OtsuThreshold, AdaptiveThreshold};

// Phase 3 re-exports
pub use histogram::{Histogram, HistogramEqualization, Clahe, HistogramMatching};
pub use contours::{
    FindContours, Contour, ContourFeatures, ApproxPolyDP,
    Point, Rect, ContourRetrievalMode, ContourApproximation, Moments,
};

// Phase 3 color space types
pub use color::{Hsl, Lab, Xyz, YCbCr};
