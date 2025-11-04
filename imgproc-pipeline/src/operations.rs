use crate::config::{AdaptiveMethod, InterpolationMethod, StructuringElementType, ThresholdType};
use crate::error::{Error, Result};
use std::collections::HashMap;

/// Registry of available operations
pub struct OperationRegistry;

impl OperationRegistry {
    /// Lists all available operations
    pub fn list_operations() -> Vec<&'static str> {
        vec![
            "grayscale",
            "rgb_to_gray",
            "resize",
            "crop",
            "rotate",
            "gaussian_blur",
            "median_filter",
            "bilateral_filter",
            "sobel",
            "canny",
            "threshold",
            "otsu_threshold",
            "adaptive_threshold",
            "erode",
            "dilate",
            "opening",
            "closing",
            // Advanced filtering operations
            "anisotropic_diffusion",
            "unsharp_mask",
            // Histogram operations
            "clahe",
            "histogram_equalization",
            "histogram_matching",
            // Edge detection operations
            "prewitt",
            "scharr",
            "laplacian_of_gaussian",
            "difference_of_gaussians",
            "zero_crossing",
            // Morphological operations
            "morph_gradient",
            "tophat",
            "blackhat",
        ]
    }

    /// Checks if an operation exists
    pub fn operation_exists(name: &str) -> bool {
        Self::list_operations().contains(&name)
    }
}

/// Represents an executable operation with parameters
pub enum Operation {
    /// Convert RGB to grayscale
    RgbToGray,
    /// Resize image
    Resize {
        width: u32,
        height: u32,
        interpolation: InterpolationMethod,
    },
    /// Crop image
    Crop {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// Rotate image
    Rotate {
        angle: f32,
    },
    /// Gaussian blur
    GaussianBlur {
        kernel_size: u32,
        sigma: f32,
    },
    /// Median filter
    MedianFilter {
        kernel_size: u32,
    },
    /// Bilateral filter
    BilateralFilter {
        diameter: u32,
        sigma_color: f32,
        sigma_space: f32,
    },
    /// Sobel edge detection
    Sobel,
    /// Canny edge detection
    Canny {
        low_threshold: f32,
        high_threshold: f32,
    },
    /// Threshold
    Threshold {
        threshold: u8,
        max_value: u8,
        threshold_type: ThresholdType,
    },
    /// Otsu's threshold
    OtsuThreshold,
    /// Adaptive threshold
    AdaptiveThreshold {
        max_value: u8,
        method: AdaptiveMethod,
        block_size: u32,
        constant: f32,
    },
    /// Erosion
    Erode {
        kernel_size: u32,
        element: StructuringElementType,
    },
    /// Dilation
    Dilate {
        kernel_size: u32,
        element: StructuringElementType,
    },
    /// Opening
    Opening {
        kernel_size: u32,
        element: StructuringElementType,
    },
    /// Closing
    Closing {
        kernel_size: u32,
        element: StructuringElementType,
    },
    /// Anisotropic diffusion
    AnisotropicDiffusion {
        iterations: u32,
        kappa: f32,
        lambda: f32,
        option: String,
    },
    /// Unsharp mask
    UnsharpMask {
        amount: f32,
        radius: f32,
        threshold: u8,
    },
    /// CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Clahe {
        clip_limit: f32,
        tile_grid_size: (u32, u32),
    },
    /// Histogram equalization
    HistogramEqualization,
    /// Histogram matching
    HistogramMatching {
        reference_image: String,
    },
    /// Prewitt edge detection
    Prewitt,
    /// Scharr edge detection
    Scharr,
    /// Laplacian of Gaussian
    LaplacianOfGaussian {
        sigma: f32,
        kernel_size: u32,
    },
    /// Difference of Gaussians
    DifferenceOfGaussians {
        sigma1: f32,
        sigma2: f32,
        kernel_size: u32,
    },
    /// Zero crossing detection
    ZeroCrossing {
        method: String,
        threshold: f32,
    },
    /// Morphological gradient
    MorphGradient {
        kernel_size: u32,
        shape: StructuringElementType,
        gradient_type: String,
    },
    /// Top-hat transform
    TopHat {
        kernel_size: u32,
        shape: StructuringElementType,
        variant: String,
    },
    /// Black-hat transform
    BlackHat {
        kernel_size: u32,
        shape: StructuringElementType,
    },
}

impl Operation {
    /// Creates an operation from a name and parameters
    pub fn from_config(
        name: &str,
        params: &HashMap<String, serde_yaml::Value>,
    ) -> Result<Self> {
        match name {
            "grayscale" | "rgb_to_gray" => Ok(Operation::RgbToGray),

            "resize" => {
                let width = get_u32_param(params, "width", name)?;
                let height = get_u32_param(params, "height", name)?;
                let interpolation = get_interpolation_param(params, "interpolation")
                    .unwrap_or(InterpolationMethod::Bilinear);
                Ok(Operation::Resize {
                    width,
                    height,
                    interpolation,
                })
            }

            "crop" => {
                let x = get_u32_param(params, "x", name)?;
                let y = get_u32_param(params, "y", name)?;
                let width = get_u32_param(params, "width", name)?;
                let height = get_u32_param(params, "height", name)?;
                Ok(Operation::Crop { x, y, width, height })
            }

            "rotate" => {
                let angle = get_f32_param(params, "angle", name)?;
                Ok(Operation::Rotate { angle })
            }

            "gaussian_blur" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                let sigma = get_f32_param(params, "sigma", name)?;
                Ok(Operation::GaussianBlur { kernel_size, sigma })
            }

            "median_filter" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                Ok(Operation::MedianFilter { kernel_size })
            }

            "bilateral_filter" => {
                let diameter = get_u32_param(params, "diameter", name)?;
                let sigma_color = get_f32_param(params, "sigma_color", name)?;
                let sigma_space = get_f32_param(params, "sigma_space", name)?;
                Ok(Operation::BilateralFilter {
                    diameter,
                    sigma_color,
                    sigma_space,
                })
            }

            "sobel" => Ok(Operation::Sobel),

            "canny" => {
                let low_threshold = get_f32_param(params, "low_threshold", name)?;
                let high_threshold = get_f32_param(params, "high_threshold", name)?;
                Ok(Operation::Canny {
                    low_threshold,
                    high_threshold,
                })
            }

            "threshold" => {
                let threshold = get_u8_param(params, "threshold", name)?;
                let max_value = get_u8_param(params, "max_value", name)?;
                let threshold_type = get_threshold_type_param(params, "type")
                    .unwrap_or(ThresholdType::Binary);
                Ok(Operation::Threshold {
                    threshold,
                    max_value,
                    threshold_type,
                })
            }

            "otsu_threshold" => Ok(Operation::OtsuThreshold),

            "adaptive_threshold" => {
                let max_value = get_u8_param(params, "max_value", name)?;
                let method = get_adaptive_method_param(params, "method")
                    .unwrap_or(AdaptiveMethod::Mean);
                let block_size = get_u32_param(params, "block_size", name)?;
                let constant = get_f32_param(params, "constant", name)?;
                Ok(Operation::AdaptiveThreshold {
                    max_value,
                    method,
                    block_size,
                    constant,
                })
            }

            "erode" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                let element = get_structuring_element_param(params, "element")
                    .unwrap_or(StructuringElementType::Rectangle);
                Ok(Operation::Erode { kernel_size, element })
            }

            "dilate" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                let element = get_structuring_element_param(params, "element")
                    .unwrap_or(StructuringElementType::Rectangle);
                Ok(Operation::Dilate { kernel_size, element })
            }

            "opening" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                let element = get_structuring_element_param(params, "element")
                    .unwrap_or(StructuringElementType::Rectangle);
                Ok(Operation::Opening { kernel_size, element })
            }

            "closing" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                let element = get_structuring_element_param(params, "element")
                    .unwrap_or(StructuringElementType::Rectangle);
                Ok(Operation::Closing { kernel_size, element })
            }

            "anisotropic_diffusion" => {
                let iterations = get_u32_param(params, "iterations", name)?;
                let kappa = get_f32_param(params, "kappa", name)?;
                let lambda = get_f32_param(params, "lambda", name)?;
                let option = get_string_param(params, "option").unwrap_or_else(|| "option1".to_string());
                Ok(Operation::AnisotropicDiffusion {
                    iterations,
                    kappa,
                    lambda,
                    option,
                })
            }

            "unsharp_mask" => {
                let amount = get_f32_param(params, "amount", name)?;
                let radius = get_f32_param(params, "radius", name)?;
                let threshold = get_u8_param(params, "threshold", name)?;
                Ok(Operation::UnsharpMask {
                    amount,
                    radius,
                    threshold,
                })
            }

            "clahe" => {
                let clip_limit = get_f32_param(params, "clip_limit", name)?;
                let tile_grid_size = get_tuple_u32_param(params, "tile_grid_size", name)?;
                Ok(Operation::Clahe {
                    clip_limit,
                    tile_grid_size,
                })
            }

            "histogram_equalization" => Ok(Operation::HistogramEqualization),

            "histogram_matching" => {
                let reference_image = get_string_param(params, "reference_image")
                    .ok_or_else(|| Error::MissingParameter {
                        operation: name.to_string(),
                        parameter: "reference_image".to_string(),
                    })?;
                Ok(Operation::HistogramMatching { reference_image })
            }

            "prewitt" => Ok(Operation::Prewitt),

            "scharr" => Ok(Operation::Scharr),

            "laplacian_of_gaussian" => {
                let sigma = get_f32_param(params, "sigma", name)?;
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                Ok(Operation::LaplacianOfGaussian { sigma, kernel_size })
            }

            "difference_of_gaussians" => {
                let sigma1 = get_f32_param(params, "sigma1", name)?;
                let sigma2 = get_f32_param(params, "sigma2", name)?;
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                Ok(Operation::DifferenceOfGaussians {
                    sigma1,
                    sigma2,
                    kernel_size,
                })
            }

            "zero_crossing" => {
                let method = get_string_param(params, "method").unwrap_or_else(|| "laplacian".to_string());
                let threshold = get_f32_param(params, "threshold", name)?;
                Ok(Operation::ZeroCrossing { method, threshold })
            }

            "morph_gradient" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                let shape = get_structuring_element_param(params, "shape")
                    .unwrap_or(StructuringElementType::Rectangle);
                let gradient_type = get_string_param(params, "gradient_type")
                    .unwrap_or_else(|| "basic".to_string());
                Ok(Operation::MorphGradient {
                    kernel_size,
                    shape,
                    gradient_type,
                })
            }

            "tophat" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                let shape = get_structuring_element_param(params, "shape")
                    .unwrap_or(StructuringElementType::Rectangle);
                let variant = get_string_param(params, "variant").unwrap_or_else(|| "white".to_string());
                Ok(Operation::TopHat {
                    kernel_size,
                    shape,
                    variant,
                })
            }

            "blackhat" => {
                let kernel_size = get_u32_param(params, "kernel_size", name)?;
                let shape = get_structuring_element_param(params, "shape")
                    .unwrap_or(StructuringElementType::Rectangle);
                Ok(Operation::BlackHat { kernel_size, shape })
            }

            _ => Err(Error::UnknownOperation(name.to_string())),
        }
    }
}

// Parameter extraction helpers

fn get_u32_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
    operation: &str,
) -> Result<u32> {
    params
        .get(name)
        .ok_or_else(|| Error::MissingParameter {
            operation: operation.to_string(),
            parameter: name.to_string(),
        })?
        .as_u64()
        .ok_or_else(|| Error::InvalidParameter {
            operation: operation.to_string(),
            name: name.to_string(),
            message: "Expected unsigned integer".to_string(),
        })
        .map(|v| v as u32)
}

fn get_u8_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
    operation: &str,
) -> Result<u8> {
    params
        .get(name)
        .ok_or_else(|| Error::MissingParameter {
            operation: operation.to_string(),
            parameter: name.to_string(),
        })?
        .as_u64()
        .ok_or_else(|| Error::InvalidParameter {
            operation: operation.to_string(),
            name: name.to_string(),
            message: "Expected unsigned integer 0-255".to_string(),
        })
        .and_then(|v| {
            if v <= 255 {
                Ok(v as u8)
            } else {
                Err(Error::InvalidParameter {
                    operation: operation.to_string(),
                    name: name.to_string(),
                    message: "Value must be 0-255".to_string(),
                })
            }
        })
}

fn get_f32_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
    operation: &str,
) -> Result<f32> {
    params
        .get(name)
        .ok_or_else(|| Error::MissingParameter {
            operation: operation.to_string(),
            parameter: name.to_string(),
        })?
        .as_f64()
        .ok_or_else(|| Error::InvalidParameter {
            operation: operation.to_string(),
            name: name.to_string(),
            message: "Expected floating point number".to_string(),
        })
        .map(|v| v as f32)
}

fn get_interpolation_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
) -> Option<InterpolationMethod> {
    params.get(name).and_then(|v| {
        v.as_str().and_then(|s| match s.to_lowercase().as_str() {
            "nearest" => Some(InterpolationMethod::Nearest),
            "bilinear" => Some(InterpolationMethod::Bilinear),
            "bicubic" => Some(InterpolationMethod::Bicubic),
            "lanczos" => Some(InterpolationMethod::Lanczos),
            _ => None,
        })
    })
}

fn get_threshold_type_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
) -> Option<ThresholdType> {
    params.get(name).and_then(|v| {
        v.as_str().and_then(|s| match s.to_lowercase().as_str() {
            "binary" => Some(ThresholdType::Binary),
            "binary_inv" => Some(ThresholdType::BinaryInv),
            "truncate" => Some(ThresholdType::Truncate),
            "to_zero" => Some(ThresholdType::ToZero),
            "to_zero_inv" => Some(ThresholdType::ToZeroInv),
            _ => None,
        })
    })
}

fn get_adaptive_method_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
) -> Option<AdaptiveMethod> {
    params.get(name).and_then(|v| {
        v.as_str().and_then(|s| match s.to_lowercase().as_str() {
            "mean" => Some(AdaptiveMethod::Mean),
            "gaussian" => Some(AdaptiveMethod::Gaussian),
            _ => None,
        })
    })
}

fn get_structuring_element_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
) -> Option<StructuringElementType> {
    params.get(name).and_then(|v| {
        v.as_str().and_then(|s| match s.to_lowercase().as_str() {
            "rectangle" => Some(StructuringElementType::Rectangle),
            "ellipse" => Some(StructuringElementType::Ellipse),
            "cross" => Some(StructuringElementType::Cross),
            _ => None,
        })
    })
}

fn get_string_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
) -> Option<String> {
    params.get(name).and_then(|v| v.as_str().map(|s| s.to_string()))
}

fn get_tuple_u32_param(
    params: &HashMap<String, serde_yaml::Value>,
    name: &str,
    operation: &str,
) -> Result<(u32, u32)> {
    let seq = params
        .get(name)
        .ok_or_else(|| Error::MissingParameter {
            operation: operation.to_string(),
            parameter: name.to_string(),
        })?
        .as_sequence()
        .ok_or_else(|| Error::InvalidParameter {
            operation: operation.to_string(),
            name: name.to_string(),
            message: "Expected array of two integers".to_string(),
        })?;

    if seq.len() != 2 {
        return Err(Error::InvalidParameter {
            operation: operation.to_string(),
            name: name.to_string(),
            message: "Expected array of exactly two integers".to_string(),
        });
    }

    let first = seq[0].as_u64().ok_or_else(|| Error::InvalidParameter {
        operation: operation.to_string(),
        name: name.to_string(),
        message: "Expected unsigned integer for first element".to_string(),
    })? as u32;

    let second = seq[1].as_u64().ok_or_else(|| Error::InvalidParameter {
        operation: operation.to_string(),
        name: name.to_string(),
        message: "Expected unsigned integer for second element".to_string(),
    })? as u32;

    Ok((first, second))
}
