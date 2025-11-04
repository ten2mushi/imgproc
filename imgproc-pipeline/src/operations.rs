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
