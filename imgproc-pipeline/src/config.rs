use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline version (e.g., "1.0")
    pub version: String,

    /// Pipeline name
    pub name: String,

    /// Optional description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Variables for interpolation
    #[serde(default)]
    pub variables: HashMap<String, serde_yaml::Value>,

    /// Sequence of operations
    pub pipeline: Vec<OperationConfig>,

    /// Error handling strategy
    #[serde(default)]
    pub error_handling: ErrorHandlingConfig,

    /// Output configuration
    #[serde(default)]
    pub output: OutputConfig,
}

/// Error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Strategy to use when an operation fails
    #[serde(default)]
    pub on_error: ErrorHandlingStrategy,

    /// Whether to log errors
    #[serde(default = "default_true")]
    pub log_errors: bool,
}

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            on_error: ErrorHandlingStrategy::Stop,
            log_errors: true,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Error handling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ErrorHandlingStrategy {
    /// Stop processing on first error
    Stop,
    /// Skip failed images and continue with others
    Skip,
    /// Continue processing, ignore errors
    Continue,
}

impl Default for ErrorHandlingStrategy {
    fn default() -> Self {
        Self::Stop
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format (png, jpg, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Naming pattern for output files
    #[serde(skip_serializing_if = "Option::is_none")]
    pub naming: Option<String>,

    /// Whether to save intermediate results
    #[serde(default)]
    pub save_intermediates: bool,

    /// Directory for intermediate results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intermediate_dir: Option<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: None,
            naming: None,
            save_intermediates: false,
            intermediate_dir: None,
        }
    }
}

/// Configuration for a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationConfig {
    /// Operation type (e.g., "grayscale", "resize", "blur")
    pub operation: String,

    /// Optional name for this operation instance
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Operation parameters
    #[serde(default)]
    pub params: HashMap<String, serde_yaml::Value>,

    /// Conditional execution
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
}

// Enums for common parameter types

/// Interpolation methods for resizing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
    /// Lanczos interpolation
    Lanczos,
}

/// Threshold types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThresholdType {
    /// Binary threshold
    Binary,
    /// Binary inverted threshold
    BinaryInv,
    /// Truncate
    Truncate,
    /// To zero
    ToZero,
    /// To zero inverted
    ToZeroInv,
}

/// Adaptive thresholding methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AdaptiveMethod {
    /// Mean method
    Mean,
    /// Gaussian method
    Gaussian,
}

/// Structuring element types for morphological operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StructuringElementType {
    /// Rectangle
    Rectangle,
    /// Ellipse
    Ellipse,
    /// Cross
    Cross,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_config() {
        let yaml = r#"
version: "1.0"
name: "test"
pipeline:
  - operation: grayscale
"#;
        let config: PipelineConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.version, "1.0");
        assert_eq!(config.name, "test");
        assert_eq!(config.pipeline.len(), 1);
    }

    #[test]
    fn test_parse_with_variables() {
        let yaml = r#"
version: "1.0"
name: "test"
variables:
  size: 800
  sigma: 1.5
pipeline:
  - operation: resize
    params:
      width: 800
      height: 600
"#;
        let config: PipelineConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.variables.len(), 2);
    }

    #[test]
    fn test_error_handling_config() {
        let yaml = r#"
version: "1.0"
name: "test"
error_handling:
  on_error: skip
  log_errors: false
pipeline:
  - operation: grayscale
"#;
        let config: PipelineConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.error_handling.on_error, ErrorHandlingStrategy::Skip);
        assert!(!config.error_handling.log_errors);
    }
}
