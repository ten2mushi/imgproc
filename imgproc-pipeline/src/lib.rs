
//! - YAML-based pipeline configuration
//!
//! # Example
//!
//! ```rust,no_run
//! use imgproc_pipeline::Pipeline;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load pipeline from YAML
//! let pipeline = Pipeline::from_file("pipeline.yaml")?;
//!
//! // Validate pipeline
//! pipeline.validate()?;
//!
//! // Execute pipeline
//! pipeline.execute("input.jpg", "output.png")?;
//! # Ok(())
//! # }
//! ```

mod config;
mod error;
mod executor;
mod interpolation;
mod operations;
mod validation;

pub use config::{
    AdaptiveMethod, ErrorHandlingStrategy, InterpolationMethod, OperationConfig,
    OutputConfig, PipelineConfig, StructuringElementType, ThresholdType,
};
pub use error::{Error, Result};
pub use executor::{ExecutionContext, PipelineExecutor};
pub use operations::OperationRegistry;

use std::path::Path;

/// pipeline operations are executed sequentially on input images.
pub struct Pipeline {
    config: PipelineConfig,
    executor: PipelineExecutor,
}

impl Pipeline {
    /// Creates a new pipeline from a YAML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the YAML configuration file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::IoError(e.to_string()))?;
        Self::from_yaml(&content)
    }

    /// Creates a new pipeline from a YAML string.
    ///
    /// # Arguments
    ///
    /// * `yaml` - YAML configuration as a string
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML cannot be parsed.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        let config: PipelineConfig = serde_yaml::from_str(yaml)
            .map_err(|e| Error::ParseError(format!("Failed to parse YAML: {}", e)))?;

        let executor = PipelineExecutor::new();

        Ok(Self { config, executor })
    }

    /// Validates the pipeline configuration.
    ///
    /// Checks that all operations are valid, parameters are correct, and
    /// variable references can be resolved.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate(&self) -> Result<()> {
        validation::validate_pipeline(&self.config)
    }

    /// Executes the pipeline on a single image.
    ///
    /// # Arguments
    ///
    /// * `input_path` - Path to the input image
    /// * `output_path` - Path where the output image should be saved
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails, depending on the error handling strategy.
    pub fn execute<P: AsRef<Path>>(&self, input_path: P, output_path: P) -> Result<()> {
        self.executor.execute(&self.config, input_path, output_path)
    }

    /// Executes the pipeline on a batch of images.
    ///
    /// # Arguments
    ///
    /// * `input_dir` - Directory containing input images
    /// * `output_dir` - Directory where output images should be saved
    ///
    /// # Errors
    ///
    /// Returns a vector of results, one for each processed image.
    pub fn execute_batch<P: AsRef<Path>>(
        &self,
        input_dir: P,
        output_dir: P,
    ) -> Result<Vec<Result<()>>> {
        self.executor
            .execute_batch(&self.config, input_dir, output_dir)
    }

    /// Returns the pipeline name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Returns the pipeline description, if available.
    pub fn description(&self) -> Option<&str> {
        self.config.description.as_deref()
    }

    /// Returns the pipeline version.
    pub fn version(&self) -> &str {
        &self.config.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_parse() {
        let yaml = r#"
version: "1.0"
name: "test_pipeline"
description: "A test pipeline"

variables:
  size: 800
  sigma: 1.5

pipeline:
  - operation: grayscale
  - operation: resize
    params:
      width: ${size}
      height: ${size}
      interpolation: bilinear
"#;

        let pipeline = Pipeline::from_yaml(yaml).expect("Failed to parse pipeline");
        assert_eq!(pipeline.name(), "test_pipeline");
        assert_eq!(pipeline.version(), "1.0");
    }
}
