use thiserror::Error;

/// Pipeline error type
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error
    #[error("I/O error: {0}")]
    IoError(String),

    /// YAML parsing error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Unknown operation
    #[error("Unknown operation: {0}")]
    UnknownOperation(String),

    /// Invalid parameter
    #[error("Invalid parameter '{name}' for operation '{operation}': {message}")]
    InvalidParameter {
        /// Operation name
        operation: String,
        /// Parameter name
        name: String,
        /// Error message
        message: String,
    },

    /// Missing required parameter
    #[error("Missing required parameter '{parameter}' for operation '{operation}'")]
    MissingParameter {
        /// Operation name
        operation: String,
        /// Parameter name
        parameter: String,
    },

    /// Variable not found
    #[error("Variable not found: {0}")]
    VariableNotFound(String),

    /// Execution error
    #[error("Execution error in operation '{operation}': {message}")]
    ExecutionError {
        /// Operation name
        operation: String,
        /// Error message
        message: String,
    },

    /// Image processing error
    #[error("Image processing error: {0}")]
    ImageError(String),

    /// Type conversion error
    #[error("Type conversion error: {0}")]
    ConversionError(String),
}

/// Result type for pipeline operations
pub type Result<T> = std::result::Result<T, Error>;

/// Convert imgproc-core errors to pipeline errors
impl From<imgproc_core::Error> for Error {
    fn from(err: imgproc_core::Error) -> Self {
        Error::ImageError(err.to_string())
    }
}
