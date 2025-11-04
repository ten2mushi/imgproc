use crate::config::PipelineConfig;
use crate::error::{Error, Result};
use crate::interpolation;
use crate::operations::{Operation, OperationRegistry};

/// Validates a complete pipeline configuration.
///
/// Checks:
/// - Pipeline version is supported
/// - All operations are known
/// - All required parameters are present
/// - Variable references are valid
/// - Parameter types and values are correct
pub fn validate_pipeline(config: &PipelineConfig) -> Result<()> {
    // Check version
    if config.version != "1.0" {
        return Err(Error::ValidationError(format!(
            "Unsupported pipeline version: {}. Only '1.0' is supported.",
            config.version
        )));
    }

    // Check that pipeline is not empty
    if config.pipeline.is_empty() {
        return Err(Error::ValidationError(
            "Pipeline must contain at least one operation".to_string(),
        ));
    }

    // Validate each operation
    for (index, op_config) in config.pipeline.iter().enumerate() {
        validate_operation(op_config, &config.variables, index)?;
    }

    Ok(())
}

/// Validates a single operation configuration.
fn validate_operation(
    op_config: &crate::config::OperationConfig,
    variables: &std::collections::HashMap<String, serde_yaml::Value>,
    index: usize,
) -> Result<()> {
    let op_name = &op_config.operation;

    // Check if operation exists
    if !OperationRegistry::operation_exists(op_name) {
        return Err(Error::UnknownOperation(format!(
            "Operation '{}' at index {} is not recognized. Available operations: {}",
            op_name,
            index,
            OperationRegistry::list_operations().join(", ")
        )));
    }

    // Interpolate parameters
    let interpolated_params = interpolation::interpolate_params(&op_config.params, variables)?;

    // Try to construct the operation (this validates parameters)
    Operation::from_config(op_name, &interpolated_params).map_err(|e| {
        Error::ValidationError(format!(
            "Invalid configuration for operation '{}' at index {}: {}",
            op_name, index, e
        ))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_validate_empty_pipeline() {
        let config = PipelineConfig {
            version: "1.0".to_string(),
            name: "test".to_string(),
            description: None,
            variables: HashMap::new(),
            pipeline: vec![],
            error_handling: Default::default(),
            output: Default::default(),
        };

        let result = validate_pipeline(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_unknown_operation() {
        let config = PipelineConfig {
            version: "1.0".to_string(),
            name: "test".to_string(),
            description: None,
            variables: HashMap::new(),
            pipeline: vec![crate::config::OperationConfig {
                operation: "unknown_operation".to_string(),
                name: None,
                params: HashMap::new(),
                condition: None,
            }],
            error_handling: Default::default(),
            output: Default::default(),
        };

        let result = validate_pipeline(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_valid_pipeline() {
        let config = PipelineConfig {
            version: "1.0".to_string(),
            name: "test".to_string(),
            description: None,
            variables: HashMap::new(),
            pipeline: vec![crate::config::OperationConfig {
                operation: "grayscale".to_string(),
                name: None,
                params: HashMap::new(),
                condition: None,
            }],
            error_handling: Default::default(),
            output: Default::default(),
        };

        let result = validate_pipeline(&config);
        assert!(result.is_ok());
    }
}
