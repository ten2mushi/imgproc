use crate::error::Result;

/// Trait for image processing operations
pub trait Operation: Send + Sync {
    /// Input type for the operation
    type Input;
    /// Output type for the operation
    type Output;

    /// Apply the operation to an input
    fn apply(&self, input: Self::Input) -> Result<Self::Output>;

    /// Get operation metadata
    fn metadata(&self) -> OperationMetadata {
        OperationMetadata::default()
    }
}

/// Trait for in-place operations
pub trait InPlaceOperation: Send + Sync {
    /// Target type for the operation
    type Target;

    /// Apply the operation in-place
    fn apply_inplace(&self, target: &mut Self::Target) -> Result<()>;
}

/// Metadata about an operation
#[derive(Debug, Clone, Default)]
pub struct OperationMetadata {
    /// Operation name
    pub name: String,
    /// Operation description
    pub description: String,
    /// Whether the operation is parallelizable
    pub parallelizable: bool,
    /// Whether the operation preserves dimensions
    pub preserves_dimensions: bool,
    /// Whether the operation preserves color type
    pub preserves_color_type: bool,
}

/// Builder trait for operations with configurable parameters
pub trait OperationBuilder {
    /// The operation type being built
    type Operation;

    /// Build the operation with validated parameters
    fn build(self) -> Result<Self::Operation>;
}

/// Trait for composable operations
pub trait Compose<Other> {
    /// Output type of composition
    type Output;

    /// Compose this operation with another
    fn compose(self, other: Other) -> Self::Output;
}

/// Trait for operations that can be converted to dynamic dispatch
pub trait IntoDynamic {
    /// Input type
    type Input;
    /// Output type
    type Output;

    /// Convert to dynamic operation
    fn into_dynamic(self) -> Box<dyn Operation<Input = Self::Input, Output = Self::Output>>;
}