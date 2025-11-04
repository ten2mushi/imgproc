use crate::config::{ErrorHandlingStrategy, InterpolationMethod, PipelineConfig, AdaptiveMethod, StructuringElementType, ThresholdType};
use crate::error::{Error, Result};
use crate::interpolation;
use crate::operations::Operation;
use imgproc_core::{Gray, Image, Rect, Rgb};
use imgproc_ops::{
    color, edges, filters, geometry, morphology, segment,
};
use std::path::{Path, PathBuf};

// Type aliases for convenience
type Gray8 = Gray<u8>;
type Rgb8 = Rgb<u8>;

/// Execution context for a pipeline run
pub struct ExecutionContext {
    /// Current image being processed (grayscale)
    pub current_gray: Option<Image<Gray8>>,
    /// Current image being processed (RGB)
    pub current_rgb: Option<Image<Rgb8>>,
    /// Input file name (for variable substitution)
    pub input_filename: String,
}

impl ExecutionContext {
    /// Creates a new execution context
    pub fn new() -> Self {
        Self {
            current_gray: None,
            current_rgb: None,
            input_filename: String::new(),
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline executor
pub struct PipelineExecutor;

impl PipelineExecutor {
    /// Creates a new pipeline executor
    pub fn new() -> Self {
        Self
    }

    /// Executes a pipeline on a single image
    pub fn execute<P: AsRef<Path>>(
        &self,
        config: &PipelineConfig,
        input_path: P,
        output_path: P,
    ) -> Result<()> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        log::info!("Executing pipeline '{}' on {:?}", config.name, input_path);

        // Create execution context
        let mut context = ExecutionContext::new();
        context.input_filename = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
            .to_string();

        // Load input image - try grayscale first, then RGB
        match imgproc_io::read_image_gray(input_path) {
            Ok(img) => {
                context.current_gray = Some(img);
            }
            Err(_) => {
                let img = imgproc_io::read_image_rgb(input_path)
                    .map_err(|e| Error::IoError(format!("Failed to load image: {}", e)))?;
                context.current_rgb = Some(img);
            }
        }

        // Execute each operation in sequence
        for (index, op_config) in config.pipeline.iter().enumerate() {
            log::debug!("Executing operation {} of {}: {}", index + 1, config.pipeline.len(), op_config.operation);

            match self.execute_operation(&mut context, op_config, &config.variables) {
                Ok(()) => {
                    // Optionally save intermediate results
                    if config.output.save_intermediates {
                        if let Some(ref intermediate_dir) = config.output.intermediate_dir {
                            let intermediate_path = Path::new(intermediate_dir)
                                .join(format!("{}_{:02}_{}.png",
                                    context.input_filename,
                                    index + 1,
                                    op_config.name.as_deref().unwrap_or(&op_config.operation)));

                            self.save_current_image(&context, &intermediate_path)?;
                        }
                    }
                }
                Err(e) => {
                    if config.error_handling.log_errors {
                        log::error!("Operation '{}' failed: {}", op_config.operation, e);
                    }

                    match config.error_handling.on_error {
                        ErrorHandlingStrategy::Stop => return Err(e),
                        ErrorHandlingStrategy::Skip => {
                            log::warn!("Skipping failed operation and continuing...");
                            continue;
                        }
                        ErrorHandlingStrategy::Continue => {
                            log::warn!("Ignoring error and continuing...");
                            continue;
                        }
                    }
                }
            }
        }

        // Save final output
        self.save_current_image(&context, output_path)?;

        log::info!("Pipeline execution complete. Output saved to {:?}", output_path);
        Ok(())
    }

    /// Executes a pipeline on a batch of images
    pub fn execute_batch<P: AsRef<Path>>(
        &self,
        config: &PipelineConfig,
        input_dir: P,
        output_dir: P,
    ) -> Result<Vec<Result<()>>> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        // Ensure output directory exists
        std::fs::create_dir_all(output_dir)
            .map_err(|e| Error::IoError(format!("Failed to create output directory: {}", e)))?;

        // Find all image files in input directory
        let image_files: Vec<PathBuf> = std::fs::read_dir(input_dir)
            .map_err(|e| Error::IoError(format!("Failed to read input directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| matches!(s.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "tif"))
                        .unwrap_or(false)
            })
            .collect();

        log::info!(
            "Processing batch of {} images from {:?}",
            image_files.len(),
            input_dir
        );

        // Process each image
        let mut results = Vec::new();
        for input_path in image_files {
            let filename = input_path.file_name().unwrap();
            let output_path = output_dir.join(filename);

            let result = self.execute(config, &input_path, &output_path);
            results.push(result);
        }

        Ok(results)
    }

    /// Executes a single operation
    fn execute_operation(
        &self,
        context: &mut ExecutionContext,
        op_config: &crate::config::OperationConfig,
        variables: &std::collections::HashMap<String, serde_yaml::Value>,
    ) -> Result<()> {
        // Interpolate parameters
        let interpolated_params = interpolation::interpolate_params(&op_config.params, variables)?;

        // Create operation
        let operation = Operation::from_config(&op_config.operation, &interpolated_params)?;

        // Execute operation
        self.apply_operation(context, operation)
    }

    /// Applies an operation to the current image in the context
    fn apply_operation(&self, context: &mut ExecutionContext, operation: Operation) -> Result<()> {
        match operation {
            Operation::RgbToGray => {
                if let Some(rgb_img) = context.current_rgb.take() {
                    let gray_img = color::rgb_to_gray(&rgb_img)?;
                    context.current_gray = Some(gray_img);
                } else if context.current_gray.is_some() {
                    // Already grayscale, no-op
                    log::debug!("Image is already grayscale, skipping conversion");
                } else {
                    return Err(Error::ExecutionError {
                        operation: "rgb_to_gray".to_string(),
                        message: "No image in context".to_string(),
                    });
                }
            }

            Operation::Resize { width, height, interpolation } => {
                let interp = match interpolation {
                    InterpolationMethod::Nearest => geometry::Interpolation::Nearest,
                    InterpolationMethod::Bilinear => geometry::Interpolation::Bilinear,
                    _ => geometry::Interpolation::Bilinear, // Default for unsupported
                };

                if let Some(gray_img) = context.current_gray.take() {
                    let resizer = geometry::Resize::new(width, height, interp);
                    let resized = resizer.resize_gray(&gray_img)?;
                    context.current_gray = Some(resized);
                } else if let Some(rgb_img) = context.current_rgb.take() {
                    // Convert to grayscale first for resize
                    let gray_img = color::rgb_to_gray(&rgb_img)?;
                    let resizer = geometry::Resize::new(width, height, interp);
                    let resized = resizer.resize_gray(&gray_img)?;
                    context.current_gray = Some(resized);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "resize".to_string(),
                        message: "No image in context".to_string(),
                    });
                }
            }

            Operation::Crop { x, y, width, height } => {
                let rect = Rect::new(x, y, width, height);
                let cropper = geometry::Crop::new(rect);

                if let Some(gray_img) = context.current_gray.take() {
                    let cropped = cropper.crop_gray(&gray_img)?;
                    context.current_gray = Some(cropped);
                } else if let Some(rgb_img) = context.current_rgb.take() {
                    // Convert to grayscale first for crop
                    let gray_img = color::rgb_to_gray(&rgb_img)?;
                    let cropped = cropper.crop_gray(&gray_img)?;
                    context.current_gray = Some(cropped);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "crop".to_string(),
                        message: "No image in context".to_string(),
                    });
                }
            }

            Operation::Rotate { angle } => {
                let rotator = geometry::Rotate::new(angle);

                if let Some(gray_img) = context.current_gray.take() {
                    let rotated = rotator.rotate_gray(&gray_img)?;
                    context.current_gray = Some(rotated);
                } else if let Some(rgb_img) = context.current_rgb.take() {
                    // Convert to grayscale first for rotate
                    let gray_img = color::rgb_to_gray(&rgb_img)?;
                    let rotated = rotator.rotate_gray(&gray_img)?;
                    context.current_gray = Some(rotated);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "rotate".to_string(),
                        message: "No image in context".to_string(),
                    });
                }
            }

            Operation::GaussianBlur { kernel_size, sigma } => {
                let blur = filters::GaussianBlur::new(kernel_size, sigma)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let blurred = blur.apply_gray(&gray_img)?;
                    context.current_gray = Some(blurred);
                } else if let Some(rgb_img) = context.current_rgb.take() {
                    // Convert to grayscale first for blur
                    let gray_img = color::rgb_to_gray(&rgb_img)?;
                    let blurred = blur.apply_gray(&gray_img)?;
                    context.current_gray = Some(blurred);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "gaussian_blur".to_string(),
                        message: "No image in context".to_string(),
                    });
                }
            }

            Operation::MedianFilter { kernel_size } => {
                let filter = filters::MedianFilter::new(kernel_size)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let filtered = filter.apply_gray(&gray_img)?;
                    context.current_gray = Some(filtered);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "median_filter".to_string(),
                        message: "Only grayscale images supported for median filter".to_string(),
                    });
                }
            }

            Operation::BilateralFilter { diameter, sigma_color, sigma_space } => {
                let filter = filters::BilateralFilter::new(diameter, sigma_color, sigma_space)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let filtered = filter.apply_gray(&gray_img)?;
                    context.current_gray = Some(filtered);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "bilateral_filter".to_string(),
                        message: "Only grayscale images supported for bilateral filter".to_string(),
                    });
                }
            }

            Operation::Sobel => {
                let sobel = edges::Sobel::new();

                if let Some(gray_img) = context.current_gray.take() {
                    let edges = sobel.apply_gray(&gray_img)?;
                    context.current_gray = Some(edges);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "sobel".to_string(),
                        message: "Only grayscale images supported for Sobel".to_string(),
                    });
                }
            }

            Operation::Canny { low_threshold, high_threshold } => {
                let canny = edges::Canny::new(low_threshold, high_threshold)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let edges = canny.apply_gray(&gray_img)?;
                    context.current_gray = Some(edges);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "canny".to_string(),
                        message: "Only grayscale images supported for Canny".to_string(),
                    });
                }
            }

            Operation::Threshold { threshold, max_value, threshold_type } => {
                let thresh_type = match threshold_type {
                    ThresholdType::Binary => segment::ThresholdType::Binary,
                    ThresholdType::BinaryInv => segment::ThresholdType::BinaryInv,
                    ThresholdType::Truncate => segment::ThresholdType::Truncate,
                    ThresholdType::ToZero => segment::ThresholdType::ToZero,
                    ThresholdType::ToZeroInv => segment::ThresholdType::ToZeroInv,
                };
                let thresholder = segment::Threshold::new(threshold, max_value, thresh_type);

                if let Some(gray_img) = context.current_gray.take() {
                    let thresholded = thresholder.apply_gray(&gray_img)?;
                    context.current_gray = Some(thresholded);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "threshold".to_string(),
                        message: "Only grayscale images supported for threshold".to_string(),
                    });
                }
            }

            Operation::OtsuThreshold => {
                let otsu = segment::OtsuThreshold::new();

                if let Some(gray_img) = context.current_gray.take() {
                    let thresholded = otsu.apply_gray(&gray_img)?;
                    context.current_gray = Some(thresholded);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "otsu_threshold".to_string(),
                        message: "Only grayscale images supported for Otsu threshold".to_string(),
                    });
                }
            }

            Operation::AdaptiveThreshold { max_value, method, block_size, constant } => {
                let adapt_method = match method {
                    AdaptiveMethod::Mean => segment::AdaptiveMethod::Mean,
                    AdaptiveMethod::Gaussian => segment::AdaptiveMethod::Gaussian,
                };
                let adaptive = segment::AdaptiveThreshold::new(max_value, adapt_method, block_size, constant)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let thresholded = adaptive.apply_gray(&gray_img)?;
                    context.current_gray = Some(thresholded);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "adaptive_threshold".to_string(),
                        message: "Only grayscale images supported for adaptive threshold".to_string(),
                    });
                }
            }

            Operation::Erode { kernel_size, element } => {
                let struct_elem = match element {
                    StructuringElementType::Rectangle => morphology::StructuringElement::Rectangle,
                    StructuringElementType::Ellipse => morphology::StructuringElement::Ellipse,
                    StructuringElementType::Cross => morphology::StructuringElement::Cross,
                };
                let erode = morphology::Erode::new(kernel_size, struct_elem)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let eroded = erode.apply_gray(&gray_img)?;
                    context.current_gray = Some(eroded);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "erode".to_string(),
                        message: "Only grayscale images supported for erosion".to_string(),
                    });
                }
            }

            Operation::Dilate { kernel_size, element } => {
                let struct_elem = match element {
                    StructuringElementType::Rectangle => morphology::StructuringElement::Rectangle,
                    StructuringElementType::Ellipse => morphology::StructuringElement::Ellipse,
                    StructuringElementType::Cross => morphology::StructuringElement::Cross,
                };
                let dilate = morphology::Dilate::new(kernel_size, struct_elem)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let dilated = dilate.apply_gray(&gray_img)?;
                    context.current_gray = Some(dilated);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "dilate".to_string(),
                        message: "Only grayscale images supported for dilation".to_string(),
                    });
                }
            }

            Operation::Opening { kernel_size, element } => {
                let struct_elem = match element {
                    StructuringElementType::Rectangle => morphology::StructuringElement::Rectangle,
                    StructuringElementType::Ellipse => morphology::StructuringElement::Ellipse,
                    StructuringElementType::Cross => morphology::StructuringElement::Cross,
                };
                let opening = morphology::MorphOpen::new(kernel_size, struct_elem)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let opened = opening.apply_gray(&gray_img)?;
                    context.current_gray = Some(opened);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "opening".to_string(),
                        message: "Only grayscale images supported for opening".to_string(),
                    });
                }
            }

            Operation::Closing { kernel_size, element } => {
                let struct_elem = match element {
                    StructuringElementType::Rectangle => morphology::StructuringElement::Rectangle,
                    StructuringElementType::Ellipse => morphology::StructuringElement::Ellipse,
                    StructuringElementType::Cross => morphology::StructuringElement::Cross,
                };
                let closing = morphology::MorphClose::new(kernel_size, struct_elem)?;

                if let Some(gray_img) = context.current_gray.take() {
                    let closed = closing.apply_gray(&gray_img)?;
                    context.current_gray = Some(closed);
                } else {
                    return Err(Error::ExecutionError {
                        operation: "closing".to_string(),
                        message: "Only grayscale images supported for closing".to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Saves the current image from context
    fn save_current_image(&self, context: &ExecutionContext, output_path: &Path) -> Result<()> {
        if let Some(ref gray_img) = context.current_gray {
            imgproc_io::write_image_gray(gray_img, output_path)
                .map_err(|e| Error::IoError(format!("Failed to save image: {}", e)))?;
        } else if let Some(ref rgb_img) = context.current_rgb {
            imgproc_io::write_image_rgb(rgb_img, output_path)
                .map_err(|e| Error::IoError(format!("Failed to save image: {}", e)))?;
        } else {
            return Err(Error::ExecutionError {
                operation: "save".to_string(),
                message: "No image in context to save".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for PipelineExecutor {
    fn default() -> Self {
        Self::new()
    }
}
