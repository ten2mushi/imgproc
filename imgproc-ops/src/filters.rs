use imgproc_core::{
    error::{Error, Result},
    image::Image,
    pixel::Gray8,
    traits::Operation,
};

/// Gaussian blur filter
#[derive(Debug, Clone)]
pub struct GaussianBlur {
    /// Kernel size (must be odd)
    pub kernel_size: u32,
    /// Standard deviation
    pub sigma: f32,
}

impl GaussianBlur {
    /// Create a new Gaussian blur filter
    ///
    /// # Errors
    /// Returns error if kernel size is even or sigma is invalid
    pub fn new(kernel_size: u32, sigma: f32) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        if sigma <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "sigma".to_string(),
                message: "Must be positive".to_string(),
            });
        }
        Ok(Self { kernel_size, sigma })
    }

    /// Generate 1D Gaussian kernel
    fn generate_kernel(&self) -> Vec<f32> {
        let size = self.kernel_size as i32;
        let center = size / 2;
        let mut kernel = vec![0.0; size as usize];
        let mut sum = 0.0;

        for i in 0..size {
            let x = (i - center) as f32;
            let value = (-x * x / (2.0 * self.sigma * self.sigma)).exp();
            kernel[i as usize] = value;
            sum += value;
        }

        // Normalize
        for value in &mut kernel {
            *value /= sum;
        }

        kernel
    }

    /// Apply Gaussian blur to grayscale image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let kernel = self.generate_kernel();
        let mut temp = Image::new(input.width(), input.height())?;
        let mut output = Image::new(input.width(), input.height())?;

        let radius = (self.kernel_size / 2) as i32;

        // Horizontal pass
        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0.0;

                for k in -radius..=radius {
                    let px = (x as i32 + k).clamp(0, input.width() as i32 - 1) as u32;
                    let pixel = input.get_pixel(px, y)?;
                    sum += pixel.value as f32 * kernel[(k + radius) as usize];
                }

                temp.set_pixel(x, y, Gray8::new(sum.round() as u8))?;
            }
        }

        // Vertical pass
        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0.0;

                for k in -radius..=radius {
                    let py = (y as i32 + k).clamp(0, input.height() as i32 - 1) as u32;
                    let pixel = temp.get_pixel(x, py)?;
                    sum += pixel.value as f32 * kernel[(k + radius) as usize];
                }

                output.set_pixel(x, y, Gray8::new(sum.round() as u8))?;
            }
        }

        Ok(output)
    }
}

impl Operation for GaussianBlur {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Median filter for noise reduction
#[derive(Debug, Clone)]
pub struct MedianFilter {
    /// Kernel size (must be odd)
    pub kernel_size: u32,
}

impl MedianFilter {
    /// Create a new median filter
    ///
    /// # Errors
    /// Returns error if kernel size is even
    pub fn new(kernel_size: u32) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self { kernel_size })
    }

    /// Apply median filter to grayscale image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;
        let radius = (self.kernel_size / 2) as i32;

        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut values = Vec::new();

                for ky in -radius..=radius {
                    for kx in -radius..=radius {
                        let px = (x as i32 + kx).clamp(0, input.width() as i32 - 1) as u32;
                        let py = (y as i32 + ky).clamp(0, input.height() as i32 - 1) as u32;
                        let pixel = input.get_pixel(px, py)?;
                        values.push(pixel.value);
                    }
                }

                values.sort_unstable();
                let median = values[values.len() / 2];
                output.set_pixel(x, y, Gray8::new(median))?;
            }
        }

        Ok(output)
    }
}

impl Operation for MedianFilter {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Bilateral filter for edge-preserving smoothing
#[derive(Debug, Clone)]
pub struct BilateralFilter {
    /// Filter diameter
    pub diameter: u32,
    /// Sigma for spatial kernel
    pub sigma_space: f32,
    /// Sigma for intensity kernel
    pub sigma_color: f32,
}

impl BilateralFilter {
    /// Create a new bilateral filter
    ///
    /// # Errors
    /// Returns error if parameters are invalid
    pub fn new(diameter: u32, sigma_space: f32, sigma_color: f32) -> Result<Self> {
        if diameter == 0 {
            return Err(Error::InvalidParameter {
                name: "diameter".to_string(),
                message: "Must be positive".to_string(),
            });
        }
        if sigma_space <= 0.0 || sigma_color <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "sigma".to_string(),
                message: "Must be positive".to_string(),
            });
        }
        Ok(Self { diameter, sigma_space, sigma_color })
    }

    /// Apply bilateral filter to grayscale image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;
        let radius = (self.diameter / 2) as i32;

        for y in 0..input.height() {
            for x in 0..input.width() {
                let center_pixel = input.get_pixel(x, y)?;
                let center_value = center_pixel.value as f32;

                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for ky in -radius..=radius {
                    for kx in -radius..=radius {
                        let px = (x as i32 + kx).clamp(0, input.width() as i32 - 1) as u32;
                        let py = (y as i32 + ky).clamp(0, input.height() as i32 - 1) as u32;

                        let pixel = input.get_pixel(px, py)?;
                        let pixel_value = pixel.value as f32;

                        // Spatial weight
                        let spatial_dist = ((kx * kx + ky * ky) as f32).sqrt();
                        let spatial_weight = (-spatial_dist * spatial_dist / (2.0 * self.sigma_space * self.sigma_space)).exp();

                        // Color weight
                        let color_dist = (pixel_value - center_value).abs();
                        let color_weight = (-color_dist * color_dist / (2.0 * self.sigma_color * self.sigma_color)).exp();

                        let weight = spatial_weight * color_weight;
                        weighted_sum += pixel_value * weight;
                        weight_sum += weight;
                    }
                }

                let result = (weighted_sum / weight_sum).round() as u8;
                output.set_pixel(x, y, Gray8::new(result))?;
            }
        }

        Ok(output)
    }
}

impl Operation for BilateralFilter {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Box filter for fast uniform averaging
///
/// The box filter is a simple averaging filter that replaces each pixel with the
/// average of pixels in a rectangular neighborhood. It's implemented using separable
/// 1D passes for efficiency, making it faster than Gaussian blur for simple smoothing.
///
/// # Use Cases
/// - Fast image smoothing
/// - Noise reduction
/// - Downsampling preprocessing
/// - Background estimation
///
/// # Example
/// ```no_run
/// use imgproc_ops::filters::BoxFilter;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let filter = BoxFilter::new(5)?;
/// let result = filter.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct BoxFilter {
    /// Kernel size (must be odd)
    pub kernel_size: u32,
}

impl BoxFilter {
    /// Create a new box filter
    ///
    /// # Parameters
    /// - `kernel_size`: Size of the averaging kernel (must be odd, typically 3-11)
    ///
    /// # Errors
    /// Returns error if kernel size is even or zero
    pub fn new(kernel_size: u32) -> Result<Self> {
        if kernel_size == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be positive".to_string(),
            });
        }
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self { kernel_size })
    }

    /// Apply box filter to grayscale image using separable implementation
    ///
    /// # Performance
    /// O(w*h*k) where k is kernel size, thanks to separable implementation.
    /// Much faster than naive O(w*h*k²) approach.
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut temp = Image::new(input.width(), input.height())?;
        let mut output = Image::new(input.width(), input.height())?;

        let radius = (self.kernel_size / 2) as i32;
        let divisor = self.kernel_size as f32;

        // Horizontal pass - accumulate sum across rows
        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0.0;

                for k in -radius..=radius {
                    let px = (x as i32 + k).clamp(0, input.width() as i32 - 1) as u32;
                    let pixel = input.get_pixel(px, y)?;
                    sum += pixel.value as f32;
                }

                let avg = sum / divisor;
                temp.set_pixel(x, y, Gray8::new(avg.round() as u8))?;
            }
        }

        // Vertical pass - accumulate sum down columns
        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0.0;

                for k in -radius..=radius {
                    let py = (y as i32 + k).clamp(0, input.height() as i32 - 1) as u32;
                    let pixel = temp.get_pixel(x, py)?;
                    sum += pixel.value as f32;
                }

                let avg = sum / divisor;
                output.set_pixel(x, y, Gray8::new(avg.round() as u8))?;
            }
        }

        Ok(output)
    }
}

impl Operation for BoxFilter {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Laplacian kernel type for edge detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianKernel {
    /// 4-connected kernel (cross pattern)
    ///  0  1  0
    ///  1 -4  1
    ///  0  1  0
    Type4,
    /// 8-connected kernel (includes diagonals)
    ///  1  1  1
    ///  1 -8  1
    ///  1  1  1
    Type8,
}

/// Laplacian filter for edge detection
///
/// The Laplacian is a second-order derivative operator that detects edges by
/// finding zero-crossings in the second derivative of the image. It's isotropic
/// (rotation-invariant) and responds to edges in all orientations.
///
/// # Mathematical Foundation
/// The Laplacian operator: ∇²I = ∂²I/∂x² + ∂²I/∂y²
///
/// # Use Cases
/// - Edge detection
/// - Image sharpening (when combined with original)
/// - Blob detection
/// - Feature enhancement
///
/// # Note
/// The Laplacian is sensitive to noise. Consider applying Gaussian blur first
/// for Laplacian of Gaussian (LoG) which is more stable.
///
/// # Example
/// ```no_run
/// use imgproc_ops::filters::{Laplacian, LaplacianKernel};
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let filter = Laplacian::new(LaplacianKernel::Type4, true)?;
/// let edges = filter.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Laplacian {
    /// Kernel type (4-connected or 8-connected)
    pub kernel_type: LaplacianKernel,
    /// Whether to normalize output to 0-255 range
    pub normalize: bool,
}

impl Laplacian {
    /// Create a new Laplacian filter
    ///
    /// # Parameters
    /// - `kernel_type`: Type of kernel (Type4 or Type8)
    /// - `normalize`: If true, normalizes output to full 0-255 range for visualization
    ///
    /// # Errors
    /// Currently doesn't return errors but follows API convention
    pub fn new(kernel_type: LaplacianKernel, normalize: bool) -> Result<Self> {
        Ok(Self { kernel_type, normalize })
    }

    /// Get the Laplacian kernel for the specified type
    fn get_kernel(&self) -> (Vec<Vec<i32>>, i32) {
        match self.kernel_type {
            LaplacianKernel::Type4 => {
                (vec![
                    vec![0, 1, 0],
                    vec![1, -4, 1],
                    vec![0, 1, 0],
                ], 4)
            }
            LaplacianKernel::Type8 => {
                (vec![
                    vec![1, 1, 1],
                    vec![1, -8, 1],
                    vec![1, 1, 1],
                ], 8)
            }
        }
    }

    /// Apply Laplacian filter to grayscale image
    ///
    /// # Returns
    /// Image with edges highlighted. If normalize is true, output is scaled
    /// to 0-255 range, otherwise absolute values are used and clamped.
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let (kernel, _) = self.get_kernel();
        let mut output = Image::new(input.width(), input.height())?;

        // Store raw Laplacian values for optional normalization
        let mut raw_values = Vec::with_capacity((input.width() * input.height()) as usize);

        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0i32;

                // Apply 3x3 kernel
                for ky in -1..=1 {
                    for kx in -1..=1 {
                        let px = (x as i32 + kx).clamp(0, input.width() as i32 - 1) as u32;
                        let py = (y as i32 + ky).clamp(0, input.height() as i32 - 1) as u32;
                        let pixel = input.get_pixel(px, py)?;
                        let kernel_val = kernel[(ky + 1) as usize][(kx + 1) as usize];
                        sum += pixel.value as i32 * kernel_val;
                    }
                }

                raw_values.push(sum);
            }
        }

        // Normalize or take absolute values
        if self.normalize {
            // Find min and max for normalization
            let min_val = *raw_values.iter().min().unwrap_or(&0);
            let max_val = *raw_values.iter().max().unwrap_or(&0);
            let range = max_val - min_val;

            if range > 0 {
                for (idx, &val) in raw_values.iter().enumerate() {
                    let y = idx as u32 / input.width();
                    let x = idx as u32 % input.width();
                    let normalized = ((val - min_val) as f32 / range as f32 * 255.0).round() as u8;
                    output.set_pixel(x, y, Gray8::new(normalized))?;
                }
            } else {
                // All values same, output zeros
                for y in 0..input.height() {
                    for x in 0..input.width() {
                        output.set_pixel(x, y, Gray8::new(0))?;
                    }
                }
            }
        } else {
            // Use absolute values, clamped to 0-255
            for (idx, &val) in raw_values.iter().enumerate() {
                let y = idx as u32 / input.width();
                let x = idx as u32 % input.width();
                let abs_val = val.abs().min(255) as u8;
                output.set_pixel(x, y, Gray8::new(abs_val))?;
            }
        }

        Ok(output)
    }
}

impl Operation for Laplacian {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Scharr filter for edge detection with better rotational symmetry
///
/// The Scharr operator is an improved version of the Sobel operator with better
/// rotational invariance. It uses optimized 3x3 kernels that provide more accurate
/// gradient magnitude estimation, especially for edges at different orientations.
///
/// # Mathematical Foundation
/// Scharr kernels use weights [-3, 0, 3; -10, 0, 10; -3, 0, 3] for X-direction
/// and the transpose for Y-direction, providing better angular accuracy than Sobel.
///
/// # Use Cases
/// - Gradient-based edge detection
/// - Optical flow computation
/// - Structure from motion
/// - Stereo vision
/// - Feature detection requiring accurate gradients
///
/// # Advantages over Sobel
/// - Better rotational invariance (more accurate at all angles)
/// - More accurate gradient magnitude
/// - Preferred for applications requiring precise gradient computation
///
/// # Example
/// ```no_run
/// use imgproc_ops::filters::Scharr;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let filter = Scharr::new(true);
/// let (magnitude, direction) = filter.apply_with_direction(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Scharr {
    /// Whether to compute gradient direction in addition to magnitude
    pub compute_direction: bool,
}

impl Scharr {
    /// Create a new Scharr filter
    ///
    /// # Parameters
    /// - `compute_direction`: If true, also compute gradient direction (angles)
    pub fn new(compute_direction: bool) -> Self {
        Self { compute_direction }
    }

    /// Apply Scharr filter to grayscale image, returning magnitude only
    ///
    /// # Returns
    /// Image with gradient magnitude (edge strength)
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let (magnitude, _) = self.apply_with_direction(input)?;
        Ok(magnitude)
    }

    /// Apply Scharr filter and optionally compute gradient direction
    ///
    /// # Returns
    /// Tuple of (magnitude_image, optional_direction_vector)
    /// - Magnitude: Edge strength at each pixel (0-255)
    /// - Direction: Optional vector of gradient angles in radians [-π, π]
    pub fn apply_with_direction(&self, input: &Image<Gray8>)
        -> Result<(Image<Gray8>, Option<Vec<f32>>)> {

        let mut magnitude = Image::new(input.width(), input.height())?;
        let mut direction = if self.compute_direction {
            Some(Vec::with_capacity((input.width() * input.height()) as usize))
        } else {
            None
        };

        // Scharr kernels (better rotational symmetry than Sobel)
        // X-direction:
        // -3   0   3
        // -10  0  10
        // -3   0   3
        let kernel_x = [
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3],
        ];

        // Y-direction:
        // -3  -10  -3
        //  0    0   0
        //  3   10   3
        let kernel_y = [
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3],
        ];

        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut gx = 0i32;
                let mut gy = 0i32;

                // Apply 3x3 kernels
                for ky in -1..=1 {
                    for kx in -1..=1 {
                        let px = (x as i32 + kx).clamp(0, input.width() as i32 - 1) as u32;
                        let py = (y as i32 + ky).clamp(0, input.height() as i32 - 1) as u32;
                        let pixel = input.get_pixel(px, py)?;
                        let val = pixel.value as i32;

                        let kx_idx = (kx + 1) as usize;
                        let ky_idx = (ky + 1) as usize;

                        gx += val * kernel_x[ky_idx][kx_idx];
                        gy += val * kernel_y[ky_idx][kx_idx];
                    }
                }

                // Compute magnitude
                let mag = ((gx as f32).powi(2) + (gy as f32).powi(2)).sqrt();
                let mag_clamped = mag.min(255.0) as u8;
                magnitude.set_pixel(x, y, Gray8::new(mag_clamped))?;

                // Optionally compute direction
                if let Some(ref mut dir) = direction {
                    let angle = (gy as f32).atan2(gx as f32);
                    dir.push(angle);
                }
            }
        }

        Ok((magnitude, direction))
    }
}

impl Operation for Scharr {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Unsharp masking for image sharpening
///
/// Unsharp masking is a classic image sharpening technique that enhances edges
/// by subtracting a blurred version of the image from the original, then adding
/// this difference back to the original at a controlled strength.
///
/// # Mathematical Foundation
/// sharpened = original + amount × (original - blurred)
///
/// # Use Cases
/// - Image sharpening for print and display
/// - Detail enhancement
/// - Defocus correction
/// - Medical image enhancement
/// - Microscopy image processing
///
/// # Parameters Guide
/// - `amount`: Sharpening strength (0.5-2.0 typical)
///   - 0.5-1.0: Subtle sharpening
///   - 1.0-1.5: Moderate sharpening
///   - 1.5-2.0: Strong sharpening
/// - `radius`: Blur kernel radius (1.0-5.0 typical)
///   - Small (1.0-2.0): Fine detail enhancement
///   - Medium (2.0-3.0): General purpose
///   - Large (3.0-5.0): Broad edge enhancement
/// - `threshold`: Minimum difference to sharpen (0-10 typical)
///   - 0: Sharpen everything
///   - 2-5: Avoid amplifying noise
///   - 5-10: Only sharpen strong edges
///
/// # Example
/// ```no_run
/// use imgproc_ops::filters::UnsharpMask;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let filter = UnsharpMask::new(1.5, 2.0, 2)?;
/// let sharpened = filter.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct UnsharpMask {
    /// Sharpening strength (typically 0.5-2.0)
    pub amount: f32,
    /// Gaussian blur radius (typically 1.0-5.0)
    pub radius: f32,
    /// Minimum difference to sharpen (0-10 typical, 0 = sharpen all)
    pub threshold: u8,
}

impl UnsharpMask {
    /// Create a new unsharp mask filter
    ///
    /// # Parameters
    /// - `amount`: Sharpening strength (0.5-2.0 recommended, must be non-negative)
    /// - `radius`: Gaussian blur radius (1.0-5.0 recommended, must be positive)
    /// - `threshold`: Minimum difference for sharpening (0-10 typical)
    ///
    /// # Errors
    /// Returns error if amount is negative or radius is not positive
    pub fn new(amount: f32, radius: f32, threshold: u8) -> Result<Self> {
        if amount < 0.0 {
            return Err(Error::InvalidParameter {
                name: "amount".to_string(),
                message: "Must be non-negative".to_string(),
            });
        }
        if radius <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "radius".to_string(),
                message: "Must be positive".to_string(),
            });
        }
        Ok(Self { amount, radius, threshold })
    }

    /// Apply unsharp masking to grayscale image
    ///
    /// # Algorithm
    /// 1. Create blurred version using Gaussian blur
    /// 2. Compute difference (detail) = original - blurred
    /// 3. If |difference| > threshold: add scaled difference back to original
    /// 4. Clamp result to valid range [0, 255]
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        // Determine kernel size from radius (should be odd)
        // Rule of thumb: kernel_size = 2 * ceil(3 * sigma) + 1
        let sigma = self.radius;
        let kernel_size = (2.0 * (3.0 * sigma).ceil() + 1.0) as u32;
        let kernel_size = if kernel_size % 2 == 0 { kernel_size + 1 } else { kernel_size };

        // Apply Gaussian blur
        let blur = GaussianBlur::new(kernel_size, sigma)?;
        let blurred = blur.apply_gray(input)?;

        // Create output image
        let mut output = Image::new(input.width(), input.height())?;

        // Apply unsharp mask formula
        for y in 0..input.height() {
            for x in 0..input.width() {
                let original = input.get_pixel(x, y)?.value as f32;
                let blurred_val = blurred.get_pixel(x, y)?.value as f32;

                // Compute difference (detail)
                let difference = original - blurred_val;

                // Apply threshold - only sharpen if difference is significant
                let sharpened = if difference.abs() > self.threshold as f32 {
                    original + self.amount * difference
                } else {
                    original
                };

                // Clamp to valid range
                let result = sharpened.clamp(0.0, 255.0).round() as u8;
                output.set_pixel(x, y, Gray8::new(result))?;
            }
        }

        Ok(output)
    }
}

impl Operation for UnsharpMask {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Derivative order for Gaussian derivatives
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeOrder {
    /// First order derivative
    First,
    /// Second order derivative
    Second,
}

/// Direction for Gaussian derivatives
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeDirection {
    /// X-direction (horizontal)
    X,
    /// Y-direction (vertical)
    Y,
    /// XX - second derivative in X
    XX,
    /// XY - mixed partial derivative
    XY,
    /// YY - second derivative in Y
    YY,
}

/// Gaussian derivative filter
///
/// Computes image derivatives by convolving with derivatives of Gaussian kernels.
/// This approach is more stable than computing derivatives directly, as the Gaussian
/// smoothing reduces noise sensitivity.
///
/// # Mathematical Foundation
/// First derivative of Gaussian:
/// G'(x) = -x/σ² × exp(-x²/(2σ²))
///
/// Second derivative of Gaussian:
/// G''(x) = (x²/σ⁴ - 1/σ²) × exp(-x²/(2σ²))
///
/// # Use Cases
/// - Multi-scale edge detection
/// - Feature detection (corners, blobs)
/// - Optical flow computation
/// - Structure tensor computation
/// - Scale-space analysis
///
/// # Advantages
/// - More stable than direct derivatives (noise reduction)
/// - Separable implementation (efficient)
/// - Multi-scale capability
/// - Theoretically sound (scale-space theory)
///
/// # Example
/// ```no_run
/// use imgproc_ops::filters::{GaussianDerivative, DerivativeOrder, DerivativeDirection};
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let filter = GaussianDerivative::new(1.5, DerivativeOrder::First, DerivativeDirection::X)?;
/// let gradient_x = filter.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct GaussianDerivative {
    /// Standard deviation of Gaussian
    pub sigma: f32,
    /// Derivative order (first or second)
    pub order: DerivativeOrder,
    /// Derivative direction
    pub direction: DerivativeDirection,
}

impl GaussianDerivative {
    /// Create a new Gaussian derivative filter
    ///
    /// # Parameters
    /// - `sigma`: Standard deviation (typically 0.5-5.0)
    /// - `order`: Derivative order (First or Second)
    /// - `direction`: Direction (X, Y, XX, XY, YY)
    ///
    /// # Errors
    /// Returns error if sigma is not positive or if direction doesn't match order
    pub fn new(sigma: f32, order: DerivativeOrder, direction: DerivativeDirection) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "sigma".to_string(),
                message: "Must be positive".to_string(),
            });
        }

        // Validate direction matches order
        match (order, direction) {
            (DerivativeOrder::First, DerivativeDirection::X) |
            (DerivativeOrder::First, DerivativeDirection::Y) |
            (DerivativeOrder::Second, DerivativeDirection::XX) |
            (DerivativeOrder::Second, DerivativeDirection::XY) |
            (DerivativeOrder::Second, DerivativeDirection::YY) => {
                Ok(Self { sigma, order, direction })
            }
            _ => Err(Error::InvalidParameter {
                name: "direction".to_string(),
                message: format!("Direction {:?} not valid for order {:?}", direction, order),
            })
        }
    }

    /// Generate 1D Gaussian kernel
    fn generate_gaussian_kernel(&self, size: usize) -> Vec<f32> {
        let center = size as i32 / 2;
        let mut kernel = vec![0.0; size];
        let mut sum = 0.0;

        for i in 0..size {
            let x = (i as i32 - center) as f32;
            let value = (-x * x / (2.0 * self.sigma * self.sigma)).exp();
            kernel[i] = value;
            sum += value;
        }

        // Normalize
        for value in &mut kernel {
            *value /= sum;
        }

        kernel
    }

    /// Generate 1D first derivative of Gaussian kernel
    fn generate_first_derivative_kernel(&self, size: usize) -> Vec<f32> {
        let center = size as i32 / 2;
        let mut kernel = vec![0.0; size];
        let sigma_sq = self.sigma * self.sigma;

        for i in 0..size {
            let x = (i as i32 - center) as f32;
            let gauss = (-x * x / (2.0 * sigma_sq)).exp();
            kernel[i] = -x / sigma_sq * gauss;
        }

        kernel
    }

    /// Generate 1D second derivative of Gaussian kernel
    fn generate_second_derivative_kernel(&self, size: usize) -> Vec<f32> {
        let center = size as i32 / 2;
        let mut kernel = vec![0.0; size];
        let sigma_sq = self.sigma * self.sigma;
        let sigma_quad = sigma_sq * sigma_sq;

        for i in 0..size {
            let x = (i as i32 - center) as f32;
            let gauss = (-x * x / (2.0 * sigma_sq)).exp();
            kernel[i] = (x * x / sigma_quad - 1.0 / sigma_sq) * gauss;
        }

        kernel
    }

    /// Apply Gaussian derivative to grayscale image
    ///
    /// Uses separable implementation for efficiency
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        // Determine kernel size: 6*sigma + 1 (ensure odd)
        let kernel_size = ((6.0 * self.sigma).ceil() as u32).max(3);
        let kernel_size = if kernel_size % 2 == 0 { kernel_size + 1 } else { kernel_size };
        let size = kernel_size as usize;

        let (kernel_x, kernel_y) = match self.direction {
            DerivativeDirection::X => {
                // First derivative in X: G'(x) * G(y)
                (self.generate_first_derivative_kernel(size),
                 self.generate_gaussian_kernel(size))
            }
            DerivativeDirection::Y => {
                // First derivative in Y: G(x) * G'(y)
                (self.generate_gaussian_kernel(size),
                 self.generate_first_derivative_kernel(size))
            }
            DerivativeDirection::XX => {
                // Second derivative in X: G''(x) * G(y)
                (self.generate_second_derivative_kernel(size),
                 self.generate_gaussian_kernel(size))
            }
            DerivativeDirection::XY => {
                // Mixed derivative: G'(x) * G'(y)
                (self.generate_first_derivative_kernel(size),
                 self.generate_first_derivative_kernel(size))
            }
            DerivativeDirection::YY => {
                // Second derivative in Y: G(x) * G''(y)
                (self.generate_gaussian_kernel(size),
                 self.generate_second_derivative_kernel(size))
            }
        };

        // Apply separable convolution
        let mut temp = vec![0.0; (input.width() * input.height()) as usize];
        let radius = (kernel_size / 2) as i32;

        // Horizontal pass
        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0.0;

                for k in -radius..=radius {
                    let px = (x as i32 + k).clamp(0, input.width() as i32 - 1) as u32;
                    let pixel = input.get_pixel(px, y)?;
                    sum += pixel.value as f32 * kernel_x[(k + radius) as usize];
                }

                temp[(y * input.width() + x) as usize] = sum;
            }
        }

        // Vertical pass and create output
        let mut output = Image::new(input.width(), input.height())?;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut results = Vec::with_capacity((input.width() * input.height()) as usize);

        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0.0;

                for k in -radius..=radius {
                    let py = (y as i32 + k).clamp(0, input.height() as i32 - 1) as u32;
                    let idx = (py * input.width() + x) as usize;
                    sum += temp[idx] * kernel_y[(k + radius) as usize];
                }

                results.push(sum);
                min_val = min_val.min(sum);
                max_val = max_val.max(sum);
            }
        }

        // Normalize to 0-255 range
        let range = max_val - min_val;
        if range > 0.0 {
            for (idx, &val) in results.iter().enumerate() {
                let y = idx as u32 / input.width();
                let x = idx as u32 % input.width();
                let normalized = ((val - min_val) / range * 255.0).round() as u8;
                output.set_pixel(x, y, Gray8::new(normalized))?;
            }
        } else {
            // All values same, output zeros
            for y in 0..input.height() {
                for x in 0..input.width() {
                    output.set_pixel(x, y, Gray8::new(128))?;
                }
            }
        }

        Ok(output)
    }
}

impl Operation for GaussianDerivative {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Diffusion function option for anisotropic diffusion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionOption {
    /// Option 1: g(∇I) = exp(-(∇I/K)²)
    /// Preserves wide regions over edges
    Option1,
    /// Option 2: g(∇I) = 1 / (1 + (∇I/K)²)
    /// Better edge preservation
    Option2,
}

/// Anisotropic diffusion filter (Perona-Malik)
///
/// Anisotropic diffusion is an edge-preserving noise reduction technique that
/// smooths homogeneous regions while preserving and even enhancing edges. It
/// works by solving a partial differential equation (PDE) where the diffusion
/// coefficient depends on the local gradient magnitude.
///
/// # Mathematical Foundation
/// ∂I/∂t = div(g(||∇I||) × ∇I)
///
/// where g is a diffusion function that decreases at edges:
/// - Option 1: g(x) = exp(-(x/κ)²) - favors wide regions
/// - Option 2: g(x) = 1/(1 + (x/κ)²) - favors high contrast edges
///
/// # Use Cases
/// - Medical image denoising (MRI, CT, ultrasound)
/// - Edge-preserving smoothing
/// - Speckle reduction
/// - Image restoration
/// - Preprocessing for segmentation
///
/// # Parameters Guide
/// - `iterations`: Number of diffusion steps (10-50 typical)
///   - More iterations = more smoothing
///   - 10-20: Light smoothing
///   - 20-40: Moderate smoothing
///   - 40-100: Strong smoothing
/// - `kappa`: Conduction coefficient (1.0-100.0 typical)
///   - Controls edge sensitivity
///   - Lower values: preserve more edges
///   - Higher values: more aggressive smoothing
///   - Rule of thumb: estimate gradient magnitude at edges
/// - `lambda`: Integration constant (0.0-0.25 typical)
///   - Controls stability and convergence speed
///   - Must be ≤ 0.25 for numerical stability
///   - 0.2: safe and commonly used
///
/// # Example
/// ```no_run
/// use imgproc_ops::filters::{AnisotropicDiffusion, DiffusionOption};
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let filter = AnisotropicDiffusion::new(20, 10.0, 0.2, DiffusionOption::Option2)?;
/// let smoothed = filter.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct AnisotropicDiffusion {
    /// Number of iterations
    pub iterations: u32,
    /// Conduction coefficient (edge threshold)
    pub kappa: f32,
    /// Integration constant (time step)
    pub lambda: f32,
    /// Diffusion function option
    pub option: DiffusionOption,
}

impl AnisotropicDiffusion {
    /// Create a new anisotropic diffusion filter
    ///
    /// # Parameters
    /// - `iterations`: Number of diffusion iterations (typically 10-50)
    /// - `kappa`: Conduction coefficient (typically 5.0-50.0)
    /// - `lambda`: Integration constant (must be ≤ 0.25, typically 0.2)
    /// - `option`: Diffusion function (Option1 or Option2)
    ///
    /// # Errors
    /// Returns error if kappa or iterations are zero, or if lambda > 0.25
    pub fn new(iterations: u32, kappa: f32, lambda: f32, option: DiffusionOption) -> Result<Self> {
        if iterations == 0 {
            return Err(Error::InvalidParameter {
                name: "iterations".to_string(),
                message: "Must be positive".to_string(),
            });
        }
        if kappa <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "kappa".to_string(),
                message: "Must be positive".to_string(),
            });
        }
        if lambda <= 0.0 || lambda > 0.25 {
            return Err(Error::InvalidParameter {
                name: "lambda".to_string(),
                message: "Must be in range (0.0, 0.25] for numerical stability".to_string(),
            });
        }
        Ok(Self { iterations, kappa, lambda, option })
    }

    /// Compute diffusion coefficient based on gradient magnitude
    fn diffusion_coefficient(&self, gradient: f32) -> f32 {
        match self.option {
            DiffusionOption::Option1 => {
                // g(x) = exp(-(x/K)²)
                let ratio = gradient / self.kappa;
                (-ratio * ratio).exp()
            }
            DiffusionOption::Option2 => {
                // g(x) = 1 / (1 + (x/K)²)
                let ratio = gradient / self.kappa;
                1.0 / (1.0 + ratio * ratio)
            }
        }
    }

    /// Apply anisotropic diffusion to grayscale image
    ///
    /// # Algorithm
    /// Iteratively updates each pixel based on gradients to neighbors:
    /// I(t+1) = I(t) + λ × Σ[g(∇I) × ∇I] over 4-connected neighbors
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let width = input.width();
        let height = input.height();

        // Convert to f32 for processing
        let mut current = vec![0.0f32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let pixel = input.get_pixel(x, y)?;
                current[(y * width + x) as usize] = pixel.value as f32;
            }
        }

        // Iterate diffusion process
        for _ in 0..self.iterations {
            let mut next = current.clone();

            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) as usize;
                    let center = current[idx];

                    // Compute gradients and diffusion to 4-connected neighbors
                    let mut diffusion_sum = 0.0;

                    // North
                    if y > 0 {
                        let north_idx = ((y - 1) * width + x) as usize;
                        let gradient = (current[north_idx] - center).abs();
                        let coeff = self.diffusion_coefficient(gradient);
                        diffusion_sum += coeff * (current[north_idx] - center);
                    }

                    // South
                    if y < height - 1 {
                        let south_idx = ((y + 1) * width + x) as usize;
                        let gradient = (current[south_idx] - center).abs();
                        let coeff = self.diffusion_coefficient(gradient);
                        diffusion_sum += coeff * (current[south_idx] - center);
                    }

                    // West
                    if x > 0 {
                        let west_idx = (y * width + x - 1) as usize;
                        let gradient = (current[west_idx] - center).abs();
                        let coeff = self.diffusion_coefficient(gradient);
                        diffusion_sum += coeff * (current[west_idx] - center);
                    }

                    // East
                    if x < width - 1 {
                        let east_idx = (y * width + x + 1) as usize;
                        let gradient = (current[east_idx] - center).abs();
                        let coeff = self.diffusion_coefficient(gradient);
                        diffusion_sum += coeff * (current[east_idx] - center);
                    }

                    // Update with diffusion
                    next[idx] = center + self.lambda * diffusion_sum;
                }
            }

            current = next;
        }

        // Convert back to u8 and create output
        let mut output = Image::new(width, height)?;
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let val = current[idx].clamp(0.0, 255.0).round() as u8;
                output.set_pixel(x, y, Gray8::new(val))?;
            }
        }

        Ok(output)
    }
}

impl Operation for AnisotropicDiffusion {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_filter_creation() {
        assert!(BoxFilter::new(5).is_ok());
        assert!(BoxFilter::new(4).is_err()); // Even size
        assert!(BoxFilter::new(0).is_err()); // Zero size
    }

    #[test]
    fn test_laplacian_creation() {
        assert!(Laplacian::new(LaplacianKernel::Type4, true).is_ok());
        assert!(Laplacian::new(LaplacianKernel::Type8, false).is_ok());
    }

    #[test]
    fn test_scharr_creation() {
        let filter = Scharr::new(false);
        assert!(!filter.compute_direction);
        let filter = Scharr::new(true);
        assert!(filter.compute_direction);
    }

    #[test]
    fn test_unsharp_mask_validation() {
        assert!(UnsharpMask::new(1.5, 2.0, 2).is_ok());
        assert!(UnsharpMask::new(-0.5, 2.0, 2).is_err()); // Negative amount
        assert!(UnsharpMask::new(1.5, 0.0, 2).is_err()); // Zero radius
    }

    #[test]
    fn test_gaussian_derivative_validation() {
        use DerivativeOrder::*;
        use DerivativeDirection::*;

        assert!(GaussianDerivative::new(1.5, First, X).is_ok());
        assert!(GaussianDerivative::new(1.5, First, Y).is_ok());
        assert!(GaussianDerivative::new(1.5, Second, XX).is_ok());
        assert!(GaussianDerivative::new(1.5, Second, XY).is_ok());
        assert!(GaussianDerivative::new(1.5, Second, YY).is_ok());

        // Invalid combinations
        assert!(GaussianDerivative::new(1.5, First, XX).is_err());
        assert!(GaussianDerivative::new(1.5, Second, X).is_err());

        // Invalid sigma
        assert!(GaussianDerivative::new(0.0, First, X).is_err());
    }

    #[test]
    fn test_anisotropic_diffusion_validation() {
        assert!(AnisotropicDiffusion::new(20, 10.0, 0.2, DiffusionOption::Option1).is_ok());
        assert!(AnisotropicDiffusion::new(20, 10.0, 0.2, DiffusionOption::Option2).is_ok());

        assert!(AnisotropicDiffusion::new(0, 10.0, 0.2, DiffusionOption::Option1).is_err()); // Zero iterations
        assert!(AnisotropicDiffusion::new(20, 0.0, 0.2, DiffusionOption::Option1).is_err()); // Zero kappa
        assert!(AnisotropicDiffusion::new(20, 10.0, 0.3, DiffusionOption::Option1).is_err()); // Lambda too large
        assert!(AnisotropicDiffusion::new(20, 10.0, 0.0, DiffusionOption::Option1).is_err()); // Lambda zero
    }
}