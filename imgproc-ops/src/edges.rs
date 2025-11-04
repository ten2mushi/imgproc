use imgproc_core::{
    error::{Error, Result},
    image::Image,
    pixel::Gray8,
    traits::Operation,
};
use crate::filters::{GaussianBlur, Laplacian, LaplacianKernel};

/// Sobel edge detection
#[derive(Debug, Clone)]
pub struct Sobel {
    /// Whether to return magnitude only or include direction
    pub magnitude_only: bool,
}

impl Sobel {
    /// Create a new Sobel edge detector
    #[must_use]
    pub const fn new() -> Self {
        Self { magnitude_only: true }
    }

    /// Apply Sobel edge detection
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;

        // Sobel kernels
        let sobel_x = [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ];

        let sobel_y = [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ];

        for y in 1..input.height() - 1 {
            for x in 1..input.width() - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                // Apply Sobel kernels
                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let pixel = input.get_pixel(px, py)?;
                        let value = pixel.value as f32;

                        gx += value * sobel_x[ky as usize][kx as usize];
                        gy += value * sobel_y[ky as usize][kx as usize];
                    }
                }

                // Calculate magnitude
                let magnitude = (gx * gx + gy * gy).sqrt();
                let magnitude = magnitude.min(255.0) as u8;

                output.set_pixel(x, y, Gray8::new(magnitude))?;
            }
        }

        // Handle borders
        for x in 0..input.width() {
            output.set_pixel(x, 0, Gray8::new(0))?;
            output.set_pixel(x, input.height() - 1, Gray8::new(0))?;
        }
        for y in 0..input.height() {
            output.set_pixel(0, y, Gray8::new(0))?;
            output.set_pixel(input.width() - 1, y, Gray8::new(0))?;
        }

        Ok(output)
    }
}

impl Default for Sobel {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for Sobel {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Canny edge detection
#[derive(Debug, Clone)]
pub struct Canny {
    /// Low threshold for hysteresis
    pub low_threshold: f32,
    /// High threshold for hysteresis
    pub high_threshold: f32,
    /// Gaussian blur sigma for preprocessing
    pub blur_sigma: f32,
}

impl Canny {
    /// Create a new Canny edge detector
    pub fn new(low_threshold: f32, high_threshold: f32) -> Result<Self> {
        if low_threshold >= high_threshold {
            return Err(Error::InvalidParameter {
                name: "threshold".to_string(),
                message: "Low threshold must be less than high threshold".to_string(),
            });
        }
        Ok(Self {
            low_threshold,
            high_threshold,
            blur_sigma: 1.0,
        })
    }

    /// Apply Canny edge detection
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        // Step 1: Gaussian blur
        let blur = GaussianBlur::new(5, self.blur_sigma)?;
        let smoothed = blur.apply_gray(input)?;

        // Step 2: Gradient calculation (using Sobel)
        let mut gradient_mag = Image::new(input.width(), input.height())?;
        let mut gradient_dir = vec![0.0; (input.width() * input.height()) as usize];

        for y in 1..input.height() - 1 {
            for x in 1..input.width() - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                // Sobel kernels
                gx += smoothed.get_pixel(x - 1, y - 1)?.value as f32 * -1.0;
                gx += smoothed.get_pixel(x - 1, y)?.value as f32 * -2.0;
                gx += smoothed.get_pixel(x - 1, y + 1)?.value as f32 * -1.0;
                gx += smoothed.get_pixel(x + 1, y - 1)?.value as f32 * 1.0;
                gx += smoothed.get_pixel(x + 1, y)?.value as f32 * 2.0;
                gx += smoothed.get_pixel(x + 1, y + 1)?.value as f32 * 1.0;

                gy += smoothed.get_pixel(x - 1, y - 1)?.value as f32 * -1.0;
                gy += smoothed.get_pixel(x, y - 1)?.value as f32 * -2.0;
                gy += smoothed.get_pixel(x + 1, y - 1)?.value as f32 * -1.0;
                gy += smoothed.get_pixel(x - 1, y + 1)?.value as f32 * 1.0;
                gy += smoothed.get_pixel(x, y + 1)?.value as f32 * 2.0;
                gy += smoothed.get_pixel(x + 1, y + 1)?.value as f32 * 1.0;

                let magnitude = (gx * gx + gy * gy).sqrt();
                let direction = gy.atan2(gx);

                gradient_mag.set_pixel(x, y, Gray8::new(magnitude.min(255.0) as u8))?;
                gradient_dir[(y * input.width() + x) as usize] = direction;
            }
        }

        // Step 3: Non-maximum suppression
        let mut suppressed = Image::new(input.width(), input.height())?;

        for y in 1..input.height() - 1 {
            for x in 1..input.width() - 1 {
                let mag = gradient_mag.get_pixel(x, y)?.value as f32;
                let dir = gradient_dir[(y * input.width() + x) as usize];

                // Quantize direction to 0, 45, 90, or 135 degrees
                let dir_discrete = ((dir * 4.0 / std::f32::consts::PI).round() as i32).rem_euclid(4);

                let (dx1, dy1, dx2, dy2) = match dir_discrete {
                    0 => (1, 0, -1, 0),    // Horizontal
                    1 => (1, -1, -1, 1),   // Diagonal /
                    2 => (0, 1, 0, -1),    // Vertical
                    _ => (1, 1, -1, -1),   // Diagonal \
                };

                let mag1 = gradient_mag.get_pixel(
                    (x as i32 + dx1).clamp(0, input.width() as i32 - 1) as u32,
                    (y as i32 + dy1).clamp(0, input.height() as i32 - 1) as u32
                )?.value as f32;

                let mag2 = gradient_mag.get_pixel(
                    (x as i32 + dx2).clamp(0, input.width() as i32 - 1) as u32,
                    (y as i32 + dy2).clamp(0, input.height() as i32 - 1) as u32
                )?.value as f32;

                if mag >= mag1 && mag >= mag2 {
                    suppressed.set_pixel(x, y, Gray8::new(mag as u8))?;
                }
            }
        }

        // Step 4: Hysteresis thresholding
        let mut output = Image::new(input.width(), input.height())?;
        let mut strong = vec![vec![false; input.width() as usize]; input.height() as usize];

        // Mark strong edges
        for y in 0..input.height() {
            for x in 0..input.width() {
                let mag = suppressed.get_pixel(x, y)?.value as f32;
                if mag >= self.high_threshold {
                    strong[y as usize][x as usize] = true;
                    output.set_pixel(x, y, Gray8::new(255))?;
                }
            }
        }

        // Trace weak edges connected to strong edges
        let mut changed = true;
        while changed {
            changed = false;
            for y in 1..input.height() - 1 {
                for x in 1..input.width() - 1 {
                    if strong[y as usize][x as usize] {
                        continue;
                    }

                    let mag = suppressed.get_pixel(x, y)?.value as f32;
                    if mag >= self.low_threshold {
                        // Check if connected to a strong edge
                        let mut connected = false;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dx == 0 && dy == 0 {
                                    continue;
                                }
                                let nx = (x as i32 + dx) as usize;
                                let ny = (y as i32 + dy) as usize;
                                if strong[ny][nx] {
                                    connected = true;
                                    break;
                                }
                            }
                            if connected {
                                break;
                            }
                        }

                        if connected {
                            strong[y as usize][x as usize] = true;
                            output.set_pixel(x, y, Gray8::new(255))?;
                            changed = true;
                        }
                    }
                }
            }
        }

        Ok(output)
    }
}

impl Operation for Canny {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Prewitt edge detection
///
/// The Prewitt operator is a gradient-based edge detection method similar to Sobel
/// but with simpler kernel weights. It computes the gradient magnitude by convolving
/// the image with two 3x3 kernels (one for horizontal edges, one for vertical).
///
/// # Mathematical Foundation
/// The Prewitt operator uses the following kernels:
/// ```text
/// X-direction:        Y-direction:
/// -1  0  1            -1  -1  -1
/// -1  0  1             0   0   0
/// -1  0  1             1   1   1
/// ```
///
/// # Use Cases
/// - Edge detection (simpler alternative to Sobel)
/// - Gradient computation
/// - Feature extraction
/// - Image analysis
///
/// # Example
/// ```no_run
/// use imgproc_ops::edges::Prewitt;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let prewitt = Prewitt::new();
/// let edges = prewitt.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Prewitt {
    /// Whether to return magnitude only or include direction
    pub magnitude_only: bool,
}

impl Prewitt {
    /// Create a new Prewitt edge detector
    ///
    /// Returns a Prewitt detector that computes edge magnitude only.
    #[must_use]
    pub const fn new() -> Self {
        Self { magnitude_only: true }
    }

    /// Create a Prewitt edge detector with direction option
    ///
    /// # Parameters
    /// - `magnitude_only`: If true, only returns edge magnitude; if false, direction info could be preserved
    #[must_use]
    pub const fn with_direction(magnitude_only: bool) -> Self {
        Self { magnitude_only }
    }

    /// Apply Prewitt edge detection to a grayscale image
    ///
    /// # Errors
    /// Returns an error if image operations fail
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;

        // Prewitt kernels
        let prewitt_x = [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ];

        let prewitt_y = [
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ];

        for y in 1..input.height() - 1 {
            for x in 1..input.width() - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                // Apply Prewitt kernels
                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let pixel = input.get_pixel(px, py)?;
                        let value = pixel.value as f32;

                        gx += value * prewitt_x[ky as usize][kx as usize];
                        gy += value * prewitt_y[ky as usize][kx as usize];
                    }
                }

                // Calculate magnitude
                let magnitude = (gx * gx + gy * gy).sqrt();
                let magnitude = magnitude.min(255.0) as u8;

                output.set_pixel(x, y, Gray8::new(magnitude))?;
            }
        }

        // Handle borders
        for x in 0..input.width() {
            output.set_pixel(x, 0, Gray8::new(0))?;
            output.set_pixel(x, input.height() - 1, Gray8::new(0))?;
        }
        for y in 0..input.height() {
            output.set_pixel(0, y, Gray8::new(0))?;
            output.set_pixel(input.width() - 1, y, Gray8::new(0))?;
        }

        Ok(output)
    }
}

impl Default for Prewitt {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for Prewitt {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Laplacian of Gaussian (LoG) edge detection
///
/// The Laplacian of Gaussian (LoG) operator combines Gaussian smoothing with Laplacian
/// second derivative for robust blob and edge detection. It first applies Gaussian blur
/// to reduce noise, then applies the Laplacian operator to detect regions of rapid
/// intensity change.
///
/// # Mathematical Foundation
/// The LoG operator is defined as: ∇²(G * I) where G is the Gaussian kernel and I is the image.
/// This is implemented as two sequential operations: Gaussian blur followed by Laplacian.
///
/// # Use Cases
/// - Blob detection
/// - Scale-space analysis
/// - Feature detection
/// - Zero-crossing edge detection
/// - Multi-scale edge analysis
///
/// # Example
/// ```no_run
/// use imgproc_ops::edges::LaplacianOfGaussian;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let log = LaplacianOfGaussian::new(2.0, 5)?;
/// let edges = log.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct LaplacianOfGaussian {
    /// Gaussian blur sigma
    pub sigma: f32,
    /// Kernel size for Gaussian blur
    pub kernel_size: u32,
}

impl LaplacianOfGaussian {
    /// Create a new Laplacian of Gaussian operator
    ///
    /// # Parameters
    /// - `sigma`: Standard deviation of Gaussian blur (typically 1.0-3.0)
    /// - `kernel_size`: Size of Gaussian kernel (must be odd, typically 3, 5, or 7)
    ///
    /// # Errors
    /// Returns an error if parameters are invalid
    pub fn new(sigma: f32, kernel_size: u32) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "sigma".to_string(),
                message: "Sigma must be positive".to_string(),
            });
        }
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Kernel size must be odd".to_string(),
            });
        }
        Ok(Self { sigma, kernel_size })
    }

    /// Apply Laplacian of Gaussian to a grayscale image
    ///
    /// # Errors
    /// Returns an error if image operations fail
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        // Step 1: Apply Gaussian blur
        let blur = GaussianBlur::new(self.kernel_size, self.sigma)?;
        let blurred = blur.apply_gray(input)?;

        // Step 2: Apply Laplacian
        let laplacian = Laplacian::new(LaplacianKernel::Type8, true)?;
        laplacian.apply_gray(&blurred)
    }
}

impl Operation for LaplacianOfGaussian {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Difference of Gaussians (DoG) edge detection
///
/// The Difference of Gaussians (DoG) operator is a fast approximation of the Laplacian of
/// Gaussian (LoG). It computes the difference between two Gaussian-blurred versions of the
/// image with different standard deviations. This creates a band-pass filter that enhances
/// edges at multiple scales.
///
/// # Mathematical Foundation
/// DoG = G(σ₁) - G(σ₂) where σ₁ < σ₂ (typically σ₂ = 1.6 × σ₁)
///
/// # Use Cases
/// - Multi-scale edge detection
/// - SIFT preprocessing
/// - Blob detection
/// - Fast approximation of LoG
/// - Scale-space analysis
///
/// # Example
/// ```no_run
/// use imgproc_ops::edges::DifferenceOfGaussians;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let dog = DifferenceOfGaussians::new(1.0, 1.6, 5)?;
/// let edges = dog.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct DifferenceOfGaussians {
    /// Smaller sigma (finer scale)
    pub sigma1: f32,
    /// Larger sigma (coarser scale)
    pub sigma2: f32,
    /// Kernel size for both Gaussian blurs
    pub kernel_size: u32,
}

impl DifferenceOfGaussians {
    /// Create a new Difference of Gaussians operator
    ///
    /// # Parameters
    /// - `sigma1`: Smaller standard deviation (finer scale)
    /// - `sigma2`: Larger standard deviation (coarser scale)
    /// - `kernel_size`: Size of Gaussian kernels (must be odd)
    ///
    /// # Errors
    /// Returns an error if parameters are invalid
    pub fn new(sigma1: f32, sigma2: f32, kernel_size: u32) -> Result<Self> {
        if sigma1 <= 0.0 || sigma2 <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "sigma".to_string(),
                message: "Sigma values must be positive".to_string(),
            });
        }
        if sigma1 >= sigma2 {
            return Err(Error::InvalidParameter {
                name: "sigma".to_string(),
                message: "sigma1 must be less than sigma2".to_string(),
            });
        }
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Kernel size must be odd".to_string(),
            });
        }
        Ok(Self { sigma1, sigma2, kernel_size })
    }

    /// Create a new DoG operator with ratio-based sigmas
    ///
    /// # Parameters
    /// - `sigma`: Base standard deviation
    /// - `ratio`: Ratio between sigma2 and sigma1 (typically 1.6)
    /// - `kernel_size`: Size of Gaussian kernels (must be odd)
    ///
    /// # Errors
    /// Returns an error if parameters are invalid
    pub fn with_ratio(sigma: f32, ratio: f32, kernel_size: u32) -> Result<Self> {
        if ratio <= 1.0 {
            return Err(Error::InvalidParameter {
                name: "ratio".to_string(),
                message: "Ratio must be greater than 1.0".to_string(),
            });
        }
        Self::new(sigma, sigma * ratio, kernel_size)
    }

    /// Apply Difference of Gaussians to a grayscale image
    ///
    /// # Errors
    /// Returns an error if image operations fail
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        // Apply two Gaussian blurs
        let blur1 = GaussianBlur::new(self.kernel_size, self.sigma1)?;
        let blur2 = GaussianBlur::new(self.kernel_size, self.sigma2)?;

        let blurred1 = blur1.apply_gray(input)?;
        let blurred2 = blur2.apply_gray(input)?;

        // Subtract: blur1 - blur2 (take absolute difference)
        let mut output = Image::new(input.width(), input.height())?;
        for y in 0..input.height() {
            for x in 0..input.width() {
                let v1 = blurred1.get_pixel(x, y)?.value as i16;
                let v2 = blurred2.get_pixel(x, y)?.value as i16;
                let diff = (v1 - v2).abs() as u8;
                output.set_pixel(x, y, Gray8::new(diff))?;
            }
        }

        Ok(output)
    }
}

impl Operation for DifferenceOfGaussians {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Method for zero-crossing detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZeroCrossingMethod {
    /// Use Laplacian filter with specified kernel type
    Laplacian {
        /// Type of Laplacian kernel (4-connected or 8-connected)
        kernel_type: LaplacianKernel,
    },
    /// Use Laplacian of Gaussian with specified sigma
    LaplacianOfGaussian {
        /// Standard deviation of Gaussian blur
        sigma: f32,
    },
}

/// Zero-crossing detection for edge detection
///
/// Zero-crossing detection finds edges by locating points where the second derivative
/// (Laplacian or LoG) changes sign. These zero-crossings correspond to edges in the
/// original image. A threshold is used to filter weak edges.
///
/// # Mathematical Foundation
/// Edges are detected where the second derivative crosses zero:
/// - Positive to negative crossing (bright to dark edge)
/// - Negative to positive crossing (dark to bright edge)
///
/// # Use Cases
/// - Edge detection with precise localization
/// - Blob detection
/// - Feature extraction
/// - Marr-Hildreth edge detection
///
/// # Example
/// ```no_run
/// use imgproc_ops::edges::{ZeroCrossing, ZeroCrossingMethod};
/// use imgproc_ops::filters::LaplacianKernel;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// let img = Image::<Gray8>::new(640, 480)?;
/// let method = ZeroCrossingMethod::LaplacianOfGaussian { sigma: 2.0 };
/// let zc = ZeroCrossing::new(method, 10.0)?;
/// let edges = zc.apply_gray(&img)?;
/// # Ok::<(), imgproc_core::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ZeroCrossing {
    /// Method for computing second derivative
    pub method: ZeroCrossingMethod,
    /// Minimum gradient magnitude threshold to consider
    pub threshold: f32,
}

impl ZeroCrossing {
    /// Create a new zero-crossing detector
    ///
    /// # Parameters
    /// - `method`: Method for computing second derivative (Laplacian or LoG)
    /// - `threshold`: Minimum gradient magnitude to consider as edge
    ///
    /// # Errors
    /// Returns an error if parameters are invalid
    pub fn new(method: ZeroCrossingMethod, threshold: f32) -> Result<Self> {
        if threshold < 0.0 {
            return Err(Error::InvalidParameter {
                name: "threshold".to_string(),
                message: "Threshold must be non-negative".to_string(),
            });
        }
        Ok(Self { method, threshold })
    }

    /// Create a zero-crossing detector using Laplacian
    ///
    /// # Parameters
    /// - `threshold`: Minimum gradient magnitude to consider as edge
    ///
    /// # Errors
    /// Returns an error if threshold is invalid
    pub fn with_laplacian(threshold: f32) -> Result<Self> {
        Self::new(
            ZeroCrossingMethod::Laplacian {
                kernel_type: LaplacianKernel::Type8,
            },
            threshold,
        )
    }

    /// Create a zero-crossing detector using Laplacian of Gaussian
    ///
    /// # Parameters
    /// - `sigma`: Standard deviation of Gaussian blur
    /// - `threshold`: Minimum gradient magnitude to consider as edge
    ///
    /// # Errors
    /// Returns an error if parameters are invalid
    pub fn with_log(sigma: f32, threshold: f32) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "sigma".to_string(),
                message: "Sigma must be positive".to_string(),
            });
        }
        Self::new(
            ZeroCrossingMethod::LaplacianOfGaussian { sigma },
            threshold,
        )
    }

    /// Apply zero-crossing detection to a grayscale image
    ///
    /// # Errors
    /// Returns an error if image operations fail
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        // Step 1: Find zero-crossings
        let mut output = Image::new(input.width(), input.height())?;

        // We need to work with signed values to detect zero-crossings
        // Since the Laplacian with normalize=false gives us absolute values,
        // we need to recompute with access to signed values
        let signed_values = self.compute_signed_laplacian(input)?;

        for y in 1..input.height() - 1 {
            for x in 1..input.width() - 1 {
                let center_idx = (y * input.width() + x) as usize;
                let center_value = signed_values[center_idx];

                // Check 4 neighbors for sign change
                let mut is_zero_crossing = false;
                let neighbors = [
                    (x.wrapping_sub(1), y),  // left
                    (x + 1, y),               // right
                    (x, y.wrapping_sub(1)),  // top
                    (x, y + 1),               // bottom
                ];

                for (nx, ny) in neighbors {
                    if nx < input.width() && ny < input.height() {
                        let neighbor_idx = (ny * input.width() + nx) as usize;
                        let neighbor_value = signed_values[neighbor_idx];

                        // Check for zero-crossing (sign change)
                        if (center_value > 0.0 && neighbor_value < 0.0)
                            || (center_value < 0.0 && neighbor_value > 0.0)
                        {
                            // Check gradient magnitude
                            if (center_value - neighbor_value).abs() > self.threshold {
                                is_zero_crossing = true;
                                break;
                            }
                        }
                    }
                }

                if is_zero_crossing {
                    output.set_pixel(x, y, Gray8::new(255))?;
                }
            }
        }

        Ok(output)
    }

    /// Compute signed Laplacian values
    fn compute_signed_laplacian(&self, input: &Image<Gray8>) -> Result<Vec<f32>> {
        let mut values = vec![0.0; (input.width() * input.height()) as usize];

        // First apply Gaussian blur if using LoG
        let working_image = match self.method {
            ZeroCrossingMethod::LaplacianOfGaussian { sigma } => {
                let kernel_size = ((6.0 * sigma).ceil() as u32).max(3) | 1;
                let blur = GaussianBlur::new(kernel_size, sigma)?;
                blur.apply_gray(input)?
            }
            ZeroCrossingMethod::Laplacian { .. } => input.clone(),
        };

        // Get kernel based on method
        let kernel = match self.method {
            ZeroCrossingMethod::Laplacian { kernel_type } => {
                match kernel_type {
                    LaplacianKernel::Type4 => vec![
                        vec![0, 1, 0],
                        vec![1, -4, 1],
                        vec![0, 1, 0],
                    ],
                    LaplacianKernel::Type8 => vec![
                        vec![1, 1, 1],
                        vec![1, -8, 1],
                        vec![1, 1, 1],
                    ],
                }
            }
            ZeroCrossingMethod::LaplacianOfGaussian { .. } => {
                // Default to Type8 for LoG
                vec![
                    vec![1, 1, 1],
                    vec![1, -8, 1],
                    vec![1, 1, 1],
                ]
            }
        };

        // Apply Laplacian kernel
        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0i32;

                for ky in -1..=1 {
                    for kx in -1..=1 {
                        let px = (x as i32 + kx).clamp(0, input.width() as i32 - 1) as u32;
                        let py = (y as i32 + ky).clamp(0, input.height() as i32 - 1) as u32;
                        let pixel = working_image.get_pixel(px, py)?;
                        let kernel_val = kernel[(ky + 1) as usize][(kx + 1) as usize];
                        sum += pixel.value as i32 * kernel_val;
                    }
                }

                let idx = (y * input.width() + x) as usize;
                values[idx] = sum as f32;
            }
        }

        Ok(values)
    }
}

impl Operation for ZeroCrossing {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}