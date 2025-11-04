use imgproc_core::{
    error::{Error, Result},
    image::Image,
    pixel::Gray8,
    traits::Operation,
};

/// Simple threshold operation
#[derive(Debug, Clone)]
pub struct Threshold {
    /// Threshold value
    pub threshold: u8,
    /// Maximum value to assign
    pub max_value: u8,
    /// Threshold type
    pub threshold_type: ThresholdType,
}

/// Types of thresholding
#[derive(Debug, Clone, Copy)]
pub enum ThresholdType {
    /// Binary threshold: value > threshold ? max_value : 0
    Binary,
    /// Binary inverted: value > threshold ? 0 : max_value
    BinaryInv,
    /// Truncate: value > threshold ? threshold : value
    Truncate,
    /// To zero: value > threshold ? value : 0
    ToZero,
    /// To zero inverted: value > threshold ? 0 : value
    ToZeroInv,
}

impl Threshold {
    /// Create a new threshold operation
    #[must_use]
    pub const fn new(threshold: u8, max_value: u8, threshold_type: ThresholdType) -> Self {
        Self {
            threshold,
            max_value,
            threshold_type,
        }
    }

    /// Apply threshold to grayscale image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;

        for (x, y, pixel) in input.enumerate_pixels() {
            let value = pixel.value;
            let new_value = match self.threshold_type {
                ThresholdType::Binary => {
                    if value > self.threshold {
                        self.max_value
                    } else {
                        0
                    }
                }
                ThresholdType::BinaryInv => {
                    if value > self.threshold {
                        0
                    } else {
                        self.max_value
                    }
                }
                ThresholdType::Truncate => {
                    if value > self.threshold {
                        self.threshold
                    } else {
                        value
                    }
                }
                ThresholdType::ToZero => {
                    if value > self.threshold {
                        value
                    } else {
                        0
                    }
                }
                ThresholdType::ToZeroInv => {
                    if value > self.threshold {
                        0
                    } else {
                        value
                    }
                }
            };

            output.set_pixel(x, y, Gray8::new(new_value))?;
        }

        Ok(output)
    }
}

impl Operation for Threshold {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Otsu's automatic thresholding
#[derive(Debug, Clone)]
pub struct OtsuThreshold {
    /// Maximum value to assign
    pub max_value: u8,
}

impl OtsuThreshold {
    /// Create a new Otsu threshold operation
    #[must_use]
    pub const fn new() -> Self {
        Self { max_value: 255 }
    }

    /// Calculate Otsu's threshold value
    pub fn calculate_threshold(&self, input: &Image<Gray8>) -> u8 {
        // Calculate histogram
        let mut histogram = vec![0u64; 256];
        for pixel in input.pixels() {
            histogram[pixel.value as usize] += 1;
        }

        let total_pixels = input.len() as f64;
        let mut sum = 0.0;
        for (i, &count) in histogram.iter().enumerate() {
            sum += i as f64 * count as f64;
        }

        let mut sum_background = 0.0;
        let mut weight_background = 0.0;
        let mut maximum_variance = 0.0;
        let mut optimal_threshold = 0u8;

        for threshold in 0..256 {
            weight_background += histogram[threshold] as f64;
            if weight_background == 0.0 {
                continue;
            }

            let weight_foreground = total_pixels - weight_background;
            if weight_foreground == 0.0 {
                break;
            }

            sum_background += threshold as f64 * histogram[threshold] as f64;

            let mean_background = sum_background / weight_background;
            let mean_foreground = (sum - sum_background) / weight_foreground;

            // Calculate between-class variance
            let variance = weight_background * weight_foreground
                * (mean_background - mean_foreground)
                * (mean_background - mean_foreground);

            if variance > maximum_variance {
                maximum_variance = variance;
                optimal_threshold = threshold as u8;
            }
        }

        optimal_threshold
    }

    /// Apply Otsu thresholding to grayscale image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let threshold = self.calculate_threshold(input);
        let simple_threshold = Threshold::new(threshold, self.max_value, ThresholdType::Binary);
        simple_threshold.apply_gray(input)
    }
}

impl Default for OtsuThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl Operation for OtsuThreshold {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Adaptive threshold operation
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    /// Maximum value to assign
    pub max_value: u8,
    /// Method for calculating threshold
    pub method: AdaptiveMethod,
    /// Size of the neighborhood area
    pub block_size: u32,
    /// Constant subtracted from mean
    pub constant: f32,
}

/// Adaptive threshold calculation methods
#[derive(Debug, Clone, Copy)]
pub enum AdaptiveMethod {
    /// Use mean of neighborhood
    Mean,
    /// Use Gaussian-weighted mean of neighborhood
    Gaussian,
}

impl AdaptiveThreshold {
    /// Create a new adaptive threshold operation
    ///
    /// # Errors
    /// Returns error if block size is even
    pub fn new(max_value: u8, method: AdaptiveMethod, block_size: u32, constant: f32) -> Result<Self> {
        if block_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "block_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self {
            max_value,
            method,
            block_size,
            constant,
        })
    }

    /// Apply adaptive threshold to grayscale image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;
        let radius = (self.block_size / 2) as i32;

        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut sum = 0.0;
                let mut count = 0.0;

                // Calculate local threshold
                for ky in -radius..=radius {
                    for kx in -radius..=radius {
                        let px = (x as i32 + kx).clamp(0, input.width() as i32 - 1) as u32;
                        let py = (y as i32 + ky).clamp(0, input.height() as i32 - 1) as u32;

                        let pixel = input.get_pixel(px, py)?;
                        let weight = match self.method {
                            AdaptiveMethod::Mean => 1.0,
                            AdaptiveMethod::Gaussian => {
                                let dist_sq = (kx * kx + ky * ky) as f32;
                                (-dist_sq / (2.0 * (radius as f32 / 3.0).powi(2))).exp()
                            }
                        };

                        sum += pixel.value as f32 * weight;
                        count += weight;
                    }
                }

                let local_threshold = (sum / count - self.constant) as u8;
                let pixel_value = input.get_pixel(x, y)?.value;

                let new_value = if pixel_value > local_threshold {
                    self.max_value
                } else {
                    0
                };

                output.set_pixel(x, y, Gray8::new(new_value))?;
            }
        }

        Ok(output)
    }
}

impl Operation for AdaptiveThreshold {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}