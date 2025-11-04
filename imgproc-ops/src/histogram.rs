use imgproc_core::{
    error::{Error, Result},
    image::Image,
    pixel::Gray8,
};

// ============================================================================
// Histogram Data Structure
// ============================================================================

/// Histogram representation for 8-bit images
///
/// Contains 256 bins for grayscale values 0-255
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Histogram bins (256 for 8-bit images)
    bins: Vec<u64>,
}

impl Histogram {
    /// Compute histogram from grayscale image
    ///
    /// # Arguments
    /// * `image` - Input grayscale image
    ///
    /// # Returns
    /// * Histogram with 256 bins
    pub fn compute(image: &Image<Gray8>) -> Self {
        let mut bins = vec![0u64; 256];

        for pixel in image.pixels() {
            bins[pixel.value as usize] += 1;
        }

        Self { bins }
    }

    /// Get count for a specific bin
    ///
    /// # Arguments
    /// * `value` - Pixel value (0-255)
    ///
    /// # Returns
    /// * Count of pixels with this value
    #[must_use]
    pub fn get_bin(&self, value: u8) -> u64 {
        self.bins[value as usize]
    }

    /// Get total number of pixels
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        self.bins.iter().sum()
    }

    /// Normalize histogram to [0.0, 1.0] range
    ///
    /// # Returns
    /// * Normalized histogram values
    #[must_use]
    pub fn normalize(&self) -> Vec<f64> {
        let total = self.total_pixels() as f64;
        self.bins.iter().map(|&count| count as f64 / total).collect()
    }

    /// Compute cumulative distribution function (CDF)
    #[must_use]
    pub fn cdf(&self) -> Vec<u64> {
        let mut cdf = Vec::with_capacity(256);
        let mut cumsum = 0u64;

        for &count in &self.bins {
            cumsum += count;
            cdf.push(cumsum);
        }

        cdf
    }

    /// Get histogram bins
    #[must_use]
    pub fn bins(&self) -> &[u64] {
        &self.bins
    }
}

// ============================================================================
// Histogram Equalization
// ============================================================================

/// Global histogram equalization
///
/// Redistributes pixel intensities to achieve uniform histogram distribution,
/// enhancing global contrast.
///
/// # Algorithm
/// 1. Compute histogram
/// 2. Calculate cumulative distribution function (CDF)
/// 3. Normalize CDF to [0, 255]
/// 4. Map each pixel value through the normalized CDF
///
/// # References
/// - Gonzalez & Woods, "Digital Image Processing"
#[derive(Debug, Clone)]
pub struct HistogramEqualization;

impl HistogramEqualization {
    /// Create a new histogram equalization operation
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Apply histogram equalization to grayscale image
    ///
    /// # Arguments
    /// * `input` - Input grayscale image
    ///
    /// # Returns
    /// * Equalized image with enhanced contrast
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        // Compute histogram
        let hist = Histogram::compute(input);
        let cdf = hist.cdf();
        let total_pixels = hist.total_pixels();

        // Find minimum non-zero CDF value
        let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);

        // Build lookup table
        let mut lut = [0u8; 256];
        for (i, &cdf_val) in cdf.iter().enumerate() {
            if cdf_val > 0 {
                let normalized = ((cdf_val - cdf_min) as f64 / (total_pixels - cdf_min) as f64) * 255.0;
                lut[i] = normalized.round().clamp(0.0, 255.0) as u8;
            }
        }

        // Apply lookup table
        let mut output = Image::new(input.width(), input.height())?;
        for (x, y, pixel) in input.enumerate_pixels() {
            let new_value = lut[pixel.value as usize];
            output.set_pixel(x, y, Gray8::new(new_value))?;
        }

        Ok(output)
    }
}

impl Default for HistogramEqualization {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CLAHE (Contrast Limited Adaptive Histogram Equalization)
// ============================================================================

/// Contrast Limited Adaptive Histogram Equalization
///
/// Performs local histogram equalization with contrast limiting to prevent
/// over-amplification of noise. The image is divided into tiles, and histogram
/// equalization is applied to each tile with contrast limiting.
///
/// # Algorithm
/// 1. Divide image into non-overlapping tiles
/// 2. For each tile:
///    a. Compute histogram
///    b. Clip histogram at clip_limit
///    c. Redistribute clipped pixels uniformly
///    d. Compute equalization mapping
/// 3. Interpolate between tile mappings for smooth transitions
///
/// # References
/// - Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"
///   Graphics Gems IV, Academic Press
#[derive(Debug, Clone)]
pub struct Clahe {
    /// Contrast clipping limit (typical: 2.0-4.0)
    pub clip_limit: f64,
    /// Number of tiles in each dimension (e.g., (8, 8))
    pub tile_grid_size: (u32, u32),
}

impl Clahe {
    /// Create a new CLAHE operation
    ///
    /// # Arguments
    /// * `clip_limit` - Contrast limit (typical: 2.0-4.0, higher = more contrast)
    /// * `tile_grid_size` - Grid size for tiles (e.g., (8, 8))
    ///
    /// # Returns
    /// * CLAHE operation or error if parameters invalid
    pub fn new(clip_limit: f64, tile_grid_size: (u32, u32)) -> Result<Self> {
        if clip_limit <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "clip_limit".to_string(),
                message: "Clip limit must be positive".to_string(),
            });
        }

        if tile_grid_size.0 == 0 || tile_grid_size.1 == 0 {
            return Err(Error::InvalidParameter {
                name: "tile_grid_size".to_string(),
                message: "Grid size must be non-zero".to_string(),
            });
        }

        Ok(Self {
            clip_limit,
            tile_grid_size,
        })
    }

    /// Apply CLAHE to grayscale image
    ///
    /// # Arguments
    /// * `input` - Input grayscale image
    ///
    /// # Returns
    /// * Contrast-enhanced image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let (grid_x, grid_y) = self.tile_grid_size;
        let width = input.width();
        let height = input.height();

        let tile_width = (width + grid_x - 1) / grid_x;
        let tile_height = (height + grid_y - 1) / grid_y;

        // Compute equalization mappings for each tile
        let mut tile_mappings = Vec::new();

        for ty in 0..grid_y {
            for tx in 0..grid_x {
                let x_start = tx * tile_width;
                let y_start = ty * tile_height;
                let x_end = (x_start + tile_width).min(width);
                let y_end = (y_start + tile_height).min(height);

                // Compute histogram for this tile
                let mut hist = vec![0u64; 256];
                let mut pixel_count = 0u64;

                for y in y_start..y_end {
                    for x in x_start..x_end {
                        if let Ok(pixel) = input.get_pixel(x, y) {
                            hist[pixel.value as usize] += 1;
                            pixel_count += 1;
                        }
                    }
                }

                // Apply contrast limiting
                let clip_value = (self.clip_limit * pixel_count as f64 / 256.0).round() as u64;
                let clipped_hist = Self::clip_histogram(&hist, clip_value);

                // Compute CDF and mapping
                let mapping = Self::compute_mapping(&clipped_hist, pixel_count);
                tile_mappings.push(mapping);
            }
        }

        // Apply bilinear interpolation
        let mut output = Image::new(width, height)?;

        for y in 0..height {
            for x in 0..width {
                let pixel = input.get_pixel(x, y)?;
                let value = self.interpolate_value(
                    x,
                    y,
                    pixel.value,
                    &tile_mappings,
                    tile_width,
                    tile_height,
                    grid_x,
                    grid_y,
                );
                output.set_pixel(x, y, Gray8::new(value))?;
            }
        }

        Ok(output)
    }

    /// Clip histogram at specified limit and redistribute excess
    fn clip_histogram(hist: &[u64], clip_limit: u64) -> Vec<u64> {
        let mut clipped = hist.to_vec();
        let mut excess = 0u64;

        // Clip bins and accumulate excess
        for bin in &mut clipped {
            if *bin > clip_limit {
                excess += *bin - clip_limit;
                *bin = clip_limit;
            }
        }

        // Redistribute excess uniformly
        let redistribution = excess / 256;
        let remainder = excess % 256;

        for bin in &mut clipped {
            *bin += redistribution;
        }

        // Distribute remainder
        for i in 0..remainder as usize {
            clipped[i] += 1;
        }

        clipped
    }

    /// Compute equalization mapping from clipped histogram
    fn compute_mapping(hist: &[u64], total_pixels: u64) -> [u8; 256] {
        let mut mapping = [0u8; 256];
        let mut cumsum = 0u64;

        for (i, &count) in hist.iter().enumerate() {
            cumsum += count;
            let normalized = if total_pixels > 0 {
                (cumsum as f64 / total_pixels as f64) * 255.0
            } else {
                i as f64
            };
            mapping[i] = normalized.round().clamp(0.0, 255.0) as u8;
        }

        mapping
    }

    /// Interpolate value using bilinear interpolation between tiles
    #[allow(clippy::too_many_arguments)]
    fn interpolate_value(
        &self,
        x: u32,
        y: u32,
        value: u8,
        tile_mappings: &[[u8; 256]],
        tile_width: u32,
        tile_height: u32,
        grid_x: u32,
        grid_y: u32,
    ) -> u8 {
        // Find tile coordinates
        let tile_x_f = x as f64 / tile_width as f64;
        let tile_y_f = y as f64 / tile_height as f64;

        let tile_x0 = tile_x_f.floor() as u32;
        let tile_y0 = tile_y_f.floor() as u32;
        let tile_x1 = (tile_x0 + 1).min(grid_x - 1);
        let tile_y1 = (tile_y0 + 1).min(grid_y - 1);

        // Interpolation weights
        let wx = tile_x_f - tile_x0 as f64;
        let wy = tile_y_f - tile_y0 as f64;

        // Get mapped values from surrounding tiles
        let v00 = tile_mappings[(tile_y0 * grid_x + tile_x0) as usize][value as usize] as f64;
        let v01 = tile_mappings[(tile_y0 * grid_x + tile_x1) as usize][value as usize] as f64;
        let v10 = tile_mappings[(tile_y1 * grid_x + tile_x0) as usize][value as usize] as f64;
        let v11 = tile_mappings[(tile_y1 * grid_x + tile_x1) as usize][value as usize] as f64;

        // Bilinear interpolation
        let v0 = v00 * (1.0 - wx) + v01 * wx;
        let v1 = v10 * (1.0 - wx) + v11 * wx;
        let result = v0 * (1.0 - wy) + v1 * wy;

        result.round().clamp(0.0, 255.0) as u8
    }
}

// ============================================================================
// Histogram Matching
// ============================================================================

/// Histogram matching (specification)
///
/// Adjusts the histogram of an image to match a reference histogram.
/// Useful for color/tone transfer and standardization.
///
/// # Algorithm
/// 1. Compute CDF of input image
/// 2. Compute CDF of reference histogram
/// 3. For each intensity level, find the reference level with closest CDF
/// 4. Map input intensities to matched reference intensities
#[derive(Debug, Clone)]
pub struct HistogramMatching {
    /// Reference histogram to match
    pub reference_histogram: Histogram,
}

impl HistogramMatching {
    /// Create a new histogram matching operation
    ///
    /// # Arguments
    /// * `reference_image` - Reference image whose histogram to match
    ///
    /// # Returns
    /// * Histogram matching operation
    #[must_use]
    pub fn new(reference_image: &Image<Gray8>) -> Self {
        let reference_histogram = Histogram::compute(reference_image);
        Self { reference_histogram }
    }

    /// Create from pre-computed histogram
    #[must_use]
    pub const fn from_histogram(reference_histogram: Histogram) -> Self {
        Self { reference_histogram }
    }

    /// Apply histogram matching to grayscale image
    ///
    /// # Arguments
    /// * `input` - Input grayscale image
    ///
    /// # Returns
    /// * Image with histogram matched to reference
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        // Compute CDFs
        let input_hist = Histogram::compute(input);
        let input_cdf = input_hist.cdf();
        let ref_cdf = self.reference_histogram.cdf();

        let input_total = input_hist.total_pixels() as f64;
        let ref_total = self.reference_histogram.total_pixels() as f64;

        // Build mapping table
        let mut lut = [0u8; 256];

        for i in 0..256 {
            let input_cdf_value = input_cdf[i] as f64 / input_total;

            // Find closest reference CDF value
            let mut best_match = 0usize;
            let mut min_diff = f64::MAX;

            for j in 0..256 {
                let ref_cdf_value = ref_cdf[j] as f64 / ref_total;
                let diff = (input_cdf_value - ref_cdf_value).abs();

                if diff < min_diff {
                    min_diff = diff;
                    best_match = j;
                }
            }

            lut[i] = best_match as u8;
        }

        // Apply mapping
        let mut output = Image::new(input.width(), input.height())?;

        for (x, y, pixel) in input.enumerate_pixels() {
            let new_value = lut[pixel.value as usize];
            output.set_pixel(x, y, Gray8::new(new_value))?;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_compute() {
        let mut img = Image::new(10, 10).unwrap();
        for y in 0..10 {
            for x in 0..10 {
                img.set_pixel(x, y, Gray8::new((x + y) as u8 * 10)).unwrap();
            }
        }

        let hist = Histogram::compute(&img);
        assert_eq!(hist.total_pixels(), 100);
    }

    #[test]
    fn test_histogram_equalization() {
        let mut img = Image::new(10, 10).unwrap();
        // Create image with limited dynamic range
        for y in 0..10 {
            for x in 0..10 {
                img.set_pixel(x, y, Gray8::new(100)).unwrap();
            }
        }

        let eq = HistogramEqualization::new();
        let result = eq.apply_gray(&img).unwrap();
        assert_eq!(result.width(), 10);
        assert_eq!(result.height(), 10);
    }

    #[test]
    fn test_clahe_creation() {
        let clahe = Clahe::new(2.0, (8, 8));
        assert!(clahe.is_ok());

        let clahe = Clahe::new(0.0, (8, 8));
        assert!(clahe.is_err());

        let clahe = Clahe::new(2.0, (0, 8));
        assert!(clahe.is_err());
    }
}
