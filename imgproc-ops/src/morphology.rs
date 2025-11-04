use imgproc_core::{
    error::{Error, Result},
    image::Image,
    pixel::Gray8,
    traits::Operation,
};

/// Structuring element shapes
#[derive(Debug, Clone, Copy)]
pub enum StructuringElement {
    /// Rectangle shape
    Rectangle,
    /// Ellipse shape
    Ellipse,
    /// Cross shape
    Cross,
}

/// Erosion operation
#[derive(Debug, Clone)]
pub struct Erode {
    /// Kernel size
    pub kernel_size: u32,
    /// Structuring element shape
    pub shape: StructuringElement,
}

impl Erode {
    /// Create a new erosion operation
    ///
    /// # Errors
    /// Returns error if kernel size is even
    pub fn new(kernel_size: u32, shape: StructuringElement) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self { kernel_size, shape })
    }

    /// Apply erosion to binary/grayscale image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;
        let radius = (self.kernel_size / 2) as i32;

        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut min_value = 255u8;

                for ky in -radius..=radius {
                    for kx in -radius..=radius {
                        if !self.is_in_structuring_element(kx, ky, radius) {
                            continue;
                        }

                        let px = (x as i32 + kx).clamp(0, input.width() as i32 - 1) as u32;
                        let py = (y as i32 + ky).clamp(0, input.height() as i32 - 1) as u32;

                        let pixel = input.get_pixel(px, py)?;
                        min_value = min_value.min(pixel.value);
                    }
                }

                output.set_pixel(x, y, Gray8::new(min_value))?;
            }
        }

        Ok(output)
    }

    fn is_in_structuring_element(&self, x: i32, y: i32, radius: i32) -> bool {
        match self.shape {
            StructuringElement::Rectangle => true,
            StructuringElement::Cross => x == 0 || y == 0,
            StructuringElement::Ellipse => {
                let fx = x as f32 / radius as f32;
                let fy = y as f32 / radius as f32;
                fx * fx + fy * fy <= 1.0
            }
        }
    }
}

impl Operation for Erode {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Dilation operation
#[derive(Debug, Clone)]
pub struct Dilate {
    /// Kernel size
    pub kernel_size: u32,
    /// Structuring element shape
    pub shape: StructuringElement,
}

impl Dilate {
    /// Create a new dilation operation
    ///
    /// # Errors
    /// Returns error if kernel size is even
    pub fn new(kernel_size: u32, shape: StructuringElement) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self { kernel_size, shape })
    }

    /// Apply dilation to binary/grayscale image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;
        let radius = (self.kernel_size / 2) as i32;

        for y in 0..input.height() {
            for x in 0..input.width() {
                let mut max_value = 0u8;

                for ky in -radius..=radius {
                    for kx in -radius..=radius {
                        if !self.is_in_structuring_element(kx, ky, radius) {
                            continue;
                        }

                        let px = (x as i32 + kx).clamp(0, input.width() as i32 - 1) as u32;
                        let py = (y as i32 + ky).clamp(0, input.height() as i32 - 1) as u32;

                        let pixel = input.get_pixel(px, py)?;
                        max_value = max_value.max(pixel.value);
                    }
                }

                output.set_pixel(x, y, Gray8::new(max_value))?;
            }
        }

        Ok(output)
    }

    fn is_in_structuring_element(&self, x: i32, y: i32, radius: i32) -> bool {
        match self.shape {
            StructuringElement::Rectangle => true,
            StructuringElement::Cross => x == 0 || y == 0,
            StructuringElement::Ellipse => {
                let fx = x as f32 / radius as f32;
                let fy = y as f32 / radius as f32;
                fx * fx + fy * fy <= 1.0
            }
        }
    }
}

impl Operation for Dilate {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Morphological opening (erosion followed by dilation)
#[derive(Debug, Clone)]
pub struct MorphOpen {
    /// Kernel size
    pub kernel_size: u32,
    /// Structuring element shape
    pub shape: StructuringElement,
}

impl MorphOpen {
    /// Create a new opening operation
    ///
    /// # Errors
    /// Returns error if kernel size is even
    pub fn new(kernel_size: u32, shape: StructuringElement) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self { kernel_size, shape })
    }

    /// Apply opening to image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let erode = Erode::new(self.kernel_size, self.shape)?;
        let dilate = Dilate::new(self.kernel_size, self.shape)?;

        let eroded = erode.apply_gray(input)?;
        dilate.apply_gray(&eroded)
    }
}

impl Operation for MorphOpen {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Morphological closing (dilation followed by erosion)
#[derive(Debug, Clone)]
pub struct MorphClose {
    /// Kernel size
    pub kernel_size: u32,
    /// Structuring element shape
    pub shape: StructuringElement,
}

impl MorphClose {
    /// Create a new closing operation
    ///
    /// # Errors
    /// Returns error if kernel size is even
    pub fn new(kernel_size: u32, shape: StructuringElement) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self { kernel_size, shape })
    }

    /// Apply closing to image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let dilate = Dilate::new(self.kernel_size, self.shape)?;
        let erode = Erode::new(self.kernel_size, self.shape)?;

        let dilated = dilate.apply_gray(input)?;
        erode.apply_gray(&dilated)
    }
}

impl Operation for MorphClose {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Type of morphological gradient
#[derive(Debug, Clone, Copy)]
pub enum GradientType {
    /// Basic gradient: dilation - erosion
    Basic,
    /// Internal gradient: original - erosion
    Internal,
    /// External gradient: dilation - original
    External,
}

/// Morphological gradient operation
///
/// Computes edge detection via morphology by computing the difference between
/// dilation and erosion operations.
///
/// # Variants
/// - Basic: `gradient = dilation(I) - erosion(I)` - Full gradient
/// - Internal: `gradient = I - erosion(I)` - Inner boundary
/// - External: `gradient = dilation(I) - I` - Outer boundary
///
/// # Use Cases
/// - Edge detection in binary images
/// - Boundary extraction
/// - Feature enhancement for segmentation
///
/// # Example
/// ```rust,no_run
/// use imgproc_ops::morphology::{MorphGradient, GradientType, StructuringElement};
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// # fn example() -> imgproc_core::error::Result<()> {
/// let image = Image::<Gray8>::new(100, 100)?;
/// let gradient = MorphGradient::new(5, StructuringElement::Ellipse, GradientType::Basic)?;
/// let result = gradient.apply_gray(&image)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MorphGradient {
    /// Kernel size
    pub kernel_size: u32,
    /// Structuring element shape
    pub shape: StructuringElement,
    /// Type of gradient to compute
    pub gradient_type: GradientType,
}

impl MorphGradient {
    /// Create a new morphological gradient operation
    ///
    /// # Errors
    /// Returns error if kernel size is even
    pub fn new(kernel_size: u32, shape: StructuringElement, gradient_type: GradientType) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self { kernel_size, shape, gradient_type })
    }

    /// Apply morphological gradient to image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        match self.gradient_type {
            GradientType::Basic => {
                let dilate = Dilate::new(self.kernel_size, self.shape)?;
                let erode = Erode::new(self.kernel_size, self.shape)?;
                let dilated = dilate.apply_gray(input)?;
                let eroded = erode.apply_gray(input)?;
                subtract_images(&dilated, &eroded)
            }
            GradientType::Internal => {
                let erode = Erode::new(self.kernel_size, self.shape)?;
                let eroded = erode.apply_gray(input)?;
                subtract_images(input, &eroded)
            }
            GradientType::External => {
                let dilate = Dilate::new(self.kernel_size, self.shape)?;
                let dilated = dilate.apply_gray(input)?;
                subtract_images(&dilated, input)
            }
        }
    }
}

impl Operation for MorphGradient {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Type of top-hat transform
#[derive(Debug, Clone, Copy)]
pub enum TopHatVariant {
    /// White top-hat: original - opening (extracts bright features)
    White,
    /// Black top-hat: closing - original (extracts dark features)
    Black,
}

/// Top-hat transform operation
///
/// Extracts features smaller than the structuring element by computing the
/// difference between the original image and its morphological opening/closing.
///
/// # Variants
/// - White top-hat: `I - opening(I)` - Extracts bright features smaller than SE
/// - Black top-hat: `closing(I) - I` - Extracts dark features smaller than SE
///
/// # Use Cases
/// - Removing uneven illumination
/// - Small feature extraction
/// - Background correction
/// - Document image enhancement
///
/// # Example
/// ```rust,no_run
/// use imgproc_ops::morphology::{TopHat, TopHatVariant, StructuringElement};
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// # fn example() -> imgproc_core::error::Result<()> {
/// let image = Image::<Gray8>::new(100, 100)?;
/// let tophat = TopHat::new(15, StructuringElement::Ellipse, TopHatVariant::White)?;
/// let result = tophat.apply_gray(&image)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TopHat {
    /// Kernel size
    pub kernel_size: u32,
    /// Structuring element shape
    pub shape: StructuringElement,
    /// Type of top-hat transform
    pub variant: TopHatVariant,
}

impl TopHat {
    /// Create a new top-hat transform operation
    ///
    /// # Errors
    /// Returns error if kernel size is even
    pub fn new(kernel_size: u32, shape: StructuringElement, variant: TopHatVariant) -> Result<Self> {
        if kernel_size % 2 == 0 {
            return Err(Error::InvalidParameter {
                name: "kernel_size".to_string(),
                message: "Must be odd".to_string(),
            });
        }
        Ok(Self { kernel_size, shape, variant })
    }

    /// Apply top-hat transform to image
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        match self.variant {
            TopHatVariant::White => {
                let opening = MorphOpen::new(self.kernel_size, self.shape)?;
                let opened = opening.apply_gray(input)?;
                subtract_images(input, &opened)
            }
            TopHatVariant::Black => {
                let closing = MorphClose::new(self.kernel_size, self.shape)?;
                let closed = closing.apply_gray(input)?;
                subtract_images(&closed, input)
            }
        }
    }
}

impl Operation for TopHat {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Skeletonization method
#[derive(Debug, Clone, Copy)]
pub enum SkeletonMethod {
    /// Zhang-Suen algorithm (fast iterative thinning)
    ZhangSuen,
}

/// Skeletonization operation
///
/// Reduces binary shapes to 1-pixel-wide skeletal representations while
/// preserving topology and connectivity.
///
/// # Algorithm: Zhang-Suen
/// A two-pass iterative thinning algorithm that:
/// 1. Examines each foreground pixel (value > 127)
/// 2. Marks pixels for deletion based on 8-neighbor conditions
/// 3. Deletes marked pixels in two sub-iterations
/// 4. Repeats until no more changes occur
///
/// # Use Cases
/// - Fingerprint analysis
/// - Character recognition (OCR)
/// - Shape analysis
/// - Network/graph extraction from images
///
/// # Example
/// ```rust,no_run
/// use imgproc_ops::morphology::{Skeletonize, SkeletonMethod};
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// # fn example() -> imgproc_core::error::Result<()> {
/// let binary_image = Image::<Gray8>::new(100, 100)?;
/// let skeleton = Skeletonize::new(SkeletonMethod::ZhangSuen, Some(100))?;
/// let result = skeleton.apply_gray(&binary_image)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Skeletonize {
    /// Skeletonization method
    pub method: SkeletonMethod,
    /// Maximum iterations (None for unlimited)
    pub max_iterations: Option<u32>,
}

impl Skeletonize {
    /// Create a new skeletonization operation
    ///
    /// # Errors
    /// Returns error if max_iterations is 0
    pub fn new(method: SkeletonMethod, max_iterations: Option<u32>) -> Result<Self> {
        if let Some(max) = max_iterations {
            if max == 0 {
                return Err(Error::InvalidParameter {
                    name: "max_iterations".to_string(),
                    message: "Must be greater than 0".to_string(),
                });
            }
        }
        Ok(Self { method, max_iterations })
    }

    /// Apply skeletonization to binary image
    ///
    /// Pixels with value > 127 are considered foreground.
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        match self.method {
            SkeletonMethod::ZhangSuen => self.zhang_suen(input),
        }
    }

    fn zhang_suen(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut image = input.clone();
        let width = image.width() as i32;
        let height = image.height() as i32;
        let max_iter = self.max_iterations.unwrap_or(1000);

        for _ in 0..max_iter {
            let mut changed = false;

            // Sub-iteration 1
            let mut to_delete = Vec::new();
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let p1 = get_pixel_value(&image, x, y);
                    if p1 < 128 {
                        continue;
                    }

                    let (p2, p3, p4, p5, p6, p7, p8, p9) = get_neighbors(&image, x, y);

                    let n = count_neighbors(p2, p3, p4, p5, p6, p7, p8, p9);
                    let s = count_transitions(p2, p3, p4, p5, p6, p7, p8, p9);

                    // Check conditions using logical AND instead of multiplication to avoid overflow
                    let cond1 = p2 < 128 || p4 < 128 || p6 < 128;
                    let cond2 = p4 < 128 || p6 < 128 || p8 < 128;

                    if (2..=6).contains(&n) && s == 1 && cond1 && cond2 {
                        to_delete.push((x as u32, y as u32));
                    }
                }
            }

            for (x, y) in &to_delete {
                image.set_pixel(*x, *y, Gray8::new(0))?;
                changed = true;
            }

            // Sub-iteration 2
            to_delete.clear();
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let p1 = get_pixel_value(&image, x, y);
                    if p1 < 128 {
                        continue;
                    }

                    let (p2, p3, p4, p5, p6, p7, p8, p9) = get_neighbors(&image, x, y);

                    let n = count_neighbors(p2, p3, p4, p5, p6, p7, p8, p9);
                    let s = count_transitions(p2, p3, p4, p5, p6, p7, p8, p9);

                    // Check conditions using logical AND instead of multiplication to avoid overflow
                    let cond1 = p2 < 128 || p4 < 128 || p8 < 128;
                    let cond2 = p2 < 128 || p6 < 128 || p8 < 128;

                    if (2..=6).contains(&n) && s == 1 && cond1 && cond2 {
                        to_delete.push((x as u32, y as u32));
                    }
                }
            }

            for (x, y) in &to_delete {
                image.set_pixel(*x, *y, Gray8::new(0))?;
                changed = true;
            }

            if !changed {
                break;
            }
        }

        Ok(image)
    }
}

impl Operation for Skeletonize {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Thinning operation
///
/// Simplified iterative morphological erosion that preserves connectivity
/// and endpoint pixels. Less sophisticated than skeletonization but faster.
///
/// # Use Cases
/// - Fast shape thinning
/// - Preprocessing for feature extraction
/// - Simple skeleton approximation
///
/// # Example
/// ```rust,no_run
/// use imgproc_ops::morphology::Thinning;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// # fn example() -> imgproc_core::error::Result<()> {
/// let binary_image = Image::<Gray8>::new(100, 100)?;
/// let thinning = Thinning::new(50)?;
/// let result = thinning.apply_gray(&binary_image)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Thinning {
    /// Maximum iterations
    pub max_iterations: u32,
}

impl Thinning {
    /// Create a new thinning operation
    ///
    /// # Errors
    /// Returns error if max_iterations is 0
    pub fn new(max_iterations: u32) -> Result<Self> {
        if max_iterations == 0 {
            return Err(Error::InvalidParameter {
                name: "max_iterations".to_string(),
                message: "Must be greater than 0".to_string(),
            });
        }
        Ok(Self { max_iterations })
    }

    /// Apply thinning to binary image
    ///
    /// Pixels with value > 127 are considered foreground.
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut image = input.clone();
        let width = image.width() as i32;
        let height = image.height() as i32;

        for _ in 0..self.max_iterations {
            let mut changed = false;
            let mut to_delete = Vec::new();

            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let p1 = get_pixel_value(&image, x, y);
                    if p1 < 128 {
                        continue;
                    }

                    let (p2, p3, p4, p5, p6, p7, p8, p9) = get_neighbors(&image, x, y);
                    let n = count_neighbors(p2, p3, p4, p5, p6, p7, p8, p9);

                    // Keep endpoints (1 neighbor) and junction points (3+ neighbors)
                    // Only thin pixels with exactly 2 neighbors
                    if n == 2 {
                        let s = count_transitions(p2, p3, p4, p5, p6, p7, p8, p9);
                        // Only delete if it doesn't break connectivity
                        if s == 1 {
                            to_delete.push((x as u32, y as u32));
                        }
                    }
                }
            }

            for (x, y) in &to_delete {
                image.set_pixel(*x, *y, Gray8::new(0))?;
                changed = true;
            }

            if !changed {
                break;
            }
        }

        Ok(image)
    }
}

impl Operation for Thinning {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

/// Hit-or-Miss transform operation
///
/// Template matching for binary images. Searches for specific pixel patterns
/// by requiring certain pixels to be foreground, others to be background, and
/// allowing some to be "don't care".
///
/// # Template Encoding
/// - `1`: Must match foreground (pixel value > 127)
/// - `-1`: Must match background (pixel value <= 127)
/// - `0`: Don't care (can be anything)
///
/// # Use Cases
/// - Corner detection
/// - Endpoint detection
/// - Pattern matching in binary images
/// - Morphological feature extraction
///
/// # Example
/// ```rust,no_run
/// use imgproc_ops::morphology::HitOrMiss;
/// use imgproc_core::image::Image;
/// use imgproc_core::pixel::Gray8;
///
/// # fn example() -> imgproc_core::error::Result<()> {
/// let binary_image = Image::<Gray8>::new(100, 100)?;
///
/// // Template to detect top-left corners
/// let template = vec![
///     vec![-1, -1, -1],
///     vec![-1,  1,  1],
///     vec![-1,  1,  1],
/// ];
///
/// let hit_or_miss = HitOrMiss::new(template)?;
/// let result = hit_or_miss.apply_gray(&binary_image)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct HitOrMiss {
    /// Template: 1 = foreground, -1 = background, 0 = don't care
    pub template: Vec<Vec<i8>>,
}

impl HitOrMiss {
    /// Create a new hit-or-miss transform operation
    ///
    /// # Errors
    /// Returns error if template is empty or not rectangular
    pub fn new(template: Vec<Vec<i8>>) -> Result<Self> {
        if template.is_empty() {
            return Err(Error::InvalidParameter {
                name: "template".to_string(),
                message: "Template cannot be empty".to_string(),
            });
        }

        let width = template[0].len();
        if width == 0 {
            return Err(Error::InvalidParameter {
                name: "template".to_string(),
                message: "Template rows cannot be empty".to_string(),
            });
        }

        for row in &template {
            if row.len() != width {
                return Err(Error::InvalidParameter {
                    name: "template".to_string(),
                    message: "Template must be rectangular".to_string(),
                });
            }
            for &val in row {
                if val < -1 || val > 1 {
                    return Err(Error::InvalidParameter {
                        name: "template".to_string(),
                        message: "Template values must be -1, 0, or 1".to_string(),
                    });
                }
            }
        }

        Ok(Self { template })
    }

    /// Apply hit-or-miss transform to binary image
    ///
    /// Pixels with value > 127 are considered foreground.
    /// Output is 255 where template matches, 0 otherwise.
    pub fn apply_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;

        let template_height = self.template.len() as i32;
        let template_width = self.template[0].len() as i32;
        let offset_y = template_height / 2;
        let offset_x = template_width / 2;

        for y in 0..input.height() as i32 {
            for x in 0..input.width() as i32 {
                let mut matches = true;

                for ty in 0..template_height {
                    for tx in 0..template_width {
                        let template_val = self.template[ty as usize][tx as usize];
                        if template_val == 0 {
                            continue; // Don't care
                        }

                        let img_x = x + tx - offset_x;
                        let img_y = y + ty - offset_y;

                        // Out of bounds pixels are treated as background
                        let pixel_val = if img_x >= 0 && img_x < input.width() as i32
                            && img_y >= 0 && img_y < input.height() as i32 {
                            get_pixel_value(input, img_x, img_y)
                        } else {
                            0
                        };

                        let is_foreground = pixel_val > 127;

                        if template_val == 1 && !is_foreground {
                            matches = false;
                            break;
                        }
                        if template_val == -1 && is_foreground {
                            matches = false;
                            break;
                        }
                    }
                    if !matches {
                        break;
                    }
                }

                let result_val = if matches { 255 } else { 0 };
                output.set_pixel(x as u32, y as u32, Gray8::new(result_val))?;
            }
        }

        Ok(output)
    }
}

impl Operation for HitOrMiss {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.apply_gray(&input)
    }
}

// Helper functions

/// Subtract two images with saturation (prevents underflow)
fn subtract_images(img1: &Image<Gray8>, img2: &Image<Gray8>) -> Result<Image<Gray8>> {
    if img1.width() != img2.width() || img1.height() != img2.height() {
        return Err(Error::InvalidParameter {
            name: "images".to_string(),
            message: "Images must have the same dimensions".to_string(),
        });
    }

    let mut output = Image::new(img1.width(), img1.height())?;

    for y in 0..img1.height() {
        for x in 0..img1.width() {
            let p1 = img1.get_pixel(x, y)?.value;
            let p2 = img2.get_pixel(x, y)?.value;
            let result = p1.saturating_sub(p2);
            output.set_pixel(x, y, Gray8::new(result))?;
        }
    }

    Ok(output)
}

/// Get pixel value safely (returns 0 for out of bounds)
fn get_pixel_value(image: &Image<Gray8>, x: i32, y: i32) -> u8 {
    if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
        image.get_pixel(x as u32, y as u32).map(|p| p.value).unwrap_or(0)
    } else {
        0
    }
}

/// Get 8-neighbors in Zhang-Suen order
/// Returns (P2, P3, P4, P5, P6, P7, P8, P9) where:
/// ```text
/// P9 P2 P3
/// P8 P1 P4
/// P7 P6 P5
/// ```
fn get_neighbors(image: &Image<Gray8>, x: i32, y: i32) -> (u8, u8, u8, u8, u8, u8, u8, u8) {
    let p2 = get_pixel_value(image, x, y - 1);
    let p3 = get_pixel_value(image, x + 1, y - 1);
    let p4 = get_pixel_value(image, x + 1, y);
    let p5 = get_pixel_value(image, x + 1, y + 1);
    let p6 = get_pixel_value(image, x, y + 1);
    let p7 = get_pixel_value(image, x - 1, y + 1);
    let p8 = get_pixel_value(image, x - 1, y);
    let p9 = get_pixel_value(image, x - 1, y - 1);
    (p2, p3, p4, p5, p6, p7, p8, p9)
}

/// Count non-zero neighbors (N(P1) in Zhang-Suen algorithm)
fn count_neighbors(p2: u8, p3: u8, p4: u8, p5: u8, p6: u8, p7: u8, p8: u8, p9: u8) -> u32 {
    let mut count = 0;
    if p2 >= 128 { count += 1; }
    if p3 >= 128 { count += 1; }
    if p4 >= 128 { count += 1; }
    if p5 >= 128 { count += 1; }
    if p6 >= 128 { count += 1; }
    if p7 >= 128 { count += 1; }
    if p8 >= 128 { count += 1; }
    if p9 >= 128 { count += 1; }
    count
}

/// Count 0→1 transitions in ordered neighbors (S(P1) in Zhang-Suen algorithm)
fn count_transitions(p2: u8, p3: u8, p4: u8, p5: u8, p6: u8, p7: u8, p8: u8, p9: u8) -> u32 {
    let neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]; // Wrap around
    let mut transitions = 0;

    for i in 0..8 {
        let curr_is_bg = neighbors[i] < 128;
        let next_is_fg = neighbors[i + 1] >= 128;
        if curr_is_bg && next_is_fg {
            transitions += 1;
        }
    }

    transitions
}