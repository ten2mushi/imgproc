//! Color space conversion operations.

use imgproc_core::{
    color::{rgb_to_gray_u8, rgba_to_gray_u8, gray_to_rgb, gray_to_rgba, rgb_to_rgba, rgba_to_rgb},
    error::Result,
    image::Image,
    pixel::{Gray8, Rgb8, Rgba8},
    traits::Operation,
    ColorSpace,
};

/// Color conversion operation
#[derive(Debug, Clone)]
pub struct ColorConvert {
    /// Target color space
    pub target: ColorSpace,
}

impl ColorConvert {
    /// Create a new color conversion operation
    #[must_use]
    pub const fn new(target: ColorSpace) -> Self {
        Self { target }
    }

    /// Convert RGB to grayscale
    pub fn rgb_to_gray(input: &Image<Rgb8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;

        for (x, y, pixel) in input.enumerate_pixels() {
            let gray = rgb_to_gray_u8(pixel);
            output.set_pixel(x, y, gray)?;
        }

        Ok(output)
    }

    /// Convert RGBA to grayscale
    pub fn rgba_to_gray(input: &Image<Rgba8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(input.width(), input.height())?;

        for (x, y, pixel) in input.enumerate_pixels() {
            let gray = rgba_to_gray_u8(pixel);
            output.set_pixel(x, y, gray)?;
        }

        Ok(output)
    }

    /// Convert grayscale to RGB
    pub fn gray_to_rgb(input: &Image<Gray8>) -> Result<Image<Rgb8>> {
        let mut output = Image::new(input.width(), input.height())?;

        for (x, y, pixel) in input.enumerate_pixels() {
            let rgb = gray_to_rgb(pixel);
            output.set_pixel(x, y, rgb)?;
        }

        Ok(output)
    }

    /// Convert grayscale to RGBA
    pub fn gray_to_rgba(input: &Image<Gray8>) -> Result<Image<Rgba8>> {
        let mut output = Image::new(input.width(), input.height())?;

        for (x, y, pixel) in input.enumerate_pixels() {
            let rgba = gray_to_rgba(pixel);
            output.set_pixel(x, y, rgba)?;
        }

        Ok(output)
    }

    /// Convert RGB to RGBA
    pub fn rgb_to_rgba(input: &Image<Rgb8>) -> Result<Image<Rgba8>> {
        let mut output = Image::new(input.width(), input.height())?;

        for (x, y, pixel) in input.enumerate_pixels() {
            let rgba = rgb_to_rgba(pixel);
            output.set_pixel(x, y, rgba)?;
        }

        Ok(output)
    }

    /// Convert RGBA to RGB (discarding alpha)
    pub fn rgba_to_rgb(input: &Image<Rgba8>) -> Result<Image<Rgb8>> {
        let mut output = Image::new(input.width(), input.height())?;

        for (x, y, pixel) in input.enumerate_pixels() {
            let rgb = rgba_to_rgb(pixel);
            output.set_pixel(x, y, rgb)?;
        }

        Ok(output)
    }
}

// Implementation of Operation trait for RGB to Gray conversion
impl Operation for ColorConvert {
    type Input = Image<Rgb8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        Self::rgb_to_gray(&input)
    }
}

// Convenience functions at module level
/// Convert RGB image to grayscale
pub fn rgb_to_gray(input: &Image<Rgb8>) -> Result<Image<Gray8>> {
    ColorConvert::rgb_to_gray(input)
}

// ============================================================================
// Phase 3: Extended Color Space Conversions
// ============================================================================

/// HSL color representation (Hue, Saturation, Lightness)
///
/// - H: 0-360 degrees
/// - S: 0.0-1.0
/// - L: 0.0-1.0
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hsl {
    /// Hue component (0-360 degrees)
    pub h: f32,
    /// Saturation component (0.0-1.0)
    pub s: f32,
    /// Lightness component (0.0-1.0)
    pub l: f32,
}

impl Hsl {
    /// Create a new HSL color
    #[must_use]
    pub const fn new(h: f32, s: f32, l: f32) -> Self {
        Self { h, s, l }
    }
}

/// LAB color representation (CIELAB)
///
/// - L: 0-100 (lightness)
/// - a: -128 to +127 (green to red)
/// - b: -128 to +127 (blue to yellow)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lab {
    /// Lightness component (0-100)
    pub l: f32,
    /// Green-Red component (-128 to +127)
    pub a: f32,
    /// Blue-Yellow component (-128 to +127)
    pub b: f32,
}

impl Lab {
    /// Create a new LAB color
    #[must_use]
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }
}

/// XYZ color representation (CIE 1931 XYZ)
///
/// Tristimulus values representing human color perception
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyz {
    /// X component
    pub x: f32,
    /// Y component
    pub y: f32,
    /// Z component
    pub z: f32,
}

impl Xyz {
    /// Create a new XYZ color
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// YCbCr color representation (Luma + Chroma)
///
/// Used in JPEG compression and video processing
/// - Y: 16-235 (luma)
/// - Cb: 16-240 (blue chroma)
/// - Cr: 16-240 (red chroma)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YCbCr {
    /// Luma component (Y)
    pub y: f32,
    /// Blue chroma component (Cb)
    pub cb: f32,
    /// Red chroma component (Cr)
    pub cr: f32,
}

impl YCbCr {
    /// Create a new YCbCr color
    #[must_use]
    pub const fn new(y: f32, cb: f32, cr: f32) -> Self {
        Self { y, cb, cr }
    }
}

// ============================================================================
// RGB to HSL Conversion
// ============================================================================

/// Convert RGB (0-255) to HSL
///
/// Reference: Standard HSL conversion formulas
///
/// # Arguments
/// * `rgb` - RGB pixel with values 0-255
///
/// # Returns
/// * HSL with H: 0-360°, S: 0-1, L: 0-1
pub fn rgb_to_hsl(rgb: &Rgb8) -> Hsl {
    let r = f32::from(rgb.r) / 255.0;
    let g = f32::from(rgb.g) / 255.0;
    let b = f32::from(rgb.b) / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Lightness
    let l = (max + min) / 2.0;

    // Saturation
    let s = if delta == 0.0 {
        0.0
    } else if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    // Hue
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    Hsl::new(h, s, l)
}

/// Helper function for HSL to RGB conversion
fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// Convert HSL to RGB (0-255)
///
/// Reference: Standard HSL conversion formulas
///
/// # Arguments
/// * `hsl` - HSL color with H: 0-360°, S: 0-1, L: 0-1
///
/// # Returns
/// * RGB pixel with values 0-255
pub fn hsl_to_rgb(hsl: &Hsl) -> Rgb8 {
    if hsl.s == 0.0 {
        // Achromatic (gray)
        let value = (hsl.l * 255.0) as u8;
        return Rgb8::new(value, value, value);
    }

    let q = if hsl.l < 0.5 {
        hsl.l * (1.0 + hsl.s)
    } else {
        hsl.l + hsl.s - hsl.l * hsl.s
    };
    let p = 2.0 * hsl.l - q;

    let h = hsl.h / 360.0;

    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

    Rgb8::new(
        (r * 255.0).clamp(0.0, 255.0) as u8,
        (g * 255.0).clamp(0.0, 255.0) as u8,
        (b * 255.0).clamp(0.0, 255.0) as u8,
    )
}

// ============================================================================
// RGB to XYZ Conversion (via sRGB)
// ============================================================================

/// Convert sRGB component to linear RGB
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear RGB component to sRGB
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert RGB (0-255) to XYZ
///
/// Reference: sRGB to XYZ transformation matrix (D65 illuminant, 2° observer)
///
/// # Arguments
/// * `rgb` - RGB pixel with values 0-255
///
/// # Returns
/// * XYZ color space values
pub fn rgb_to_xyz(rgb: &Rgb8) -> Xyz {
    // Normalize to 0-1
    let r = f32::from(rgb.r) / 255.0;
    let g = f32::from(rgb.g) / 255.0;
    let b = f32::from(rgb.b) / 255.0;

    // Convert to linear RGB
    let r_linear = srgb_to_linear(r);
    let g_linear = srgb_to_linear(g);
    let b_linear = srgb_to_linear(b);

    // Apply sRGB to XYZ matrix (D65 illuminant)
    let x = r_linear * 0.4124564 + g_linear * 0.3575761 + b_linear * 0.1804375;
    let y = r_linear * 0.2126729 + g_linear * 0.7151522 + b_linear * 0.0721750;
    let z = r_linear * 0.0193339 + g_linear * 0.1191920 + b_linear * 0.9503041;

    Xyz::new(x * 100.0, y * 100.0, z * 100.0) // Scale to 0-100 range
}

/// Convert XYZ to RGB (0-255)
///
/// Reference: XYZ to sRGB transformation (inverse)
///
/// # Arguments
/// * `xyz` - XYZ color space values (0-100 range)
///
/// # Returns
/// * RGB pixel with values 0-255
pub fn xyz_to_rgb(xyz: &Xyz) -> Rgb8 {
    // Normalize from 0-100 to 0-1
    let x = xyz.x / 100.0;
    let y = xyz.y / 100.0;
    let z = xyz.z / 100.0;

    // Apply XYZ to sRGB matrix
    let r_linear = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    let g_linear = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b_linear = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

    // Convert to sRGB
    let r = linear_to_srgb(r_linear);
    let g = linear_to_srgb(g_linear);
    let b = linear_to_srgb(b_linear);

    Rgb8::new(
        (r * 255.0).clamp(0.0, 255.0) as u8,
        (g * 255.0).clamp(0.0, 255.0) as u8,
        (b * 255.0).clamp(0.0, 255.0) as u8,
    )
}

// ============================================================================
// XYZ to LAB Conversion
// ============================================================================

/// Helper function for LAB conversion
fn lab_f(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA_CUBED: f32 = DELTA * DELTA * DELTA;

    if t > DELTA_CUBED {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Helper function for inverse LAB conversion
fn lab_f_inv(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;

    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// Convert XYZ to LAB
///
/// Reference: CIE 1976 L*a*b* standard (D65 illuminant, 2° observer)
///
/// # Arguments
/// * `xyz` - XYZ color space values
///
/// # Returns
/// * LAB color space values (L: 0-100, a: -128 to +127, b: -128 to +127)
pub fn xyz_to_lab(xyz: &Xyz) -> Lab {
    // D65 reference white
    const XN: f32 = 95.047;
    const YN: f32 = 100.000;
    const ZN: f32 = 108.883;

    let fx = lab_f(xyz.x / XN);
    let fy = lab_f(xyz.y / YN);
    let fz = lab_f(xyz.z / ZN);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    Lab::new(l, a, b)
}

/// Convert LAB to XYZ
///
/// Reference: Inverse CIE 1976 L*a*b* conversion
///
/// # Arguments
/// * `lab` - LAB color space values
///
/// # Returns
/// * XYZ color space values
pub fn lab_to_xyz(lab: &Lab) -> Xyz {
    // D65 reference white
    const XN: f32 = 95.047;
    const YN: f32 = 100.000;
    const ZN: f32 = 108.883;

    let fy = (lab.l + 16.0) / 116.0;
    let fx = lab.a / 500.0 + fy;
    let fz = fy - lab.b / 200.0;

    let x = XN * lab_f_inv(fx);
    let y = YN * lab_f_inv(fy);
    let z = ZN * lab_f_inv(fz);

    Xyz::new(x, y, z)
}

/// Convert RGB to LAB (via XYZ)
///
/// # Arguments
/// * `rgb` - RGB pixel with values 0-255
///
/// # Returns
/// * LAB color space values
pub fn rgb_to_lab(rgb: &Rgb8) -> Lab {
    let xyz = rgb_to_xyz(rgb);
    xyz_to_lab(&xyz)
}

/// Convert LAB to RGB (via XYZ)
///
/// # Arguments
/// * `lab` - LAB color space values
///
/// # Returns
/// * RGB pixel with values 0-255
pub fn lab_to_rgb(lab: &Lab) -> Rgb8 {
    let xyz = lab_to_xyz(lab);
    xyz_to_rgb(&xyz)
}

// ============================================================================
// RGB to YCbCr Conversion
// ============================================================================

/// Convert RGB to YCbCr
///
/// Reference: ITU-R BT.601 standard (JPEG)
///
/// # Arguments
/// * `rgb` - RGB pixel with values 0-255
///
/// # Returns
/// * YCbCr color space values
pub fn rgb_to_ycbcr(rgb: &Rgb8) -> YCbCr {
    let r = f32::from(rgb.r);
    let g = f32::from(rgb.g);
    let b = f32::from(rgb.b);

    // ITU-R BT.601 conversion
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 128.0 + (-0.168736 * r - 0.331264 * g + 0.5 * b);
    let cr = 128.0 + (0.5 * r - 0.418688 * g - 0.081312 * b);

    YCbCr::new(y, cb, cr)
}

/// Convert YCbCr to RGB
///
/// Reference: ITU-R BT.601 standard (inverse)
///
/// # Arguments
/// * `ycbcr` - YCbCr color space values
///
/// # Returns
/// * RGB pixel with values 0-255
pub fn ycbcr_to_rgb(ycbcr: &YCbCr) -> Rgb8 {
    let y = ycbcr.y;
    let cb = ycbcr.cb - 128.0;
    let cr = ycbcr.cr - 128.0;

    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;

    Rgb8::new(
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

// ============================================================================
// Image-level Conversion Functions
// ============================================================================

/// Convert RGB image to HSL
pub fn rgb_image_to_hsl(input: &Image<Rgb8>) -> Result<Vec<Hsl>> {
    let mut result = Vec::with_capacity((input.width() * input.height()) as usize);

    for pixel in input.pixels() {
        result.push(rgb_to_hsl(pixel));
    }

    Ok(result)
}

/// Convert HSL to RGB image
pub fn hsl_to_rgb_image(hsl_data: &[Hsl], width: u32, height: u32) -> Result<Image<Rgb8>> {
    if hsl_data.len() != (width * height) as usize {
        return Err(imgproc_core::error::Error::InvalidParameter {
            name: "hsl_data".to_string(),
            message: "Data length does not match image dimensions".to_string(),
        });
    }

    let mut output = Image::new(width, height)?;

    for (idx, hsl) in hsl_data.iter().enumerate() {
        let x = (idx as u32) % width;
        let y = (idx as u32) / width;
        let rgb = hsl_to_rgb(hsl);
        output.set_pixel(x, y, rgb)?;
    }

    Ok(output)
}

/// Convert RGB image to LAB
pub fn rgb_image_to_lab(input: &Image<Rgb8>) -> Result<Vec<Lab>> {
    let mut result = Vec::with_capacity((input.width() * input.height()) as usize);

    for pixel in input.pixels() {
        result.push(rgb_to_lab(pixel));
    }

    Ok(result)
}

/// Convert LAB to RGB image
pub fn lab_to_rgb_image(lab_data: &[Lab], width: u32, height: u32) -> Result<Image<Rgb8>> {
    if lab_data.len() != (width * height) as usize {
        return Err(imgproc_core::error::Error::InvalidParameter {
            name: "lab_data".to_string(),
            message: "Data length does not match image dimensions".to_string(),
        });
    }

    let mut output = Image::new(width, height)?;

    for (idx, lab) in lab_data.iter().enumerate() {
        let x = (idx as u32) % width;
        let y = (idx as u32) / width;
        let rgb = lab_to_rgb(lab);
        output.set_pixel(x, y, rgb)?;
    }

    Ok(output)
}

/// Convert RGB image to XYZ
pub fn rgb_image_to_xyz(input: &Image<Rgb8>) -> Result<Vec<Xyz>> {
    let mut result = Vec::with_capacity((input.width() * input.height()) as usize);

    for pixel in input.pixels() {
        result.push(rgb_to_xyz(pixel));
    }

    Ok(result)
}

/// Convert XYZ to RGB image
pub fn xyz_to_rgb_image(xyz_data: &[Xyz], width: u32, height: u32) -> Result<Image<Rgb8>> {
    if xyz_data.len() != (width * height) as usize {
        return Err(imgproc_core::error::Error::InvalidParameter {
            name: "xyz_data".to_string(),
            message: "Data length does not match image dimensions".to_string(),
        });
    }

    let mut output = Image::new(width, height)?;

    for (idx, xyz) in xyz_data.iter().enumerate() {
        let x = (idx as u32) % width;
        let y = (idx as u32) / width;
        let rgb = xyz_to_rgb(xyz);
        output.set_pixel(x, y, rgb)?;
    }

    Ok(output)
}

/// Convert RGB image to YCbCr
pub fn rgb_image_to_ycbcr(input: &Image<Rgb8>) -> Result<Vec<YCbCr>> {
    let mut result = Vec::with_capacity((input.width() * input.height()) as usize);

    for pixel in input.pixels() {
        result.push(rgb_to_ycbcr(pixel));
    }

    Ok(result)
}

/// Convert YCbCr to RGB image
pub fn ycbcr_to_rgb_image(ycbcr_data: &[YCbCr], width: u32, height: u32) -> Result<Image<Rgb8>> {
    if ycbcr_data.len() != (width * height) as usize {
        return Err(imgproc_core::error::Error::InvalidParameter {
            name: "ycbcr_data".to_string(),
            message: "Data length does not match image dimensions".to_string(),
        });
    }

    let mut output = Image::new(width, height)?;

    for (idx, ycbcr) in ycbcr_data.iter().enumerate() {
        let x = (idx as u32) % width;
        let y = (idx as u32) / width;
        let rgb = ycbcr_to_rgb(ycbcr);
        output.set_pixel(x, y, rgb)?;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hsl_and_back() {
        let rgb = Rgb8::new(128, 64, 192);
        let hsl = rgb_to_hsl(&rgb);
        let rgb_back = hsl_to_rgb(&hsl);

        // Allow small rounding errors
        assert!((rgb.r as i16 - rgb_back.r as i16).abs() <= 2);
        assert!((rgb.g as i16 - rgb_back.g as i16).abs() <= 2);
        assert!((rgb.b as i16 - rgb_back.b as i16).abs() <= 2);
    }

    #[test]
    fn test_rgb_to_lab_and_back() {
        let rgb = Rgb8::new(128, 64, 192);
        let lab = rgb_to_lab(&rgb);
        let rgb_back = lab_to_rgb(&lab);

        // LAB conversions may have larger errors due to gamut clipping
        assert!((rgb.r as i16 - rgb_back.r as i16).abs() <= 5);
        assert!((rgb.g as i16 - rgb_back.g as i16).abs() <= 5);
        assert!((rgb.b as i16 - rgb_back.b as i16).abs() <= 5);
    }

    #[test]
    fn test_rgb_to_ycbcr_and_back() {
        let rgb = Rgb8::new(128, 64, 192);
        let ycbcr = rgb_to_ycbcr(&rgb);
        let rgb_back = ycbcr_to_rgb(&ycbcr);

        // Allow small rounding errors
        assert!((rgb.r as i16 - rgb_back.r as i16).abs() <= 2);
        assert!((rgb.g as i16 - rgb_back.g as i16).abs() <= 2);
        assert!((rgb.b as i16 - rgb_back.b as i16).abs() <= 2);
    }
}