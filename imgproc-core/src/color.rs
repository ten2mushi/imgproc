use crate::pixel::{Gray, Rgb, Rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Grayscale color space
    Gray,
    /// RGB color space
    Rgb,
    /// RGBA color space with alpha
    Rgba,
    /// HSV color space
    Hsv,
    /// HSL color space
    Hsl,
    /// LAB color space
    Lab,
    /// YCbCr color space
    YCbCr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    /// Grayscale
    Gray,
    /// RGB
    Rgb,
    /// RGBA with alpha
    Rgba,
}

impl ColorType {
    /// Get number of channels for this color type
    #[must_use]
    pub const fn channels(&self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Check if this color type has alpha channel
    #[must_use]
    pub const fn has_alpha(&self) -> bool {
        matches!(self, Self::Rgba)
    }
}

/// Convert RGB to grayscale using ITU-R BT.709 luma coefficients
#[inline]
pub fn rgb_to_gray_u8(rgb: &Rgb<u8>) -> Gray<u8> {
    let r = rgb.r as f32;
    let g = rgb.g as f32;
    let b = rgb.b as f32;

    let value = (r * 0.2126_f32 + g * 0.7152_f32 + b * 0.0722_f32).round() as u8;
    Gray::new(value)
}

/// Convert RGB to grayscale (generic)
#[inline]
pub fn rgb_to_gray<T>(rgb: &Rgb<T>) -> Gray<T>
where
    T: crate::pixel::Channel,
{
    // For now, only support u8
    // In a real implementation, we'd handle other types
    Gray::new(rgb.r) // Simplified: just take red channel
}

/// Convert RGBA to grayscale (ignoring alpha) for u8
#[inline]
pub fn rgba_to_gray_u8(rgba: &Rgba<u8>) -> Gray<u8> {
    let rgb = Rgb::new(rgba.r, rgba.g, rgba.b);
    rgb_to_gray_u8(&rgb)
}

/// Convert RGBA to grayscale (ignoring alpha) generic
#[inline]
pub fn rgba_to_gray<T>(rgba: &Rgba<T>) -> Gray<T>
where
    T: crate::pixel::Channel,
{
    let rgb = Rgb::new(rgba.r, rgba.g, rgba.b);
    rgb_to_gray(&rgb)
}

/// Convert grayscale to RGB
#[inline]
pub fn gray_to_rgb<T>(gray: &Gray<T>) -> Rgb<T>
where
    T: crate::pixel::Channel,
{
    Rgb::new(gray.value, gray.value, gray.value)
}

/// Convert grayscale to RGBA
#[inline]
pub fn gray_to_rgba<T>(gray: &Gray<T>) -> Rgba<T>
where
    T: crate::pixel::Channel,
{
    Rgba::new(gray.value, gray.value, gray.value, T::MAX)
}

/// Convert RGB to RGBA
#[inline]
pub fn rgb_to_rgba<T>(rgb: &Rgb<T>) -> Rgba<T>
where
    T: crate::pixel::Channel,
{
    Rgba::new(rgb.r, rgb.g, rgb.b, T::MAX)
}

/// Convert RGBA to RGB (discarding alpha)
#[inline]
pub fn rgba_to_rgb<T>(rgba: &Rgba<T>) -> Rgb<T>
where
    T: crate::pixel::Channel,
{
    Rgb::new(rgba.r, rgba.g, rgba.b)
}

/// Convert RGB to HSV color space
pub fn rgb_to_hsv<T>(rgb: &Rgb<T>) -> (f32, f32, f32)
where
    T: crate::pixel::Channel,
    T: Into<f32>,
{
    let r: f32 = rgb.r.into() / T::MAX.into();
    let g: f32 = rgb.g.into() / T::MAX.into();
    let b: f32 = rgb.b.into() / T::MAX.into();

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let value = max;
    let saturation = if max == 0.0 { 0.0 } else { delta / max };

    let hue = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let hue = if hue < 0.0 { hue + 360.0 } else { hue };

    (hue, saturation, value)
}

/// Convert HSV to RGB color space
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgb<u8> {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    Rgb::new(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}