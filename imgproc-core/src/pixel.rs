use bytemuck::{Pod, Zeroable};
use num_traits::{Bounded, NumCast, Zero};

/// Trait for pixel channel types
pub trait Channel:
    Copy + Clone + Zero + Bounded + NumCast + PartialEq + PartialOrd + Pod + Zeroable + Send + Sync + 'static
{
    /// Maximum value for this channel type
    const MAX: Self;
    /// Minimum value for this channel type
    const MIN: Self;
}

impl Channel for u8 {
    const MAX: Self = u8::MAX;
    const MIN: Self = u8::MIN;
}

impl Channel for u16 {
    const MAX: Self = u16::MAX;
    const MIN: Self = u16::MIN;
}

impl Channel for f32 {
    const MAX: Self = 1.0;
    const MIN: Self = 0.0;
}

/// Trait for pixel types
pub trait Pixel: Copy + Clone + Send + Sync + 'static {
    /// Channel type for this pixel
    type Channel: Channel;

    /// Number of channels
    const CHANNELS: usize;

    /// Whether this pixel has an alpha channel
    const HAS_ALPHA: bool = false;

    /// Create a pixel from channel values
    fn from_channels(channels: &[Self::Channel]) -> Self;

    /// Get channels as slice
    fn channels(&self) -> &[Self::Channel];

    /// Get mutable channels as slice
    fn channels_mut(&mut self) -> &mut [Self::Channel];

    /// Create a black/zero pixel
    fn black() -> Self {
        Self::from_channels(&vec![Self::Channel::MIN; Self::CHANNELS])
    }

    /// Create a white/max pixel
    fn white() -> Self {
        Self::from_channels(&vec![Self::Channel::MAX; Self::CHANNELS])
    }
}

/// Grayscale pixel type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gray<T: Channel> {
    /// Luminance value
    pub value: T,
}

// Implement Pod and Zeroable manually for concrete types
unsafe impl Zeroable for Gray<u8> {}
unsafe impl Pod for Gray<u8> {}
unsafe impl Zeroable for Gray<u16> {}
unsafe impl Pod for Gray<u16> {}
unsafe impl Zeroable for Gray<f32> {}
unsafe impl Pod for Gray<f32> {}

impl<T: Channel> Gray<T> {
    /// Create a new grayscale pixel
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Channel> Pixel for Gray<T> {
    type Channel = T;
    const CHANNELS: usize = 1;

    fn from_channels(channels: &[Self::Channel]) -> Self {
        Self::new(channels[0])
    }

    fn channels(&self) -> &[Self::Channel] {
        std::slice::from_ref(&self.value)
    }

    fn channels_mut(&mut self) -> &mut [Self::Channel] {
        std::slice::from_mut(&mut self.value)
    }
}

/// RGB pixel type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb<T: Channel> {
    /// Red channel
    pub r: T,
    /// Green channel
    pub g: T,
    /// Blue channel
    pub b: T,
}

// Implement Pod and Zeroable manually for concrete types
unsafe impl Zeroable for Rgb<u8> {}
unsafe impl Pod for Rgb<u8> {}
unsafe impl Zeroable for Rgb<u16> {}
unsafe impl Pod for Rgb<u16> {}
unsafe impl Zeroable for Rgb<f32> {}
unsafe impl Pod for Rgb<f32> {}

impl<T: Channel> Rgb<T> {
    /// Create a new RGB pixel
    #[must_use]
    pub const fn new(r: T, g: T, b: T) -> Self {
        Self { r, g, b }
    }
}

impl<T: Channel> Pixel for Rgb<T> {
    type Channel = T;
    const CHANNELS: usize = 3;

    fn from_channels(channels: &[Self::Channel]) -> Self {
        Self::new(channels[0], channels[1], channels[2])
    }

    fn channels(&self) -> &[Self::Channel] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const T, 3)
        }
    }

    fn channels_mut(&mut self) -> &mut [Self::Channel] {
        unsafe {
            std::slice::from_raw_parts_mut(self as *mut _ as *mut T, 3)
        }
    }
}

/// RGBA pixel type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba<T: Channel> {
    /// Red channel
    pub r: T,
    /// Green channel
    pub g: T,
    /// Blue channel
    pub b: T,
    /// Alpha channel
    pub a: T,
}

// Implement Pod and Zeroable manually for concrete types
unsafe impl Zeroable for Rgba<u8> {}
unsafe impl Pod for Rgba<u8> {}
unsafe impl Zeroable for Rgba<u16> {}
unsafe impl Pod for Rgba<u16> {}
unsafe impl Zeroable for Rgba<f32> {}
unsafe impl Pod for Rgba<f32> {}

impl<T: Channel> Rgba<T> {
    /// Create a new RGBA pixel
    #[must_use]
    pub const fn new(r: T, g: T, b: T, a: T) -> Self {
        Self { r, g, b, a }
    }
}

impl<T: Channel> Pixel for Rgba<T> {
    type Channel = T;
    const CHANNELS: usize = 4;
    const HAS_ALPHA: bool = true;

    fn from_channels(channels: &[Self::Channel]) -> Self {
        Self::new(channels[0], channels[1], channels[2], channels[3])
    }

    fn channels(&self) -> &[Self::Channel] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const T, 4)
        }
    }

    fn channels_mut(&mut self) -> &mut [Self::Channel] {
        unsafe {
            std::slice::from_raw_parts_mut(self as *mut _ as *mut T, 4)
        }
    }
}

// Common type aliases
/// 8-bit grayscale pixel
pub type Gray8 = Gray<u8>;
/// 16-bit grayscale pixel
pub type Gray16 = Gray<u16>;
/// 32-bit float grayscale pixel
pub type GrayF32 = Gray<f32>;

/// 8-bit RGB pixel
pub type Rgb8 = Rgb<u8>;
/// 16-bit RGB pixel
pub type Rgb16 = Rgb<u16>;
/// 32-bit float RGB pixel
pub type RgbF32 = Rgb<f32>;

/// 8-bit RGBA pixel
pub type Rgba8 = Rgba<u8>;
/// 16-bit RGBA pixel
pub type Rgba16 = Rgba<u16>;
/// 32-bit float RGBA pixel
pub type RgbaF32 = Rgba<f32>;