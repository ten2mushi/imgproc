use imgproc_core::{
    error::{Error, Result},
    image::Image,
    pixel::Gray8,
    traits::Operation,
    Rect,
};
use std::f32::consts::PI;

/// Resize operation with different interpolation methods
#[derive(Debug, Clone)]
pub struct Resize {
    /// Target width
    pub width: u32,
    /// Target height
    pub height: u32,
    /// Interpolation method
    pub interpolation: Interpolation,
}

/// Interpolation methods for resizing
#[derive(Debug, Clone, Copy)]
pub enum Interpolation {
    /// Nearest neighbor interpolation
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
    /// Lanczos interpolation
    Lanczos,
}

impl Resize {
    /// Create a new resize operation
    #[must_use]
    pub const fn new(width: u32, height: u32, interpolation: Interpolation) -> Self {
        Self { width, height, interpolation }
    }

    /// Resize grayscale image
    pub fn resize_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let mut output = Image::new(self.width, self.height)?;

        let x_scale = input.width() as f32 / self.width as f32;
        let y_scale = input.height() as f32 / self.height as f32;

        match self.interpolation {
            Interpolation::Nearest => {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let src_x = (x as f32 * x_scale) as u32;
                        let src_y = (y as f32 * y_scale) as u32;

                        let src_x = src_x.min(input.width() - 1);
                        let src_y = src_y.min(input.height() - 1);

                        let pixel = input.get_pixel(src_x, src_y)?;
                        output.set_pixel(x, y, *pixel)?;
                    }
                }
            }
            Interpolation::Bilinear => {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let src_x = x as f32 * x_scale;
                        let src_y = y as f32 * y_scale;

                        let x0 = src_x.floor() as u32;
                        let x1 = (x0 + 1).min(input.width() - 1);
                        let y0 = src_y.floor() as u32;
                        let y1 = (y0 + 1).min(input.height() - 1);

                        let fx = src_x - x0 as f32;
                        let fy = src_y - y0 as f32;

                        let p00 = input.get_pixel(x0, y0)?.value as f32;
                        let p10 = input.get_pixel(x1, y0)?.value as f32;
                        let p01 = input.get_pixel(x0, y1)?.value as f32;
                        let p11 = input.get_pixel(x1, y1)?.value as f32;

                        let value = p00 * (1.0 - fx) * (1.0 - fy) +
                                   p10 * fx * (1.0 - fy) +
                                   p01 * (1.0 - fx) * fy +
                                   p11 * fx * fy;

                        output.set_pixel(x, y, Gray8::new(value.round() as u8))?;
                    }
                }
            }
            _ => {
                // For now, fall back to bilinear for other methods
                // TODO: Implement bicubic and Lanczos
                return self.new_with_interpolation(Interpolation::Bilinear).resize_gray(input);
            }
        }

        Ok(output)
    }

    fn new_with_interpolation(&self, interpolation: Interpolation) -> Self {
        Self {
            width: self.width,
            height: self.height,
            interpolation,
        }
    }
}

impl Operation for Resize {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.resize_gray(&input)
    }
}

/// Crop operation
#[derive(Debug, Clone)]
pub struct Crop {
    /// Crop rectangle
    pub rect: Rect,
}

impl Crop {
    /// Create a new crop operation
    #[must_use]
    pub const fn new(rect: Rect) -> Self {
        Self { rect }
    }

    /// Crop grayscale image
    pub fn crop_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        if self.rect.x + self.rect.width > input.width() ||
           self.rect.y + self.rect.height > input.height() {
            return Err(Error::InvalidCrop {
                x: self.rect.x,
                y: self.rect.y,
                width: self.rect.width,
                height: self.rect.height,
            });
        }

        let mut output = Image::new(self.rect.width, self.rect.height)?;

        for y in 0..self.rect.height {
            for x in 0..self.rect.width {
                let src_x = self.rect.x + x;
                let src_y = self.rect.y + y;
                let pixel = input.get_pixel(src_x, src_y)?;
                output.set_pixel(x, y, *pixel)?;
            }
        }

        Ok(output)
    }
}

impl Operation for Crop {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.crop_gray(&input)
    }
}

/// Rotate operation
#[derive(Debug, Clone)]
pub struct Rotate {
    /// Rotation angle in degrees
    pub angle: f32,
    /// Fill value for empty pixels
    pub fill_value: u8,
}

impl Rotate {
    /// Create a new rotate operation
    #[must_use]
    pub const fn new(angle: f32) -> Self {
        Self { angle, fill_value: 0 }
    }

    /// Rotate grayscale image
    pub fn rotate_gray(&self, input: &Image<Gray8>) -> Result<Image<Gray8>> {
        let angle_rad = self.angle * PI / 180.0;
        let cos_angle = angle_rad.cos();
        let sin_angle = angle_rad.sin();

        // Calculate output dimensions
        let in_width = input.width() as f32;
        let in_height = input.height() as f32;

        let out_width = (in_width * cos_angle.abs() + in_height * sin_angle.abs()).ceil() as u32;
        let out_height = (in_width * sin_angle.abs() + in_height * cos_angle.abs()).ceil() as u32;

        let mut output = Image::from_pixel(out_width, out_height, Gray8::new(self.fill_value))?;

        let cx_in = in_width / 2.0;
        let cy_in = in_height / 2.0;
        let cx_out = out_width as f32 / 2.0;
        let cy_out = out_height as f32 / 2.0;

        for y_out in 0..out_height {
            for x_out in 0..out_width {
                // Translate to center
                let x = x_out as f32 - cx_out;
                let y = y_out as f32 - cy_out;

                // Inverse rotation to find source pixel
                let src_x = x * cos_angle + y * sin_angle + cx_in;
                let src_y = -x * sin_angle + y * cos_angle + cy_in;

                // Check bounds and sample
                if src_x >= 0.0 && src_x < in_width && src_y >= 0.0 && src_y < in_height {
                    // Bilinear interpolation
                    let x0 = src_x.floor() as u32;
                    let x1 = (x0 + 1).min(input.width() - 1);
                    let y0 = src_y.floor() as u32;
                    let y1 = (y0 + 1).min(input.height() - 1);

                    let fx = src_x - x0 as f32;
                    let fy = src_y - y0 as f32;

                    let p00 = input.get_pixel(x0, y0)?.value as f32;
                    let p10 = input.get_pixel(x1, y0)?.value as f32;
                    let p01 = input.get_pixel(x0, y1)?.value as f32;
                    let p11 = input.get_pixel(x1, y1)?.value as f32;

                    let value = p00 * (1.0 - fx) * (1.0 - fy) +
                               p10 * fx * (1.0 - fy) +
                               p01 * (1.0 - fx) * fy +
                               p11 * fx * fy;

                    output.set_pixel(x_out, y_out, Gray8::new(value.round() as u8))?;
                }
            }
        }

        Ok(output)
    }
}

impl Operation for Rotate {
    type Input = Image<Gray8>;
    type Output = Image<Gray8>;

    fn apply(&self, input: Self::Input) -> Result<Self::Output> {
        self.rotate_gray(&input)
    }
}