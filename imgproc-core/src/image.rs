use crate::{
    error::{Error, Result},
    pixel::Pixel,
    Rect,
};
use std::marker::PhantomData;

/// Core image structure with row-major storage
#[derive(Debug, Clone)]
pub struct Image<P: Pixel> {
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Pixel data in row-major order
    data: Vec<P>,
}

impl<P: Pixel> Image<P> {
    /// Create a new image with specified dimensions
    ///
    /// # Errors
    /// Returns error if dimensions are zero or allocation fails
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        let size = (width as usize) * (height as usize);
        if size > isize::MAX as usize {
            return Err(Error::AllocationFailed { size });
        }

        Ok(Self {
            width,
            height,
            data: vec![P::black(); size],
        })
    }

    /// Create an image from raw data
    ///
    /// # Errors
    /// Returns error if data length doesn't match dimensions
    pub fn from_raw(width: u32, height: u32, data: Vec<P>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(Error::BufferSizeMismatch {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self { width, height, data })
    }

    /// Create an image filled with a specific pixel value
    pub fn from_pixel(width: u32, height: u32, pixel: P) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions { width, height });
        }

        let size = (width as usize) * (height as usize);
        Ok(Self {
            width,
            height,
            data: vec![pixel; size],
        })
    }

    /// Get image width
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Get image height
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Get image dimensions as tuple
    #[inline]
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get total number of pixels
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if image is empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get pixel at coordinates
    ///
    /// # Errors
    /// Returns error if coordinates are out of bounds
    pub fn get_pixel(&self, x: u32, y: u32) -> Result<&P> {
        if x >= self.width || y >= self.height {
            return Err(Error::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let index = (y as usize) * (self.width as usize) + (x as usize);
        Ok(&self.data[index])
    }

    /// Get mutable pixel at coordinates
    ///
    /// # Errors
    /// Returns error if coordinates are out of bounds
    pub fn get_pixel_mut(&mut self, x: u32, y: u32) -> Result<&mut P> {
        if x >= self.width || y >= self.height {
            return Err(Error::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let index = (y as usize) * (self.width as usize) + (x as usize);
        Ok(&mut self.data[index])
    }

    /// Set pixel at coordinates
    ///
    /// # Errors
    /// Returns error if coordinates are out of bounds
    pub fn set_pixel(&mut self, x: u32, y: u32, pixel: P) -> Result<()> {
        if x >= self.width || y >= self.height {
            return Err(Error::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let index = (y as usize) * (self.width as usize) + (x as usize);
        self.data[index] = pixel;
        Ok(())
    }

    /// Get pixel without bounds checking
    ///
    /// # Safety
    /// Caller must ensure coordinates are within bounds
    #[inline]
    pub unsafe fn get_pixel_unchecked(&self, x: u32, y: u32) -> &P {
        let index = (y as usize) * (self.width as usize) + (x as usize);
        self.data.get_unchecked(index)
    }

    /// Get mutable pixel without bounds checking
    ///
    /// # Safety
    /// Caller must ensure coordinates are within bounds
    #[inline]
    pub unsafe fn get_pixel_unchecked_mut(&mut self, x: u32, y: u32) -> &mut P {
        let index = (y as usize) * (self.width as usize) + (x as usize);
        self.data.get_unchecked_mut(index)
    }

    /// Get raw pixel data
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[P] {
        &self.data
    }

    /// Get mutable raw pixel data
    #[inline]
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [P] {
        &mut self.data
    }

    /// Consume image and return raw data
    #[inline]
    #[must_use]
    pub fn into_raw(self) -> Vec<P> {
        self.data
    }

    /// Get a row of pixels
    pub fn row(&self, y: u32) -> Result<&[P]> {
        if y >= self.height {
            return Err(Error::OutOfBounds {
                x: 0,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let start = (y as usize) * (self.width as usize);
        let end = start + (self.width as usize);
        Ok(&self.data[start..end])
    }

    /// Get a mutable row of pixels
    pub fn row_mut(&mut self, y: u32) -> Result<&mut [P]> {
        if y >= self.height {
            return Err(Error::OutOfBounds {
                x: 0,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let start = (y as usize) * (self.width as usize);
        let end = start + (self.width as usize);
        Ok(&mut self.data[start..end])
    }

    /// Create a view into a sub-region of the image
    pub fn view(&self, rect: Rect) -> Result<ImageView<'_, P>> {
        if rect.x + rect.width > self.width || rect.y + rect.height > self.height {
            return Err(Error::InvalidCrop {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
            });
        }
        Ok(ImageView {
            image: self,
            rect,
            _phantom: PhantomData,
        })
    }

    /// Create a mutable view into a sub-region of the image
    pub fn view_mut(&mut self, rect: Rect) -> Result<ImageViewMut<'_, P>> {
        if rect.x + rect.width > self.width || rect.y + rect.height > self.height {
            return Err(Error::InvalidCrop {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
            });
        }
        Ok(ImageViewMut {
            image: self,
            rect,
            _phantom: PhantomData,
        })
    }

    /// Iterate over all pixels
    #[inline]
    pub fn pixels(&self) -> impl Iterator<Item = &P> {
        self.data.iter()
    }

    /// Iterate over all pixels mutably
    #[inline]
    pub fn pixels_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.data.iter_mut()
    }

    /// Iterate over pixels with coordinates
    pub fn enumerate_pixels(&self) -> impl Iterator<Item = (u32, u32, &P)> {
        self.data.iter().enumerate().map(move |(i, p)| {
            let x = (i % self.width as usize) as u32;
            let y = (i / self.width as usize) as u32;
            (x, y, p)
        })
    }
}

/// Immutable view into a sub-region of an image
pub struct ImageView<'a, P: Pixel> {
    image: &'a Image<P>,
    rect: Rect,
    _phantom: PhantomData<P>,
}

impl<'a, P: Pixel> ImageView<'a, P> {
    /// Get view width
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.rect.width
    }

    /// Get view height
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.rect.height
    }

    /// Get pixel at view-relative coordinates
    pub fn get_pixel(&self, x: u32, y: u32) -> Result<&P> {
        if x >= self.rect.width || y >= self.rect.height {
            return Err(Error::OutOfBounds {
                x,
                y,
                width: self.rect.width,
                height: self.rect.height,
            });
        }
        self.image.get_pixel(self.rect.x + x, self.rect.y + y)
    }
}

/// Mutable view into a sub-region of an image
pub struct ImageViewMut<'a, P: Pixel> {
    image: &'a mut Image<P>,
    rect: Rect,
    _phantom: PhantomData<P>,
}

impl<'a, P: Pixel> ImageViewMut<'a, P> {
    /// Get view width
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.rect.width
    }

    /// Get view height
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.rect.height
    }

    /// Get pixel at view-relative coordinates
    pub fn get_pixel(&self, x: u32, y: u32) -> Result<&P> {
        if x >= self.rect.width || y >= self.rect.height {
            return Err(Error::OutOfBounds {
                x,
                y,
                width: self.rect.width,
                height: self.rect.height,
            });
        }
        self.image.get_pixel(self.rect.x + x, self.rect.y + y)
    }

    /// Set pixel at view-relative coordinates
    pub fn set_pixel(&mut self, x: u32, y: u32, pixel: P) -> Result<()> {
        if x >= self.rect.width || y >= self.rect.height {
            return Err(Error::OutOfBounds {
                x,
                y,
                width: self.rect.width,
                height: self.rect.height,
            });
        }
        self.image.set_pixel(self.rect.x + x, self.rect.y + y, pixel)
    }
}