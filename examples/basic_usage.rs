//! Basic usage example demonstrating core operations

use imgproc::prelude::*;

fn main() -> Result<()> {
    println!("imgproc - Image Processing Library");
    println!("===================================\n");

    // Example 1: Creating images
    println!("1. Creating a new image...");
    let mut img = Image::<Gray8>::new(100, 100)?;
    println!("   Created {}x{} grayscale image", img.width(), img.height());

    // Example 2: Setting pixels
    println!("\n2. Setting pixel values...");
    for y in 0..100 {
        for x in 0..100 {
            let value = ((x + y) * 255 / 200) as u8;
            img.set_pixel(x, y, Gray8::new(value))?;
        }
    }
    println!("   Filled image with gradient pattern");

    // Example 3: Gaussian blur
    println!("\n3. Applying Gaussian blur...");
    let blur = GaussianBlur::new(5, 1.5)?;
    let blurred = blur.apply_gray(&img)?;
    println!("   Applied 5x5 Gaussian blur with sigma=1.5");

    // Example 4: Median filter
    println!("\n4. Applying median filter...");
    let median = MedianFilter::new(5)?;
    let filtered = median.apply_gray(&img)?;
    println!("   Applied 5x5 median filter");

    // Example 5: Sobel edge detection
    println!("\n5. Detecting edges with Sobel...");
    let sobel = Sobel::new();
    let edges = sobel.apply_gray(&blurred)?;
    println!("   Detected edges using Sobel operator");

    // Example 6: Thresholding
    println!("\n6. Applying threshold...");
    use imgproc::ops::segment::ThresholdType;
    let threshold = Threshold::new(128, 255, ThresholdType::Binary);
    let binary = threshold.apply_gray(&img)?;
    println!("   Applied binary threshold at 128");

    // Example 7: Otsu's threshold
    println!("\n7. Applying Otsu's automatic threshold...");
    let otsu = OtsuThreshold::new();
    let threshold_value = otsu.calculate_threshold(&img);
    println!("   Calculated optimal threshold: {}", threshold_value);
    let binary_otsu = otsu.apply_gray(&img)?;
    println!("   Applied Otsu's threshold");

    // Example 8: Morphological operations
    println!("\n8. Applying morphological operations...");
    use imgproc::ops::morphology::StructuringElement;
    let dilate = Dilate::new(5, StructuringElement::Rectangle)?;
    let dilated = dilate.apply_gray(&binary)?;
    println!("   Applied dilation with 5x5 rectangle");

    let erode = Erode::new(5, StructuringElement::Ellipse)?;
    let eroded = erode.apply_gray(&binary)?;
    println!("   Applied erosion with 5x5 ellipse");

    // Example 9: Resize operation
    println!("\n9. Resizing image...");
    use imgproc::ops::geometry::Interpolation;
    let resize = Resize::new(50, 50, Interpolation::Bilinear);
    let resized = resize.resize_gray(&img)?;
    println!("   Resized from {}x{} to {}x{}",
             img.width(), img.height(),
             resized.width(), resized.height());

    // Example 10: Crop operation
    println!("\n10. Cropping image...");
    let crop_rect = Rect::new(25, 25, 50, 50);
    let crop = Crop::new(crop_rect);
    let cropped = crop.crop_gray(&img)?;
    println!("    Cropped region: {}x{}", cropped.width(), cropped.height());

    // Example 11: Image statistics
    println!("\n11. Image statistics...");
    let mut min_val = 255u8;
    let mut max_val = 0u8;
    let mut sum = 0u64;

    for pixel in img.pixels() {
        min_val = min_val.min(pixel.value);
        max_val = max_val.max(pixel.value);
        sum += pixel.value as u64;
    }

    let mean = sum / (img.width() as u64 * img.height() as u64);
    println!("    Min: {}, Max: {}, Mean: {}", min_val, max_val, mean);

    println!("\n✓ All operations completed successfully!");
    println!("\nNote: To process actual image files, use:");
    println!("  imgproc::io::read_image_gray(\"input.jpg\")");
    println!("  imgproc::io::write_image_gray(&img, \"output.png\")");

    Ok(())
}
