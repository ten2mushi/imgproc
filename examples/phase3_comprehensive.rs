//! Phase 3 Comprehensive Test - Extended Operations
//!
//! This example demonstrates all Phase 3 operations including:
//! - Extended color spaces (HSL, LAB, XYZ, YCbCr)
//! - Histogram operations (Equalization, CLAHE, Matching)
//! - Contour detection and analysis
//!
//! Builds on top of Phase 1 and 2 operations with 40+ total operations.

use imgproc::prelude::*;
use imgproc::ops::color::{
    rgb_to_hsl, hsl_to_rgb, rgb_to_lab, lab_to_rgb,
    rgb_to_xyz, xyz_to_rgb, rgb_to_ycbcr, ycbcr_to_rgb,
    rgb_image_to_hsl, hsl_to_rgb_image,
    rgb_image_to_lab, lab_to_rgb_image,
    rgb_image_to_xyz, xyz_to_rgb_image,
    rgb_image_to_ycbcr, ycbcr_to_rgb_image,
};
use imgproc::ops::histogram::{Histogram, HistogramEqualization, Clahe, HistogramMatching};
use imgproc::ops::contours::{FindContours, ContourFeatures, ApproxPolyDP};
use imgproc::ops::contours::{ContourRetrievalMode, ContourApproximation};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=======================================================");
    println!("  imgproc - Phase 3 Comprehensive Operation Test");
    println!("  Testing: Extended Color Spaces, Histograms, Contours");
    println!("=======================================================\n");

    let input_path = "input/IMG-20251103-WA0001.jpg";
    let output_dir = "output";

    // ========================================================================
    // Load Images
    // ========================================================================

    println!("📂 Loading input image: {}", input_path);
    let start = Instant::now();
    let img_rgb = imgproc::io::read_image_rgb(input_path)?;
    let img_gray = imgproc::ops::color::rgb_to_gray(&img_rgb)?;
    println!("   ✓ Loaded RGB {}x{} and converted to grayscale ({:.2}ms)\n",
             img_rgb.width(), img_rgb.height(),
             start.elapsed().as_secs_f64() * 1000.0);

    let mut operation_count = 0;

    // ========================================================================
    // PHASE 3: Extended Color Spaces
    // ========================================================================

    println!("🎨 Phase 3A: Extended Color Space Conversions\n");

    // Operation 25: RGB to HSL and back
    operation_count += 1;
    println!("{}️⃣  RGB to HSL color space conversion...", operation_count);
    let start = Instant::now();
    let hsl_data = rgb_image_to_hsl(&img_rgb)?;
    let img_hsl_back = hsl_to_rgb_image(&hsl_data, img_rgb.width(), img_rgb.height())?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_rgb(&img_hsl_back, &format!("{}/25_hsl_conversion.png", output_dir))?;
    println!("   ✓ Converted RGB→HSL→RGB (HSL: H=0-360°, S=0-1, L=0-1) ({:.2}ms)\n", elapsed);

    // Operation 26: RGB to LAB and back
    operation_count += 1;
    println!("{}️⃣  RGB to LAB (CIELAB) color space conversion...", operation_count);
    let start = Instant::now();
    let lab_data = rgb_image_to_lab(&img_rgb)?;
    let img_lab_back = lab_to_rgb_image(&lab_data, img_rgb.width(), img_rgb.height())?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_rgb(&img_lab_back, &format!("{}/26_lab_conversion.png", output_dir))?;
    println!("   ✓ Converted RGB→LAB→RGB (Perceptually uniform) ({:.2}ms)\n", elapsed);

    // Operation 27: RGB to XYZ and back
    operation_count += 1;
    println!("{}️⃣  RGB to XYZ (CIE 1931) color space conversion...", operation_count);
    let start = Instant::now();
    let xyz_data = rgb_image_to_xyz(&img_rgb)?;
    let img_xyz_back = xyz_to_rgb_image(&xyz_data, img_rgb.width(), img_rgb.height())?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_rgb(&img_xyz_back, &format!("{}/27_xyz_conversion.png", output_dir))?;
    println!("   ✓ Converted RGB→XYZ→RGB (Tristimulus values) ({:.2}ms)\n", elapsed);

    // Operation 28: RGB to YCbCr and back
    operation_count += 1;
    println!("{}️⃣  RGB to YCbCr color space conversion...", operation_count);
    let start = Instant::now();
    let ycbcr_data = rgb_image_to_ycbcr(&img_rgb)?;
    let img_ycbcr_back = ycbcr_to_rgb_image(&ycbcr_data, img_rgb.width(), img_rgb.height())?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_rgb(&img_ycbcr_back, &format!("{}/28_ycbcr_conversion.png", output_dir))?;
    println!("   ✓ Converted RGB→YCbCr→RGB (JPEG standard) ({:.2}ms)\n", elapsed);

    // ========================================================================
    // PHASE 3: Histogram Operations
    // ========================================================================

    println!("📊 Phase 3B: Histogram Operations\n");

    // Operation 29: Compute histogram
    operation_count += 1;
    println!("{}️⃣  Computing histogram...", operation_count);
    let start = Instant::now();
    let hist = Histogram::compute(&img_gray);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let normalized = hist.normalize();
    let max_bin_value = normalized.iter().copied().fold(0.0_f64, f64::max);
    println!("   ✓ Computed 256-bin histogram");
    println!("      Total pixels: {}", hist.total_pixels());
    println!("      Max bin probability: {:.4} ({:.2}ms)\n", max_bin_value, elapsed);

    // Operation 30: Histogram equalization
    operation_count += 1;
    println!("{}️⃣  Applying histogram equalization...", operation_count);
    let start = Instant::now();
    let eq = HistogramEqualization::new();
    let img_equalized = eq.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&img_equalized, &format!("{}/30_histogram_equalized.png", output_dir))?;
    println!("   ✓ Enhanced global contrast ({:.2}ms)\n", elapsed);

    // Operation 31: CLAHE with different parameters
    operation_count += 1;
    println!("{}️⃣  Applying CLAHE (2x2 grid, clip=2.0)...", operation_count);
    let start = Instant::now();
    let clahe1 = Clahe::new(2.0, (2, 2))?;
    let img_clahe1 = clahe1.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&img_clahe1, &format!("{}/31_clahe_2x2.png", output_dir))?;
    println!("   ✓ Adaptive equalization with 2x2 tiles ({:.2}ms)\n", elapsed);

    operation_count += 1;
    println!("{}️⃣  Applying CLAHE (8x8 grid, clip=3.0)...", operation_count);
    let start = Instant::now();
    let clahe2 = Clahe::new(3.0, (8, 8))?;
    let img_clahe2 = clahe2.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&img_clahe2, &format!("{}/32_clahe_8x8.png", output_dir))?;
    println!("   ✓ Adaptive equalization with 8x8 tiles ({:.2}ms)\n", elapsed);

    // Operation 33: Histogram matching
    operation_count += 1;
    println!("{}️⃣  Applying histogram matching...", operation_count);
    let start = Instant::now();
    // Use the equalized image as reference
    let matcher = HistogramMatching::new(&img_equalized);
    let img_matched = matcher.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&img_matched, &format!("{}/33_histogram_matched.png", output_dir))?;
    println!("   ✓ Matched histogram to reference ({:.2}ms)\n", elapsed);

    // ========================================================================
    // PHASE 3: Contour Detection and Analysis
    // ========================================================================

    println!("🔍 Phase 3C: Contour Detection and Analysis\n");

    // First create a binary image for contour detection
    let otsu = OtsuThreshold::new();
    let binary_img = otsu.apply_gray(&img_gray)?;

    // Operation 34: Find contours (external mode)
    operation_count += 1;
    println!("{}️⃣  Detecting contours (external mode)...", operation_count);
    let start = Instant::now();
    let finder_external = FindContours::new(
        ContourRetrievalMode::External,
        ContourApproximation::Simple,
    );
    let contours_external = finder_external.find_contours(&binary_img)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("   ✓ Found {} external contours ({:.2}ms)\n", contours_external.len(), elapsed);

    // Operation 35: Find contours (all with hierarchy)
    operation_count += 1;
    println!("{}️⃣  Detecting all contours with hierarchy...", operation_count);
    let start = Instant::now();
    let finder_tree = FindContours::new(
        ContourRetrievalMode::Tree,
        ContourApproximation::Simple,
    );
    let contours_all = finder_tree.find_contours(&binary_img)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("   ✓ Found {} total contours ({:.2}ms)\n", contours_all.len(), elapsed);

    // Operation 36: Compute contour areas and sort
    operation_count += 1;
    println!("{}️⃣  Computing contour areas...", operation_count);
    let start = Instant::now();
    let mut contour_areas: Vec<(usize, f64)> = contours_all
        .iter()
        .enumerate()
        .map(|(i, c)| (i, ContourFeatures::area(c)))
        .collect();
    contour_areas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("   ✓ Computed areas for {} contours", contour_areas.len());
    if !contour_areas.is_empty() {
        println!("      Largest contour: {:.1} pixels", contour_areas[0].1);
        println!("      Smallest contour: {:.1} pixels ({:.2}ms)\n",
                 contour_areas.last().map(|x| x.1).unwrap_or(0.0), elapsed);
    } else {
        println!("      No contours found ({:.2}ms)\n", elapsed);
    }

    // Operation 37: Compute bounding boxes
    operation_count += 1;
    println!("{}️⃣  Computing bounding boxes for top 5 contours...", operation_count);
    let start = Instant::now();
    let mut bbox_count = 0;
    for &(idx, area) in contour_areas.iter().take(5) {
        let contour = &contours_all[idx];
        let bbox = ContourFeatures::bounding_box(contour);
        println!("      Contour {}: area={:.1}, bbox={}x{} at ({},{})",
                 idx, area, bbox.width, bbox.height, bbox.x, bbox.y);
        bbox_count += 1;
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("   ✓ Computed {} bounding boxes ({:.2}ms)\n", bbox_count, elapsed);

    // Operation 38: Compute convex hulls
    operation_count += 1;
    println!("{}️⃣  Computing convex hulls...", operation_count);
    let start = Instant::now();
    let mut hull_count = 0;
    for &(idx, _) in contour_areas.iter().take(5) {
        let contour = &contours_all[idx];
        let hull = ContourFeatures::convex_hull(contour);
        println!("      Contour {}: {} points → {} hull points",
                 idx, contour.len(), hull.len());
        hull_count += 1;
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("   ✓ Computed {} convex hulls ({:.2}ms)\n", hull_count, elapsed);

    // Operation 39: Compute shape descriptors
    operation_count += 1;
    println!("{}️⃣  Computing shape descriptors...", operation_count);
    let start = Instant::now();
    for &(idx, _) in contour_areas.iter().take(3) {
        let contour = &contours_all[idx];
        let circularity = ContourFeatures::circularity(contour);
        let solidity = ContourFeatures::solidity(contour);
        let extent = ContourFeatures::extent(contour);
        let aspect_ratio = ContourFeatures::aspect_ratio(contour);

        println!("      Contour {}:", idx);
        println!("         Circularity: {:.3} (1.0 = perfect circle)", circularity);
        println!("         Solidity: {:.3} (ratio to convex hull)", solidity);
        println!("         Extent: {:.3} (ratio to bounding box)", extent);
        println!("         Aspect ratio: {:.2}", aspect_ratio);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("   ✓ Computed shape descriptors ({:.2}ms)\n", elapsed);

    // Operation 40: Contour approximation (Douglas-Peucker)
    operation_count += 1;
    println!("{}️⃣  Approximating contours (Douglas-Peucker)...", operation_count);
    let start = Instant::now();
    for &(idx, _) in contour_areas.iter().take(3) {
        let contour = &contours_all[idx];
        let approx = ApproxPolyDP::new(2.0)?;
        let simplified = approx.approximate(contour);
        println!("      Contour {}: {} points → {} simplified points (epsilon=2.0)",
                 idx, contour.len(), simplified.len());
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("   ✓ Approximated contours ({:.2}ms)\n", elapsed);

    // ========================================================================
    // Summary
    // ========================================================================

    println!("=======================================================");
    println!("✅ Phase 3: {} extended operations completed!", operation_count);
    println!("=======================================================");
    println!("\n📁 Phase 3 output files saved to: {}/", output_dir);
    println!("\nPhase 3 Operations Summary:");
    println!("  🎨 Extended Color Spaces:");
    println!("     • HSL (Hue, Saturation, Lightness)");
    println!("     • LAB (CIELAB - perceptually uniform)");
    println!("     • XYZ (CIE 1931 tristimulus)");
    println!("     • YCbCr (JPEG/video compression standard)");
    println!("\n  📊 Histogram Operations:");
    println!("     • Histogram computation (256 bins)");
    println!("     • Global histogram equalization");
    println!("     • CLAHE (Contrast Limited Adaptive HE)");
    println!("     • Histogram matching/specification");
    println!("\n  🔍 Contour Analysis:");
    println!("     • Contour detection (border following)");
    println!("     • Bounding box calculation");
    println!("     • Convex hull computation");
    println!("     • Shape descriptors (circularity, solidity, extent)");
    println!("     • Douglas-Peucker approximation");
    println!("\n🎉 Phase 3 comprehensive test complete!");
    println!("\nTotal operations tested across all phases: 40+");
    println!("  - Phase 1: 24 operations (basic image processing)");
    println!("  - Phase 2: Pipeline system + CLI");
    println!("  - Phase 3: {} extended operations (color, histogram, contours)", operation_count);

    Ok(())
}
