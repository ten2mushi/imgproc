//! Test all operations with a real input image
//!
//! This example loads an actual image file and applies various operations,
//! saving each result to the output directory for visual inspection.

use imgproc::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("===========================================");
    println!("  imgproc - Comprehensive Operation Test");
    println!("===========================================\n");

    // let input_path = "input/IMG-20251103-WA0001.jpg";
    let input_path = "input/IMG-20251026-WA0000.jpg";
    // let input_path = "input/k0m0r3b1_a_knight_of_Malta_in_the_style_of_Winslow_Homer_waterc_b8ab7b93-db9b-4208-9e41-19ad0332237c.png";
    let output_dir = "output";

    // Load the input image as RGB
    println!("📂 Loading input image: {}", input_path);
    let start = Instant::now();
    let img_rgb = imgproc::io::read_image_rgb(input_path)?;
    println!("   ✓ Loaded RGB image: {}x{} ({:.2}ms)\n",
             img_rgb.width(), img_rgb.height(),
             start.elapsed().as_secs_f64() * 1000.0);

    // Save original as reference
    println!("1️⃣  Saving original RGB image...");
    imgproc::io::write_image_rgb(&img_rgb, &format!("{}/01_original_rgb.png", output_dir))?;
    println!("   ✓ Saved: 01_original_rgb.png\n");

    // Convert to grayscale
    println!("2️⃣  Converting to grayscale...");
    let start = Instant::now();
    let img_gray = imgproc::ops::color::rgb_to_gray(&img_rgb)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&img_gray, &format!("{}/02_grayscale.png", output_dir))?;
    println!("   ✓ Converted and saved ({:.2}ms)\n", elapsed);

    // Test Gaussian blur with different parameters
    println!("3️⃣  Applying Gaussian blur (kernel=5, sigma=1.5)...");
    let start = Instant::now();
    let blur1 = GaussianBlur::new(5, 1.5)?;
    let blurred1 = blur1.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&blurred1, &format!("{}/03_gaussian_blur_5x5_s1.5.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("4️⃣  Applying Gaussian blur (kernel=11, sigma=3.0)...");
    let start = Instant::now();
    let blur2 = GaussianBlur::new(11, 3.0)?;
    let blurred2 = blur2.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&blurred2, &format!("{}/04_gaussian_blur_11x11_s3.0.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test median filter
    println!("5️⃣  Applying median filter (kernel=5)...");
    let start = Instant::now();
    let median = MedianFilter::new(5)?;
    let filtered = median.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&filtered, &format!("{}/05_median_filter_5x5.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test bilateral filter
    println!("6️⃣  Applying bilateral filter (preserves edges)...");
    let start = Instant::now();
    let bilateral = BilateralFilter::new(9, 75.0, 75.0)?;
    let bilateral_result = bilateral.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&bilateral_result, &format!("{}/06_bilateral_filter.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // ========================================================================
    // NEW FILTERING OPERATIONS
    // ========================================================================

    // Test Box Filter
    println!("6️⃣a  Applying box filter (fast averaging - kernel=7)...");
    let start = Instant::now();
    let box_filter = BoxFilter::new(7)?;
    let box_result = box_filter.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&box_result, &format!("{}/06a_box_filter_7x7.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Laplacian Filter (4-connected)
    println!("6️⃣b  Applying Laplacian filter (4-connected, normalized)...");
    let start = Instant::now();
    let laplacian4 = Laplacian::new(LaplacianKernel::Type4, true)?;
    let laplacian4_result = laplacian4.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&laplacian4_result, &format!("{}/06b_laplacian_type4.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Laplacian Filter (8-connected)
    println!("6️⃣c  Applying Laplacian filter (8-connected, normalized)...");
    let start = Instant::now();
    let laplacian8 = Laplacian::new(LaplacianKernel::Type8, true)?;
    let laplacian8_result = laplacian8.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&laplacian8_result, &format!("{}/06c_laplacian_type8.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Scharr Filter
    println!("6️⃣d  Applying Scharr filter (improved edge detection)...");
    let start = Instant::now();
    let scharr = Scharr::new(false);
    let scharr_result = scharr.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&scharr_result, &format!("{}/06d_scharr_edges.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Unsharp Masking (subtle sharpening)
    println!("6️⃣e  Applying unsharp masking (subtle sharpening - amount=1.0)...");
    let start = Instant::now();
    let unsharp1 = UnsharpMask::new(1.0, 2.0, 2)?;
    let unsharp1_result = unsharp1.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&unsharp1_result, &format!("{}/06e_unsharp_mask_subtle.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Unsharp Masking (strong sharpening)
    println!("6️⃣f  Applying unsharp masking (strong sharpening - amount=1.8)...");
    let start = Instant::now();
    let unsharp2 = UnsharpMask::new(1.8, 2.5, 3)?;
    let unsharp2_result = unsharp2.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&unsharp2_result, &format!("{}/06f_unsharp_mask_strong.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Gaussian Derivatives (X direction)
    println!("6️⃣g  Applying Gaussian derivative (first order, X direction)...");
    let start = Instant::now();
    let gauss_deriv_x = GaussianDerivative::new(1.5, DerivativeOrder::First, DerivativeDirection::X)?;
    let gauss_deriv_x_result = gauss_deriv_x.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&gauss_deriv_x_result, &format!("{}/06g_gaussian_derivative_dx.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Gaussian Derivatives (Y direction)
    println!("6️⃣h  Applying Gaussian derivative (first order, Y direction)...");
    let start = Instant::now();
    let gauss_deriv_y = GaussianDerivative::new(1.5, DerivativeOrder::First, DerivativeDirection::Y)?;
    let gauss_deriv_y_result = gauss_deriv_y.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&gauss_deriv_y_result, &format!("{}/06h_gaussian_derivative_dy.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Gaussian Derivatives (XX - second order)
    println!("6️⃣i  Applying Gaussian derivative (second order, XX)...");
    let start = Instant::now();
    let gauss_deriv_xx = GaussianDerivative::new(1.5, DerivativeOrder::Second, DerivativeDirection::XX)?;
    let gauss_deriv_xx_result = gauss_deriv_xx.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&gauss_deriv_xx_result, &format!("{}/06i_gaussian_derivative_dxx.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Anisotropic Diffusion (light smoothing)
    println!("6️⃣j  Applying anisotropic diffusion (light - 15 iterations)...");
    let start = Instant::now();
    let aniso1 = AnisotropicDiffusion::new(15, 15.0, 0.2, DiffusionOption::Option2)?;
    let aniso1_result = aniso1.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&aniso1_result, &format!("{}/06j_anisotropic_diffusion_light.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Anisotropic Diffusion (moderate smoothing)
    println!("6️⃣k  Applying anisotropic diffusion (moderate - 30 iterations)...");
    let start = Instant::now();
    let aniso2 = AnisotropicDiffusion::new(30, 15.0, 0.2, DiffusionOption::Option2)?;
    let aniso2_result = aniso2.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&aniso2_result, &format!("{}/06k_anisotropic_diffusion_moderate.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // ========================================================================
    // END NEW FILTERING OPERATIONS
    // ========================================================================

    // Test Sobel edge detection
    println!("7️⃣  Detecting edges with Sobel...");
    let start = Instant::now();
    let sobel = Sobel::new();
    let edges_sobel = sobel.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_sobel, &format!("{}/07_sobel_edges.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    // Test Canny edge detection with different thresholds
    println!("8️⃣  Detecting edges with Canny (low=50, high=150)...");
    let start = Instant::now();
    let canny1 = Canny::new(50.0, 150.0)?;
    let edges_canny1 = canny1.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_canny1, &format!("{}/08_canny_edges_50_150.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    println!("9️⃣  Detecting edges with Canny (low=100, high=200)...");
    let start = Instant::now();
    let canny2 = Canny::new(100.0, 200.0)?;
    let edges_canny2 = canny2.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_canny2, &format!("{}/09_canny_edges_100_200.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    // ========================================================================
    // NEW EDGE DETECTION OPERATIONS
    // ========================================================================

    println!("9️⃣a  Detecting edges with Prewitt (simpler than Sobel)...");
    let start = Instant::now();
    let prewitt = Prewitt::new();
    let edges_prewitt = prewitt.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_prewitt, &format!("{}/09a_prewitt_edges.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    println!("9️⃣b  Detecting blobs with Laplacian of Gaussian (LoG, sigma=2.0)...");
    let start = Instant::now();
    let log = LaplacianOfGaussian::new(2.0, 5)?;
    let edges_log = log.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_log, &format!("{}/09b_log_edges_sigma2.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    println!("9️⃣c  Multi-scale edge detection with Difference of Gaussians (DoG, sigma1=1.0, sigma2=1.6)...");
    let start = Instant::now();
    let dog = DifferenceOfGaussians::new(1.0, 1.6, 5)?;
    let edges_dog = dog.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_dog, &format!("{}/09c_dog_edges_1.0_1.6.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    println!("9️⃣d  Multi-scale edge detection with DoG using ratio (sigma=1.5, ratio=1.6)...");
    let start = Instant::now();
    let dog_ratio = DifferenceOfGaussians::with_ratio(1.5, 1.6, 7)?;
    let edges_dog_ratio = dog_ratio.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_dog_ratio, &format!("{}/09d_dog_edges_ratio.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    println!("9️⃣e  Zero-crossing detection with Laplacian (threshold=10.0)...");
    let start = Instant::now();
    let zc_laplacian = ZeroCrossing::with_laplacian(10.0)?;
    let edges_zc_laplacian = zc_laplacian.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_zc_laplacian, &format!("{}/09e_zero_crossing_laplacian.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    println!("9️⃣f  Zero-crossing detection with LoG (sigma=2.0, threshold=15.0)...");
    let start = Instant::now();
    let zc_log = ZeroCrossing::with_log(2.0, 15.0)?;
    let edges_zc_log = zc_log.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&edges_zc_log, &format!("{}/09f_zero_crossing_log.png", output_dir))?;
    println!("   ✓ Detected and saved ({:.2}ms)\n", elapsed);

    // ========================================================================
    // END NEW EDGE DETECTION OPERATIONS
    // ========================================================================

    // Test binary threshold
    println!("🔟  Applying binary threshold (value=128)...");
    let start = Instant::now();
    use imgproc::ops::segment::ThresholdType;
    let threshold = Threshold::new(128, 255, ThresholdType::Binary);
    let binary = threshold.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&binary, &format!("{}/10_binary_threshold_128.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test Otsu's automatic threshold
    println!("1️⃣1️⃣  Applying Otsu's automatic threshold...");
    let start = Instant::now();
    let otsu = OtsuThreshold::new();
    let threshold_value = otsu.calculate_threshold(&img_gray);
    let binary_otsu = otsu.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&binary_otsu, &format!("{}/11_otsu_threshold_{}.png", output_dir, threshold_value))?;
    println!("   ✓ Optimal threshold: {} ({:.2}ms)\n", threshold_value, elapsed);

    // Test adaptive threshold
    println!("1️⃣2️⃣  Applying adaptive threshold (Mean method)...");
    let start = Instant::now();
    use imgproc::ops::segment::AdaptiveMethod;
    let adaptive = AdaptiveThreshold::new(255, AdaptiveMethod::Mean, 11, 2.0)?;
    let binary_adaptive = adaptive.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&binary_adaptive, &format!("{}/12_adaptive_threshold_mean.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣3️⃣  Applying adaptive threshold (Gaussian method)...");
    let start = Instant::now();
    let adaptive_gauss = AdaptiveThreshold::new(255, AdaptiveMethod::Gaussian, 11, 2.0)?;
    let binary_adaptive_gauss = adaptive_gauss.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&binary_adaptive_gauss, &format!("{}/13_adaptive_threshold_gaussian.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // Test morphological operations (using Otsu binary image)
    println!("1️⃣4️⃣  Applying erosion (5x5 rectangle)...");
    let start = Instant::now();
    use imgproc::ops::morphology::StructuringElement;
    let erode = Erode::new(5, StructuringElement::Rectangle)?;
    let eroded = erode.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&eroded, &format!("{}/14_erosion_5x5_rect.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣5️⃣  Applying dilation (5x5 rectangle)...");
    let start = Instant::now();
    let dilate = Dilate::new(5, StructuringElement::Rectangle)?;
    let dilated = dilate.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&dilated, &format!("{}/15_dilation_5x5_rect.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣6️⃣  Applying opening (erosion → dilation)...");
    let start = Instant::now();
    let opening = MorphOpen::new(5, StructuringElement::Ellipse)?;
    let opened = opening.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&opened, &format!("{}/16_opening_5x5_ellipse.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣7️⃣  Applying closing (dilation → erosion)...");
    let start = Instant::now();
    let closing = MorphClose::new(5, StructuringElement::Ellipse)?;
    let closed = closing.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&closed, &format!("{}/17_closing_5x5_ellipse.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // ========================================================================
    // NEW MORPHOLOGICAL OPERATIONS
    // ========================================================================

    println!("1️⃣7️⃣a  Applying morphological gradient (basic - dilation - erosion)...");
    let start = Instant::now();
    use imgproc::ops::morphology::GradientType;
    let morph_grad_basic = MorphGradient::new(5, StructuringElement::Ellipse, GradientType::Basic)?;
    let grad_basic = morph_grad_basic.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&grad_basic, &format!("{}/17a_morph_gradient_basic.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣7️⃣b  Applying morphological gradient (internal - original - erosion)...");
    let start = Instant::now();
    let morph_grad_internal = MorphGradient::new(3, StructuringElement::Ellipse, GradientType::Internal)?;
    let grad_internal = morph_grad_internal.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&grad_internal, &format!("{}/17b_morph_gradient_internal.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣7️⃣c  Applying morphological gradient (external - dilation - original)...");
    let start = Instant::now();
    let morph_grad_external = MorphGradient::new(3, StructuringElement::Ellipse, GradientType::External)?;
    let grad_external = morph_grad_external.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&grad_external, &format!("{}/17c_morph_gradient_external.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣7️⃣d  Applying white top-hat transform (extracts bright features)...");
    let start = Instant::now();
    use imgproc::ops::morphology::TopHatVariant;
    let white_tophat = TopHat::new(15, StructuringElement::Ellipse, TopHatVariant::White)?;
    let tophat_white = white_tophat.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&tophat_white, &format!("{}/17d_tophat_white.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣7️⃣e  Applying black top-hat transform (extracts dark features)...");
    let start = Instant::now();
    let black_tophat = TopHat::new(15, StructuringElement::Ellipse, TopHatVariant::Black)?;
    let tophat_black = black_tophat.apply_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&tophat_black, &format!("{}/17e_tophat_black.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣7️⃣f  Applying skeletonization (Zhang-Suen algorithm)...");
    let start = Instant::now();
    use imgproc::ops::morphology::SkeletonMethod;
    let skeletonize = Skeletonize::new(SkeletonMethod::ZhangSuen, Some(100))?;
    let skeleton = skeletonize.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&skeleton, &format!("{}/17f_skeleton_zhangsuen.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣7️⃣g  Applying thinning (simplified iterative)...");
    let start = Instant::now();
    let thinning = Thinning::new(50)?;
    let thinned = thinning.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&thinned, &format!("{}/17g_thinning.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    println!("1️⃣7️⃣h  Applying hit-or-miss transform (corner detection)...");
    let start = Instant::now();
    // Template to detect top-left corners in binary images
    let corner_template = vec![
        vec![-1, -1, -1],
        vec![-1,  1,  1],
        vec![-1,  1,  0],
    ];
    let hit_or_miss = HitOrMiss::new(corner_template)?;
    let corners = hit_or_miss.apply_gray(&binary_otsu)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&corners, &format!("{}/17h_hit_or_miss_corners.png", output_dir))?;
    println!("   ✓ Applied and saved ({:.2}ms)\n", elapsed);

    // ========================================================================
    // END NEW MORPHOLOGICAL OPERATIONS
    // ========================================================================

    // Test geometric transformations
    println!("1️⃣8️⃣  Resizing image (50% scale with bilinear)...");
    let start = Instant::now();
    use imgproc::ops::geometry::Interpolation;
    let new_width = img_gray.width() / 2;
    let new_height = img_gray.height() / 2;
    let resize = Resize::new(new_width, new_height, Interpolation::Bilinear);
    let resized = resize.resize_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&resized, &format!("{}/18_resized_50percent_bilinear.png", output_dir))?;
    println!("   ✓ Resized {}x{} → {}x{} ({:.2}ms)\n",
             img_gray.width(), img_gray.height(),
             resized.width(), resized.height(), elapsed);

    println!("1️⃣9️⃣  Resizing image (200% scale with nearest)...");
    let start = Instant::now();
    let resize_up = Resize::new(img_gray.width() * 2, img_gray.height() * 2, Interpolation::Nearest);
    let resized_up = resize_up.resize_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&resized_up, &format!("{}/19_resized_200percent_nearest.png", output_dir))?;
    println!("   ✓ Resized {}x{} → {}x{} ({:.2}ms)\n",
             img_gray.width(), img_gray.height(),
             resized_up.width(), resized_up.height(), elapsed);

    // Test crop
    println!("2️⃣0️⃣  Cropping center region...");
    let start = Instant::now();
    let crop_w = img_gray.width() / 2;
    let crop_h = img_gray.height() / 2;
    let crop_x = img_gray.width() / 4;
    let crop_y = img_gray.height() / 4;
    let crop_rect = Rect::new(crop_x, crop_y, crop_w, crop_h);
    let crop = Crop::new(crop_rect);
    let cropped = crop.crop_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&cropped, &format!("{}/20_cropped_center.png", output_dir))?;
    println!("   ✓ Cropped region {}x{} at ({}, {}) ({:.2}ms)\n",
             cropped.width(), cropped.height(), crop_x, crop_y, elapsed);

    // Test rotate
    println!("2️⃣1️⃣  Rotating image (45 degrees)...");
    let start = Instant::now();
    let rotate = Rotate::new(45.0);
    let rotated = rotate.rotate_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&rotated, &format!("{}/21_rotated_45deg.png", output_dir))?;
    println!("   ✓ Rotated and saved ({:.2}ms)\n", elapsed);

    println!("2️⃣2️⃣  Rotating image (90 degrees)...");
    let start = Instant::now();
    let rotate90 = Rotate::new(90.0);
    let rotated90 = rotate90.rotate_gray(&img_gray)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    imgproc::io::write_image_gray(&rotated90, &format!("{}/22_rotated_90deg.png", output_dir))?;
    println!("   ✓ Rotated and saved ({:.2}ms)\n", elapsed);

    // Test composite operations
    println!("2️⃣3️⃣  Creating artistic effect (blur + edges overlay)...");
    let start = Instant::now();
    // Blur the image
    let blur_artistic = GaussianBlur::new(7, 2.0)?;
    let blurred_artistic = blur_artistic.apply_gray(&img_gray)?;
    // Detect edges
    let edges_artistic = sobel.apply_gray(&img_gray)?;
    // Combine: Use edges as a "sketch" overlay
    // For demonstration, we'll just save both, but you could combine them
    imgproc::io::write_image_gray(&blurred_artistic, &format!("{}/23a_artistic_blurred.png", output_dir))?;
    imgproc::io::write_image_gray(&edges_artistic, &format!("{}/23b_artistic_edges.png", output_dir))?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("   ✓ Created artistic variations ({:.2}ms)\n", elapsed);

    // Calculate and display image statistics
    println!("2️⃣4️⃣  Calculating image statistics...");
    let mut min_val = 255u8;
    let mut max_val = 0u8;
    let mut sum = 0u64;
    let mut histogram = vec![0u32; 256];

    for pixel in img_gray.pixels() {
        let val = pixel.value;
        min_val = min_val.min(val);
        max_val = max_val.max(val);
        sum += val as u64;
        histogram[val as usize] += 1;
    }

    let pixel_count = img_gray.width() as u64 * img_gray.height() as u64;
    let mean = sum / pixel_count;

    // Find median
    let mut cumsum = 0u32;
    let mut median = 0u8;
    let half_pixels = pixel_count as u32 / 2;
    for (i, &count) in histogram.iter().enumerate() {
        cumsum += count;
        if cumsum >= half_pixels {
            median = i as u8;
            break;
        }
    }

    println!("   📊 Statistics:");
    println!("      - Dimensions: {}x{}", img_gray.width(), img_gray.height());
    println!("      - Min value: {}", min_val);
    println!("      - Max value: {}", max_val);
    println!("      - Mean value: {}", mean);
    println!("      - Median value: {}", median);
    println!("      - Total pixels: {}\n", pixel_count);

    // Final summary
    println!("===========================================");
    println!("✅ All 49 operations completed successfully!");
    println!("===========================================");
    println!("\n📁 Output files saved to: {}/", output_dir);
    println!("\nGenerated images:");
    println!("  • Original RGB and grayscale conversions");
    println!("  • Gaussian blur (2 variants)");
    println!("  • Median and bilateral filters");
    println!("  • NEW: Box filter (fast averaging)");
    println!("  • NEW: Laplacian edge detection (2 variants: 4-connected, 8-connected)");
    println!("  • NEW: Scharr edge detection (improved Sobel)");
    println!("  • NEW: Unsharp masking (2 variants: subtle and strong sharpening)");
    println!("  • NEW: Gaussian derivatives (3 variants: dX, dY, dXX)");
    println!("  • NEW: Anisotropic diffusion (2 variants: light and moderate smoothing)");
    println!("  • Edge detection (Sobel, 2x Canny)");
    println!("  • NEW: Prewitt edge detection");
    println!("  • NEW: Laplacian of Gaussian (LoG) blob detection");
    println!("  • NEW: Difference of Gaussians (DoG) multi-scale edge detection (2 variants)");
    println!("  • NEW: Zero-crossing detection (2 variants: Laplacian and LoG)");
    println!("  • Thresholding (binary, Otsu, 2x adaptive)");
    println!("  • Morphology (erosion, dilation, opening, closing)");
    println!("  • NEW: Morphological gradient (3 variants: basic, internal, external)");
    println!("  • NEW: Top-hat transforms (white and black)");
    println!("  • NEW: Skeletonization (Zhang-Suen algorithm)");
    println!("  • NEW: Thinning (simplified iterative)");
    println!("  • NEW: Hit-or-Miss transform (corner detection)");
    println!("  • Geometric transforms (2x resize, crop, 2x rotate)");
    println!("  • Artistic effects");
    println!("\n🎉 Image processing library test complete!");
    println!("   Total new filtering operations demonstrated: 11");
    println!("   - 1x Box Filter");
    println!("   - 2x Laplacian (Type4, Type8)");
    println!("   - 1x Scharr");
    println!("   - 2x Unsharp Mask (subtle, strong)");
    println!("   - 3x Gaussian Derivatives (dX, dY, dXX)");
    println!("   - 2x Anisotropic Diffusion (light, moderate)");
    println!("\n   Total new edge detection operations demonstrated: 6");
    println!("   - 1x Prewitt");
    println!("   - 1x Laplacian of Gaussian (LoG)");
    println!("   - 2x Difference of Gaussians (DoG)");
    println!("   - 2x Zero-Crossing Detection (Laplacian and LoG)");
    println!("\n   Total new morphological operations demonstrated: 8");
    println!("   - 3x Morphological Gradient (basic, internal, external)");
    println!("   - 2x Top-Hat Transform (white, black)");
    println!("   - 1x Skeletonization (Zhang-Suen)");
    println!("   - 1x Thinning");
    println!("   - 1x Hit-or-Miss Transform");

    Ok(())
}
