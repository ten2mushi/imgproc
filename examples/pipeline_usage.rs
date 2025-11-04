//! Example demonstrating pipeline usage
//!
//! This example shows how to use the YAML-based pipeline system to process images.

use imgproc_pipeline::Pipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("===================================================");
    println!("    imgproc Pipeline System Demonstration");
    println!("===================================================\n");

    let input_path = "input/IMG-20251103-WA0001.jpg";
    let output_dir = "output";

    // Ensure output directory exists
    std::fs::create_dir_all(output_dir)?;

    // Example 1: Basic Preprocessing Pipeline
    println!("1. Running Basic Preprocessing Pipeline...");
    println!("   - Convert to grayscale");
    println!("   - Resize to 800x800");
    println!("   - Apply Gaussian blur\n");

    let pipeline1 = Pipeline::from_file("pipelines/basic_preprocessing.yaml")?;
    println!("   Loaded pipeline: {}", pipeline1.name());
    if let Some(desc) = pipeline1.description() {
        println!("   Description: {}\n", desc);
    }

    // Validate before execution
    pipeline1.validate()?;
    println!("   ✓ Pipeline validation passed");

    let output1 = format!("{}/pipeline_basic.png", output_dir);
    pipeline1.execute(input_path, &output1)?;
    println!("   ✓ Pipeline executed successfully");
    println!("   Output: {}\n", output1);

    // Example 2: Edge Detection Pipeline
    println!("2. Running Edge Detection Pipeline...");
    println!("   - Convert to grayscale");
    println!("   - Resize to 1024x1024");
    println!("   - Gaussian blur");
    println!("   - Canny edge detection\n");

    let pipeline2 = Pipeline::from_file("pipelines/edge_detection.yaml")?;
    println!("   Loaded pipeline: {}", pipeline2.name());
    pipeline2.validate()?;
    println!("   ✓ Pipeline validation passed");

    let output2 = format!("{}/pipeline_edges.png", output_dir);
    pipeline2.execute(input_path, &output2)?;
    println!("   ✓ Pipeline executed successfully");
    println!("   Output: {}\n", output2);

    // Example 3: Document Scanning Pipeline
    println!("3. Running Document Scanning Pipeline...");
    println!("   - Convert to grayscale");
    println!("   - Resize to A4 dimensions");
    println!("   - Median filter for noise");
    println!("   - Adaptive threshold");
    println!("   - Morphological closing and opening\n");

    let pipeline3 = Pipeline::from_file("pipelines/document_scanning.yaml")?;
    println!("   Loaded pipeline: {}", pipeline3.name());
    pipeline3.validate()?;
    println!("   ✓ Pipeline validation passed");

    let output3 = format!("{}/pipeline_document.png", output_dir);
    pipeline3.execute(input_path, &output3)?;
    println!("   ✓ Pipeline executed successfully");
    println!("   Output: {}\n", output3);

    // Example 4: Photo Enhancement Pipeline
    println!("4. Running Photo Enhancement Pipeline...");
    println!("   - Convert to grayscale");
    println!("   - Resize to 1920x1920");
    println!("   - Bilateral filter (edge-preserving)");
    println!("   - Otsu auto-threshold\n");

    let pipeline4 = Pipeline::from_file("pipelines/photo_enhancement.yaml")?;
    println!("   Loaded pipeline: {}", pipeline4.name());
    pipeline4.validate()?;
    println!("   ✓ Pipeline validation passed");

    let output4 = format!("{}/pipeline_enhanced.png", output_dir);
    pipeline4.execute(input_path, &output4)?;
    println!("   ✓ Pipeline executed successfully");
    println!("   Output: {}\n", output4);

    // Example 5: Morphological Analysis Pipeline
    println!("5. Running Morphological Analysis Pipeline...");
    println!("   - Convert to grayscale");
    println!("   - Resize to 512x512");
    println!("   - Threshold");
    println!("   - Erode, Dilate, Opening, Closing\n");

    let pipeline5 = Pipeline::from_file("pipelines/morphological_analysis.yaml")?;
    println!("   Loaded pipeline: {}", pipeline5.name());
    pipeline5.validate()?;
    println!("   ✓ Pipeline validation passed");

    let output5 = format!("{}/pipeline_morph.png", output_dir);
    pipeline5.execute(input_path, &output5)?;
    println!("   ✓ Pipeline executed successfully");
    println!("   Output: {}\n", output5);

    println!("===================================================");
    println!("✅ All pipelines executed successfully!");
    println!("===================================================");
    println!("\nGenerated outputs:");
    println!("  • {}", output1);
    println!("  • {}", output2);
    println!("  • {}", output3);
    println!("  • {}", output4);
    println!("  • {}", output5);
    println!("\nPipeline Features Demonstrated:");
    println!("  ✓ YAML-based configuration");
    println!("  ✓ Variable interpolation");
    println!("  ✓ Sequential operation execution");
    println!("  ✓ Multiple operation types");
    println!("  ✓ Error handling strategies");
    println!("  ✓ Pipeline validation");

    Ok(())
}
