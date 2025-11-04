//! imgproc CLI - Command-line interface for image processing
//!
//! This binary provides a comprehensive command-line interface for all image
//! processing operations, including individual operations and pipeline execution.

use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;

mod commands;
mod utils;

use commands::{convert, edges, filters, morphology, pipeline, resize, segment};

/// imgproc - High-performance image processing toolkit
#[derive(Parser)]
#[command(name = "imgproc")]
#[command(author = "Image Processing Team")]
#[command(version = "0.1.0")]
#[command(about = "High-performance image processing toolkit", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Enable JSON output mode
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert color spaces
    #[command(alias = "conv")]
    Convert(convert::ConvertArgs),

    /// Resize images
    Resize(resize::ResizeArgs),

    /// Apply filters (blur, median, bilateral)
    #[command(alias = "filt")]
    Filter(filters::FilterArgs),

    /// Detect edges (Sobel, Canny)
    Edges(edges::EdgesArgs),

    /// Segmentation and thresholding
    #[command(alias = "seg")]
    Segment(segment::SegmentArgs),

    /// Morphological operations
    #[command(alias = "morph")]
    Morphology(morphology::MorphologyArgs),

    /// Execute YAML pipeline
    #[command(alias = "pipe")]
    Pipeline(pipeline::PipelineArgs),

    /// List available operations
    List,

    /// Validate pipeline configuration
    Validate {
        /// Pipeline YAML file to validate
        #[arg(value_name = "FILE")]
        pipeline: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    // Execute command
    let result = match &cli.command {
        Commands::Convert(args) => convert::execute(args, cli.json),
        Commands::Resize(args) => resize::execute(args, cli.json),
        Commands::Filter(args) => filters::execute(args, cli.json),
        Commands::Edges(args) => edges::execute(args, cli.json),
        Commands::Segment(args) => segment::execute(args, cli.json),
        Commands::Morphology(args) => morphology::execute(args, cli.json),
        Commands::Pipeline(args) => pipeline::execute(args, cli.json),
        Commands::List => list_operations(),
        Commands::Validate { pipeline } => validate_pipeline(pipeline),
    };

    // Handle result
    if let Err(e) = result {
        if cli.json {
            let error = serde_json::json!({
                "success": false,
                "error": e.to_string()
            });
            println!("{}", serde_json::to_string_pretty(&error).unwrap_or_default());
        } else {
            eprintln!("{} {}", "Error:".red().bold(), e);
        }
        std::process::exit(1);
    }
}

fn list_operations() -> anyhow::Result<()> {
    println!("{}", "Available Operations:".cyan().bold());
    println!();

    println!("{}", "Color Conversion:".yellow().bold());
    println!("  • grayscale     - Convert to grayscale");
    println!();

    println!("{}", "Geometric:".yellow().bold());
    println!("  • resize        - Resize images with various interpolation methods");
    println!("  • crop          - Crop image regions");
    println!("  • rotate        - Rotate images by angle");
    println!();

    println!("{}", "Filters:".yellow().bold());
    println!("  • gaussian_blur - Gaussian smoothing");
    println!("  • median        - Median filter for noise reduction");
    println!("  • bilateral     - Edge-preserving bilateral filter");
    println!();

    println!("{}", "Edge Detection:".yellow().bold());
    println!("  • sobel         - Sobel edge detection");
    println!("  • canny         - Canny edge detection");
    println!();

    println!("{}", "Segmentation:".yellow().bold());
    println!("  • threshold     - Simple thresholding");
    println!("  • otsu          - Otsu automatic thresholding");
    println!("  • adaptive      - Adaptive thresholding");
    println!();

    println!("{}", "Morphology:".yellow().bold());
    println!("  • erode         - Morphological erosion");
    println!("  • dilate        - Morphological dilation");
    println!("  • opening       - Erosion followed by dilation");
    println!("  • closing       - Dilation followed by erosion");
    println!();

    println!("{}", "Pipeline:".yellow().bold());
    println!("  • pipeline      - Execute YAML-configured pipeline");
    println!();

    Ok(())
}

fn validate_pipeline(path: &PathBuf) -> anyhow::Result<()> {
    println!("{} {}", "Validating pipeline:".cyan(), path.display());

    let pipeline = imgproc_pipeline::Pipeline::from_file(path)?;

    println!("  {} {}", "Name:".green(), pipeline.name());
    if let Some(desc) = pipeline.description() {
        println!("  {} {}", "Description:".green(), desc);
    }
    println!("  {} {}", "Version:".green(), pipeline.version());

    // Validate
    pipeline.validate()?;

    println!();
    println!("{}", "✓ Pipeline is valid".green().bold());

    Ok(())
}
