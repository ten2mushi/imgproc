//! Color conversion command

use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct ConvertArgs {
    /// Input image file
    #[arg(short, long, value_name = "FILE")]
    pub input: PathBuf,

    /// Output image file
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// Convert to grayscale
    #[arg(long, group = "conversion")]
    pub grayscale: bool,
}

pub fn execute(args: &ConvertArgs, json_output: bool) -> anyhow::Result<()> {
    use crate::utils::{ensure_output_dir, output_json, print_info, print_success};

    if !json_output {
        print_info(&format!("Converting: {}", args.input.display()));
    }

    ensure_output_dir(&args.output)?;

    if args.grayscale {
        // Load RGB image
        let img = imgproc_io::read_image_rgb(&args.input)?;

        // Convert to grayscale
        let gray = imgproc_ops::color::rgb_to_gray(&img)?;

        // Save
        imgproc_io::write_image_gray(&gray, &args.output)?;

        if json_output {
            output_json(
                true,
                serde_json::json!({
                    "operation": "grayscale",
                    "output": args.output.display().to_string()
                }),
            );
        } else {
            print_success(&format!("Saved to: {}", args.output.display()));
        }
    }

    Ok(())
}
