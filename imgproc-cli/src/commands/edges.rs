//! Edge detection commands

use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Args)]
pub struct EdgesArgs {
    #[command(subcommand)]
    pub command: EdgesCommands,
}

#[derive(Subcommand)]
pub enum EdgesCommands {
    /// Sobel edge detection
    Sobel {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Canny edge detection
    Canny {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "50.0")]
        low_threshold: f32,
        #[arg(long, default_value = "150.0")]
        high_threshold: f32,
    },
}

pub fn execute(args: &EdgesArgs, json_output: bool) -> anyhow::Result<()> {
    use crate::utils::{ensure_output_dir, output_json, print_info, print_success};

    match &args.command {
        EdgesCommands::Sobel { input, output } => {
            if !json_output {
                print_info("Detecting edges with Sobel");
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let sobel = imgproc_ops::edges::Sobel::new();
            let result = sobel.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "method": "sobel",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
        EdgesCommands::Canny { input, output, low_threshold, high_threshold } => {
            if !json_output {
                print_info(&format!("Detecting edges with Canny ({}/{})", low_threshold, high_threshold));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let canny = imgproc_ops::edges::Canny::new(*low_threshold, *high_threshold)?;
            let result = canny.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "method": "canny",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
    }

    Ok(())
}
