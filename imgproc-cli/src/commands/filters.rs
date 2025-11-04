//! Filter commands

use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Args)]
pub struct FilterArgs {
    #[command(subcommand)]
    pub command: FilterCommands,
}

#[derive(Subcommand)]
pub enum FilterCommands {
    /// Gaussian blur
    Gaussian {
        /// Input image
        #[arg(short, long)]
        input: PathBuf,
        /// Output image
        #[arg(short, long)]
        output: PathBuf,
        /// Kernel size (odd number)
        #[arg(short, long, default_value = "5")]
        kernel_size: u32,
        /// Sigma value
        #[arg(short, long, default_value = "1.5")]
        sigma: f32,
    },
    /// Median filter
    Median {
        /// Input image
        #[arg(short, long)]
        input: PathBuf,
        /// Output image
        #[arg(short, long)]
        output: PathBuf,
        /// Kernel size (odd number)
        #[arg(short, long, default_value = "5")]
        kernel_size: u32,
    },
    /// Bilateral filter
    Bilateral {
        /// Input image
        #[arg(short, long)]
        input: PathBuf,
        /// Output image
        #[arg(short, long)]
        output: PathBuf,
        /// Diameter
        #[arg(short, long, default_value = "9")]
        diameter: u32,
        /// Sigma color
        #[arg(long, default_value = "75.0")]
        sigma_color: f32,
        /// Sigma space
        #[arg(long, default_value = "75.0")]
        sigma_space: f32,
    },
}

pub fn execute(args: &FilterArgs, json_output: bool) -> anyhow::Result<()> {
    use crate::utils::{ensure_output_dir, output_json, print_info, print_success};

    match &args.command {
        FilterCommands::Gaussian { input, output, kernel_size, sigma } => {
            if !json_output {
                print_info(&format!("Applying Gaussian blur (k={}, σ={})", kernel_size, sigma));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let blur = imgproc_ops::filters::GaussianBlur::new(*kernel_size, *sigma)?;
            let result = blur.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "filter": "gaussian",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
        FilterCommands::Median { input, output, kernel_size } => {
            if !json_output {
                print_info(&format!("Applying median filter (k={})", kernel_size));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let filter = imgproc_ops::filters::MedianFilter::new(*kernel_size)?;
            let result = filter.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "filter": "median",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
        FilterCommands::Bilateral { input, output, diameter, sigma_color, sigma_space } => {
            if !json_output {
                print_info("Applying bilateral filter");
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let filter = imgproc_ops::filters::BilateralFilter::new(*diameter, *sigma_color, *sigma_space)?;
            let result = filter.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "filter": "bilateral",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
    }

    Ok(())
}
