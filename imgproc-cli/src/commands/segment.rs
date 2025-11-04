use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Args)]
pub struct SegmentArgs {
    #[command(subcommand)]
    pub command: SegmentCommands,
}

#[derive(Subcommand)]
pub enum SegmentCommands {
    /// Simple threshold
    Threshold {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "128")]
        threshold: u8,
        #[arg(long, default_value = "255")]
        max_value: u8,
    },
    /// Otsu automatic threshold
    Otsu {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Adaptive threshold
    Adaptive {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "11")]
        block_size: u32,
        #[arg(long, default_value = "2.0")]
        constant: f32,
    },
}

pub fn execute(args: &SegmentArgs, json_output: bool) -> anyhow::Result<()> {
    use crate::utils::{ensure_output_dir, output_json, print_info, print_success};

    match &args.command {
        SegmentCommands::Threshold { input, output, threshold, max_value } => {
            if !json_output {
                print_info(&format!("Applying threshold ({})", threshold));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let thresh = imgproc_ops::segment::Threshold::new(*threshold, *max_value, imgproc_ops::segment::ThresholdType::Binary);
            let result = thresh.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "method": "threshold",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
        SegmentCommands::Otsu { input, output } => {
            if !json_output {
                print_info("Applying Otsu threshold");
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let otsu = imgproc_ops::segment::OtsuThreshold::new();
            let result = otsu.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "method": "otsu",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
        SegmentCommands::Adaptive { input, output, block_size, constant } => {
            if !json_output {
                print_info(&format!("Applying adaptive threshold (block={})", block_size));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let adaptive = imgproc_ops::segment::AdaptiveThreshold::new(255, imgproc_ops::segment::AdaptiveMethod::Gaussian, *block_size, *constant)?;
            let result = adaptive.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "method": "adaptive",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
    }

    Ok(())
}
