//! Morphological operations

use clap::{Args, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Args)]
pub struct MorphologyArgs {
    #[command(subcommand)]
    pub command: MorphologyCommands,
}

#[derive(Clone, ValueEnum)]
pub enum StructuringElement {
    Rectangle,
    Ellipse,
    Cross,
}

#[derive(Subcommand)]
pub enum MorphologyCommands {
    /// Erosion
    Erode {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "5")]
        kernel_size: u32,
        #[arg(long, default_value = "rectangle")]
        element: StructuringElement,
    },
    /// Dilation
    Dilate {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "5")]
        kernel_size: u32,
        #[arg(long, default_value = "rectangle")]
        element: StructuringElement,
    },
    /// Opening
    Opening {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "5")]
        kernel_size: u32,
        #[arg(long, default_value = "rectangle")]
        element: StructuringElement,
    },
    /// Closing
    Closing {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long, default_value = "5")]
        kernel_size: u32,
        #[arg(long, default_value = "rectangle")]
        element: StructuringElement,
    },
}

fn to_struct_elem(elem: &StructuringElement) -> imgproc_ops::morphology::StructuringElement {
    match elem {
        StructuringElement::Rectangle => imgproc_ops::morphology::StructuringElement::Rectangle,
        StructuringElement::Ellipse => imgproc_ops::morphology::StructuringElement::Ellipse,
        StructuringElement::Cross => imgproc_ops::morphology::StructuringElement::Cross,
    }
}

pub fn execute(args: &MorphologyArgs, json_output: bool) -> anyhow::Result<()> {
    use crate::utils::{ensure_output_dir, output_json, print_info, print_success};

    match &args.command {
        MorphologyCommands::Erode { input, output, kernel_size, element } => {
            if !json_output {
                print_info(&format!("Applying erosion (k={})", kernel_size));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let erode = imgproc_ops::morphology::Erode::new(*kernel_size, to_struct_elem(element))?;
            let result = erode.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "operation": "erode",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
        MorphologyCommands::Dilate { input, output, kernel_size, element } => {
            if !json_output {
                print_info(&format!("Applying dilation (k={})", kernel_size));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let dilate = imgproc_ops::morphology::Dilate::new(*kernel_size, to_struct_elem(element))?;
            let result = dilate.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "operation": "dilate",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
        MorphologyCommands::Opening { input, output, kernel_size, element } => {
            if !json_output {
                print_info(&format!("Applying opening (k={})", kernel_size));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let opening = imgproc_ops::morphology::MorphOpen::new(*kernel_size, to_struct_elem(element))?;
            let result = opening.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "operation": "opening",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
        MorphologyCommands::Closing { input, output, kernel_size, element } => {
            if !json_output {
                print_info(&format!("Applying closing (k={})", kernel_size));
            }

            ensure_output_dir(output)?;
            let img = imgproc_io::read_image_gray(input)?;
            let closing = imgproc_ops::morphology::MorphClose::new(*kernel_size, to_struct_elem(element))?;
            let result = closing.apply_gray(&img)?;
            imgproc_io::write_image_gray(&result, output)?;

            if json_output {
                output_json(true, serde_json::json!({
                    "operation": "closing",
                    "output": output.display().to_string()
                }));
            } else {
                print_success(&format!("Saved to: {}", output.display()));
            }
        }
    }

    Ok(())
}
