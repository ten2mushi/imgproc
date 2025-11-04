use clap::{Args, ValueEnum};
use std::path::PathBuf;

#[derive(Args)]
pub struct ResizeArgs {
    /// Input image file
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output image file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Target width
    #[arg(short, long)]
    pub width: u32,

    /// Target height
    #[arg(long)]
    pub height: u32,

    /// Interpolation method
    #[arg(long, default_value = "bilinear")]
    pub method: Interpolation,
}

#[derive(Clone, ValueEnum)]
pub enum Interpolation {
    Nearest,
    Bilinear,
}

pub fn execute(args: &ResizeArgs, json_output: bool) -> anyhow::Result<()> {
    use crate::utils::{ensure_output_dir, output_json, print_info, print_success};

    if !json_output {
        print_info(&format!("Resizing: {} to {}x{}", args.input.display(), args.width, args.height));
    }

    ensure_output_dir(&args.output)?;

    let img = imgproc_io::read_image_gray(&args.input)?;

    let interp = match args.method {
        Interpolation::Nearest => imgproc_ops::geometry::Interpolation::Nearest,
        Interpolation::Bilinear => imgproc_ops::geometry::Interpolation::Bilinear,
    };

    let resizer = imgproc_ops::geometry::Resize::new(args.width, args.height, interp);
    let resized = resizer.resize_gray(&img)?;

    imgproc_io::write_image_gray(&resized, &args.output)?;

    if json_output {
        output_json(
            true,
            serde_json::json!({
                "operation": "resize",
                "width": args.width,
                "height": args.height,
                "output": args.output.display().to_string()
            }),
        );
    } else {
        print_success(&format!("Saved to: {}", args.output.display()));
    }

    Ok(())
}
