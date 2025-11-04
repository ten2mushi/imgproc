use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct PipelineArgs {
    /// Pipeline YAML configuration file
    #[arg(short = 'c', long = "config", value_name = "FILE")]
    pub pipeline: PathBuf,

    /// Input image or directory
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output image or directory
    #[arg(short, long)]
    pub output: PathBuf,

    /// Process directory (batch mode)
    #[arg(long)]
    pub batch: bool,
}

pub fn execute(args: &PipelineArgs, json_output: bool) -> anyhow::Result<()> {
    use crate::utils::{create_progress_bar, output_json, print_info, print_success};

    // Load and validate pipeline
    let pipeline = imgproc_pipeline::Pipeline::from_file(&args.pipeline)?;

    if !json_output {
        print_info(&format!("Loaded pipeline: {}", pipeline.name()));
        if let Some(desc) = pipeline.description() {
            println!("  {}", desc);
        }
    }

    pipeline.validate()?;

    if !json_output {
        print_info("Pipeline validated successfully");
    }

    if args.batch {
        // Batch processing
        if !json_output {
            print_info(&format!("Processing batch from: {}", args.input.display()));
        }

        let results = pipeline.execute_batch(&args.input, &args.output)?;

        let success_count = results.iter().filter(|r| r.is_ok()).count();
        let total_count = results.len();

        if json_output {
            output_json(
                true,
                serde_json::json!({
                    "mode": "batch",
                    "total": total_count,
                    "successful": success_count,
                    "failed": total_count - success_count
                }),
            );
        } else {
            if success_count == total_count {
                print_success(&format!("Processed {} images successfully", total_count));
            } else {
                println!("Processed {}/{} images successfully", success_count, total_count);
            }
        }
    } else {
        // Single image processing
        if !json_output {
            let pb = create_progress_bar(1, "Processing");
            pipeline.execute(&args.input, &args.output)?;
            pb.finish_with_message("Done");
        } else {
            pipeline.execute(&args.input, &args.output)?;
        }

        if json_output {
            output_json(
                true,
                serde_json::json!({
                    "mode": "single",
                    "output": args.output.display().to_string()
                }),
            );
        } else {
            print_success(&format!("Saved to: {}", args.output.display()));
        }
    }

    Ok(())
}
