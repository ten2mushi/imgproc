//! Utility functions for CLI

use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

/// Creates a progress bar with standard styling
pub fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("#>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Prints success message
pub fn print_success(message: &str) {
    println!("{} {}", "✓".green().bold(), message);
}

/// Prints error message
pub fn print_error(message: &str) {
    eprintln!("{} {}", "✗".red().bold(), message);
}

/// Prints info message
pub fn print_info(message: &str) {
    println!("{} {}", "ℹ".cyan().bold(), message);
}

/// Outputs JSON result
pub fn output_json(success: bool, data: serde_json::Value) {
    let result = serde_json::json!({
        "success": success,
        "data": data
    });
    println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
}

/// Ensures output directory exists
pub fn ensure_output_dir(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
