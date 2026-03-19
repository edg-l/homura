//! Progress and stats display for CLI output.
//!
//! Provides spinners, progress bars, and a structured stats summary using
//! `indicatif` for bars and `console` for colored text.
//!
//! All output goes to stderr. When stderr is not a terminal, indicatif
//! automatically disables progress bar rendering.

use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

use crate::generate::GenerationStats;

// ── Spinner ───────────────────────────────────────────────────────────────────

/// Create a spinner with the given message.
///
/// Returns an `indicatif::ProgressBar` configured as a spinner.
/// Call `finish_spinner` when done.
pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("valid spinner template"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_message(msg.to_string());
    pb
}

/// Finish a spinner with a green checkmark and a completion message.
pub fn finish_spinner(pb: &ProgressBar, msg: &str) {
    pb.finish_with_message(format!("{} {}", style("").green(), msg));
}

// ── Compile progress ─────────────────────────────────────────────────────────

/// Create a progress bar for kernel compilation.
///
/// Displays: `Compiling kernels [====>   ] 45/86 (eta: 2s)`
pub fn compile_progress(total: usize) -> ProgressBar {
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:30.cyan/dim}] {pos}/{len} ({eta})")
            .expect("valid bar template")
            .progress_chars("=>-"),
    );
    pb.set_message("Compiling kernels");
    pb
}

/// Finish a compile progress bar showing total kernel count and elapsed time.
pub fn finish_compile(pb: &ProgressBar, ms: u64) {
    let total = pb.length().unwrap_or(0);
    pb.finish_with_message(format!(
        "{} Compiled {total} kernels in {ms}ms",
        style("").green(),
    ));
}

// ── Decode progress ───────────────────────────────────────────────────────────

/// Create a progress bar for autoregressive decoding.
///
/// Displays: `  [12/100] 13.2 tok/s`
pub fn decode_progress(max_tokens: usize) -> ProgressBar {
    let pb = ProgressBar::new(max_tokens as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  [{pos}/{len}] {msg}")
            .expect("valid decode template"),
    );
    pb
}

/// Update the decode progress bar with the current step and performance metrics.
pub fn update_decode(pb: &ProgressBar, step: usize, tok_s: f64, _ms: f64) {
    pb.set_position(step as u64);
    pb.set_message(format!("{} tok/s", style(format!("{tok_s:.1}")).green()));
}

/// Finish the decode progress bar, clearing it from display.
pub fn finish_decode(pb: &ProgressBar) {
    pb.finish_and_clear();
}

// ── Stats summary ─────────────────────────────────────────────────────────────

/// Print a structured, colored generation stats summary to stderr.
///
/// Replaces the previous one-liner `── done ──` with a multi-line block:
///
/// ```text
///   ── done ────────────────────────────────
///   prefill   0.22s  (15 tokens)
///   decode    51 tok  in 5.24s
///             9.9 avg  5.7 min  11.9 max tok/s
///             103 ms/tok
///   total     5.47s
///   ────────────────────────────────────────
/// ```
pub fn print_stats(stats: &GenerationStats) {
    let sep = style("────────────────────────────────────────").dim();
    let header = style("── done ────────────────────────────────")
        .bold()
        .magenta();

    let prefill_s = stats.prefill_time.as_secs_f64();
    let n = stats.decode_times.len();

    eprintln!("\n  {header}");

    // Prefill line
    eprintln!(
        "  {}   {}  ({} tokens)",
        style("prefill").dim(),
        style(format!("{prefill_s:.2}s")).cyan(),
        style(stats.prompt_tokens).cyan(),
    );

    if n == 0 {
        let total = prefill_s;
        eprintln!(
            "  {}     {}",
            style("total").dim(),
            style(format!("{total:.2}s")).cyan(),
        );
        eprintln!("  {sep}");
        return;
    }

    // Decode stats
    let tok_s: Vec<f64> = stats
        .decode_times
        .iter()
        .map(|d| 1.0 / d.as_secs_f64())
        .collect();
    let decode_total: f64 = stats.decode_times.iter().map(|d| d.as_secs_f64()).sum();
    let avg_tok_s = tok_s.iter().sum::<f64>() / n as f64;
    let min_tok_s = tok_s.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_tok_s = tok_s.iter().cloned().fold(0.0f64, f64::max);
    let avg_ms = decode_total * 1000.0 / n as f64;
    let total = prefill_s + decode_total;

    eprintln!(
        "  {}      {} tok  in {}",
        style("decode").dim(),
        style(n).cyan(),
        style(format!("{decode_total:.2}s")).cyan(),
    );
    eprintln!(
        "            {}  {}  {} tok/s",
        style(format!("{avg_tok_s:.1} avg")).green(),
        style(format!("{min_tok_s:.1} min")).green(),
        style(format!("{max_tok_s:.1} max")).green(),
    );
    eprintln!(
        "            {}",
        style(format!("{avg_ms:.0} ms/tok")).yellow(),
    );

    if let Some(seed) = stats.seed {
        eprintln!("  {}       {}", style("seed").dim(), style(seed).dim());
    }

    eprintln!(
        "  {}       {}",
        style("total").dim(),
        style(format!("{total:.2}s")).cyan(),
    );
    eprintln!("  {sep}");
}
