//! Progress and stats display for CLI output.
//!
//! Provides spinners, progress bars, and a structured stats summary using
//! `indicatif` for bars and `console` for colored text.
//!
//! All output goes to stderr. When stderr is not a terminal, indicatif
//! automatically disables progress bar rendering.

use console::{Term, style};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

use crate::generate::GenerationStats;

// ── Spinner ───────────────────────────────────────────────────────────────────

/// Create a spinner with the given message.
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
    pb.finish_with_message(format!("{} {}", style("✓").green(), msg));
}

// ── Compile progress ─────────────────────────────────────────────────────────

/// Create a progress bar for kernel compilation.
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
        style("✓").green(),
    ));
}

// ── Decode progress ───────────────────────────────────────────────────────────

/// Create a progress bar for autoregressive decoding.
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

/// Build the stats box lines (without ANSI colors, for width measurement,
/// and with colors for display).
fn build_stats_lines(stats: &GenerationStats) -> Vec<(String, String)> {
    // Returns Vec<(plain_text, colored_text)>
    let prefill_s = stats.prefill_time.as_secs_f64();
    let n = stats.decode_times.len();

    let mut lines = Vec::new();

    if n == 0 {
        lines.push((
            format!("prefill {prefill_s:.2}s ({} tok)", stats.prompt_tokens),
            format!(
                "{} {}  ({} tok)",
                style("prefill").dim(),
                style(format!("{prefill_s:.2}s")).cyan(),
                style(stats.prompt_tokens).cyan(),
            ),
        ));
        lines.push((
            format!("total {prefill_s:.2}s"),
            format!(
                "{}   {}",
                style("total").dim(),
                style(format!("{prefill_s:.2}s")).cyan(),
            ),
        ));
        return lines;
    }

    let tok_s: Vec<f64> = stats
        .decode_times
        .iter()
        .map(|d| 1.0 / d.as_secs_f64())
        .collect();
    let decode_total: f64 = stats.decode_times.iter().map(|d| d.as_secs_f64()).sum();
    let avg_tok_s = tok_s.iter().sum::<f64>() / n as f64;
    let avg_ms = decode_total * 1000.0 / n as f64;
    let total = prefill_s + decode_total;

    lines.push((
        format!("{n} tok  {avg_tok_s:.1} tok/s  {avg_ms:.0}ms/tok"),
        format!(
            "{} tok  {} tok/s  {}",
            style(n).cyan().bold(),
            style(format!("{avg_tok_s:.1}")).green().bold(),
            style(format!("{avg_ms:.0}ms/tok")).yellow(),
        ),
    ));
    lines.push((
        format!("{total:.2}s total  {prefill_s:.2}s prefill"),
        format!(
            "{} total  {} prefill",
            style(format!("{total:.2}s")).cyan(),
            style(format!("{prefill_s:.2}s")).dim(),
        ),
    ));

    if stats.context_max > 0 {
        let pct = stats.context_used as f64 / stats.context_max as f64 * 100.0;
        let pct_style = if pct > 80.0 {
            style(format!("{pct:.0}%")).red()
        } else if pct > 50.0 {
            style(format!("{pct:.0}%")).yellow()
        } else {
            style(format!("{pct:.0}%")).dim()
        };
        lines.push((
            format!(
                "ctx {}/{} ({}%)",
                stats.context_used, stats.context_max, pct as u32
            ),
            format!(
                "{} {}/{}  {}",
                style("ctx").dim(),
                style(stats.context_used).dim(),
                style(stats.context_max).dim(),
                pct_style,
            ),
        ));
    }

    lines
}

/// Count how many terminal lines a string occupies, accounting for line wrapping.
fn count_display_lines(text: &str, term_width: usize) -> usize {
    if term_width == 0 {
        return text.lines().count().max(1);
    }
    let mut total = 0;
    for line in text.split('\n') {
        let len = console::measure_text_width(line);
        if len == 0 {
            total += 1;
        } else {
            total += (len + term_width - 1) / term_width;
        }
    }
    total.max(1)
}

/// Print stats as a right-aligned overlay on the generated text.
///
/// Uses ANSI cursor movement to go back up and write the stats box
/// on the right side of the already-printed text. Falls back to the
/// block-below style if the terminal is too narrow or not a TTY.
pub fn print_stats(stats: &GenerationStats, generated_text: &str) {
    let term = Term::stderr();
    let is_tty = atty::is(atty::Stream::Stderr) && atty::is(atty::Stream::Stdout);

    let stats_lines = build_stats_lines(stats);
    if stats_lines.is_empty() {
        return;
    }

    // Fixed box width for consistent positioning across turns.
    let box_inner = 32; // fixed inner width
    let box_outer = box_inner + 2; // border chars

    let term_width = if is_tty { term.size().1 as usize } else { 0 };

    // Need: terminal wide enough for text + box, and enough output lines to overlay on
    let min_width = box_outer + 20; // at least 20 chars for text on the left
    let output_lines = if is_tty && !generated_text.is_empty() {
        count_display_lines(generated_text, term_width)
    } else {
        0
    };
    // +1 for the trailing newline after output
    let box_height = stats_lines.len() + 2; // top border + content + bottom border

    let use_side = is_tty && term_width >= min_width && output_lines >= box_height;

    if use_side {
        let col = term_width - box_outer; // 0-indexed column where box starts

        // Position box at the bottom of the output.
        // Cursor is on the line after the trailing println!(), so move up
        // by box_height to align the box with the last few lines of output.
        eprint!("\x1b[{}A", box_height);

        // Top border
        eprint!("\x1b[{}G\x1b[2m╭{}╮\x1b[0m", col + 1, "─".repeat(box_inner));
        eprint!("\n");

        // Content lines
        for (plain, colored) in &stats_lines {
            let content_width = box_inner.saturating_sub(2); // space for padding inside │ │
            let padding = content_width.saturating_sub(plain.len());
            eprint!(
                "\x1b[{}G\x1b[2m│\x1b[0m {colored}{} \x1b[2m│\x1b[0m",
                col + 1,
                " ".repeat(padding),
            );
            eprint!("\n");
        }

        // Bottom border -- on the same line as the last output line
        eprint!("\x1b[{}G\x1b[2m╰{}╯\x1b[0m", col + 1, "─".repeat(box_inner));
        eprintln!();
    } else {
        // Fallback: compact line below output
        let n = stats.decode_times.len();
        if n == 0 {
            let prefill_s = stats.prefill_time.as_secs_f64();
            eprintln!(
                "\n  {} {} ({} tok)",
                style("prefill").dim(),
                style(format!("{prefill_s:.2}s")).cyan(),
                style(stats.prompt_tokens).cyan(),
            );
        } else {
            let decode_total: f64 = stats.decode_times.iter().map(|d| d.as_secs_f64()).sum();
            let avg_tok_s: f64 = stats
                .decode_times
                .iter()
                .map(|d| 1.0 / d.as_secs_f64())
                .sum::<f64>()
                / n as f64;
            let avg_ms = decode_total * 1000.0 / n as f64;
            let total = stats.prefill_time.as_secs_f64() + decode_total;
            let ctx_str = if stats.context_max > 0 {
                let pct = stats.context_used as f64 / stats.context_max as f64 * 100.0;
                let pct_styled = if pct > 80.0 {
                    style(format!("{pct:.0}%")).red()
                } else if pct > 50.0 {
                    style(format!("{pct:.0}%")).yellow()
                } else {
                    style(format!("{pct:.0}%")).dim()
                };
                format!(
                    "  {} {}/{}",
                    pct_styled,
                    style(stats.context_used).dim(),
                    style(stats.context_max).dim(),
                )
            } else {
                String::new()
            };
            eprintln!(
                "\n  {} tok  {} tok/s  {}  {}{}",
                style(n).cyan().bold(),
                style(format!("{avg_tok_s:.1}")).green().bold(),
                style(format!("{avg_ms:.0}ms/tok")).yellow(),
                style(format!("{total:.2}s")).dim(),
                ctx_str,
            );
        }
    }
}
