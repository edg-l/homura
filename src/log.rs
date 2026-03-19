//! Homura logging module.
//!
//! Level-filtered format-string logging to stderr with ANSI colors.
//! Controlled by `HOMURA_LOG` env var (default: `info`).
//!
//! Levels: error, warn, info, debug.
//!
//! ```ignore
//! log::info!("loaded model in {:.2}s", elapsed);
//! log::compile!("plan", "passes done: {}ms", ms);
//! ```

use std::sync::OnceLock;
use std::time::Instant;

// ── Log levels ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
}

/// Global log level, parsed once from `HOMURA_LOG` env var.
pub fn max_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        match std::env::var("HOMURA_LOG")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "error" => Level::Error,
            "warn" => Level::Warn,
            "debug" => Level::Debug,
            _ => Level::Info,
        }
    })
}

#[inline]
pub fn enabled(level: Level) -> bool {
    level <= max_level()
}

// ── Timestamp ────────────────────────────────────────────────────────────────

/// Seconds since process start.
pub fn ts() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

// ── ANSI colors ──────────────────────────────────────────────────────────────

pub const RED: &str = "\x1b[31m";
pub const YELLOW: &str = "\x1b[33m";
pub const GREEN: &str = "\x1b[32m";
pub const CYAN: &str = "\x1b[36m";
pub const MAGENTA: &str = "\x1b[35m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const BOLD_MAGENTA: &str = "\x1b[1;35m";
pub const RESET: &str = "\x1b[0m";

// ── Macros ───────────────────────────────────────────────────────────────────

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Level::Error) {
            eprintln!(
                "{}error:{} {}",
                $crate::log::RED,
                $crate::log::RESET,
                format_args!($($arg)*)
            );
        }
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Level::Warn) {
            eprintln!(
                "{}warn:{} {}",
                $crate::log::YELLOW,
                $crate::log::RESET,
                format_args!($($arg)*)
            );
        }
    };
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Level::Info) {
            eprintln!("{}", format_args!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Level::Debug) {
            eprintln!(
                "{}{}{}",
                $crate::log::DIM,
                format_args!($($arg)*),
                $crate::log::RESET,
            );
        }
    };
}

/// Compile-progress with automatic timestamp: `[   1.23s] [label] msg`
#[macro_export]
macro_rules! log_compile {
    ($label:expr, $($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Level::Info) {
            eprintln!(
                "[{:>8.2}s] [{}] {}",
                $crate::log::ts(),
                $label,
                format_args!($($arg)*)
            );
        }
    };
}

pub use crate::log_compile as compile;
pub use crate::log_debug as debug;
pub use crate::log_error as error;
pub use crate::log_info as info;
pub use crate::log_warn as warn;
