//! Auto-pin the process to a single L3 cache domain (CCD) on multi-CCD CPUs.
//!
//! On AMD Zen 3D / Zen 4 chiplet designs, each CCD has its own L3 cache.
//! When OS threads migrate between CCDs, L3 contents are invalidated and
//! bandwidth-bound inference slows down 2-3x. Pinning to one CCD eliminates
//! this variance.
//!
//! Set `HOMURA_NO_PIN=1` to disable auto-pinning.

#[cfg(target_os = "linux")]
mod inner {
    use std::collections::BTreeSet;
    use std::path::Path;

    /// Pin the current process to a single CCD.
    ///
    /// No-op if: only one CCD, topology unreadable, or already pinned.
    pub fn pin_to_single_ccd() {
        if std::env::var("HOMURA_NO_PIN").is_ok() {
            log_debug!("CCD pinning disabled (HOMURA_NO_PIN)");
            return;
        }

        let ccds = match discover_ccds() {
            Some(c) if c.len() >= 2 => c,
            _ => return,
        };

        // If user already pinned via taskset, don't override.
        if let Some(current) = current_affinity_set() {
            for ccd in &ccds {
                let ccd_set: BTreeSet<usize> = ccd.iter().copied().collect();
                if current.is_subset(&ccd_set) {
                    log_debug!("already pinned to a single CCD, skipping");
                    return;
                }
            }
        }

        // Pick the best CCD for bandwidth-bound inference.
        //
        // On X3D, the V-Cache CCD has 3x larger L3 but lower boost clocks.
        // Since model weights (~1GB+) exceed even the V-Cache L3 (96MB),
        // the extra cache doesn't help -- prefer the higher-clocking CCD
        // (smaller L3). On symmetric systems, all CCDs are equal.
        //
        // Tie-break: fewest logical cores, then lowest CPU ID (stable).
        let best = ccds
            .iter()
            .min_by_key(|c| {
                let l3_kb = l3_size_kb(c[0]);
                (l3_kb, c.len(), c[0])
            })
            .unwrap();

        if set_affinity(best) {
            let core_str = format_cpu_list(best);
            log_info!("pinned to CCD cores {} ({} threads)", core_str, best.len());
        }
    }

    /// Discover unique L3 cache domains by reading sysfs.
    fn discover_ccds() -> Option<Vec<Vec<usize>>> {
        let cpu_dir = Path::new("/sys/devices/system/cpu");
        if !cpu_dir.exists() {
            return None;
        }

        let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
        for entry in std::fs::read_dir(cpu_dir).ok()? {
            let entry = entry.ok()?;
            let name = entry.file_name();
            let name_str = name.to_str()?;
            if !name_str.starts_with("cpu") || !name_str[3..].chars().next()?.is_ascii_digit() {
                continue;
            }
            let list_path = entry.path().join("cache/index3/shared_cpu_list");
            if let Ok(contents) = std::fs::read_to_string(&list_path) {
                let cpus = parse_cpu_list(contents.trim());
                if !cpus.is_empty() {
                    seen.insert(cpus);
                }
            }
        }

        if seen.is_empty() {
            None
        } else {
            Some(seen.into_iter().collect())
        }
    }

    /// Read L3 cache size in KB for a given CPU, or u64::MAX on failure.
    fn l3_size_kb(cpu: usize) -> u64 {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index3/size");
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| {
                let s = s.trim().trim_end_matches('K');
                s.parse::<u64>().ok()
            })
            .unwrap_or(u64::MAX)
    }

    /// Parse "0-5,12-17" into a sorted Vec of CPU IDs.
    fn parse_cpu_list(s: &str) -> Vec<usize> {
        let mut cpus = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((lo, hi)) = part.split_once('-') {
                if let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                    cpus.extend(lo..=hi);
                }
            } else if let Ok(cpu) = part.parse::<usize>() {
                cpus.push(cpu);
            }
        }
        cpus.sort();
        cpus.dedup();
        cpus
    }

    /// Get the current process affinity mask as a set of CPU IDs.
    fn current_affinity_set() -> Option<BTreeSet<usize>> {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            let ret = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
            if ret != 0 {
                return None;
            }
            let mut cpus = BTreeSet::new();
            for i in 0..libc::CPU_SETSIZE as usize {
                if libc::CPU_ISSET(i, &set) {
                    cpus.insert(i);
                }
            }
            Some(cpus)
        }
    }

    /// Set process affinity to the given CPU list.
    fn set_affinity(cpus: &[usize]) -> bool {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &cpu in cpus {
                if cpu < libc::CPU_SETSIZE as usize {
                    libc::CPU_SET(cpu, &mut set);
                }
            }
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
        }
    }

    /// Format a CPU list back into compact notation: [6,7,8,18,19,20] -> "6-8,18-20".
    fn format_cpu_list(cpus: &[usize]) -> String {
        if cpus.is_empty() {
            return String::new();
        }
        let mut ranges: Vec<String> = Vec::new();
        let mut start = cpus[0];
        let mut end = cpus[0];
        for &cpu in &cpus[1..] {
            if cpu == end + 1 {
                end = cpu;
            } else {
                if start == end {
                    ranges.push(format!("{start}"));
                } else {
                    ranges.push(format!("{start}-{end}"));
                }
                start = cpu;
                end = cpu;
            }
        }
        if start == end {
            ranges.push(format!("{start}"));
        } else {
            ranges.push(format!("{start}-{end}"));
        }
        ranges.join(",")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_parse_cpu_list() {
            assert_eq!(
                parse_cpu_list("0-5,12-17"),
                vec![0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
            );
            assert_eq!(
                parse_cpu_list("6-11,18-23"),
                vec![6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]
            );
            assert_eq!(parse_cpu_list("0"), vec![0]);
            assert_eq!(parse_cpu_list("0,2,4"), vec![0, 2, 4]);
            assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
        }

        #[test]
        fn test_format_cpu_list() {
            assert_eq!(
                format_cpu_list(&[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]),
                "0-5,12-17"
            );
            assert_eq!(
                format_cpu_list(&[6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]),
                "6-11,18-23"
            );
            assert_eq!(format_cpu_list(&[0]), "0");
            assert_eq!(format_cpu_list(&[0, 2, 4]), "0,2,4");
        }

        #[test]
        fn test_discover_ccds() {
            // Just verify it doesn't crash; actual result depends on hardware.
            let ccds = discover_ccds();
            if let Some(ref ccds) = ccds {
                eprintln!("discovered {} CCDs:", ccds.len());
                for (i, ccd) in ccds.iter().enumerate() {
                    eprintln!("  CCD{i}: {}", format_cpu_list(ccd));
                }
            } else {
                eprintln!("no CCD topology found (expected on non-Linux or non-AMD)");
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub use inner::pin_to_single_ccd;

#[cfg(not(target_os = "linux"))]
pub fn pin_to_single_ccd() {}
