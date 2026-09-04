//! Configuration and global coefficient storage for the Lightyear integrator.
//!
//! This module handles:
//! - Global spherical harmonics coefficient storage using lock-free arc-swap
//! - Loading coefficients from EGM files or in-memory arrays
//! - Thread pool initialization for parallel batch processing

use anyhow::Context as _;
use arc_swap::ArcSwap;
use satpy_core::{pack_gravity_coeffs, PackedGravityCoeffs};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// Global Spherical Harmonics Coefficient Storage
// ============================================================================

/// Global coefficients using arc-swap for lock-free reads.
///
/// This allows multiple threads to read coefficients simultaneously without
/// any locking overhead, while writes atomically swap the entire coefficient set.
pub static GLOBAL_COEFFS: std::sync::LazyLock<ArcSwap<GlobalCoeffs>> =
    std::sync::LazyLock::new(|| ArcSwap::from_pointee(GlobalCoeffs::Unloaded));

#[derive(Clone)]
pub enum GlobalCoeffs {
    /// No gravity coefficients have been installed.
    Unloaded,
    /// A complete, validated gravity snapshot.
    Loaded(Arc<PackedGravityCoeffs>),
}

impl GlobalCoeffs {
    /// Clone a coherent coefficient snapshot if global gravity is loaded.
    #[must_use]
    pub(crate) fn loaded_snapshot(&self) -> Option<Arc<PackedGravityCoeffs>> {
        match self {
            Self::Unloaded => None,
            Self::Loaded(coefficients) => Some(Arc::clone(coefficients)),
        }
    }
}

/// Get global coefficients for use in integration.
///
/// Returns None if coefficients haven't been loaded yet.
#[must_use]
pub fn get_global_coeffs_packed() -> Option<Arc<PackedGravityCoeffs>> {
    let global = GLOBAL_COEFFS.load();
    global.loaded_snapshot()
}

// ============================================================================
// Coefficient Loading Functions
// ============================================================================

/// Load spherical harmonics constants from an EGM file.
///
/// # Errors
///
/// Returns an error when the file cannot be read or a coefficient record is
/// malformed or cannot fit in the requested table.
pub fn load_constants(path: &str, order: usize) -> anyhow::Result<i32> {
    let path_obj = Path::new(path);
    if !path_obj.exists() {
        anyhow::bail!("Constants file not found: {path}");
    }

    let file = File::open(path_obj).context("Failed to open file")?;

    let reader = BufReader::new(file);

    let packed = packed_constants_from_reader(reader, order)?;
    publish_constants(packed);
    Ok(0)
}

/// Parse spherical harmonics constants from embedded EGM bytes without
/// changing the process-global coefficient snapshot.
///
/// # Errors
///
/// Returns an error when a coefficient record is malformed or cannot fit in
/// the requested table.
pub fn packed_constants_from_bytes(
    bytes: &[u8],
    order: usize,
) -> anyhow::Result<Arc<PackedGravityCoeffs>> {
    packed_constants_from_reader(BufReader::new(bytes), order)
}

/// Load spherical harmonics constants from embedded EGM bytes.
///
/// # Errors
///
/// Returns an error when a coefficient record is malformed or cannot fit in
/// the requested table.
pub fn load_constants_from_bytes(bytes: &[u8], order: usize) -> anyhow::Result<i32> {
    let packed = packed_constants_from_bytes(bytes, order)?;
    publish_constants(packed);
    Ok(0)
}

fn publish_constants(packed: Arc<PackedGravityCoeffs>) {
    GLOBAL_COEFFS.store(Arc::new(GlobalCoeffs::Loaded(packed)));
}

fn checked_add(left: usize, right: usize, context: &str) -> anyhow::Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow::anyhow!("coefficient {context} overflows usize"))
}

fn checked_mul(left: usize, right: usize, context: &str) -> anyhow::Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow::anyhow!("coefficient {context} overflows usize"))
}

fn exact_usize_as_f64(value: usize, context: &str) -> anyhow::Result<f64> {
    u32::try_from(value)
        .map(f64::from)
        .map_err(|_| anyhow::anyhow!("coefficient {context} exceeds supported degree range"))
}

fn packed_constants_from_reader(
    reader: impl BufRead,
    order: usize,
) -> anyhow::Result<Arc<PackedGravityCoeffs>> {
    // Match C++ coefficient table sizing: load_order = required_order + 2
    // This ensures we have enough room for the (l+1, m+1) accesses.
    let stride = checked_add(order, 2, "table stride")?;
    let total_size = checked_mul(stride, stride, "table size")?;
    let mut c_coeffs = vec![0.0; total_size];
    let mut s_coeffs = vec![0.0; total_size];

    // C[0,0] = 1.0 is the point-mass gravity term (must be set explicitly)
    *c_coeffs
        .first_mut()
        .ok_or_else(|| anyhow::anyhow!("coefficient table is unexpectedly empty"))? = 1.0;

    // Precompute ln(n!) up to (2*order) for normalization.
    // C++ uses: nrm = sqrt( gamma(l+m+1) / ((2 - delta(m,0))*(2l+1)*gamma(l-m+1)) )
    let max_fact = checked_mul(2, stride, "normalization limit")?;
    let factorial_count = checked_add(max_fact, 1, "normalization table length")?;
    let mut ln_fact = vec![0.0f64; factorial_count];
    let mut previous_log_factorial = 0.0;
    for (index, log_factorial) in ln_fact.iter_mut().enumerate().skip(1) {
        let index_f64 = exact_usize_as_f64(index, "normalization index")?;
        *log_factorial = previous_log_factorial + index_f64.ln();
        previous_log_factorial = *log_factorial;
    }

    for line in reader.lines() {
        let line = line.context("Read error")?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        let [degree_text, order_text, c_text, s_text, ..] = fields.as_slice() else {
            continue;
        };

        let degree = degree_text
            .parse::<usize>()
            .with_context(|| format!("invalid coefficient degree `{degree_text}`"))?;
        let degree_order = order_text
            .parse::<usize>()
            .with_context(|| format!("invalid coefficient order `{order_text}`"))?;
        let c_value = c_text
            .replace(['D', 'd'], "E")
            .parse::<f64>()
            .with_context(|| format!("invalid cosine coefficient `{c_text}`"))?;
        let s_value = s_text
            .replace(['D', 'd'], "E")
            .parse::<f64>()
            .with_context(|| format!("invalid sine coefficient `{s_text}`"))?;

        if degree <= order && degree_order <= degree {
            let row_start = checked_mul(degree, stride, "row offset")?;
            let coefficient_index = checked_add(row_start, degree_order, "coefficient index")?;
            if coefficient_index < total_size {
                // Apply the same normalization as C++ sphericals_native.hpp
                let delta_order_zero = if degree_order == 0 { 1.0 } else { 0.0 };
                let degree_f64 = exact_usize_as_f64(degree, "degree")?;
                let denominator = (2.0 - delta_order_zero) * (2.0 * degree_f64 + 1.0);
                let numerator_index = checked_add(degree, degree_order, "numerator index")?;
                let denominator_index = degree
                    .checked_sub(degree_order)
                    .ok_or_else(|| anyhow::anyhow!("coefficient order exceeds degree"))?;
                let log_factorial_numerator =
                    ln_fact.get(numerator_index).copied().ok_or_else(|| {
                        anyhow::anyhow!("coefficient numerator index is out of range")
                    })?;
                let log_factorial_denominator =
                    ln_fact.get(denominator_index).copied().ok_or_else(|| {
                        anyhow::anyhow!("coefficient denominator index is out of range")
                    })?;
                let log_normalization = 0.5
                    * ((log_factorial_numerator - log_factorial_denominator) - denominator.ln());
                let normalization = log_normalization.exp();

                let c_slot = c_coeffs
                    .get_mut(coefficient_index)
                    .ok_or_else(|| anyhow::anyhow!("coefficient cosine index is out of range"))?;
                let s_slot = s_coeffs
                    .get_mut(coefficient_index)
                    .ok_or_else(|| anyhow::anyhow!("coefficient sine index is out of range"))?;
                *c_slot = c_value / normalization;
                *s_slot = s_value / normalization;
            }
        }
    }

    // Raw loader scratch dies here; callers receive only validated immutable
    // packed authority.
    Ok(Arc::new(pack_gravity_coeffs(
        &c_coeffs, &s_coeffs, stride, order,
    )?))
}

/// Serialises every lib test that PUBLISHES to [`GLOBAL_COEFFS`] and then
/// reads it back. The lib test binary runs its tests on parallel threads, and
/// `session.rs`, `batch.rs`, and (historically) `rhs.rs` each install their own
/// pack — synthetic in the first two, sealed DIR-R6 in the third — so an
/// unserialised install-then-read can observe ANOTHER test's coefficients.
/// `ArcSwap` makes the swap atomic, never coherent across two tests; this lock
/// supplies the coherence. Hold the returned guard for the WHOLE test (bind it
/// to a named `_guard`, never `_`, which drops immediately). Tests that only
/// need coefficient VALUES should use [`packed_constants_from_bytes`] (or a
/// locally built pack) instead and skip both the global and this lock.
#[cfg(test)]
pub(crate) fn lock_global_coeffs_for_test() -> std::sync::MutexGuard<'static, ()> {
    static GLOBAL_COEFFS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A panicking holder poisons the mutex; the lock's only job is mutual
    // exclusion, so the poison flag carries no information — take the guard.
    GLOBAL_COEFFS_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::{
        get_global_coeffs_packed, load_constants_from_bytes, packed_constants_from_bytes,
        GlobalCoeffs, GLOBAL_COEFFS,
    };
    use std::process::Command;
    use std::sync::Arc;

    fn run_isolated_global_coeffs_test(
        child_environment: &str,
        test_name: &str,
    ) -> anyhow::Result<()> {
        let test_binary = std::env::current_exe()
            .map_err(|error| anyhow::anyhow!("cannot locate config test binary: {error}"))?;
        let output = Command::new(test_binary)
            .args(["--exact", test_name])
            .env(child_environment, "1")
            .output()
            .map_err(|error| anyhow::anyhow!("cannot run isolated config test: {error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::ensure!(
            output.status.success()
                && stdout.contains("running 1 test")
                && stdout.contains(test_name),
            "isolated config test `{test_name}` failed or matched no test\nstdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }

    #[test]
    fn unloaded_global_coefficients_do_not_fabricate_a_pack() {
        let coefficients = GlobalCoeffs::Unloaded;

        assert!(coefficients.loaded_snapshot().is_none());
    }

    #[test]
    fn loader_propagates_packed_gravity_validation_error() -> anyhow::Result<()> {
        let Err(error) = load_constants_from_bytes(b"", satpy_core::MAX_ORDER + 1) else {
            anyhow::bail!("unsupported packed gravity order must fail closed");
        };

        anyhow::ensure!(
            error.downcast_ref::<satpy_core::GravityError>()
                == Some(&satpy_core::GravityError::UnsupportedOrder),
            "unsupported packed order must retain GravityError::UnsupportedOrder"
        );
        Ok(())
    }

    #[test]
    fn malformed_numeric_coefficient_fails_closed() -> anyhow::Result<()> {
        let Err(error) = load_constants_from_bytes(b"2 0 not-a-number 0.0\n", 2) else {
            anyhow::bail!("malformed production coefficient must not become zero");
        };

        anyhow::ensure!(
            error.to_string().contains("coefficient"),
            "malformed coefficient error lost coefficient context: {error}"
        );
        Ok(())
    }

    #[test]
    fn local_byte_parse_leaves_published_identity_unchanged() -> anyhow::Result<()> {
        const CHILD_ENV: &str = "ND_CONFIG_LOCAL_PARSE_CHILD";
        const TEST_NAME: &str =
            "config::tests::local_byte_parse_leaves_published_identity_unchanged";
        if std::env::var_os(CHILD_ENV).is_none() {
            return run_isolated_global_coeffs_test(CHILD_ENV, TEST_NAME);
        }
        let before = GLOBAL_COEFFS.load_full();

        let local = packed_constants_from_bytes(b"2 0 -4.84165143790815e-4 0.0\n", 2)?;
        let after = GLOBAL_COEFFS.load_full();

        anyhow::ensure!(
            Arc::ptr_eq(&before, &after),
            "local parser changed global identity"
        );
        anyhow::ensure!(
            local.max_order() == 2,
            "local parser returned wrong gravity order"
        );
        Ok(())
    }

    #[test]
    fn publishing_byte_loader_replaces_global_identity() -> anyhow::Result<()> {
        const CHILD_ENV: &str = "ND_CONFIG_PUBLISHING_LOADER_CHILD";
        const TEST_NAME: &str = "config::tests::publishing_byte_loader_replaces_global_identity";
        if std::env::var_os(CHILD_ENV).is_none() {
            return run_isolated_global_coeffs_test(CHILD_ENV, TEST_NAME);
        }

        load_constants_from_bytes(b"", 0)?;
        let first = get_global_coeffs_packed()
            .ok_or_else(|| anyhow::anyhow!("first publishing load left gravity unloaded"))?;
        load_constants_from_bytes(b"2 0 -4.84165143790815e-4 0.0\n", 2)?;
        let second = get_global_coeffs_packed()
            .ok_or_else(|| anyhow::anyhow!("second publishing load left gravity unloaded"))?;

        anyhow::ensure!(
            !Arc::ptr_eq(&first, &second),
            "publishing loader retained stale global identity"
        );
        anyhow::ensure!(
            second.max_order() == 2,
            "publishing loader installed wrong order"
        );
        Ok(())
    }
}
