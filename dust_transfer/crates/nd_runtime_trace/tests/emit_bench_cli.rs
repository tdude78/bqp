use std::path::PathBuf;
use std::process::Command;

use serde::Deserialize;

#[derive(Deserialize)]
struct BenchResult {
    schema: String,
    mode: String,
    iterations: u64,
    producer_attempts: u64,
    elapsed_ns: u64,
    record_size_bytes: usize,
    queue_capacity: usize,
    trace_disposition: Option<String>,
    accepted: u64,
    dropped_full: u64,
    dropped_disconnected: u64,
    dropped_reentrant: u64,
    trace_path: Option<PathBuf>,
}

#[test]
fn disabled_benchmark_emits_one_machine_readable_result() {
    let output = Command::new(env!("CARGO_BIN_EXE_emit_bench"))
        .args(["--mode", "disabled", "--iterations", "4"])
        .output()
        .expect("run emit benchmark");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 benchmark output");
    assert_eq!(stdout.lines().count(), 1);
    let result: BenchResult = serde_json::from_str(stdout.trim()).expect("JSON result");
    assert_eq!(result.schema, "nasa_dust.runtime_trace.emit_bench.v1");
    assert_eq!(result.mode, "disabled");
    assert_eq!(result.iterations, 4);
    assert_eq!(result.producer_attempts, 4);
    assert!(result.elapsed_ns > 0);
}

#[test]
fn accepted_benchmark_reports_exact_trace_accounting() {
    let trace_dir = tempfile::tempdir().expect("temporary benchmark trace root");
    let output = Command::new(env!("CARGO_BIN_EXE_emit_bench"))
        .args([
            "--mode",
            "accepted",
            "--iterations",
            "4",
            "--trace-dir",
            trace_dir.path().to_str().expect("UTF-8 trace root"),
            "--binary-sha256",
            &"11".repeat(32),
            "--revision-sha256",
            &"22".repeat(32),
        ])
        .output()
        .expect("run accepted emit benchmark");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let result: BenchResult =
        serde_json::from_slice(&output.stdout).expect("JSON benchmark result");
    assert_eq!(result.mode, "accepted");
    assert_eq!(result.producer_attempts, 4);
    assert_eq!(result.accepted, 5);
    assert!(result.record_size_bytes > 0);
    assert_eq!(result.dropped_full, 0);
    assert_eq!(result.dropped_disconnected, 0);
    assert_eq!(result.trace_disposition.as_deref(), Some("complete"));
    assert!(result
        .trace_path
        .as_deref()
        .and_then(std::path::Path::to_str)
        .is_some_and(|path| path.ends_with(".complete.ndjson")));
    assert!(result
        .trace_path
        .as_deref()
        .and_then(std::path::Path::file_name)
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|name| name.starts_with("emit-bench-")));
}

#[test]
fn saturated_benchmark_reports_loss_without_stalling() {
    let trace_dir = tempfile::tempdir().expect("temporary benchmark trace root");
    let output = Command::new(env!("CARGO_BIN_EXE_emit_bench"))
        .args([
            "--mode",
            "saturated",
            "--iterations",
            "10000",
            "--trace-dir",
            trace_dir.path().to_str().expect("UTF-8 trace root"),
            "--binary-sha256",
            &"33".repeat(32),
        ])
        .output()
        .expect("run saturated emit benchmark");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: BenchResult =
        serde_json::from_slice(&output.stdout).expect("JSON benchmark result");
    assert_eq!(result.mode, "saturated");
    assert_eq!(result.queue_capacity, 1);
    assert_eq!(result.trace_disposition.as_deref(), Some("lossy"));
    assert!(result.dropped_full > 0);
    assert_eq!(
        result.accepted
            + result.dropped_full
            + result.dropped_disconnected
            + result.dropped_reentrant,
        10_001
    );
}
