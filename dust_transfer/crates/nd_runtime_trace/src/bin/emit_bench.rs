use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use nd_runtime_trace::{
    emit, sha256_from_hex, CommandOutcome, CommandRole, ContextId, SchedulerIdentity, TraceConfig,
    TraceDisposition, TraceEvent, TraceSession, DEFAULT_QUEUE_CAPACITY, TRACE_RECORD_SIZE_BYTES,
};
use serde::Serialize;

const RESULT_SCHEMA: &str = "nasa_dust.runtime_trace.emit_bench.v1";

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    iterations: u64,
    #[arg(long)]
    trace_dir: Option<PathBuf>,
    #[arg(long)]
    binary_sha256: Option<String>,
    #[arg(long)]
    revision_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Disabled,
    Accepted,
    Saturated,
}

#[derive(Serialize)]
struct ResultLine {
    schema: &'static str,
    mode: Mode,
    iterations: u64,
    producer_attempts: u64,
    elapsed_ns: u64,
    record_size_bytes: usize,
    queue_capacity: usize,
    binary_sha256: Option<String>,
    revision_sha256: Option<String>,
    trace_disposition: Option<TraceDisposition>,
    accepted: u64,
    dropped_full: u64,
    dropped_disconnected: u64,
    dropped_reentrant: u64,
    trace_path: Option<PathBuf>,
}

fn main() {
    match run(Args::parse()) {
        Ok(result) => {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            if serde_json::to_writer(&mut lock, &result).is_err() || writeln!(lock).is_err() {
                std::process::exit(2);
            }
        }
        Err(error) => {
            let _ = writeln!(std::io::stderr().lock(), "{error}");
            std::process::exit(2);
        }
    }
}

fn run(args: Args) -> anyhow::Result<ResultLine> {
    let event = TraceEvent::Notice {
        context: ContextId::PROCESS,
        code: 1,
        detail: 1,
    };
    if args.mode == Mode::Disabled {
        let started = Instant::now();
        for _ in 0..args.iterations {
            emit(event);
        }
        return Ok(ResultLine {
            schema: RESULT_SCHEMA,
            mode: args.mode,
            iterations: args.iterations,
            producer_attempts: args.iterations,
            elapsed_ns: elapsed_ns(started),
            record_size_bytes: TRACE_RECORD_SIZE_BYTES,
            queue_capacity: 0,
            binary_sha256: args.binary_sha256,
            revision_sha256: args.revision_sha256,
            trace_disposition: None,
            accepted: 0,
            dropped_full: 0,
            dropped_disconnected: 0,
            dropped_reentrant: 0,
            trace_path: None,
        });
    }

    let binary_sha256_text = args
        .binary_sha256
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--binary-sha256 is required for enabled modes"))?;
    let binary_sha256 = sha256_from_hex(binary_sha256_text)
        .ok_or_else(|| anyhow::anyhow!("--binary-sha256 must be 64 hexadecimal characters"))?;
    let revision_sha256 = args
        .revision_sha256
        .as_deref()
        .map(|value| {
            sha256_from_hex(value).ok_or_else(|| {
                anyhow::anyhow!("--revision-sha256 must be 64 hexadecimal characters")
            })
        })
        .transpose()?;
    let trace_dir = args.trace_dir.clone().unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("build/runtime-trace-bench/emit")
    });
    let iterations = usize::try_from(args.iterations)
        .map_err(|_| anyhow::anyhow!("--iterations does not fit this platform"))?;
    let queue_capacity = match args.mode {
        Mode::Disabled => 0,
        Mode::Accepted => iterations.max(DEFAULT_QUEUE_CAPACITY),
        Mode::Saturated => 1,
    };
    let mut config = TraceConfig::new(
        trace_dir,
        CommandRole::EmitBench,
        binary_sha256,
        revision_sha256,
        SchedulerIdentity::default(),
        Vec::new(),
        0,
    );
    config.queue_capacity = queue_capacity;
    let started_trace = TraceSession::start(config)?;
    let binding = started_trace.controller.bind(ContextId::PROCESS);
    let started = Instant::now();
    for _ in 0..args.iterations {
        emit(event);
    }
    let elapsed_ns = elapsed_ns(started);
    drop(binding);
    drop(started_trace.drivers);
    let report = started_trace
        .session
        .finish(CommandOutcome::Succeeded, None);
    Ok(ResultLine {
        schema: RESULT_SCHEMA,
        mode: args.mode,
        iterations: args.iterations,
        producer_attempts: args.iterations,
        elapsed_ns,
        record_size_bytes: TRACE_RECORD_SIZE_BYTES,
        queue_capacity,
        binary_sha256: args.binary_sha256,
        revision_sha256: args.revision_sha256,
        trace_disposition: Some(report.disposition),
        accepted: report.accepted,
        dropped_full: report.dropped_full,
        dropped_disconnected: report.dropped_disconnected,
        dropped_reentrant: report.dropped_reentrant,
        trace_path: report.path,
    })
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
