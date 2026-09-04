# Performance Baseline (2026-01-01)

## Before Optimizations

Captured with maturin release build on Apple M3 Pro.

| Test | Time (us) | Notes |
|------|-----------|-------|
| single_component | 4,433 | 1 component, 48x720 grid |
| two_component | 6,937 | 2 components |
| five_component | 10,645 | 5 components ring |
| eight_anisotropic | 15,866 | 8 anisotropic components |
| sixteen_component | 28,076 | 16 components |

## Test Configuration

- Grid: 48 radial x 720 angular (default)
- Success probability: 0.9
- Debris covariance: Identity matrix

## Hardware

- Apple M3 Pro (10-core)
- macOS Darwin
- Python 3.14t (free-threading build)
- Rust 1.75+ with release optimizations

## Notes

- Times include all 5 phases of computation
- Phase 3 (gradient descent) dominates cost (~85%)
- These are the targets for optimization
