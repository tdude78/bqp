//! Batch output alignment.
//!
//! One function, one caller (`batch.rs`). It used to be two: a private
//! `write_aligned` and a `write_batch_output_aligned` that forwarded to it
//! with the same four arguments in the same order.

use crate::types::IntegrationResult;

/// Write batch integration output, aligning integrator results with `t_eval` times.
///
/// This function handles the alignment between integration-native output times
/// and requested evaluation times (`t_eval`). It uses a two-pointer merge algorithm
/// to match times within a tolerance of `1e-9` efficiently.
///
/// # Arguments
/// * `out_chunk` - Output slice to write six-element state vectors (`n_times * 6` long).
/// * `t_eval` - Requested evaluation times.
/// * `result` - Integration result containing times and states.
/// * `direction` - Integration direction (`+1.0` forward, `-1.0` backward).
#[must_use]
pub fn write_batch_output_aligned(
    out_chunk: &mut [f64],
    t_eval: &[f64],
    result: &IntegrationResult,
    direction: f64,
) -> bool {
    let n_times = t_eval.len();
    let tol = 1e-9;

    // Missing output must remain conspicuously unusable.
    out_chunk.fill(f64::NAN);

    if n_times == 0 {
        return out_chunk.is_empty();
    }
    let Some(expected_values) = n_times.checked_mul(6) else {
        return false;
    };
    if out_chunk.len() != expected_values {
        return false;
    }
    let forward = direction >= 0.0;
    let mut eval_rows = t_eval.iter().zip(out_chunk.chunks_exact_mut(6));
    let mut result_rows = result.times.iter().zip(&result.states);
    let mut eval_row = eval_rows.next();
    let mut result_row = result_rows.next();
    let mut matched = 0usize;
    while let (Some((eval_time, output)), Some((result_time, state))) =
        (eval_row.take(), result_row.take())
    {
        let dt = eval_time - result_time;
        if dt.abs() <= tol {
            output.copy_from_slice(state);
            let Some(next_matched) = matched.checked_add(1) else {
                return false;
            };
            matched = next_matched;
            eval_row = eval_rows.next();
            result_row = result_rows.next();
        } else if (forward && dt < -tol) || (!forward && dt > tol) {
            eval_row = eval_rows.next();
            result_row = Some((result_time, state));
        } else {
            eval_row = Some((eval_time, output));
            result_row = result_rows.next();
        }
    }
    let complete = matched == n_times;
    if !complete {
        // Callers must never mistake a partially written trajectory for a
        // complete one after an error is ignored at a higher boundary.
        out_chunk.fill(f64::NAN);
    }
    complete
}

#[cfg(test)]
mod tests {
    use super::write_batch_output_aligned;
    use crate::types::IntegrationResult;

    fn state_with(seed: f64) -> [f64; 6] {
        [
            seed,
            seed + 1.0,
            seed + 2.0,
            seed + 3.0,
            seed + 4.0,
            seed + 5.0,
        ]
    }

    #[test]
    fn test_write_batch_output_aligned_forward_exact() {
        let mut out = vec![0.0f64; 3 * 6];
        let t_eval = [0.0, 1.0, 2.0];
        let result = IntegrationResult {
            times: vec![0.0, 1.0, 2.0],
            states: vec![state_with(10.0), state_with(20.0), state_with(30.0)],
            ..IntegrationResult::default()
        };

        assert!(write_batch_output_aligned(&mut out, &t_eval, &result, 1.0));

        assert!(out
            .chunks_exact(6)
            .zip([state_with(10.0), state_with(20.0), state_with(30.0)])
            .all(|(actual, expected)| actual.iter().copied().eq(expected)));
    }

    #[test]
    fn test_write_batch_output_aligned_backward_exact() {
        let mut out = vec![0.0f64; 3 * 6];
        let t_eval = [2.0, 1.0, 0.0];
        let result = IntegrationResult {
            times: vec![2.0, 1.0, 0.0],
            states: vec![state_with(10.0), state_with(20.0), state_with(30.0)],
            ..IntegrationResult::default()
        };

        assert!(write_batch_output_aligned(&mut out, &t_eval, &result, -1.0));

        assert!(out
            .chunks_exact(6)
            .zip([state_with(10.0), state_with(20.0), state_with(30.0)])
            .all(|(actual, expected)| actual.iter().copied().eq(expected)));
    }

    #[test]
    fn test_write_batch_output_aligned_backward_unmatched_tail_is_error_and_nan() {
        let mut out = vec![1.0f64; 4 * 6];
        let t_eval = [2.0, 1.0, 0.0, -1.0];
        let result = IntegrationResult {
            times: vec![2.0, 1.0, 0.0],
            states: vec![state_with(10.0), state_with(20.0), state_with(30.0)],
            ..IntegrationResult::default()
        };

        assert!(!write_batch_output_aligned(
            &mut out, &t_eval, &result, -1.0
        ));

        assert!(out.iter().all(|value| value.is_nan()));
    }
}
