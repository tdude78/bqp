use crate::odesolve::tableau::{dop853, dopri5, rkv98, tsit5, vern7, vern9, Tableau};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Tsit5,
    Dop853,
    Rkv98,
    Dopri5,
    Vern7,
    Vern9,
}

impl Method {
    #[must_use]
    pub fn tableau(self) -> &'static Tableau {
        match self {
            Self::Tsit5 => tsit5::tableau(),
            Self::Dop853 => dop853::tableau(),
            Self::Rkv98 => rkv98::tableau(),
            Self::Dopri5 => dopri5::tableau(),
            Self::Vern7 => vern7::tableau(),
            Self::Vern9 => vern9::tableau(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorControl {
    Absolute { eps: f64 },
    Scaled { rtol: f64, atol: f64 },
}

const DEFAULT_RTOL: f64 = 1e-9;
const DEFAULT_ATOL: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
pub struct IntegratorConfig {
    pub error_control: ErrorControl,
    pub h0: Option<f64>,
    pub h_min: f64,
    pub h_max: f64,
    pub max_steps: usize,
    pub max_rejects: usize,
    pub force_eval: bool,
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        Self {
            error_control: ErrorControl::Scaled {
                rtol: DEFAULT_RTOL,
                atol: DEFAULT_ATOL,
            },
            h0: None,
            h_min: 1e-12,
            h_max: 60.0,
            max_steps: 200_000,
            max_rejects: 50,
            force_eval: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationStatus {
    Success,
    MaxStepsExceeded,
    StepUnderflow,
    InvalidInput,
    /// Never produced: the solver reports a non-finite state as
    /// [`Self::NonFiniteState`], which superseded this variant. It is kept
    /// because it is not free to remove —
    /// `crates/lightyear_odeint_rs/src/integrator.rs` maps it onto
    /// `ObservedSolverTerminalStatus::NanEncountered`, which
    /// `nd_pipeline::solver_qualification::evidence` encodes as wire code 6
    /// and must still DECODE for receipts sealed before the rename. Removing
    /// the variant would renumber a versioned schema to delete an arm that
    /// costs nothing.
    ///
    /// `allow` and not `expect` on purpose: under `scalar-leg-observer` the
    /// `From<OdeIntegrationStatus> for ObservedSolverTerminalStatus` arm makes
    /// the variant reachable, so an expectation would report itself unfulfilled
    /// in exactly that lane. Gating the attribute on the feature instead would
    /// put a feature name in this module, which
    /// `rejected_a2_and_rhs_context_api_are_absent` forbids and is right to.
    #[allow(
        dead_code,
        reason = "unconstructible, but its wire code is still decoded by sealed receipts"
    )]
    NanEncountered,
    EventTriggered,
    NonFiniteState,
    MaxRejectsExceeded,
    EventInvalid,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IntegrationStats {
    pub steps: usize,
    pub evals: usize,
    /// Accepted steps on which the PI controller asked for `h > h_max` and was
    /// clipped by the cap.
    ///
    /// # Why this is not a diagnostic of a problem
    ///
    /// Saturation is benign and stable. At the saturated fixed point the
    /// controller's demand exceeds the cap by a wide margin — for Vern9 at
    /// `err/eps ~ 1e-3` the PI factor is `0.9 * (1e3)^(0.7/9) * (1e-3)^(0.4/9)
    /// = 1.13 > 1` — so `h` sits pinned against `h_max` and recovers on the
    /// first step where the demand drops. There is no windup.
    ///
    /// # What it is for
    ///
    /// It distinguishes "the error controller chose this step" from "`h_max`
    /// chose it", which is otherwise unobservable from outside the solver. A
    /// tolerance sweep run entirely in saturation converges at the rate of the
    /// CAP, not of the tolerance: tightening `eps` buys nothing because the
    /// controller was never the binding constraint. That produces a flat
    /// error-vs-`eps` plateau which is indistinguishable, from the endpoint
    /// state alone, from a stale-cache error floor — the exact failure a
    /// convergence gate exists to catch. `saturated_steps` is what lets a test
    /// tell those two apart. See
    /// `lightyear_odeint_rs/src/integrator.rs::production_band_saturation_is_measured_not_assumed`.
    ///
    /// Counted on ACCEPTED steps only, at the point of the controller update.
    /// Rejected steps cannot saturate: the reject branch clamps its factor to
    /// `[0.1, 0.5]` and `h <= h_max` already holds, so `h * factor < h_max`
    /// always. Steps shortened to land on `tf` or on a `t_eval` point are not
    /// counted either — this measures controller DEMAND against the cap, not
    /// the size of the step actually taken.
    ///
    /// **NOT instrumented on the DOPRI5-compat path.** `lightyear_compat.rs`
    /// carries its own copy of the controller and is left alone deliberately;
    /// this field is always 0 there, which means 0 is "no information", not
    /// "no saturation". Only trust it for the `solver.rs` explicit-RK and
    /// ESDIRK loops.
    pub saturated_steps: usize,
    /// Steps accepted DESPITE failing the error test, because `h` had already
    /// reached `h_min` and could not be reduced further.
    ///
    /// # This is an error-bound violation, and it is otherwise silent
    ///
    /// The accept condition is `err_norm <= accept_threshold || h_step.abs() <=
    /// h_min`. Once the second disjunct holds, a step is taken with an
    /// arbitrarily large local error and the run still terminates
    /// `IntegrationStatus::Success`. Worse, the accept branch resets `rejects =
    /// 0`, so `max_rejects` can never fire once the integration is pinned at
    /// `h_min` — the one guard that would otherwise surface the condition is
    /// disarmed by the very thing it should catch.
    ///
    /// Any nonzero value here means the returned state does NOT satisfy the
    /// tolerance that was requested, and `stats` is the only place that is
    /// recorded.
    ///
    /// # Why this is not currently a live defect
    ///
    /// Every production call site sets `h_min = 1e-12` s
    /// (`crates/lightyear_odeint_rs/src/integrator.rs`,
    /// `crates/lightyear_odeint_rs/src/adaptive_solver.rs`). For Vern9 to fail
    /// the error test at
    /// `h = 1e-12` the ninth derivative of the state would have to be around
    /// `1e100`. The path is unreachable for any non-pathological RHS, which is
    /// why this is instrumentation rather than a hard error: turning the
    /// condition into a terminal status would risk converting a survivable
    /// numerical hiccup into a failed propagation, for a case that does not
    /// occur.
    ///
    /// Reaching it at all means the RHS has gone badly non-smooth — a
    /// discontinuous force model, a table lookup falling off its domain, a NaN
    /// upstream of the finiteness checks — and the diagnosis belongs there, not
    /// in the controller.
    ///
    /// Not instrumented on the DOPRI5-compat path, same as
    /// [`Self::saturated_steps`]; that loop has the identical construct at
    /// `lightyear_compat.rs` with `compat_h_min = 1e-10`.
    pub underflow_accepts: usize,
    /// CUMULATIVE rejected steps over the whole integration.
    ///
    /// The local `rejects` counter this is taken alongside is a CONSECUTIVE
    /// count reset to zero on every accept, because its job is to drive
    /// `max_rejects`. It therefore cannot answer "what fraction of the work
    /// was thrown away", which is a different question and the one that
    /// decides whether a step-count reduction is an engineering fix or a
    /// science decision: a rejected step costs a full stage sweep and
    /// contributes nothing to the answer.
    ///
    /// Instrumented on the `solver.rs` explicit-RK and ESDIRK loops, which is
    /// what production Vern9 runs. Always 0 on the DOPRI5-compat path, same
    /// caveat as [`Self::saturated_steps`].
    pub rejected_steps: usize,
    /// Smallest step size the CONTROLLER chose over the run; 0.0 if none was.
    ///
    /// Steps shortened to land on `tf` or a `t_eval` point are excluded, and
    /// that exclusion is the whole content of this field. Without it the
    /// statistic is a minimum over every segment endpoint in the run — it read
    /// 0.000000000 s across a 2.45M-segment census, which says how many
    /// segments ran, not how the controller behaved. See `CACHE_CLUSTER_H_S`
    /// for why the aliasing argument this field used to carry was wrong by
    /// ~2,134x.
    pub min_accepted_h: f64,
    /// Sizes of this solver entry's FIRST [`RAMP_PROBE_STEPS`] accepted steps,
    /// in order; trailing entries stay 0.0 when the entry accepted fewer.
    ///
    /// Every solver entry restarts the controller with no memory of the previous
    /// entry, so the opening steps run below whatever step size the error
    /// controller would otherwise sustain. That costs ACCEPTED steps, which is
    /// why no rejection counter can see it — measured 2026-08-05, the mass lane
    /// runs 1.79M solver entries and rejects 238 steps in total. This field is
    /// what priced [`SHORT_SPAN_H0_S`], and it stays the instrument that would
    /// catch a residual ramp on either population.
    ///
    /// Paired with [`Self::tail_h_sum`] this measures the ramp directly instead
    /// of inferring it: the first entries are what the restart costs, the tail
    /// mean is what the controller sustains, and the difference is the prize.
    /// Aggregates cannot separate those — `steps/segment` and `sat_frac` gave
    /// contradictory answers (5.0 vs 30.4 predicted steps/segment against 8.036
    /// measured), which is what this field exists to settle.
    pub first_accepted_h: [f64; RAMP_PROBE_STEPS],
    /// Sum of accepted `h` AFTER the first [`RAMP_PROBE_STEPS`], with
    /// [`Self::tail_h_count`] as its denominator. Their quotient is the step
    /// size this entry actually sustained once the ramp finished.
    pub tail_h_sum: f64,
    /// Denominator for [`Self::tail_h_sum`]. Zero when the entry never got past
    /// the ramp, which is itself the answer for very short segments.
    pub tail_h_count: usize,
    /// `|tf - t0|` for this solver entry, so the ramp above can be attributed
    /// to a segment POPULATION after the fact. Encke deviation rebases and
    /// eclipse root-refinement legs share one segment counter and have very
    /// different spans (~574 s against a 10 s clamp), so a mean taken across
    /// both describes neither.
    pub segment_span_s: f64,
    /// Accepted steps below `CACHE_CLUSTER_H_S`. That constant is an UPPER
    /// BOUND, so this is a superset of the aliasing regime and not a hazard
    /// count — read its doc before quoting this number.
    pub cache_cluster_steps: usize,
    /// Same, but excluding steps shortened to land exactly on `tf` or a
    /// `t_eval` point. Those are an endpoint artifact, not the controller
    /// collapsing, and conflating them overstates the hazard.
    pub cache_cluster_steps_untruncated: usize,
    /// The step size the controller was asking for when this entry exited, in
    /// seconds and unsigned; 0.0 when the entry accepted no step.
    ///
    /// This is the post-clamp `h` written by the accept branch, NOT the size of
    /// the last step taken. The two differ on almost every entry, because the
    /// final step is shortened to land on `tf` — reading the last `accepted_h`
    /// instead would report an endpoint remainder as the sustained rate, which
    /// is the same confusion [`Self::min_accepted_h`] was already fixed for.
    pub final_controller_h: f64,
}

/// How many opening accepted steps of each solver entry are recorded
/// individually before the tail mean takes over.
///
/// Five, because a cold `h0` climbing at the accept-branch growth cap of
/// 3.2222 covers a 3.3x deficit in ~1.02 steps and a 52x deficit in ~3.5, so
/// five brackets both ends of the observed regime with a slot to spare.
pub const RAMP_PROBE_STEPS: usize = 5;

/// UPPER BOUND on the step size below which two distinct Vern9 stage times can
/// alias to one entry of the `equinoc2eci` baseline cache.
///
/// **Both operands of the old `0.1 / 0.0624` were wrong, and it is kept as a
/// bound rather than silently retuned.** Measured 2026-08-05:
///
/// - `0.1` is the CLAMP CEILING of `baseline_cache_tol`, not its value. The
///   live tolerance is `(eps * 2.6e3).clamp(1e-9, 0.1)`
///   (`lightyear_odeint_rs/src/rhs.rs`), which at the sealed `eps = 1e-8` is
///   **2.6e-5 s** -- 3,846x tighter. The constructor's ~0.1 value is overwritten
///   by `adapt_cache_policy_for_eps` at every entry point.
/// - `0.0624` is `c[2] - c[1]` and was documented as "the closest distinct
///   nodes". It is the **8th** smallest gap. The closest distinct pair is
///   `c[7] = 0.645` / `c[11] = 0.659065` at 0.014065, and the gap that actually
///   governs a ONE-SLOT cache is the smallest CONSECUTIVE-in-evaluation-order
///   one, `c[1] - c[0] = 0.03462`.
///
/// So the real threshold at production `eps` is `2.6e-5 / 0.03462 = 7.51e-4 s`,
/// and the old constant overstated the hazard regime by **~2,134x in `h`**. Its
/// 4,720,749 has already been cited once as a hazard count; it was a count of
/// steps below 1.6 s and said nothing about aliasing.
///
/// This value is now `ceiling / correct_gap`, which is a true upper bound for
/// ANY `eps` and therefore safe to compare against — but it is a superset, not
/// a hazard count. **An `eps`-derived threshold and a tag-sharded counter are
/// what make this attributable to a lane; neither exists yet.** Until they do,
/// a nonzero `cache_cluster_steps` is not evidence of a hazard.
pub const CACHE_CLUSTER_H_S: f64 = 0.1 / 0.03462;

/// Span at or below which the opening step is `span/2` rather than `span/100`.
///
/// A solver entry restarts the controller with no memory of the previous entry,
/// so `h0` is a guess and the guess costs ACCEPTED steps when it is low — which
/// is why no rejection counter can see the cost. Measured 2026-08-05 over a
/// 2.45M-segment census, the two populations sharing this entry point behave
/// nothing alike:
///
/// | population | span/entry | `span/100` | h sustained |
/// |---|---|---|---|
/// | short (clamped root legs) | 2.126 s | 0.021 s | **2.006 s** |
/// | long (Encke rebases) | 985.866 s | 9.859 s | 112.579 s |
///
/// The short population opens **94x** below the step the controller then
/// sustains on a segment that needs about one step. The long population's guess
/// is low too, but only by ~11x, and raising it is a different and worse trade:
/// a uniform `span/10` puts long-population `h0` at 131 s against a sustained
/// 112.6 s, i.e. ABOVE equilibrium, and that variant produced 2 NaN masses and
/// four more failed propagations than doing nothing.
///
/// So this threshold exists to separate the populations, and 60 s separates them
/// with two orders of magnitude of clearance on either side (2.1 s against
/// 985.9 s). It is not a tuned edge; any value in the tens of seconds selects
/// the same two sets.
///
/// # The remaining ramp is not reachable by carrying `h` between entries
///
/// Measured 2026-08-09 on the boundary axis (`probe::observe_ramp`), two
/// independent corpora agreeing to ~0.05 pp. The residual ramp is worth 16.5%
/// of all accepted steps — 14.13% of it on the long population, which still
/// opens at 33.97 s against a sustained 122.07 s — and **every step of it sits
/// on a boundary where `reset_for_propagation` has just moved the Encke
/// baseline.** Carrying `h` across one of those was refuted at `c546130` for a
/// 10.77 m endpoint breach and two non-finite masses, and raising the guess by
/// formula instead is the `span/10` variant refuted in the table above.
///
/// The one boundary in the scalar lane that a carry WOULD be sound across —
/// the eclipse root transaction's window leg, which opens on the same
/// baseline and the same deviation the proof leg left behind — carries 3.10%
/// of steps and cannot pay: entries equal steps exactly in both corpora, no
/// entry reaches a second slot, and the mean opening `h` is 2.497e-6 s. That
/// leg spans the root's uncertainty bound, so it takes one step by
/// construction and is already at the floor.
const SHORT_SPAN_H0_S: f64 = 60.0;

/// The opening step for a segment of `span` seconds when the caller named none.
///
/// `span/100` came from the DOPRI5 compat path and is applied to both segment
/// populations; see [`SHORT_SPAN_H0_S`] for why only the short one is retargeted.
///
/// **The compat path itself keeps `dt_total/100` and must.** `lightyear_compat`
/// exists to reproduce a reference DOPRI5 step sequence, so its opening step is
/// part of the contract rather than a tunable; the apparent inconsistency with
/// this function is deliberate and is not a bug to fix.
/// # Widening the `span/2` rule to 300 s is REFUTED (R43, 2026-08-10)
///
/// The scalar strict-HF lane has a population the two above do not: the
/// eclipse bracket-replay leg re-integrates exactly one accepted step, so it
/// spans ~82 s at Vern7 — just above `SHORT_SPAN_H0_S` — and opens at
/// `span/100` ~= 0.8 s against a `dt_max` the eclipse coordinator has already
/// clamped to 10 s, burning three steps climbing back to a cap it then sits on
/// for the whole leg. Raising the threshold to production `dt_max` fixes
/// exactly that and looks, on the scalar lane, like a clean win: -3.38% corpus
/// RHS evaluations, accuracy flat, the unclamped population all but untouched,
/// and the V3 arc endpoint holding to 5.74 micrometres across 25 fewer steps.
///
/// **It loses on the lane the campaign actually spends its time in.** Measured
/// on the now-retired p2g1 hybrid measurement harness, base against the change:
///
/// ```text
///   rhs_evals   2,209,782,523 -> 2,124,574,494   -3.86%
///   steps         218,541,620 ->   209,974,584   -3.92%
///   rejected          187,202 ->       237,469  +26.85%
///   wall_s            239.736 ->       441.025   +84.0%
/// ```
///
/// The evaluation saving DOES carry over — and it is worth nothing. Rejections
/// rise 26.9%, and the cell wall nearly doubles. The wall is also no longer
/// reproducible: two runs of this arm read 271.2 s and 441.0 s (+62.6%) where
/// two runs of the base read 238.2 s and 239.7 s (+0.66%).
///
/// The cause is the failure mode `SHORT_SPAN_H0_S` already documents, on a
/// population nobody had checked. A mid-span mass-lane leg is ERROR-bound: its
/// equilibrium step is well below `span/2`, so opening there starts it ABOVE
/// equilibrium and it rejects its way back down. A mid-span eclipse leg is
/// CAP-bound: its equilibrium is far above the clamp, so opening at `span/2`
/// merely reaches the cap immediately. Both are "spans in (60, 300]" and
/// nothing available at `h0` time separates them — the distinguishing quantity
/// is the equilibrium step, which is what the integration is about to
/// discover. So "300 s is production `dt_max`, therefore `span/2` can only be
/// clipped to the cap, therefore this is clear of the refuted variant" is
/// WRONG, and it was the reasoning that motivated the change: being clipped to
/// the cap is only harmless when the cap is what binds.
///
/// Do not re-propose this without an equilibrium-step estimate at entry. The
/// eval count alone will look like a win again; the counter that catches it is
/// `rejected`, and the harness that shows it is the hybrid cell, not the arc.
fn default_h0(span: f64) -> f64 {
    if span <= SHORT_SPAN_H0_S {
        span / 2.0
    } else {
        span / 100.0
    }
}

/// How `sanitize_event` produced the event state it hands back.
///
/// Consumers mapping onto a coarser interpolation taxonomy must preserve the
/// historical string mapping: `Handler` and `Clamp` carry the handler's own
/// state (formerly `"handler"`/`"clamp"`, which downstream parsers treated as
/// "no interpolation"), while `Linear` and `LinearClamp` mark the linear
/// fallback state (formerly `"linear"`/`"linear_clamp"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SanitizedInterp {
    Handler,
    Clamp,
    Linear,
    LinearClamp,
}

#[derive(Debug, Clone)]
pub struct IntegrationEvent {
    pub t: f64,
    pub y: Vec<f64>,
    pub interp_method: SanitizedInterp,
    pub interp_error: f64,
}

#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// The time the integration actually reached. Every production caller
    /// reads `y` and `status` and knows the endpoint it asked for; the checks
    /// that the solver landed on that endpoint bit-for-bit are the only
    /// readers.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read only by the endpoint-identity checks")
    )]
    pub t: f64,
    pub y: Vec<f64>,
    pub status: IntegrationStatus,
    pub stats: IntegrationStats,
    pub event: Option<IntegrationEvent>,
}

/// Gustafsson PI step-size controller, accept branch.
///
/// One source for the block `integrate_internal` and
/// `integrate_internal_esdirk` used to carry as verbatim copies. A macro and
/// deliberately NOT an extracted function: a function boundary in this loop is
/// itself a perf lever (an `#[inline]` hint on an extracted stage measured
/// ~10% here), while macro expansion reproduces the exact token stream at both
/// sites, so codegen and the FP sequence are unchanged by construction.
///
/// Floors zero error to `accept_threshold * 0.1` to avoid degenerate growth.
/// Writes `h`, `stats`, and the PI memory (`err_prev`, `have_err_prev`,
/// `just_rejected`).
macro_rules! pi_controller_accept {
    (
        $err_norm:ident, $accept_threshold:ident, $error_control:ident,
        $order:ident, $inv_order:ident, $just_rejected:ident, $h_step:ident,
        $h_min:ident, $h_max:ident, $direction:ident, $h:ident, $stats:ident,
        $err_prev:ident, $have_err_prev:ident
    ) => {
        let eff_err = if $err_norm > 0.0 {
            $err_norm
        } else {
            $accept_threshold * 0.1
        };
        let demanded_h = match $error_control {
            ErrorControl::Absolute { eps } => {
                let factor = if $have_err_prev && $err_prev > 0.0 {
                    let alpha = 0.7 / $order;
                    let beta = 0.4 / $order;
                    0.9 * (eps / eff_err).powf(alpha) * ($err_prev / eps).powf(beta)
                } else {
                    0.9 * (eps / eff_err).powf($inv_order)
                };
                let max_growth_base = (1.0 + 4.0 * (5.0 / $order)).min(5.0);
                let max_growth = if $just_rejected {
                    max_growth_base.min(2.0)
                } else {
                    max_growth_base
                };
                let factor = factor.clamp(0.2, max_growth);
                $h_step.abs() * factor
            }
            ErrorControl::Scaled { .. } => {
                let factor = if $have_err_prev && $err_prev > 0.0 {
                    let alpha = 0.7 / $order;
                    let beta = 0.4 / $order;
                    0.9 * (1.0 / eff_err).powf(alpha) * $err_prev.powf(beta)
                } else {
                    0.9 * eff_err.powf(-$inv_order)
                };
                let max_growth_base = (1.0 + 4.0 * (5.0 / $order)).min(5.0);
                let max_growth = if $just_rejected {
                    max_growth_base.min(2.0)
                } else {
                    max_growth_base
                };
                let factor = factor.clamp(0.2, max_growth);
                $h_step.abs() * factor
            }
        };
        // The cap, not the error controller, set the next step.
        if demanded_h > $h_max {
            $stats.saturated_steps += 1;
        }
        $h = demanded_h.clamp($h_min, $h_max) * $direction;
        $stats.final_controller_h = $h.abs();
        $err_prev = $err_norm;
        $have_err_prev = true;
        $just_rejected = false;
    };
}

/// Reject-branch step cut, twin of [`pi_controller_accept!`] (same
/// macro-not-fn reasoning). Writes only `h` and `just_rejected`; the PI
/// memory (`err_prev`, `have_err_prev`) is deliberately untouched — the
/// explicit-RK reject branch in `integrate_internal` carries the full note.
macro_rules! pi_controller_reject {
    (
        $err_norm:ident, $error_control:ident, $inv_order:ident,
        $h_step:ident, $h_min:ident, $h_max:ident, $direction:ident,
        $h:ident, $just_rejected:ident
    ) => {
        $h = match $error_control {
            ErrorControl::Absolute { eps } => {
                let factor = 0.9 * (eps / $err_norm).powf($inv_order);
                let factor = factor.clamp(0.1, 0.5);
                ($h_step.abs() * factor).clamp($h_min, $h_max) * $direction
            }
            ErrorControl::Scaled { .. } => {
                let factor = 0.9 * $err_norm.powf(-$inv_order);
                let factor = factor.clamp(0.1, 0.5);
                ($h_step.abs() * factor).clamp($h_min, $h_max) * $direction
            }
        };
        $just_rejected = true;
    };
}

#[derive(Debug, Clone)]
pub struct IntegrationResultSampled {
    pub times: Vec<f64>,
    pub states: Vec<f64>,
    pub n_state: usize,
    pub status: IntegrationStatus,
    pub stats: IntegrationStats,
    pub event: Option<IntegrationEvent>,
}

pub enum SampleSink<'a> {
    Vec {
        states: Vec<f64>,
    },
    Slice {
        states: &'a mut [f64],
        written: usize,
        valid: bool,
    },
}

impl<'a> SampleSink<'a> {
    #[must_use]
    pub fn vec(capacity: usize) -> Self {
        Self::Vec {
            states: Vec::with_capacity(capacity),
        }
    }

    pub const fn slice(states: &'a mut [f64]) -> Self {
        Self::Slice {
            states,
            written: 0,
            valid: true,
        }
    }

    #[inline]
    pub fn push_state(&mut self, state: &[f64]) {
        match self {
            Self::Vec { states } => states.extend_from_slice(state),
            Self::Slice {
                states,
                written,
                valid,
            } => {
                let end = written.saturating_add(state.len());
                if let Some(destination) = states.get_mut(*written..end) {
                    destination.copy_from_slice(state);
                    *written = end;
                } else {
                    *valid = false;
                }
            }
        }
    }

    #[must_use]
    pub const fn written(&self) -> usize {
        match self {
            Self::Vec { states } => states.len(),
            Self::Slice { written, .. } => *written,
        }
    }

    #[must_use]
    pub const fn valid(&self) -> bool {
        match self {
            Self::Vec { .. } => true,
            Self::Slice { valid, .. } => *valid,
        }
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<f64> {
        match self {
            Self::Vec { states } => states,
            Self::Slice { .. } => Vec::new(),
        }
    }
}

pub trait OdeSystem {
    fn rhs(&self, t: f64, y: &[f64], dy: &mut [f64]);

    /// Called by the integrator after a step is rejected (error too large).
    /// Implementations can use this to invalidate subcycle caches that assumed
    /// a monotonic time advance.  Default is a no-op.
    fn on_step_reject(&self) {}

    /// Called once per explicit step, before any of its stages are evaluated,
    /// with the step's start time, its size, and the tableau's abscissas.
    ///
    /// Stage `i` will be evaluated at `t + nodes[i] * h`. An implementation
    /// whose `rhs` resolves something that depends on the stage TIME alone can
    /// resolve the whole set here, where the stages are independent of one
    /// another, instead of one at a time inside the stage loop, where they are
    /// not. Nothing is promised about which stages actually get evaluated: a
    /// rejected step evaluates all of them and then throws the step away, and a
    /// failing one may stop early, so this is a hint and never an obligation.
    ///
    /// Default is a no-op.
    fn prefill_stage_times(&self, _t: f64, _h: f64, _nodes: &[f64]) {}
}

#[derive(Debug, Clone)]
pub enum EventDecision {
    Continue,
    Stop { t_event: f64, y_event: Vec<f64> },
}

pub trait EventHandler {
    fn on_step(
        &mut self,
        prev_t: f64,
        prev_y: &[f64],
        prev_dy: &[f64],
        next_t: f64,
        next_y: &[f64],
        next_dy: &[f64],
    ) -> EventDecision;
}

#[inline]
fn axpy(scale: f64, x: &[f64], y: &mut [f64]) {
    if scale == 0.0 {
        return;
    }
    // Pre-sliced range: LLVM proves bounds once, then auto-vectorizes the
    // loop to vfmadd (AVX2) or fmla (NEON) with fp-contract=fast.
    for (&source, destination) in x.iter().zip(y.iter_mut()) {
        *destination = source.mul_add(scale, *destination);
    }
}

#[inline]
pub fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

pub fn sanitize_event(
    prev_t: f64,
    next_t: f64,
    prev_y: &[f64],
    next_y: &[f64],
    mut t_event: f64,
    mut y_event: Vec<f64>,
    direction: f64,
) -> Result<(f64, Vec<f64>, SanitizedInterp, f64), IntegrationStatus> {
    let n = prev_y.len();
    let (t_min, t_max) = if direction >= 0.0 {
        (prev_t, next_t)
    } else {
        (next_t, prev_t)
    };

    let mut method = if t_event.is_finite() {
        SanitizedInterp::Handler
    } else {
        t_event = next_t;
        SanitizedInterp::Clamp
    };
    if t_event < t_min {
        t_event = t_min;
        method = SanitizedInterp::Clamp;
    } else if t_event > t_max {
        t_event = t_max;
        method = SanitizedInterp::Clamp;
    }

    let denom = next_t - prev_t;
    let mut tau = if denom.abs() > 0.0 {
        (t_event - prev_t) / denom
    } else {
        0.0
    };
    tau = tau.clamp(0.0, 1.0);

    let mut y_linear = Vec::with_capacity(n);
    for (&previous, &next) in prev_y.iter().zip(next_y) {
        y_linear.push(previous + tau * (next - previous));
    }

    let mut interp_error = 0.0;
    let y_valid = y_event.len() == n && all_finite(&y_event);
    if y_valid {
        for (&event_value, &linear_value) in y_event.iter().zip(&y_linear) {
            let diff = (event_value - linear_value).abs();
            if diff > interp_error {
                interp_error = diff;
            }
        }
    } else {
        y_event = y_linear;
        method = if method == SanitizedInterp::Clamp {
            SanitizedInterp::LinearClamp
        } else {
            SanitizedInterp::Linear
        };
        interp_error = 0.0;
    }

    if !all_finite(&y_event) {
        return Err(IntegrationStatus::EventInvalid);
    }

    Ok((t_event, y_event, method, interp_error))
}

pub fn integrate_final<S: OdeSystem>(
    system: &S,
    method: Method,
    y0: &[f64],
    t0: f64,
    tf: f64,
    config: IntegratorConfig,
) -> IntegrationResult {
    integrate_final_with_scratch(
        system,
        method,
        y0,
        t0,
        tf,
        config,
        &mut SolverScratch::new(),
    )
}

/// [`integrate_final`] over a caller-owned workspace.
///
/// Identical semantics; the only difference is who owns the step-loop buffers.
/// Callers that integrate many segments should hold one [`SolverScratch`] and
/// pass it here rather than paying nine allocations per segment.
pub fn integrate_final_with_scratch<S: OdeSystem>(
    system: &S,
    method: Method,
    y0: &[f64],
    t0: f64,
    tf: f64,
    config: IntegratorConfig,
    scratch: &mut SolverScratch,
) -> IntegrationResult {
    integrate_internal(
        system, method, y0, t0, tf, None, config, None, None, None, false, scratch,
    )
}

pub fn integrate_final_with_events<S: OdeSystem, E: EventHandler>(
    system: &S,
    method: Method,
    y0: &[f64],
    t0: f64,
    tf: f64,
    config: IntegratorConfig,
    event_handler: &mut E,
) -> IntegrationResult {
    integrate_final_with_events_and_scratch(
        system,
        method,
        y0,
        t0,
        tf,
        config,
        event_handler,
        &mut SolverScratch::new(),
    )
}

/// [`integrate_final_with_events`] over a caller-owned workspace.
pub fn integrate_final_with_events_and_scratch<S: OdeSystem, E: EventHandler>(
    system: &S,
    method: Method,
    y0: &[f64],
    t0: f64,
    tf: f64,
    config: IntegratorConfig,
    event_handler: &mut E,
    scratch: &mut SolverScratch,
) -> IntegrationResult {
    integrate_internal(
        system,
        method,
        y0,
        t0,
        tf,
        None,
        config,
        Some(event_handler),
        None,
        None,
        false,
        scratch,
    )
}

pub fn integrate_sampled<S: OdeSystem>(
    system: &S,
    method: Method,
    y0: &[f64],
    t_eval: &[f64],
    config: IntegratorConfig,
) -> IntegrationResultSampled {
    let n_state = y0.len();
    let mut config = config;
    config.force_eval = true;
    let mut sink = SampleSink::vec(t_eval.len().saturating_mul(n_state));
    let result = integrate_internal(
        system,
        method,
        y0,
        t_eval.first().copied().unwrap_or(0.0),
        t_eval.last().copied().unwrap_or(0.0),
        Some(t_eval),
        config,
        None,
        None,
        Some(&mut sink),
        false,
        &mut SolverScratch::new(),
    );
    let states = sink.into_vec();

    IntegrationResultSampled {
        times: t_eval.to_vec(),
        states,
        n_state,
        status: result.status,
        stats: result.stats,
        event: result.event,
    }
}

pub fn integrate_sampled_into<S: OdeSystem>(
    system: &S,
    method: Method,
    y0: &[f64],
    t_eval: &[f64],
    config: IntegratorConfig,
    states_out: &mut [f64],
) -> IntegrationResult {
    let n_state = y0.len();
    let expected_len = t_eval.len().saturating_mul(n_state);
    if states_out.len() != expected_len {
        return IntegrationResult {
            t: t_eval.first().copied().unwrap_or(0.0),
            y: Vec::new(),
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }
    let mut config = config;
    config.force_eval = true;
    let (mut result, written, valid) = {
        let mut sink = SampleSink::slice(states_out);
        let result = integrate_internal(
            system,
            method,
            y0,
            t_eval.first().copied().unwrap_or(0.0),
            t_eval.last().copied().unwrap_or(0.0),
            Some(t_eval),
            config,
            None,
            None,
            Some(&mut sink),
            false,
            &mut SolverScratch::new(),
        );
        (result, sink.written(), sink.valid())
    };
    if !valid || written != expected_len {
        states_out.fill(0.0);
    }
    result.y.clear();
    result
}

pub fn integrate_sampled_with_events<S: OdeSystem>(
    system: &S,
    method: Method,
    y0: &[f64],
    t_eval: &[f64],
    config: IntegratorConfig,
    event_handler: &mut dyn EventHandler,
) -> IntegrationResultSampled {
    let mut config = config;
    config.force_eval = true;
    let sample_capacity = t_eval.len().saturating_add(1);
    let mut output_times: Vec<f64> = Vec::with_capacity(sample_capacity);
    let mut sink = SampleSink::vec(sample_capacity.saturating_mul(y0.len()));
    let result = integrate_internal(
        system,
        method,
        y0,
        t_eval.first().copied().unwrap_or(0.0),
        t_eval.last().copied().unwrap_or(0.0),
        Some(t_eval),
        config,
        Some(event_handler),
        Some(&mut output_times),
        Some(&mut sink),
        false,
        &mut SolverScratch::new(),
    );

    let n_state = y0.len();
    let states = sink.into_vec();

    IntegrationResultSampled {
        times: output_times,
        states,
        n_state,
        status: result.status,
        stats: result.stats,
        event: result.event,
    }
}

/// Sample accepted-step dense output without forcing RK steps onto `t_eval`.
///
/// The optional handler inspects each accepted step before its samples are
/// published, so a discarded event step cannot leak output. `None` is the
/// direct no-event form; it does not route through a fake handler.
pub fn integrate_sampled_unforced<S: OdeSystem>(
    system: &S,
    method: Method,
    y0: &[f64],
    t_eval: &[f64],
    mut config: IntegratorConfig,
    event_handler: Option<&mut dyn EventHandler>,
) -> IntegrationResultSampled {
    config.force_eval = false;
    let sample_capacity = t_eval.len().saturating_add(1);
    let mut output_times: Vec<f64> = Vec::with_capacity(sample_capacity);
    let mut sink = SampleSink::vec(sample_capacity.saturating_mul(y0.len()));
    let result = integrate_internal(
        system,
        method,
        y0,
        t_eval.first().copied().unwrap_or(0.0),
        t_eval.last().copied().unwrap_or(0.0),
        Some(t_eval),
        config,
        event_handler,
        Some(&mut output_times),
        Some(&mut sink),
        true,
        &mut SolverScratch::new(),
    );

    IntegrationResultSampled {
        times: output_times,
        states: sink.into_vec(),
        n_state: y0.len(),
        status: result.status,
        stats: result.stats,
        event: result.event,
    }
}

const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << f64::MANTISSA_DIGITS;
const F64_LOWER_WORD_SCALE: f64 = 4_294_967_296.0;

fn state_dimension_as_f64(value: usize) -> Result<f64, IntegrationStatus> {
    let value = u64::try_from(value).map_err(|_| IntegrationStatus::InvalidInput)?;
    if value > MAX_EXACT_F64_INTEGER {
        return Err(IntegrationStatus::InvalidInput);
    }
    let high_word = u32::try_from(value >> 32).map_err(|_| IntegrationStatus::InvalidInput)?;
    let low_word =
        u32::try_from(value & u64::from(u32::MAX)).map_err(|_| IntegrationStatus::InvalidInput)?;
    Ok(f64::from(high_word) * F64_LOWER_WORD_SCALE + f64::from(low_word))
}

/// Reusable step-loop workspace for [`integrate_internal`].
///
/// Nine buffers are sized from the state dimension and the tableau stage count
/// and are pure scratch: every one is fully written before it is read within a
/// step, and none escapes the call. Allocating them per entry costs nine
/// malloc/free pairs for every solver entry, and a segment reaches its endpoint
/// in under four accepted steps, so there is almost nothing to amortize them
/// over. A caller that integrates many segments -- a reusable integrator, an
/// eclipse-coordinator lane -- owns one of these instead and pays once.
///
/// # Why reuse is bit-neutral, and what the zero-fill is actually for
///
/// Reuse is safe because [`rk_step`] initializes every buffer it accumulates
/// into, at the top of each step: `y_next` and `err` are zeroed in place, and
/// `primary_error_compensation` (plus `secondary_error_compensation` when
/// `has_secondary_error`) get an explicit `fill(0.0)`. `y_tmp`, `dy_next` and
/// `dense_sample` are fully written before they are read. Nothing carries state
/// from one call to the next, so a scratch that arrives dirty produces the same
/// bits as one that arrives fresh.
///
/// [`SolverScratch::prepare`] nonetheless restores every buffer to exactly what
/// `vec![0.0; len]` would give -- `clear` then `resize`, so the surviving prefix
/// is zeroed too, not just the growth. That is **defence in depth, not a
/// correctness requirement today**: it is measured to be unnecessary on the
/// current paths (removing the `clear` does not move any test), and it exists so
/// that a future buffer which is *not* self-initializing cannot silently read a
/// previous segment's bytes. Two of the nine (`err3`, `dense_sample`) are inert
/// on the production Vern9 non-dense path, so that class of bug would be
/// invisible to any test that only checks production output.
#[derive(Default, Debug)]
pub struct SolverScratch {
    k: Vec<f64>,
    y_tmp: Vec<f64>,
    y_next: Vec<f64>,
    err: Vec<f64>,
    err3: Vec<f64>,
    primary_error_compensation: Vec<f64>,
    secondary_error_compensation: Vec<f64>,
    dy_next: Vec<f64>,
    dense_sample: Vec<f64>,
}

impl SolverScratch {
    /// Empty workspace; the first [`SolverScratch::prepare`] sizes it.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            k: Vec::new(),
            y_tmp: Vec::new(),
            y_next: Vec::new(),
            err: Vec::new(),
            err3: Vec::new(),
            primary_error_compensation: Vec::new(),
            secondary_error_compensation: Vec::new(),
            dy_next: Vec::new(),
            dense_sample: Vec::new(),
        }
    }

    /// Resize every buffer to this call's shape and zero it.
    ///
    /// Retains capacity across calls; that is the whole point. The zero-fill is
    /// what makes reuse indistinguishable from a fresh allocation.
    fn prepare(&mut self, n: usize, stage_value_count: usize) {
        let size_to = |buffer: &mut Vec<f64>, len: usize| {
            buffer.clear();
            buffer.resize(len, 0.0);
        };
        size_to(&mut self.k, stage_value_count);
        size_to(&mut self.y_tmp, n);
        size_to(&mut self.y_next, n);
        size_to(&mut self.err, n);
        size_to(&mut self.err3, n);
        size_to(&mut self.primary_error_compensation, n);
        size_to(&mut self.secondary_error_compensation, n);
        size_to(&mut self.dy_next, n);
        size_to(&mut self.dense_sample, n);
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve established IEEE operation order in the explicit RK kernel"
)]
#[expect(
    clippy::float_cmp,
    reason = "exact endpoint and direction comparisons define integration control flow"
)]
#[expect(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "the integration kernel intentionally uses standard RK notation"
)]
fn integrate_internal<S: OdeSystem>(
    system: &S,
    method: Method,
    y0: &[f64],
    t0: f64,
    tf: f64,
    t_eval: Option<&[f64]>,
    config: IntegratorConfig,
    mut event_handler: Option<&mut dyn EventHandler>,
    mut output_times: Option<&mut Vec<f64>>,
    mut sample_sink: Option<&mut SampleSink<'_>>,
    dense_unforced: bool,
    scratch: &mut SolverScratch,
) -> IntegrationResult {
    let n = y0.len();
    if n == 0 {
        return IntegrationResult {
            t: t0,
            y: Vec::new(),
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    let state_dimension = match state_dimension_as_f64(n) {
        Ok(dimension) => dimension,
        Err(status) => {
            return IntegrationResult {
                t: t0,
                y: y0.to_vec(),
                status,
                stats: IntegrationStats::default(),
                event: None,
            };
        }
    };

    if !t0.is_finite() || !tf.is_finite() {
        return IntegrationResult {
            t: t0,
            y: y0.to_vec(),
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    match config.error_control {
        ErrorControl::Absolute { eps } => {
            if !eps.is_finite() || eps <= 0.0 {
                return IntegrationResult {
                    t: t0,
                    y: y0.to_vec(),
                    status: IntegrationStatus::InvalidInput,
                    stats: IntegrationStats::default(),
                    event: None,
                };
            }
        }
        ErrorControl::Scaled { rtol, atol } => {
            if !rtol.is_finite() || !atol.is_finite() || rtol <= 0.0 || atol < 0.0 {
                return IntegrationResult {
                    t: t0,
                    y: y0.to_vec(),
                    status: IntegrationStatus::InvalidInput,
                    stats: IntegrationStats::default(),
                    event: None,
                };
            }
        }
    }

    let tableau = method.tableau();
    let stages = tableau.stages;

    let direction = if tf >= t0 { 1.0 } else { -1.0 };
    let span = (tf - t0).abs();

    if span == 0.0 {
        return IntegrationResult {
            t: tf,
            y: y0.to_vec(),
            status: IntegrationStatus::Success,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    let mut h_max = config.h_max.abs();
    if !h_max.is_finite() || h_max <= 0.0 {
        h_max = span;
    }
    let mut h_min = config.h_min.abs();
    if !h_min.is_finite() || h_min <= 0.0 {
        h_min = 1e-12;
    }
    if h_max < h_min {
        h_max = h_min;
    }

    // `segment_span_s` is recorded so the ramp above can be split by segment
    // POPULATION afterwards: Encke rebases and eclipse root legs share one
    // counter and differ by ~57x in span, so one mean over both describes
    // neither.
    let mut stats = IntegrationStats {
        segment_span_s: span,
        ..Default::default()
    };

    let mut h = config.h0.map_or_else(
        || default_h0(span).clamp(h_min, h_max) * direction,
        |h0| h0.abs().clamp(h_min, h_max) * direction,
    );

    let mut y = y0.to_vec();
    let mut t = t0;

    let mut rejects = 0usize;

    let Some(stage_value_count) = stages.checked_mul(n) else {
        return IntegrationResult {
            t,
            y,
            status: IntegrationStatus::InvalidInput,
            stats,
            event: None,
        };
    };
    scratch.prepare(n, stage_value_count);
    let SolverScratch {
        k,
        y_tmp,
        y_next,
        err,
        err3,
        primary_error_compensation,
        secondary_error_compensation,
        dy_next,
        dense_sample,
    } = scratch;

    let mut k1_cache: Option<Vec<f64>> = None;
    // PI step-size controller state (Gustafsson)
    let mut err_prev: f64 = 0.0;
    let mut have_err_prev = false;
    let mut just_rejected = false;
    // Kahan compensated summation for time variable
    let mut t_comp: f64 = 0.0;
    let has_err3 = tableau.err3.is_some();
    // Whether to fold this tableau's third-order embedded estimate into the
    // error norm at a given tolerance. Both halves are tableau properties; see
    // `Tableau::err3_min_eps`.
    let err3_min_eps = tableau.err3_min_eps;
    let use_err3_blend = move |eps: f64| -> bool {
        match (has_err3, err3_min_eps) {
            (true, Some(min_eps)) => eps >= min_eps,
            _ => false,
        }
    };

    let mut eval_idx = 0usize;
    let eval_tol = 1e-12;

    if let Some(eval_times) = t_eval {
        if eval_times.is_empty() {
            return IntegrationResult {
                t,
                y,
                status: IntegrationStatus::InvalidInput,
                stats,
                event: None,
            };
        }
        if !is_sorted_dir(eval_times, direction) {
            return IntegrationResult {
                t,
                y,
                status: IntegrationStatus::InvalidInput,
                stats,
                event: None,
            };
        }
        let (min_t, max_t) = if direction >= 0.0 { (t0, tf) } else { (tf, t0) };
        if eval_times.iter().any(|&time| time < min_t || time > max_t) {
            return IntegrationResult {
                t,
                y,
                status: IntegrationStatus::InvalidInput,
                stats,
                event: None,
            };
        }
    }

    if let Some(eval_times) = t_eval {
        if eval_times
            .get(eval_idx)
            .is_some_and(|time| time.to_bits() == t.to_bits())
        {
            push_sample(&mut sample_sink, &y);
            if let Some(times) = output_times.as_deref_mut() {
                times.push(t);
            }
            eval_idx += 1;
        }
    }

    let err_coeffs = if tableau.err.is_none() {
        tableau.b_hat.map(|b_hat| {
            tableau
                .b
                .iter()
                .zip(b_hat.iter())
                .map(|(b, b_hat)| b - b_hat)
                .collect::<Vec<f64>>()
        })
    } else {
        None
    };

    let order = f64::from(tableau.order_err) + 1.0;
    let inv_order = 1.0 / order;

    // Is stage 0 of this tableau exactly `f(t, y)`?
    //
    // True for every explicit tableau in `tableau/` (`c[0] == 0`, empty first
    // `a` row), but it is the precondition for treating a derivative evaluated
    // at a step's own base point as interchangeable with `k[0]`, so it is
    // CHECKED rather than assumed. A tableau with a non-zero `c[0]` would
    // otherwise silently receive the wrong first stage.
    let first_stage_is_rhs =
        tableau
            .c
            .first()
            .zip(tableau.a.first())
            .is_some_and(|(&first_node, first_row)| {
                first_node == 0.0 && first_row.iter().all(|&a_0j| a_0j == 0.0)
            });

    while tf != t && (tf - t).signum() == direction {
        if stats.steps >= config.max_steps {
            return IntegrationResult {
                t,
                y: outputs_or_final(t_eval, &y, n),
                status: IntegrationStatus::MaxStepsExceeded,
                stats,
                event: None,
            };
        }

        let mut h_step = h;
        let remaining = tf - t;
        if remaining.abs() < h_step.abs() {
            h_step = remaining;
        }

        if config.force_eval {
            if let Some(eval_times) = t_eval {
                if let Some(&next_eval) = eval_times.get(eval_idx) {
                    let dt_to_eval = next_eval - t;
                    if dt_to_eval.signum() == direction
                        && dt_to_eval.abs() > eval_tol
                        && dt_to_eval.abs() < h_step.abs()
                    {
                        h_step = dt_to_eval;
                    }
                }
            }
        }

        let lands_on_tf = h_step.to_bits() == remaining.to_bits();

        if h_step.abs() < h_min && !lands_on_tf {
            return IntegrationResult {
                t,
                y: outputs_or_final(t_eval, &y, n),
                status: IntegrationStatus::StepUnderflow,
                stats,
                event: None,
            };
        }

        let reuse_k1 = k1_cache.as_deref();
        let evals = match rk_step(
            system,
            tableau,
            t,
            &y,
            h_step,
            k,
            y_tmp,
            y_next,
            err,
            err3,
            primary_error_compensation,
            secondary_error_compensation,
            reuse_k1,
            err_coeffs.as_deref(),
            None, // err3_coeffs come from tableau.err3 inside rk_step
        ) {
            Ok(evaluations) => evaluations,
            Err(step_status) => {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: step_status,
                    stats,
                    event: None,
                };
            }
        };
        stats.evals += evals;

        if !all_finite(y_next) || !all_finite(err) {
            rejects += 1;
            stats.rejected_steps += 1;
            system.on_step_reject();
            k1_cache = None;
            if rejects > config.max_rejects {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            h = (h_step.abs() * 0.1).clamp(h_min, h_max) * direction;
            continue;
        }

        let (err_norm, accept_threshold, error_control) = match config.error_control {
            ErrorControl::Absolute { eps } => {
                // Max-norm: max(|err_i|), compared against eps.
                //
                // Some tableaus carry a THIRD-order embedded estimate alongside
                // the usual one, which Hairer blends in as
                // `sqrt(err5^2 + 0.01*err3^2)`. The blend is norm-independent
                // (it combines the two scalar norms, not per-component values),
                // so it applies to both the WRMS and max-norm forms.
                //
                // The tolerance below which the blend stops being trustworthy
                // is a property of the TABLEAU's coefficients, not of this
                // solver, so it is declared there (`Tableau::err3_min_eps`)
                // rather than hardcoded here. Hardcoding it meant the DEFINITION
                // of the error norm changed partway through any tolerance sweep
                // crossing the threshold, for a reason unrelated to the sweep.
                let mut max5 = 0.0f64;
                let max3 = if use_err3_blend(eps) {
                    // Single fused pass over err + err3
                    let mut max3 = 0.0f64;
                    for (&error, &third_order_error) in err.iter().zip(err3.iter()) {
                        max5 = max5.max(error.abs());
                        max3 = max3.max(third_order_error.abs());
                    }
                    Some(max3)
                } else {
                    for &error in err.iter() {
                        max5 = max5.max(error.abs());
                    }
                    None
                };
                let normalized_error =
                    max3.map_or(max5, |max3| max5.mul_add(max5, 0.01 * max3 * max3).sqrt());
                (normalized_error, eps, ErrorControl::Absolute { eps })
            }
            ErrorControl::Scaled { rtol, atol } => {
                // WRMS (Hairer-Wanner): sqrt(mean((err_i / scale_i)^2)), threshold = 1.0
                // WRMS is essential for Scaled control where components have
                // different magnitudes (position vs velocity in equinoctial elements).
                // Branch hoisted outside loop to avoid per-element branch.
                //
                // `err3_min_eps` is deliberately NOT consulted here: it is a
                // threshold on the absolute tolerance, and this control mode has
                // no `eps` to compare against — its accept threshold is a fixed
                // 1.0 with the scaling folded into the per-component denominator.
                // So the third-order blend is unconditional under `Scaled` and
                // gated under `Absolute`. That asymmetry is pre-existing and
                // untested. This comment used to except `ffi.rs` from "production
                // runs `Absolute` everywhere"; there is no `ffi.rs` anywhere in
                // the workspace, and re-checking on 2026-08-21 found no
                // production constructor of `Scaled` at all — the only ones are
                // in `crates/lightyear_odeint_rs/src/odesolve/basic_tests.rs`.
                // So the exception is void and the arm has no caller outside
                // its own tests. If `Scaled` ever becomes a production path on a
                // tableau with `err3`, the DOP853 noise problem that motivated
                // the gate needs re-deriving in WRMS terms rather than assuming
                // it does not apply.
                let mut sum5 = 0.0f64;
                let sum3 = if has_err3 {
                    let mut sum3 = 0.0f64;
                    for ((&state, &next_state), (&error, &third_order_error)) in
                        y.iter().zip(y_next.iter()).zip(err.iter().zip(err3.iter()))
                    {
                        let scale = atol + rtol * state.abs().max(next_state.abs());
                        let denom = if scale > 0.0 { scale } else { atol };
                        let e5 = error / denom;
                        let e3 = third_order_error / denom;
                        sum5 = e5.mul_add(e5, sum5);
                        sum3 = e3.mul_add(e3, sum3);
                    }
                    Some(sum3)
                } else {
                    for ((&state, &next_state), &error) in
                        y.iter().zip(y_next.iter()).zip(err.iter())
                    {
                        let scale = atol + rtol * state.abs().max(next_state.abs());
                        let denom = if scale > 0.0 { scale } else { atol };
                        let e5 = error / denom;
                        sum5 = e5.mul_add(e5, sum5);
                    }
                    None
                };
                let rms5 = (sum5 / state_dimension).sqrt();
                let normalized_error = sum3.map_or(rms5, |sum3| {
                    let rms3 = (sum3 / state_dimension).sqrt();
                    rms5.mul_add(rms5, 0.01 * rms3 * rms3).sqrt()
                });
                (normalized_error, 1.0, ErrorControl::Scaled { rtol, atol })
            }
        };

        if !err_norm.is_finite() {
            rejects += 1;
            stats.rejected_steps += 1;
            system.on_step_reject();
            k1_cache = None;
            if rejects > config.max_rejects {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            h = (h_step.abs() * 0.1).clamp(h_min, h_max) * direction;
            continue;
        }

        if err_norm <= accept_threshold || h_step.abs() <= h_min {
            if err_norm > accept_threshold {
                // Force-accepted at h_min: the error test FAILED and the step
                // was taken anyway because h cannot be reduced further. See
                // `IntegrationStats::underflow_accepts`.
                stats.underflow_accepts += 1;
            }
            // A terminal-clipped step was integrated over exactly `tf - t`;
            // publish that owned endpoint exactly. Applying the accumulated
            // Kahan carry here can move an event callback one ULP beyond `tf`.
            // Nonterminal steps retain compensated accumulation unchanged.
            let t_next = if lands_on_tf {
                tf
            } else {
                let kahan_y = h_step - t_comp;
                let next = t + kahan_y;
                t_comp = (next - t) - kahan_y;
                next
            };
            if !t_next.is_finite() {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            stats.steps += 1;
            let accepted_h = h_step.abs();
            // `h_step` differs from the controller's `h` only when the step was
            // shortened to land on an endpoint. BOTH statistics below have to
            // exclude those: an endpoint remainder is arithmetic, not the
            // controller collapsing. `min_accepted_h` omitted this test while
            // its immediate neighbour applied it, and reported 0.000000000 s
            // across a whole census as a result -- a minimum taken over 2.45M
            // segment endpoints, which measures how many segments ran rather
            // than how the controller behaved. Line 900 lets a sub-`h_min` step
            // through precisely when it lands on `tf`, and that is the door
            // those samples came in by.
            let controller_chose_h = accepted_h >= h.abs();
            if controller_chose_h
                && (stats.min_accepted_h == 0.0 || accepted_h < stats.min_accepted_h)
            {
                stats.min_accepted_h = accepted_h;
            }
            // The restart ramp, measured. `stats.steps` was incremented just
            // above, so it is this entry's 1-based accepted-step index. Only
            // controller-chosen steps count on BOTH sides: an endpoint
            // remainder in the first five slots would read as a ramp step, and
            // one in the tail would drag the sustained-rate denominator down.
            if controller_chose_h {
                let index = stats.steps.saturating_sub(1);
                if let Some(slot) = stats.first_accepted_h.get_mut(index) {
                    *slot = accepted_h;
                } else {
                    stats.tail_h_sum += accepted_h;
                    stats.tail_h_count += 1;
                }
            }
            if accepted_h < CACHE_CLUSTER_H_S {
                stats.cache_cluster_steps += 1;
                if controller_chose_h {
                    stats.cache_cluster_steps_untruncated += 1;
                }
            }
            rejects = 0;

            let Some(prev_dy) = k.get(..n) else {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::InvalidInput,
                    stats,
                    event: None,
                };
            };
            let mut next_dy_slice: Option<&[f64]> = None;

            if tableau.fsal {
                let Some(last_stage_index) = stages.checked_sub(1) else {
                    return IntegrationResult {
                        t,
                        y: outputs_or_final(t_eval, &y, n),
                        status: IntegrationStatus::InvalidInput,
                        stats,
                        event: None,
                    };
                };
                let Some(&last_stage_node) = tableau.c.get(last_stage_index) else {
                    return IntegrationResult {
                        t,
                        y: outputs_or_final(t_eval, &y, n),
                        status: IntegrationStatus::InvalidInput,
                        stats,
                        event: None,
                    };
                };
                let cache = k1_cache.get_or_insert_with(|| vec![0.0; n]);
                if (last_stage_node - 1.0).abs() <= 1e-12 {
                    let Some(last_stage_start) = last_stage_index.checked_mul(n) else {
                        return IntegrationResult {
                            t,
                            y: outputs_or_final(t_eval, &y, n),
                            status: IntegrationStatus::InvalidInput,
                            stats,
                            event: None,
                        };
                    };
                    let Some(last_stage_end) = last_stage_start.checked_add(n) else {
                        return IntegrationResult {
                            t,
                            y: outputs_or_final(t_eval, &y, n),
                            status: IntegrationStatus::InvalidInput,
                            stats,
                            event: None,
                        };
                    };
                    let Some(last_stage) = k.get(last_stage_start..last_stage_end) else {
                        return IntegrationResult {
                            t,
                            y: outputs_or_final(t_eval, &y, n),
                            status: IntegrationStatus::InvalidInput,
                            stats,
                            event: None,
                        };
                    };
                    cache.copy_from_slice(last_stage);
                } else {
                    system.rhs(t_next, y_next, cache);
                    stats.evals += 1;
                }
                next_dy_slice = Some(cache.as_slice());
            } else if (event_handler.is_some() || dense_unforced) && first_stage_is_rhs {
                // An event handler needs the derivative at the step's END
                // point. For a tableau whose stage 0 is `f(t, y)` that is the
                // NEXT step's `k[0]`, evaluated at the same `t_next` and the
                // same `y_next` the next step will start from. Handing it
                // forward as `reuse_k1` computes the same quantity once
                // instead of twice.
                //
                // This branch used to end in `k1_cache = None`, throwing the
                // value away. On the pinned strict-HF arc that discard cost one
                // evaluation on each of 342 steps.
                let cache = k1_cache.get_or_insert_with(|| vec![0.0; n]);
                system.rhs(t_next, y_next, cache);
                stats.evals += 1;
                next_dy_slice = Some(cache.as_slice());
            } else if event_handler.is_some() || dense_unforced {
                system.rhs(t_next, y_next, dy_next);
                stats.evals += 1;
                next_dy_slice = Some(dy_next);
                k1_cache = None;
            } else {
                k1_cache = None;
            }

            if let Some(handler) = event_handler.as_deref_mut() {
                let next_dy = next_dy_slice.unwrap_or(prev_dy);
                match handler.on_step(t, &y, prev_dy, t_next, y_next, next_dy) {
                    EventDecision::Continue => {}
                    EventDecision::Stop { t_event, y_event } => {
                        let sanitized =
                            sanitize_event(t, t_next, &y, y_next, t_event, y_event, direction);
                        let (t_event, y_event, method, error) = match sanitized {
                            Ok(v) => v,
                            Err(event_status) => {
                                return IntegrationResult {
                                    t,
                                    y: outputs_or_final(t_eval, &y, n),
                                    status: event_status,
                                    stats,
                                    event: None,
                                };
                            }
                        };
                        let event = IntegrationEvent {
                            t: t_event,
                            y: y_event.clone(),
                            interp_method: method,
                            interp_error: error,
                        };
                        if let Some(times) = output_times.as_deref_mut() {
                            if times
                                .last()
                                .is_none_or(|last| (t_event - *last).abs() > eval_tol)
                            {
                                times.push(t_event);
                                push_sample(&mut sample_sink, &y_event);
                            }
                        }
                        return IntegrationResult {
                            t: t_event,
                            y: outputs_or_final(t_eval, &y_event, n),
                            status: IntegrationStatus::EventTriggered,
                            stats,
                            event: Some(event),
                        };
                    }
                }
            }

            if dense_unforced {
                let Some(eval_times) = t_eval else {
                    return IntegrationResult {
                        t,
                        y: outputs_or_final(t_eval, &y, n),
                        status: IntegrationStatus::InvalidInput,
                        stats,
                        event: None,
                    };
                };
                while let Some(&sample_t) = eval_times.get(eval_idx) {
                    if direction * (sample_t - t_next) > 0.0 {
                        break;
                    }
                    if sample_t.to_bits() == t_next.to_bits() {
                        push_sample(&mut sample_sink, y_next);
                    } else {
                        let h_dense = t_next - t;
                        let tau = ((sample_t - t) / h_dense).clamp(0.0, 1.0);
                        let tau2 = tau * tau;
                        let tau3 = tau2 * tau;
                        let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
                        let h10 = tau3 - 2.0 * tau2 + tau;
                        let h01 = -2.0 * tau3 + 3.0 * tau2;
                        let h11 = tau3 - tau2;
                        let Some(next_dy) = next_dy_slice else {
                            return IntegrationResult {
                                t,
                                y: outputs_or_final(t_eval, &y, n),
                                status: IntegrationStatus::InvalidInput,
                                stats,
                                event: None,
                            };
                        };
                        if next_dy.len() != n {
                            return IntegrationResult {
                                t,
                                y: outputs_or_final(t_eval, &y, n),
                                status: IntegrationStatus::InvalidInput,
                                stats,
                                event: None,
                            };
                        }
                        for (
                            ((dense_value, state), previous_derivative),
                            (next_state, next_derivative),
                        ) in dense_sample
                            .iter_mut()
                            .zip(y.iter())
                            .zip(prev_dy.iter())
                            .zip(y_next.iter().zip(next_dy.iter()))
                        {
                            *dense_value = h00 * *state
                                + h10 * h_dense * *previous_derivative
                                + h01 * *next_state
                                + h11 * h_dense * *next_derivative;
                        }
                        push_sample(&mut sample_sink, dense_sample);
                    }
                    if let Some(times) = output_times.as_deref_mut() {
                        times.push(sample_t);
                    }
                    eval_idx += 1;
                }
            } else if let Some(eval_times) = t_eval {
                while eval_times
                    .get(eval_idx)
                    .is_some_and(|time| (*time - t_next).abs() <= eval_tol)
                {
                    push_sample(&mut sample_sink, y_next);
                    if let Some(times) = output_times.as_deref_mut() {
                        times.push(t_next);
                    }
                    eval_idx += 1;
                }
            }

            t = t_next;
            y.copy_from_slice(y_next);

            // PI step-size controller (Gustafsson) on accept.
            pi_controller_accept!(
                err_norm,
                accept_threshold,
                error_control,
                order,
                inv_order,
                just_rejected,
                h_step,
                h_min,
                h_max,
                direction,
                h,
                stats,
                err_prev,
                have_err_prev
            );
        } else {
            rejects += 1;
            stats.rejected_steps += 1;
            system.on_step_reject();
            k1_cache = None;
            if rejects > config.max_rejects {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::MaxRejectsExceeded,
                    stats,
                    event: None,
                };
            }
            pi_controller_reject!(
                err_norm,
                error_control,
                inv_order,
                h_step,
                h_min,
                h_max,
                direction,
                h,
                just_rejected
            );
            // A DISCARDED step must not touch the PI controller's memory.
            // `err_prev` and `have_err_prev` are deliberately NOT written here.
            //
            // `err_prev` feeds the I-term `(err_prev/eps)^beta` in the accept
            // branch, whose job is to estimate the TREND of the error sequence
            // along the ACCEPTED trajectory. Hairer's `dopri5.f` and `dop853.f`
            // both leave `FACOLD` untouched on rejection for exactly this
            // reason — `FACOLD` is assigned only inside the step-accepted
            // branch.
            //
            // This branch used to write `err_prev = err_norm; have_err_prev =
            // true`, which made the next accepted step difference an error
            // measured at `h` against one measured at `0.1h..0.5h`. That is not
            // a trend, it is mostly the `h^(p+1)` scaling of the step reduction
            // itself: for Vern9 (`beta = 0.4/9`) a 5x cut alone injects a
            // spurious factor of `5^(9*0.4/9) = 1.19` into the very next growth
            // decision — biased UPWARD, at the one moment the controller should
            // be conservative. The `just_rejected` cap of 2.0 partly masked it,
            // which is likely why it survived.
            //
            // # Why not `have_err_prev = false` instead
            //
            // Clearing the flag would fall back to the I-free form
            // `0.9 * (eps/err)^(1/order)` for one step. That also removes the
            // defect, but it is NOT what the reference implementations do and it
            // discards information that is still valid: the last accepted
            // error remains a legitimate trend anchor — it is the rejected one
            // that is not. The two differ in practice, not just on paper
            // (measured 55 steps against 54 on the `BumpSystem` fixture in
            // `crates/lightyear_odeint_rs/src/odesolve/basic_tests.rs`), so
            // the choice is load-bearing and is made here
            // in favour of matching Hairer.
            //
            // Note the asymmetry with the non-finite and Newton-failure
            // branches, which DO clear `have_err_prev`. That is correct and not
            // an inconsistency: those paths abandon a step whose error estimate
            // is meaningless (NaN, or no converged solution at all), so there is
            // no trustworthy history to carry forward. Here the history is fine;
            // it is only the current step that failed.
        }
    }

    if let Some(eval_times) = t_eval {
        while let Some(&eval_time) = eval_times.get(eval_idx) {
            push_sample(&mut sample_sink, &y);
            if let Some(times) = output_times.as_deref_mut() {
                times.push(eval_time);
            }
            eval_idx += 1;
        }
    }

    IntegrationResult {
        t,
        y: outputs_or_final(t_eval, &y, n),
        status: IntegrationStatus::Success,
        stats,
        event: None,
    }
}

#[inline]
fn push_sample(sample_sink: &mut Option<&mut SampleSink<'_>>, state: &[f64]) {
    if let Some(sink) = sample_sink.as_deref_mut() {
        sink.push_state(state);
    }
}

fn outputs_or_final(t_eval: Option<&[f64]>, y: &[f64], n: usize) -> Vec<f64> {
    if t_eval.is_some() {
        Vec::new()
    } else {
        let mut out = Vec::with_capacity(n);
        out.extend_from_slice(y);
        out
    }
}

/// Monotone in the sign of `direction`: nondecreasing when `direction` is
/// nonnegative, nonincreasing otherwise. An empty slice is sorted either way.
pub fn is_sorted_dir(values: &[f64], direction: f64) -> bool {
    if values.is_empty() {
        return true;
    }
    if direction >= 0.0 {
        values
            .windows(2)
            .all(|window| matches!(window, [left, right] if left <= right))
    } else {
        values
            .windows(2)
            .all(|window| matches!(window, [left, right] if left >= right))
    }
}

#[inline]
#[expect(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "the stage kernel follows standard RK notation"
)]
pub fn rk_step<S: OdeSystem>(
    system: &S,
    tableau: &Tableau,
    t: f64,
    y: &[f64],
    h: f64,
    k: &mut [f64],
    y_tmp: &mut [f64],
    y_next: &mut [f64],
    err: &mut [f64],
    err3: &mut [f64],
    primary_error_compensation: &mut [f64],
    secondary_error_compensation: &mut [f64],
    reuse_k1: Option<&[f64]>,
    primary_error_coefficients: Option<&[f64]>,
    secondary_error_coefficients: Option<&[f64]>,
) -> Result<usize, IntegrationStatus> {
    let n = y.len();
    let mut evals: usize = 0;
    let Some(stage_value_count) = tableau.stages.checked_mul(n) else {
        return Err(IntegrationStatus::InvalidInput);
    };
    if n == 0
        || k.len() != stage_value_count
        || y_tmp.len() != n
        || y_next.len() != n
        || err.len() != n
        || err3.len() != n
        || primary_error_compensation.len() < n
        || secondary_error_compensation.len() < n
        || tableau.a.len() < tableau.stages
        || tableau.b.len() < tableau.stages
        || tableau.c.len() < tableau.stages
        || tableau
            .b_hat
            .is_some_and(|embedded_weights| embedded_weights.len() < tableau.stages)
        || tableau
            .a
            .iter()
            .take(tableau.stages)
            .enumerate()
            .any(|(stage, row)| {
                row.iter()
                    .skip(stage)
                    .any(|&coefficient| coefficient != 0.0)
            })
    {
        return Err(IntegrationStatus::InvalidInput);
    }

    let primary_error_weights = tableau.err.or(primary_error_coefficients);
    let secondary_error_weights = tableau.err3.or(secondary_error_coefficients);
    if primary_error_weights.is_some_and(|weights| weights.len() < tableau.stages)
        || secondary_error_weights.is_some_and(|weights| weights.len() < tableau.stages)
    {
        return Err(IntegrationStatus::InvalidInput);
    }
    let has_secondary_error = secondary_error_weights.is_some();

    // Every stage time of this step is `t + c[i] * h` and all of them are known
    // now. Offered before the first evaluation so a system that resolves a
    // time-only quantity per stage can do the whole set at once; the default
    // implementation does nothing.
    system.prefill_stage_times(t, h, tableau.c.get(..tableau.stages).unwrap_or(&[]));

    let Some(first_stage) = k.get_mut(..n) else {
        return Err(IntegrationStatus::InvalidInput);
    };
    if let Some(k1) = reuse_k1 {
        if k1.len() != n {
            return Err(IntegrationStatus::InvalidInput);
        }
        first_stage.copy_from_slice(k1);
    } else {
        system.rhs(t, y, first_stage);
        evals = evals.saturating_add(1);
    }

    for stage in 1..tableau.stages {
        let Some(&row) = tableau.a.get(stage) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        y_tmp.copy_from_slice(y);
        for (j, a_ij) in row.iter().enumerate() {
            let scale = h * a_ij;
            if scale == 0.0 {
                continue;
            }
            let Some(start) = j.checked_mul(n) else {
                return Err(IntegrationStatus::InvalidInput);
            };
            let Some(end) = start.checked_add(n) else {
                return Err(IntegrationStatus::InvalidInput);
            };
            let Some(kj) = k.get(start..end) else {
                return Err(IntegrationStatus::InvalidInput);
            };
            axpy(scale, kj, y_tmp);
        }
        let Some(&node) = tableau.c.get(stage) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        let t_stage = t + node * h;
        let Some(start) = stage.checked_mul(n) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        let Some(end) = start.checked_add(n) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        let Some(ks) = k.get_mut(start..end) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        system.rhs(t_stage, y_tmp, ks);
        evals = evals.saturating_add(1);
    }

    y_next.copy_from_slice(y);
    for v in err.iter_mut() {
        *v = 0.0;
    }

    if has_secondary_error {
        for v in err3.iter_mut() {
            *v = 0.0;
        }
    }
    let Some(primary_compensation) = primary_error_compensation.get_mut(..n) else {
        return Err(IntegrationStatus::InvalidInput);
    };
    primary_compensation.fill(0.0);
    if has_secondary_error {
        let Some(secondary_compensation) = secondary_error_compensation.get_mut(..n) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        secondary_compensation.fill(0.0);
    }

    for stage in 0..tableau.stages {
        let Some(&bj) = tableau.b.get(stage) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        let ej = match primary_error_weights {
            Some(weights) => {
                let Some(&weight) = weights.get(stage) else {
                    return Err(IntegrationStatus::InvalidInput);
                };
                weight
            }
            None => 0.0,
        };
        let e3j = match secondary_error_weights {
            Some(weights) => {
                let Some(&weight) = weights.get(stage) else {
                    return Err(IntegrationStatus::InvalidInput);
                };
                weight
            }
            None => 0.0,
        };
        let Some(start) = stage.checked_mul(n) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        let Some(end) = start.checked_add(n) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        let Some(kj) = k.get(start..end) else {
            return Err(IntegrationStatus::InvalidInput);
        };
        if bj == 0.0 && ej == 0.0 && e3j == 0.0 {
            continue;
        }
        let h_b = h * bj;
        let h_e = h * ej;
        // y_next accumulation (no compensation needed — large magnitudes)
        if h_b != 0.0 {
            axpy(h_b, kj, y_next);
        }
        // err accumulation with Kahan compensated summation
        if h_e != 0.0 {
            for ((error, compensation), &derivative) in err
                .iter_mut()
                .zip(primary_error_compensation.iter_mut())
                .zip(kj)
            {
                let term = h_e * derivative;
                let y_k = term - *compensation;
                let accumulated = *error + y_k;
                *compensation = (accumulated - *error) - y_k;
                *error = accumulated;
            }
        }
        if has_secondary_error && e3j != 0.0 {
            let h_e3 = h * e3j;
            for ((error, compensation), &derivative) in err3
                .iter_mut()
                .zip(secondary_error_compensation.iter_mut())
                .zip(kj)
            {
                let term = h_e3 * derivative;
                let y_k = term - *compensation;
                let accumulated = *error + y_k;
                *compensation = (accumulated - *error) - y_k;
                *error = accumulated;
            }
        }
    }

    Ok(evals)
}

// =============================================================================
// ESDIRK4(3)6L[2]SA adaptive integration loop (Kennedy–Carpenter)
//
// Uses implicit_step::esdirk_step for the single-step solver with Newton
// iteration.  Step-size controlled by Gustafsson PI controller (order 4/3).
// Parameters validated against SUNDIALS ARKODE and Gustafsson 1994.
// =============================================================================

use crate::odesolve::implicit_step::{esdirk_step, ImplicitStepResult, JacobianProvider};
use crate::odesolve::tableau_esdirk::esdirk43_tableau;

/// Maximum consecutive Newton failures before aborting.
const ESDIRK_MAX_NEWTON_FAILS: usize = 3;
/// Max Newton iterations per implicit stage (SUNDIALS default: 3, max: 10).
/// Raised to 8 for robustness at tight tolerances near the eps floor.
const ESDIRK_MAX_NEWTON_ITER: usize = 8;

#[expect(
    clippy::arithmetic_side_effects,
    reason = "preserve established IEEE operation order in the implicit RK kernel"
)]
#[expect(
    clippy::float_cmp,
    reason = "exact endpoint and direction comparisons define integration control flow"
)]
#[expect(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "the integration kernel intentionally uses standard RK notation"
)]
fn integrate_internal_esdirk<S: OdeSystem, J: JacobianProvider>(
    system: &S,
    jac_provider: &J,
    y0: &[f64],
    t0: f64,
    tf: f64,
    t_eval: Option<&[f64]>,
    config: IntegratorConfig,
    mut event_handler: Option<&mut dyn EventHandler>,
    mut output_times: Option<&mut Vec<f64>>,
    mut sample_sink: Option<&mut SampleSink<'_>>,
) -> IntegrationResult {
    // ESDIRK infrastructure uses fixed [f64; 6] arrays
    if y0.len() != 6 {
        return IntegrationResult {
            t: t0,
            y: y0.to_vec(),
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    if !t0.is_finite() || !tf.is_finite() {
        return IntegrationResult {
            t: t0,
            y: y0.to_vec(),
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    // Validate error control and extract Newton tolerance (0.1 * eps, SUNDIALS nlscoef)
    let newton_tol = match config.error_control {
        ErrorControl::Absolute { eps } => {
            if !eps.is_finite() || eps <= 0.0 {
                return IntegrationResult {
                    t: t0,
                    y: y0.to_vec(),
                    status: IntegrationStatus::InvalidInput,
                    stats: IntegrationStats::default(),
                    event: None,
                };
            }
            0.1 * eps
        }
        ErrorControl::Scaled { rtol, atol } => {
            if !rtol.is_finite() || !atol.is_finite() || rtol <= 0.0 || atol < 0.0 {
                return IntegrationResult {
                    t: t0,
                    y: y0.to_vec(),
                    status: IntegrationStatus::InvalidInput,
                    stats: IntegrationStats::default(),
                    event: None,
                };
            }
            0.1 * rtol
        }
    };

    let tableau = esdirk43_tableau();
    let n: usize = 6;
    let Ok(embedded_order) = u32::try_from(tableau.order_err) else {
        return IntegrationResult {
            t: t0,
            y: y0.to_vec(),
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    };

    let direction = if tf >= t0 { 1.0 } else { -1.0 };
    let span = (tf - t0).abs();

    if span == 0.0 {
        return IntegrationResult {
            t: tf,
            y: y0.to_vec(),
            status: IntegrationStatus::Success,
            stats: IntegrationStats::default(),
            event: None,
        };
    }

    let mut h_max = config.h_max.abs();
    if !h_max.is_finite() || h_max <= 0.0 {
        h_max = span;
    }
    let mut h_min = config.h_min.abs();
    if !h_min.is_finite() || h_min <= 0.0 {
        h_min = 1e-12;
    }

    // `segment_span_s` is recorded so the ramp above can be split by segment
    // POPULATION afterwards: Encke rebases and eclipse root legs share one
    // counter and differ by ~57x in span, so one mean over both describes
    // neither.
    let mut stats = IntegrationStats {
        segment_span_s: span,
        ..Default::default()
    };

    let mut h = config.h0.map_or_else(
        || default_h0(span).clamp(h_min, h_max) * direction,
        |h0| h0.abs().clamp(h_min, h_max) * direction,
    );

    let Ok(mut y) = <[f64; 6]>::try_from(y0) else {
        return IntegrationResult {
            t: t0,
            y: y0.to_vec(),
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    };
    let mut t = t0;

    let mut rejects = 0usize;
    let mut newton_fails = 0usize;

    let mut k = [[0.0f64; 6]; 6];
    let mut y_next = [0.0f64; 6];
    let mut err_arr = [0.0f64; 6];

    let mut reuse_k0: Option<[f64; 6]> = None;
    // Gustafsson PI controller state
    let mut err_prev: f64 = 0.0;
    let mut have_err_prev = false;
    let mut just_rejected = false;
    // Kahan compensated summation for time
    let mut t_comp: f64 = 0.0;

    let eval_tol = 1e-12;
    let mut eval_idx = 0usize;

    // Validate t_eval
    if let Some(eval_times) = t_eval {
        if eval_times.is_empty() {
            return IntegrationResult {
                t,
                y: y.to_vec(),
                status: IntegrationStatus::InvalidInput,
                stats,
                event: None,
            };
        }
        if !is_sorted_dir(eval_times, direction) {
            return IntegrationResult {
                t,
                y: y.to_vec(),
                status: IntegrationStatus::InvalidInput,
                stats,
                event: None,
            };
        }
        let (min_t, max_t) = if direction >= 0.0 { (t0, tf) } else { (tf, t0) };
        if eval_times
            .iter()
            .any(|&time| time < min_t - eval_tol || time > max_t + eval_tol)
        {
            return IntegrationResult {
                t,
                y: y.to_vec(),
                status: IntegrationStatus::InvalidInput,
                stats,
                event: None,
            };
        }
    }

    // Capture initial eval point
    if let Some(eval_times) = t_eval {
        if eval_times
            .get(eval_idx)
            .is_some_and(|time| (*time - t).abs() <= eval_tol)
        {
            push_sample(&mut sample_sink, &y);
            if let Some(times) = output_times.as_deref_mut() {
                times.push(t);
            }
            eval_idx += 1;
        }
    }

    // Order for PI controller: min(p, p_hat) + 1 = min(4, 3) + 1 = 4
    let order = f64::from(embedded_order) + 1.0;
    let inv_order = 1.0 / order;

    while (tf - t).signum() == direction && (tf - t).abs() > eval_tol {
        if stats.steps >= config.max_steps {
            return IntegrationResult {
                t,
                y: outputs_or_final(t_eval, &y, n),
                status: IntegrationStatus::MaxStepsExceeded,
                stats,
                event: None,
            };
        }

        let mut h_step = h;
        let remaining = tf - t;
        if remaining.abs() < h_step.abs() {
            h_step = remaining;
        }

        // Force evaluation at next t_eval point
        if config.force_eval {
            if let Some(eval_times) = t_eval {
                if let Some(&next_eval) = eval_times.get(eval_idx) {
                    let dt_to_eval = next_eval - t;
                    if dt_to_eval.signum() == direction
                        && dt_to_eval.abs() > eval_tol
                        && dt_to_eval.abs() < h_step.abs()
                    {
                        h_step = dt_to_eval;
                    }
                }
            }
        }

        let lands_on_tf = h_step.to_bits() == remaining.to_bits();

        if h_step.abs() < h_min {
            return IntegrationResult {
                t,
                y: outputs_or_final(t_eval, &y, n),
                status: IntegrationStatus::StepUnderflow,
                stats,
                event: None,
            };
        }

        // --- ESDIRK step ---
        let step_result = esdirk_step(
            system,
            jac_provider,
            tableau,
            t,
            &y,
            h_step,
            &mut k,
            &mut y_next,
            &mut err_arr,
            newton_tol,
            ESDIRK_MAX_NEWTON_ITER,
            &mut stats,
            reuse_k0.as_ref(),
        );

        if step_result == ImplicitStepResult::InvalidInput {
            return IntegrationResult {
                t,
                y: outputs_or_final(t_eval, &y, n),
                status: IntegrationStatus::InvalidInput,
                stats,
                event: None,
            };
        }

        // Handle Newton failure (SUNDIALS cascade)
        if step_result == ImplicitStepResult::NewtonFailed {
            newton_fails += 1;
            rejects += 1;
            stats.rejected_steps += 1;
            system.on_step_reject();
            reuse_k0 = None; // Discard FSAL on Newton failure

            if newton_fails >= ESDIRK_MAX_NEWTON_FAILS {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::MaxRejectsExceeded,
                    stats,
                    event: None,
                };
            }
            if newton_fails >= 2 {
                // 2nd+ consecutive fail: reduce h by factor 0.25 (SUNDIALS ETA_CF)
                h = (h_step.abs() * 0.25).clamp(h_min, h_max) * direction;
            }
            // 1st fail: retry same h with fresh Jacobian (esdirk_step recomputes)
            have_err_prev = false;
            continue;
        }

        // Check for non-finite results
        if !all_finite(&y_next) || !all_finite(&err_arr) {
            rejects += 1;
            stats.rejected_steps += 1;
            system.on_step_reject();
            if rejects > config.max_rejects {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            h = (h_step.abs() * 0.1).clamp(h_min, h_max) * direction;
            reuse_k0 = None;
            have_err_prev = false;
            continue;
        }

        // Compute error norm
        let (err_norm, accept_threshold, error_control) = match config.error_control {
            ErrorControl::Absolute { eps } => {
                // Max-norm for Absolute control.
                //
                // The components do NOT share the same scale — an earlier
                // comment here claimed they did, and that claim was false. On
                // the 6-state Encke delta this norm maxes over three positions
                // in km and three velocities in km/s, so a single `eps` means
                // km for half the vector and km/s for the other half.
                //
                // It is kept anyway, because the inhomogeneity very nearly
                // cancels and the residual is small:
                //
                // - `|err_v| ~ w * |err_r|` with `w = |v|/|r| ~ 1.06e-3 s^-1`
                //   (the velocity error is the time-derivative of the position
                //   error), so the position components ALWAYS bind the max and
                //   the velocity components never set the step.
                // - What reaches the endpoint is not the velocity LTE itself
                //   but its integral over the remaining time `T`. Balancing
                //   that against the position LTE would need an effective
                //   `atol_v = atol_r / T = eps/7200 = 1.39e-4 * eps`; what the
                //   code achieves is `1.06e-3 * eps`. **Velocity error
                //   therefore contributes about 7.6x more endpoint position
                //   error than position LTE does** — one order, not the three
                //   the dimensional argument suggests.
                // - The factor is proportional to `eps`, so it moves the
                //   constant `C` in `E ~ C * eps^r` and leaves the convergence
                //   exponent `r` alone.
                // - Relatively it is near-homogeneous by accident: the delta
                //   magnitudes carry the same `w` factor as their errors, so
                //   both position and velocity land at ~5e-9 RELATIVE error at
                //   eps=1e-8.
                //
                // Where this reasoning does NOT hold: any augmented state whose
                // components span decades in natural scale with no such
                // cancellation — a variational/STM system carries `Phi_rv` in
                // seconds against `Phi_vr` in 1/s, ~1e6 apart. Such a caller
                // needs `Scaled` control, not this one.
                let mut max_err = 0.0f64;
                for &value in err_arr.iter().take(n) {
                    max_err = max_err.max(value.abs());
                }
                (max_err, eps, ErrorControl::Absolute { eps })
            }
            ErrorControl::Scaled { rtol, atol } => {
                // WRMS for Scaled control (components have different magnitudes)
                let n_as_f64 = 6.0;
                let sum: f64 = y
                    .iter()
                    .zip(y_next.iter())
                    .zip(err_arr.iter())
                    .map(|((&state, &next_state), &error)| {
                        let scale = atol + rtol * state.abs().max(next_state.abs());
                        let denom = if scale > 0.0 { scale } else { atol };
                        let e = error / denom;
                        e * e
                    })
                    .sum();
                (
                    (sum / n_as_f64).sqrt(),
                    1.0,
                    ErrorControl::Scaled { rtol, atol },
                )
            }
        };

        if !err_norm.is_finite() {
            rejects += 1;
            stats.rejected_steps += 1;
            system.on_step_reject();
            if rejects > config.max_rejects {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            h = (h_step.abs() * 0.1).clamp(h_min, h_max) * direction;
            reuse_k0 = None;
            have_err_prev = false;
            continue;
        }

        if err_norm <= accept_threshold || h_step.abs() <= h_min {
            // --- Step accepted ---

            if err_norm > accept_threshold {
                // Force-accepted at h_min; see `IntegrationStats::underflow_accepts`.
                stats.underflow_accepts += 1;
            }

            // A step integrated over exactly the remaining interval owns the
            // requested endpoint. Preserve compensated time accumulation for
            // every nonterminal step.
            let t_next = if lands_on_tf {
                tf
            } else {
                let kahan_y = h_step - t_comp;
                let next = t + kahan_y;
                t_comp = (next - t) - kahan_y;
                next
            };
            if !t_next.is_finite() {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::NonFiniteState,
                    stats,
                    event: None,
                };
            }
            stats.steps += 1;
            let accepted_h = h_step.abs();
            // `h_step` differs from the controller's `h` only when the step was
            // shortened to land on an endpoint. BOTH statistics below have to
            // exclude those: an endpoint remainder is arithmetic, not the
            // controller collapsing. `min_accepted_h` omitted this test while
            // its immediate neighbour applied it, and reported 0.000000000 s
            // across a whole census as a result -- a minimum taken over 2.45M
            // segment endpoints, which measures how many segments ran rather
            // than how the controller behaved. Line 900 lets a sub-`h_min` step
            // through precisely when it lands on `tf`, and that is the door
            // those samples came in by.
            let controller_chose_h = accepted_h >= h.abs();
            if controller_chose_h
                && (stats.min_accepted_h == 0.0 || accepted_h < stats.min_accepted_h)
            {
                stats.min_accepted_h = accepted_h;
            }
            // The restart ramp, measured. `stats.steps` was incremented just
            // above, so it is this entry's 1-based accepted-step index. Only
            // controller-chosen steps count on BOTH sides: an endpoint
            // remainder in the first five slots would read as a ramp step, and
            // one in the tail would drag the sustained-rate denominator down.
            if controller_chose_h {
                let index = stats.steps.saturating_sub(1);
                if let Some(slot) = stats.first_accepted_h.get_mut(index) {
                    *slot = accepted_h;
                } else {
                    stats.tail_h_sum += accepted_h;
                    stats.tail_h_count += 1;
                }
            }
            if accepted_h < CACHE_CLUSTER_H_S {
                stats.cache_cluster_steps += 1;
                if controller_chose_h {
                    stats.cache_cluster_steps_untruncated += 1;
                }
            }
            rejects = 0;
            newton_fails = 0;

            // FSAL: k[5] = f(t+h, y_next) due to stiff accuracy
            let fsal_k5 = k[5];

            // Derivative at previous point (k[0]) for event handler
            let prev_dy: [f64; 6] = reuse_k0.as_ref().copied().unwrap_or_else(|| k[0]);

            // Event handling
            if let Some(handler) = event_handler.as_deref_mut() {
                match handler.on_step(t, &y, &prev_dy, t_next, &y_next, &fsal_k5) {
                    EventDecision::Continue => {}
                    EventDecision::Stop { t_event, y_event } => {
                        let sanitized =
                            sanitize_event(t, t_next, &y, &y_next, t_event, y_event, direction);
                        let (t_event, y_event, method, error) = match sanitized {
                            Ok(v) => v,
                            Err(event_status) => {
                                return IntegrationResult {
                                    t,
                                    y: outputs_or_final(t_eval, &y, n),
                                    status: event_status,
                                    stats,
                                    event: None,
                                };
                            }
                        };
                        let event = IntegrationEvent {
                            t: t_event,
                            y: y_event.clone(),
                            interp_method: method,
                            interp_error: error,
                        };
                        if let Some(times) = output_times.as_deref_mut() {
                            if times
                                .last()
                                .is_none_or(|last| (t_event - *last).abs() > eval_tol)
                            {
                                times.push(t_event);
                                push_sample(&mut sample_sink, &y_event);
                            }
                        }
                        return IntegrationResult {
                            t: t_event,
                            y: outputs_or_final(t_eval, &y_event, n),
                            status: IntegrationStatus::EventTriggered,
                            stats,
                            event: Some(event),
                        };
                    }
                }
            }

            // Advance state
            t = t_next;
            y = y_next;
            reuse_k0 = Some(fsal_k5);

            // Capture eval points
            if let Some(eval_times) = t_eval {
                while eval_times
                    .get(eval_idx)
                    .is_some_and(|time| (*time - t).abs() <= eval_tol)
                {
                    push_sample(&mut sample_sink, &y);
                    if let Some(times) = output_times.as_deref_mut() {
                        times.push(t);
                    }
                    eval_idx += 1;
                }
            }

            // Gustafsson PI step-size controller (order 4)
            pi_controller_accept!(
                err_norm,
                accept_threshold,
                error_control,
                order,
                inv_order,
                just_rejected,
                h_step,
                h_min,
                h_max,
                direction,
                h,
                stats,
                err_prev,
                have_err_prev
            );
        } else {
            // --- Step rejected (error too large) ---
            rejects += 1;
            stats.rejected_steps += 1;
            system.on_step_reject();
            reuse_k0 = None;
            if rejects > config.max_rejects {
                return IntegrationResult {
                    t,
                    y: outputs_or_final(t_eval, &y, n),
                    status: IntegrationStatus::MaxRejectsExceeded,
                    stats,
                    event: None,
                };
            }
            pi_controller_reject!(
                err_norm,
                error_control,
                inv_order,
                h_step,
                h_min,
                h_max,
                direction,
                h,
                just_rejected
            );
            // A DISCARDED step must not touch the PI memory — `err_prev` and
            // `have_err_prev` are deliberately not written. Same reasoning as
            // the explicit-RK reject branch in `integrate_internal`, which
            // carries the full note including why this is not
            // `have_err_prev = false`.
        }
    }

    // Capture remaining eval points
    if let Some(eval_times) = t_eval {
        while let Some(&eval_time) = eval_times.get(eval_idx) {
            push_sample(&mut sample_sink, &y);
            if let Some(times) = output_times.as_deref_mut() {
                times.push(eval_time);
            }
            eval_idx += 1;
        }
    }

    IntegrationResult {
        t,
        y: outputs_or_final(t_eval, &y, n),
        status: IntegrationStatus::Success,
        stats,
        event: None,
    }
}

// =============================================================================
// ESDIRK public wrappers
// =============================================================================

pub fn integrate_final_esdirk<S: OdeSystem, J: JacobianProvider>(
    system: &S,
    jac_provider: &J,
    y0: &[f64],
    t0: f64,
    tf: f64,
    config: IntegratorConfig,
) -> IntegrationResult {
    integrate_internal_esdirk(
        system,
        jac_provider,
        y0,
        t0,
        tf,
        None,
        config,
        None,
        None,
        None,
    )
}

pub fn integrate_sampled_esdirk<S: OdeSystem, J: JacobianProvider>(
    system: &S,
    jac_provider: &J,
    y0: &[f64],
    t_eval: &[f64],
    config: IntegratorConfig,
) -> IntegrationResultSampled {
    let n_state = y0.len();
    let mut config = config;
    config.force_eval = true;
    let mut sink = SampleSink::vec(t_eval.len().saturating_mul(n_state));
    let result = integrate_internal_esdirk(
        system,
        jac_provider,
        y0,
        t_eval.first().copied().unwrap_or(0.0),
        t_eval.last().copied().unwrap_or(0.0),
        Some(t_eval),
        config,
        None,
        None,
        Some(&mut sink),
    );
    let states = sink.into_vec();

    IntegrationResultSampled {
        times: t_eval.to_vec(),
        states,
        n_state,
        status: result.status,
        stats: result.stats,
        event: result.event,
    }
}

pub fn integrate_sampled_esdirk_into<S: OdeSystem, J: JacobianProvider>(
    system: &S,
    jac_provider: &J,
    y0: &[f64],
    t_eval: &[f64],
    config: IntegratorConfig,
    states_out: &mut [f64],
) -> IntegrationResult {
    let n_state = y0.len();
    let expected_len = t_eval.len().saturating_mul(n_state);
    if states_out.len() != expected_len {
        return IntegrationResult {
            t: t_eval.first().copied().unwrap_or(0.0),
            y: Vec::new(),
            status: IntegrationStatus::InvalidInput,
            stats: IntegrationStats::default(),
            event: None,
        };
    }
    let mut config = config;
    config.force_eval = true;
    let (mut result, written, valid) = {
        let mut sink = SampleSink::slice(states_out);
        let result = integrate_internal_esdirk(
            system,
            jac_provider,
            y0,
            t_eval.first().copied().unwrap_or(0.0),
            t_eval.last().copied().unwrap_or(0.0),
            Some(t_eval),
            config,
            None,
            None,
            Some(&mut sink),
        );
        (result, sink.written(), sink.valid())
    };
    if !valid || written != expected_len {
        states_out.fill(0.0);
    }
    result.y.clear();
    result
}

pub fn integrate_sampled_with_events_esdirk<S: OdeSystem, J: JacobianProvider, E: EventHandler>(
    system: &S,
    jac_provider: &J,
    y0: &[f64],
    t_eval: &[f64],
    config: IntegratorConfig,
    event_handler: &mut E,
) -> IntegrationResultSampled {
    let mut config = config;
    config.force_eval = true;
    let sample_capacity = t_eval.len().saturating_add(1);
    let mut output_times: Vec<f64> = Vec::with_capacity(sample_capacity);
    let mut sink = SampleSink::vec(sample_capacity.saturating_mul(y0.len()));
    let result = integrate_internal_esdirk(
        system,
        jac_provider,
        y0,
        t_eval.first().copied().unwrap_or(0.0),
        t_eval.last().copied().unwrap_or(0.0),
        Some(t_eval),
        config,
        Some(event_handler),
        Some(&mut output_times),
        Some(&mut sink),
    );

    let n_state = y0.len();
    let states = sink.into_vec();

    IntegrationResultSampled {
        times: output_times,
        states,
        n_state,
        status: result.status,
        stats: result.stats,
        event: result.event,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// The threshold selects the two populations and each gets its own divisor.
    ///
    /// Asserted on both sides of `SHORT_SPAN_H0_S` and on the boundary itself,
    /// because the whole content of this change is WHICH segments are
    /// retargeted: a version that opened every span at `span/2` was measured and
    /// is worse than doing nothing (it starts the long population above its
    /// equilibrium step and produced 2 NaN masses where the baseline produced
    /// none). A test that only checked the short side would pass on that
    /// variant.
    #[test]
    fn default_h0_retargets_only_the_short_population() {
        assert!((default_h0(2.126) - 1.063).abs() < 1e-12, "short span");
        assert!(
            (default_h0(SHORT_SPAN_H0_S) - SHORT_SPAN_H0_S / 2.0).abs() < 1e-12,
            "the boundary itself is short"
        );
        let long = SHORT_SPAN_H0_S.next_up();
        assert!(
            (default_h0(long) - long / 100.0).abs() < 1e-12,
            "one ULP above the boundary is long"
        );
        assert!(
            (default_h0(985.866) - 9.85866).abs() < 1e-9,
            "the Encke rebase population keeps span/100"
        );
    }

    struct LinearSystem;

    impl OdeSystem for LinearSystem {
        fn rhs(&self, _t: f64, _y: &[f64], dy: &mut [f64]) {
            dy.fill(1.0);
        }
    }

    #[derive(Default)]
    struct CountingSystem {
        calls: Cell<usize>,
    }

    impl OdeSystem for CountingSystem {
        fn rhs(&self, _t: f64, _y: &[f64], dy: &mut [f64]) {
            self.calls.set(self.calls.get().saturating_add(1));
            dy.fill(1.0);
        }
    }

    #[derive(Default)]
    struct CountingJacobian {
        calls: Cell<usize>,
    }

    impl JacobianProvider for CountingJacobian {
        fn jacobian(&self, _t: f64, _y: &[f64], jac: &mut [[f64; 6]; 6]) {
            self.calls.set(self.calls.get().saturating_add(1));
            jac.fill([0.0; 6]);
        }
    }

    #[derive(Default)]
    struct HistoryDependentEsdirkSystem {
        calls: Cell<u32>,
        observations: RefCell<Vec<(u64, u64)>>,
    }

    impl OdeSystem for HistoryDependentEsdirkSystem {
        fn rhs(&self, t: f64, _y: &[f64], dy: &mut [f64]) {
            let call = self.calls.get().saturating_add(1);
            self.calls.set(call);
            let force = if t < 0.25 { 1.0 } else { 1_000.0 } + f64::from(call) * 1e-12;
            self.observations
                .borrow_mut()
                .push((t.to_bits(), force.to_bits()));
            dy.fill(0.0);
            if let Some(slot) = dy.first_mut() {
                *slot = force;
            }
        }
    }

    #[derive(Default)]
    struct AttemptRecordingZeroJacobian {
        attempt_times: RefCell<Vec<u64>>,
    }

    impl JacobianProvider for AttemptRecordingZeroJacobian {
        fn jacobian(&self, t: f64, _y: &[f64], jac: &mut [[f64; 6]; 6]) {
            self.attempt_times.borrow_mut().push(t.to_bits());
            jac.fill([0.0; 6]);
        }
    }

    #[test]
    fn rejected_esdirk_step_recomputes_stage_zero_after_later_rhs_history() {
        let system = HistoryDependentEsdirkSystem::default();
        let jacobian = AttemptRecordingZeroJacobian::default();
        let result = integrate_final_esdirk(
            &system,
            &jacobian,
            &[0.0; 6],
            0.0,
            1.0,
            IntegratorConfig {
                error_control: ErrorControl::Absolute { eps: 1e-6 },
                h0: Some(0.1),
                h_min: 1e-12,
                h_max: 0.5,
                max_steps: 100,
                max_rejects: 50,
                force_eval: false,
            },
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert!(
            result.stats.rejected_steps > 0,
            "fixture must reject at least one ESDIRK attempt"
        );
        let attempts = jacobian.attempt_times.borrow();
        let repeated_noninitial_time = attempts.iter().enumerate().find_map(|(index, &time)| {
            (time != 0.0_f64.to_bits() && attempts.get(index + 1..)?.contains(&time))
                .then_some(time)
        });
        let retry_time = repeated_noninitial_time
            .expect("fixture must reject after at least one accepted ESDIRK step");
        let derivatives = system
            .observations
            .borrow()
            .iter()
            .filter_map(|&(time, derivative)| (time == retry_time).then_some(derivative))
            .collect::<Vec<_>>();
        let [first, second, ..] = derivatives.as_slice() else {
            panic!(
                "retry must recompute stage zero at identical (t,y) after later stages changed RHS history; observed derivative bits {derivatives:?}"
            );
        };
        assert_ne!(
            first, second,
            "fixture must expose call-history-dependent derivatives at identical (t,y)"
        );
    }

    #[test]
    fn esdirk_invalid_step_input_propagates_without_callbacks() {
        let system = CountingSystem::default();
        let jacobian = CountingJacobian::default();
        let result = integrate_final_esdirk(
            &system,
            &jacobian,
            &[1.0; 6],
            0.0,
            1.0,
            IntegratorConfig {
                error_control: ErrorControl::Absolute {
                    eps: f64::from_bits(1),
                },
                ..IntegratorConfig::default()
            },
        );

        assert_eq!(result.status, IntegrationStatus::InvalidInput);
        assert_eq!(system.calls.get(), 0);
        assert_eq!(jacobian.calls.get(), 0);
        assert_eq!(result.stats.evals, 0);
    }

    struct ContinueHandler;

    impl EventHandler for ContinueHandler {
        fn on_step(
            &mut self,
            _prev_t: f64,
            _prev_y: &[f64],
            _prev_dy: &[f64],
            _next_t: f64,
            _next_y: &[f64],
            _next_dy: &[f64],
        ) -> EventDecision {
            EventDecision::Continue
        }
    }

    #[derive(Default)]
    struct EndpointRecorder {
        first_next_t: Cell<Option<f64>>,
        last_next_t: Cell<Option<f64>>,
    }

    impl EventHandler for EndpointRecorder {
        fn on_step(
            &mut self,
            _prev_t: f64,
            _prev_y: &[f64],
            _prev_dy: &[f64],
            next_t: f64,
            _next_y: &[f64],
            _next_dy: &[f64],
        ) -> EventDecision {
            if self.first_next_t.get().is_none() {
                self.first_next_t.set(Some(next_t));
            }
            self.last_next_t.set(Some(next_t));
            EventDecision::Continue
        }
    }

    #[test]
    fn explicit_terminal_clip_reports_exact_requested_endpoint() {
        let start = 27.0;
        let end = 54.451_202_197_643_7;
        let mut handler = EndpointRecorder::default();
        let result = integrate_final_with_events(
            &LinearSystem,
            Method::Vern9,
            &[2.0],
            start,
            end,
            IntegratorConfig {
                error_control: ErrorControl::Absolute { eps: 1.0 },
                h0: Some(0.002_400_000_000_000_000_2),
                h_max: 2.0,
                ..IntegratorConfig::default()
            },
            &mut handler,
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(result.t.to_bits(), end.to_bits());
        assert_eq!(
            handler.last_next_t.get().map(f64::to_bits),
            Some(end.to_bits())
        );
    }

    #[test]
    fn explicit_forced_interior_sample_is_not_labeled_as_terminal() {
        let samples = [0.0, 0.25, 1.0];
        let interior = 0.25_f64;
        let end = 1.0_f64;
        let mut handler = EndpointRecorder::default();
        let result = integrate_sampled_with_events(
            &LinearSystem,
            Method::Vern9,
            &[2.0],
            &samples,
            IntegratorConfig {
                error_control: ErrorControl::Absolute { eps: 1.0 },
                h0: Some(2.0),
                h_max: 2.0,
                ..IntegratorConfig::default()
            },
            &mut handler,
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(
            handler.first_next_t.get().map(f64::to_bits),
            Some(interior.to_bits())
        );
        assert_eq!(
            handler.last_next_t.get().map(f64::to_bits),
            Some(end.to_bits())
        );
    }

    #[test]
    fn explicit_backward_terminal_callback_is_exact() {
        let start = 54.451_202_197_643_7;
        let end = 27.0;
        let mut handler = EndpointRecorder::default();
        let result = integrate_final_with_events(
            &LinearSystem,
            Method::Vern9,
            &[2.0],
            start,
            end,
            IntegratorConfig {
                error_control: ErrorControl::Absolute { eps: 1.0 },
                h0: Some(0.002_400_000_000_000_000_2),
                h_max: 2.0,
                ..IntegratorConfig::default()
            },
            &mut handler,
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(result.t.to_bits(), end.to_bits());
        assert_eq!(
            handler.last_next_t.get().map(f64::to_bits),
            Some(end.to_bits())
        );
    }

    #[test]
    fn esdirk_terminal_callback_is_exact_when_final_step_is_taken() {
        let start = 27.0;
        let end = 54.451_202_197_643_7;
        let samples = [start, end];
        let mut handler = EndpointRecorder::default();
        let result = integrate_sampled_with_events_esdirk(
            &LinearSystem,
            &CountingJacobian::default(),
            &[2.0; 6],
            &samples,
            IntegratorConfig {
                error_control: ErrorControl::Absolute { eps: 1.0 },
                h0: Some(0.002_400_000_000_000_000_2),
                h_max: 2.0,
                ..IntegratorConfig::default()
            },
            &mut handler,
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(
            handler.last_next_t.get().map(f64::to_bits),
            Some(end.to_bits())
        );
    }

    struct StopFirstStep;

    impl EventHandler for StopFirstStep {
        fn on_step(
            &mut self,
            _prev_t: f64,
            _prev_y: &[f64],
            _prev_dy: &[f64],
            next_t: f64,
            next_y: &[f64],
            _next_dy: &[f64],
        ) -> EventDecision {
            EventDecision::Stop {
                t_event: next_t,
                y_event: next_y.to_vec(),
            }
        }
    }

    #[test]
    fn unforced_samples_do_not_change_accepted_trajectory_forward_or_backward() {
        let system = LinearSystem;
        let config = IntegratorConfig {
            h_max: 0.2,
            ..IntegratorConfig::default()
        };
        for (t0, tf, samples) in [
            (0.0, 2.0, vec![0.0, 0.13, 0.71, 1.37, 2.0]),
            (2.0, 0.0, vec![2.0, 1.87, 1.29, 0.63, 0.0]),
        ] {
            let mut final_handler = ContinueHandler;
            let final_result = integrate_final_with_events(
                &system,
                Method::Vern9,
                &[2.0],
                t0,
                tf,
                config,
                &mut final_handler,
            );
            let mut sampled_handler = ContinueHandler;
            let sampled = integrate_sampled_unforced(
                &system,
                Method::Vern9,
                &[2.0],
                &samples,
                config,
                Some(&mut sampled_handler),
            );
            assert_eq!(sampled.status, IntegrationStatus::Success);
            assert_eq!(sampled.stats.steps, final_result.stats.steps);
            assert_eq!(sampled.stats.evals, final_result.stats.evals);
            assert_eq!(
                sampled.states.last().copied().unwrap_or(f64::NAN).to_bits(),
                final_result
                    .y
                    .first()
                    .copied()
                    .unwrap_or(f64::NAN)
                    .to_bits()
            );
            let sampled_hostile_flag = integrate_sampled_unforced(
                &system,
                Method::Vern9,
                &[2.0],
                &samples,
                IntegratorConfig {
                    force_eval: true,
                    ..config
                },
                None,
            );
            assert_eq!(sampled_hostile_flag.stats.steps, sampled.stats.steps);
            assert_eq!(sampled_hostile_flag.stats.evals, sampled.stats.evals);
            assert_eq!(sampled_hostile_flag.times, sampled.times);
            assert_eq!(sampled_hostile_flag.states, sampled.states);
        }
    }

    #[test]
    fn unforced_samples_preserve_near_endpoint_state_and_ignore_force_eval() {
        // This is deliberately far below one nanosecond but many ULPs from
        // the endpoint.  A sampling tolerance must never replace this state
        // with the hidden endpoint state.
        let near_start = 5.0e-13;
        let near_final = 1.0 - 5.0e-13;
        let samples = [0.0, near_start, 0.4, near_final, 1.0];
        let config = IntegratorConfig {
            h_max: 0.2,
            ..IntegratorConfig::default()
        };
        let run = |force_eval| {
            integrate_sampled_unforced(
                &LinearSystem,
                Method::Vern9,
                &[2.0],
                &samples,
                IntegratorConfig {
                    force_eval,
                    ..config
                },
                None,
            )
        };

        let unforced = run(false);
        let hostile = run(true);
        assert_eq!(unforced.status, IntegrationStatus::Success);
        assert_eq!(
            unforced
                .times
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>(),
            samples
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>()
        );
        let start_state = unforced.states.first().copied().unwrap_or(f64::NAN);
        let near_start_state = unforced.states.get(1).copied().unwrap_or(f64::NAN);
        let near_final_state = unforced.states.get(3).copied().unwrap_or(f64::NAN);
        let final_state = unforced.states.get(4).copied().unwrap_or(f64::NAN);
        assert!((near_start_state - (2.0 + near_start)).abs() < 1.0e-10);
        assert!((near_final_state - (2.0 + near_final)).abs() < 1.0e-10);
        assert_ne!(start_state.to_bits(), near_start_state.to_bits());
        assert_ne!(near_final_state.to_bits(), final_state.to_bits());
        assert_eq!(hostile.stats.steps, unforced.stats.steps);
        assert_eq!(hostile.stats.evals, unforced.stats.evals);
        assert_eq!(hostile.times, unforced.times);
        assert_eq!(hostile.states, unforced.states);
    }

    #[test]
    fn unforced_allows_terminal_clip_below_h_min_for_exact_endpoint() {
        let final_time = 1.0e-3;
        let samples = [0.0, final_time];
        let result = integrate_sampled_unforced(
            &LinearSystem,
            Method::Vern9,
            &[2.0],
            &samples,
            IntegratorConfig {
                h0: Some(1.0),
                h_min: 1.0e-2,
                h_max: 1.0,
                ..IntegratorConfig::default()
            },
            None,
        );

        assert_eq!(result.status, IntegrationStatus::Success);
        assert_eq!(
            result
                .times
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>(),
            samples
                .iter()
                .map(|time| time.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(result.states.len(), samples.len());
        assert!(
            (result.states.last().copied().unwrap_or(f64::NAN) - (2.0 + final_time)).abs()
                < 1.0e-10
        );
    }

    #[test]
    fn unforced_handler_stop_publishes_no_sample_from_discarded_step() {
        let mut handler = StopFirstStep;
        let result = integrate_sampled_unforced(
            &LinearSystem,
            Method::Vern9,
            &[0.0],
            &[0.0, 0.005, 1.0],
            IntegratorConfig {
                h_max: 0.2,
                ..IntegratorConfig::default()
            },
            Some(&mut handler),
        );
        assert_eq!(result.status, IntegrationStatus::EventTriggered);
        assert_eq!(result.times.len(), 2);
        assert_eq!(result.times.first().copied(), Some(0.0));
        assert!(result.times.get(1).is_some_and(|time| *time > 0.005));
    }

    #[test]
    fn integrate_sampled_into_matches_allocating_sampled() {
        let system = LinearSystem;
        let y0 = [2.0_f64];
        let t_eval = [0.0_f64, 0.5, 1.0, 1.5, 2.0];
        let config = IntegratorConfig {
            h_max: 0.1,
            ..IntegratorConfig::default()
        };

        let allocating = integrate_sampled(&system, Method::Tsit5, &y0, &t_eval, config);
        let mut out = vec![f64::NAN; t_eval.len().saturating_mul(y0.len())];
        let sink = integrate_sampled_into(&system, Method::Tsit5, &y0, &t_eval, config, &mut out);

        assert_eq!(sink.status, allocating.status);
        assert_eq!(sink.stats.evals, allocating.stats.evals);
        assert_eq!(out.len(), allocating.states.len());
        for (actual, expected) in out.iter().zip(allocating.states.iter()) {
            assert!((actual - expected).abs() <= 1.0e-10);
        }
    }

    /// A reused [`SolverScratch`] must be bit-indistinguishable from a fresh
    /// one, including when the previous call left garbage in it.
    ///
    /// **What this test does and does not prove, stated precisely because the
    /// obvious reading is wrong.** It passes today even with the `clear()`
    /// removed from `prepare` -- checked, not assumed -- because `rk_step`
    /// zeroes `y_next`, `err` and both compensation buffers itself at the top of
    /// every step, and the remaining buffers are written before they are read.
    /// So this is not a guard on the zero-fill. What it *is* is a guard on the
    /// property the reuse depends on: that no buffer carries meaning across
    /// calls. If someone later adds a buffer that is accumulated into without
    /// being initialized, or moves an initialization out of `rk_step`, the NaN
    /// poisoning makes that fail here loudly instead of surfacing as a rare
    /// wrong answer on a tableau nobody exercises.
    #[test]
    fn reused_scratch_matches_fresh_scratch_bit_for_bit_even_when_poisoned() {
        let system = LinearSystem;
        let y0 = [2.0_f64];
        let config = IntegratorConfig {
            h_max: 0.1,
            ..IntegratorConfig::default()
        };

        let fresh = integrate_final_with_scratch(
            &system,
            Method::Tsit5,
            &y0,
            0.0,
            2.0,
            config,
            &mut SolverScratch::new(),
        );

        let mut reused = SolverScratch::new();
        // First call sizes and dirties the buffers.
        let first = integrate_final_with_scratch(
            &system,
            Method::Tsit5,
            &y0,
            0.0,
            2.0,
            config,
            &mut reused,
        );
        // Poison every buffer with a value that would propagate visibly if any
        // of them were read before being written.
        for buffer in [
            &mut reused.k,
            &mut reused.y_tmp,
            &mut reused.y_next,
            &mut reused.err,
            &mut reused.err3,
            &mut reused.primary_error_compensation,
            &mut reused.secondary_error_compensation,
            &mut reused.dy_next,
            &mut reused.dense_sample,
        ] {
            buffer.fill(f64::NAN);
        }
        assert!(
            reused.k.iter().any(|value| value.is_nan()),
            "poisoning must actually have written something, or this proves nothing"
        );
        let second = integrate_final_with_scratch(
            &system,
            Method::Tsit5,
            &y0,
            0.0,
            2.0,
            config,
            &mut reused,
        );

        let bits = |result: &IntegrationResult| {
            result
                .y
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&first), bits(&fresh), "first reuse must match fresh");
        assert_eq!(
            bits(&second),
            bits(&fresh),
            "a poisoned reused scratch must still match fresh"
        );
        assert_eq!(second.stats.evals, fresh.stats.evals);
        assert_eq!(second.status, fresh.status);
        // Non-vacuity: an empty result would satisfy every equality above.
        assert!(!bits(&fresh).is_empty());
        assert!(fresh.y.iter().all(|value| value.is_finite()));
    }

    /// The workspace must survive a change of shape between calls.
    ///
    /// Sizing is `clear` + `resize`, so growing and shrinking both have to
    /// land on exactly the requested length; a stale longer buffer would let
    /// `rk_step`'s length checks pass on a stale shape.
    #[test]
    fn reused_scratch_resizes_between_different_state_dimensions() {
        let mut scratch = SolverScratch::new();
        let config = IntegratorConfig {
            h_max: 0.1,
            ..IntegratorConfig::default()
        };

        let wide = integrate_final_with_scratch(
            &LinearSystem,
            Method::Tsit5,
            &[1.0, 2.0],
            0.0,
            1.0,
            config,
            &mut scratch,
        );
        assert_eq!(wide.y.len(), 2);
        assert_eq!(scratch.y_tmp.len(), 2);

        let narrow = integrate_final_with_scratch(
            &LinearSystem,
            Method::Tsit5,
            &[2.0],
            0.0,
            1.0,
            config,
            &mut scratch,
        );
        assert_eq!(narrow.y.len(), 1);
        assert_eq!(
            scratch.y_tmp.len(),
            1,
            "shrinking must land on the new length, not keep the old one"
        );
        assert_eq!(
            narrow.y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            integrate_final_with_scratch(
                &LinearSystem,
                Method::Tsit5,
                &[2.0],
                0.0,
                1.0,
                config,
                &mut SolverScratch::new(),
            )
            .y
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
            "a reshaped scratch must match a fresh one bit for bit"
        );
    }

    #[test]
    fn integrate_sampled_into_rejects_bad_output_len() {
        let system = LinearSystem;
        let y0 = [2.0_f64];
        let t_eval = [0.0_f64, 1.0];
        let mut out = vec![0.0_f64; 1];
        let result = integrate_sampled_into(
            &system,
            Method::Tsit5,
            &y0,
            &t_eval,
            IntegratorConfig::default(),
            &mut out,
        );

        assert_eq!(result.status, IntegrationStatus::InvalidInput);
    }

    #[test]
    fn rk_step_rejects_future_stage_coefficient_before_rhs() {
        const NODES: [f64; 2] = [0.0, 1.0];
        const FIRST_ROW: [f64; 0] = [];
        const SECOND_ROW: [f64; 2] = [1.0, 1.0];
        const STAGE_ROWS: [&[f64]; 2] = [&FIRST_ROW, &SECOND_ROW];
        const WEIGHTS: [f64; 2] = [0.5, 0.5];
        let malformed_tableau = Tableau {
            stages: 2,
            c: &NODES,
            a: &STAGE_ROWS,
            b: &WEIGHTS,
            b_hat: None,
            err: None,
            err3: None,
            err3_min_eps: None,
            order: 1,
            order_err: 1,
            fsal: false,
        };
        let system = CountingSystem::default();
        let mut stage_derivatives = [0.0; 2];
        let mut stage_state = [0.0];
        let mut next_state = [0.0];
        let mut error = [0.0];
        let mut third_order_error = [0.0];
        let mut primary_compensation = [0.0];
        let mut secondary_compensation = [0.0];

        let result = rk_step(
            &system,
            &malformed_tableau,
            0.0,
            &[1.0],
            0.1,
            &mut stage_derivatives,
            &mut stage_state,
            &mut next_state,
            &mut error,
            &mut third_order_error,
            &mut primary_compensation,
            &mut secondary_compensation,
            None,
            None,
            None,
        );

        assert_eq!(result, Err(IntegrationStatus::InvalidInput));
        assert_eq!(system.calls.get(), 0);
    }

    #[test]
    fn rk_step_rejects_short_error_weights_before_rhs() {
        const NODES: [f64; 2] = [0.0, 1.0];
        const FIRST_ROW: [f64; 0] = [];
        const SECOND_ROW: [f64; 1] = [1.0];
        const STAGE_ROWS: [&[f64]; 2] = [&FIRST_ROW, &SECOND_ROW];
        const WEIGHTS: [f64; 2] = [0.5, 0.5];
        const SHORT_ERROR_WEIGHTS: [f64; 1] = [0.0];
        let tableau = Tableau {
            stages: 2,
            c: &NODES,
            a: &STAGE_ROWS,
            b: &WEIGHTS,
            b_hat: None,
            err: None,
            err3: None,
            err3_min_eps: None,
            order: 1,
            order_err: 1,
            fsal: false,
        };
        let system = CountingSystem::default();
        let mut stage_derivatives = [0.0; 2];
        let mut stage_state = [0.0];
        let mut next_state = [0.0];
        let mut error = [0.0];
        let mut third_order_error = [0.0];
        let mut primary_compensation = [0.0];
        let mut secondary_compensation = [0.0];

        let result = rk_step(
            &system,
            &tableau,
            0.0,
            &[1.0],
            0.1,
            &mut stage_derivatives,
            &mut stage_state,
            &mut next_state,
            &mut error,
            &mut third_order_error,
            &mut primary_compensation,
            &mut secondary_compensation,
            None,
            Some(&SHORT_ERROR_WEIGHTS),
            None,
        );

        assert_eq!(result, Err(IntegrationStatus::InvalidInput));
        assert_eq!(system.calls.get(), 0);
    }

    #[test]
    fn rk_step_rejects_empty_embedded_weights_before_rhs() {
        const NODES: [f64; 2] = [0.0, 1.0];
        const FIRST_ROW: [f64; 0] = [];
        const SECOND_ROW: [f64; 1] = [1.0];
        const STAGE_ROWS: [&[f64]; 2] = [&FIRST_ROW, &SECOND_ROW];
        const WEIGHTS: [f64; 2] = [0.5, 0.5];
        const EMPTY_EMBEDDED_WEIGHTS: [f64; 0] = [];
        let tableau = Tableau {
            stages: 2,
            c: &NODES,
            a: &STAGE_ROWS,
            b: &WEIGHTS,
            b_hat: Some(&EMPTY_EMBEDDED_WEIGHTS),
            err: None,
            err3: None,
            err3_min_eps: None,
            order: 1,
            order_err: 1,
            fsal: false,
        };
        let system = CountingSystem::default();
        let mut stage_derivatives = [0.0; 2];
        let mut stage_state = [0.0];
        let mut next_state = [0.0];
        let mut error = [0.0];
        let mut third_order_error = [0.0];
        let mut primary_compensation = [0.0];
        let mut secondary_compensation = [0.0];

        let result = rk_step(
            &system,
            &tableau,
            0.0,
            &[1.0],
            0.1,
            &mut stage_derivatives,
            &mut stage_state,
            &mut next_state,
            &mut error,
            &mut third_order_error,
            &mut primary_compensation,
            &mut secondary_compensation,
            None,
            None,
            None,
        );

        assert_eq!(result, Err(IntegrationStatus::InvalidInput));
        assert_eq!(system.calls.get(), 0);
    }

    #[test]
    fn rk_step_rejects_empty_primary_error_weights_before_rhs() {
        const NODES: [f64; 2] = [0.0, 1.0];
        const FIRST_ROW: [f64; 0] = [];
        const SECOND_ROW: [f64; 1] = [1.0];
        const STAGE_ROWS: [&[f64]; 2] = [&FIRST_ROW, &SECOND_ROW];
        const WEIGHTS: [f64; 2] = [0.5, 0.5];
        const EMPTY_PRIMARY_ERROR_WEIGHTS: [f64; 0] = [];
        let tableau = Tableau {
            stages: 2,
            c: &NODES,
            a: &STAGE_ROWS,
            b: &WEIGHTS,
            b_hat: None,
            err: Some(&EMPTY_PRIMARY_ERROR_WEIGHTS),
            err3: None,
            err3_min_eps: None,
            order: 1,
            order_err: 1,
            fsal: false,
        };
        let system = CountingSystem::default();
        let mut stage_derivatives = [0.0; 2];
        let mut stage_state = [0.0];
        let mut next_state = [0.0];
        let mut error = [0.0];
        let mut third_order_error = [0.0];
        let mut primary_compensation = [0.0];
        let mut secondary_compensation = [0.0];

        let result = rk_step(
            &system,
            &tableau,
            0.0,
            &[1.0],
            0.1,
            &mut stage_derivatives,
            &mut stage_state,
            &mut next_state,
            &mut error,
            &mut third_order_error,
            &mut primary_compensation,
            &mut secondary_compensation,
            None,
            None,
            None,
        );

        assert_eq!(result, Err(IntegrationStatus::InvalidInput));
        assert_eq!(system.calls.get(), 0);
    }

    #[test]
    fn state_dimension_conversion_preserves_exact_f64_range() {
        let Ok(just_above_u32) = usize::try_from(4_294_967_296_u64) else {
            return;
        };
        let Ok(exact_limit) = usize::try_from(9_007_199_254_740_992_u64) else {
            return;
        };

        assert_eq!(state_dimension_as_f64(just_above_u32), Ok(4_294_967_296.0));
        assert_eq!(
            state_dimension_as_f64(exact_limit),
            Ok(9_007_199_254_740_992.0)
        );
    }

    #[test]
    fn production_solver_has_no_fallback_sample_sink() {
        let source = include_str!("solver.rs");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source, |(production, _)| production);
        assert!(!production.contains(concat!("fallback_", "sink")));
    }
}
