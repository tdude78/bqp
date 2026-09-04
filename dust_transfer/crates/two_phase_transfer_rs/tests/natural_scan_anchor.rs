//! Re-basing the sealed scan on a cached screening state.
//!
//! `refine_natural_conjunction_from_scan_anchor` starts the sealed scan at the
//! dense screening node a candidate's slab opens on instead of at the object
//! epoch, so a slab fourteen days out integrates 120 seconds rather than
//! fourteen days. The anchor is NOT bit-identical to the from-epoch state, so
//! what has to be pinned is that the two paths reach the same answer: the same
//! verdict, and a refined root and miss distance far inside the 25 m the
//! independent witness already admits.
//!
//! Every fixture here anchors on a state produced by
//! `natural_dense_ephemeris_arc` at production settings -- the same call the
//! v3 generator's narrowphase makes -- so the chain measured is the one that
//! flies, not a stand-in for it.

use anyhow::{ensure, Result};
use two_phase_transfer_rs::types::{BodyForceConfig, BodyRole};
use two_phase_transfer_rs::{
    NaturalConjunctionEnclosure, NaturalConjunctionFatalError, NaturalConjunctionInputError,
    NaturalConjunctionOutcome, NaturalConjunctionScanAnchor, NaturalObjectInput,
    TransferPostprocessSessionCore,
};

mod support;
use support::{strict_session, T0_JD_UTC};

/// Production narrowphase cache step and slab width.
const NODE_STEP_S: f64 = 60.0;
const SLAB_S: f64 = 120.0;
const HORIZON_S: f64 = 14.0 * 86_400.0;
/// The independent-witness position budget the refinement already enforces.
/// Every anchored-vs-from-epoch discrepancy asserted here is a fraction of it.
const WITNESS_POSITION_KM: f64 = 0.025;
/// Ceiling on how far the anchored answer may sit from the from-epoch answer.
///
/// A fortieth of the witness budget: 0.625 m of miss distance, and the same in
/// relative travel at the refined root. Deliberately far below the gate rather
/// than at it -- a change that consumed the witness budget would still be an
/// unacceptable regression here.
const AGREEMENT_KM: f64 = WITNESS_POSITION_KM / 40.0;
/// Floor proving the witness residual is a LIVE quantity, not a constant.
///
/// `residual <= WITNESS_POSITION_KM` passes for the wrong reason if the
/// residual is identically zero -- which is what a witness that returned the
/// state it is checked against would produce, making the 25 m gate vacuous and
/// this whole comparison worthless. So the residual is also required to be
/// NON-TRIVIAL.
///
/// Measured on the from-epoch route across this fixture's own offsets:
/// 3.5e-6 km at one hour, 7.7e-6 km at one day, 8.3e-4 km at seven days,
/// 3.1e-3 km at fourteen. This floor sits three orders below the smallest of
/// those, so it cannot flake, while still failing instantly on an exact zero
/// or a denormal.
const WITNESS_LIVENESS_FLOOR_KM: f64 = 1.0e-9;
/// Relative speed envelope for two co-altitude LEO objects, km/s. Converts the
/// root-offset agreement into the same units as the distance agreement.
const RELATIVE_SPEED_KM_S: f64 = 4.0;
/// Slabs tried per probe side before the fixture is declared broken.
///
/// A slab whose from-epoch propagation fails outright carries no verdict to
/// compare, so the search steps to the next-ranked slab. The cap is what stops
/// a probe that can never produce one from turning into a long silent run.
const MAX_SLAB_ATTEMPTS: usize = 4;

const fn body_force() -> BodyForceConfig {
    BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.01, 2.2, 1.3)
}

/// A conjunction pair that keeps recurring across the whole authorized arc.
///
/// The two objects share a radius and a speed, so they share a period and never
/// drift apart along track; the secondary's velocity is rotated out of plane by
/// `PLANE_TILT_RAD`, which turns their separation into a cross-track oscillation
/// passing through zero at every nodal crossing. That gives both verdicts the
/// refinement can return -- sub-kilometre slabs at the nodes, and slabs at the
/// antinodes the acceptance threshold rejects -- at any offset the arc reaches,
/// which is what lets the anchored path be compared at the far end where cache
/// deviation is largest.
#[expect(
    clippy::expect_used,
    reason = "test-only setup helper: a refused Part A authority must abort \
              the test loudly; clippy's allow-expect-in-tests covers \
              `#[test]` fns, not free helpers"
)]
fn conjunction_pair() -> (NaturalObjectInput, NaturalObjectInput) {
    /// Chosen so the cross-track amplitude straddles the 1 km acceptance
    /// threshold: `7000 * 2.86e-4` is about 2.0 km.
    const PLANE_TILT_RAD: f64 = 2.86e-4;
    let radius = 7_000.0_f64;
    let speed = (398_600.441_8_f64 / radius).sqrt();
    let primary = NaturalObjectInput::new(
        70_001,
        [0x11; 32],
        [0x12; 32],
        T0_JD_UTC,
        [radius, 0.0, 0.0, 0.0, speed, 0.0],
        body_force(),
    )
    .expect("valid primary authority");
    let secondary = NaturalObjectInput::new(
        70_002,
        [0x21; 32],
        [0x22; 32],
        T0_JD_UTC,
        [
            radius,
            0.0,
            0.0,
            0.0,
            speed * PLANE_TILT_RAD.cos(),
            speed * PLANE_TILT_RAD.sin(),
        ],
        body_force(),
    )
    .expect("valid secondary authority");
    (primary, secondary)
}

fn separation_km(left: [f64; 6], right: [f64; 6]) -> f64 {
    (left[0] - right[0])
        .hypot(left[1] - right[1])
        .hypot(left[2] - right[2])
}

/// What a refinement decided, flattened so the two paths can be compared
/// without caring which variant carried the answer.
struct Decision {
    verified: bool,
    offset_s: f64,
    miss_km: f64,
}

fn decide(
    session: &TransferPostprocessSessionCore,
    primary: &NaturalObjectInput,
    secondary: &NaturalObjectInput,
    enclosure: NaturalConjunctionEnclosure,
    anchor: Option<NaturalConjunctionScanAnchor>,
) -> Result<Option<Decision>> {
    let outcome = anchor
        .map_or_else(
            || session.refine_natural_conjunction(primary, secondary, enclosure),
            |anchor| {
                session.refine_natural_conjunction_from_scan_anchor(
                    primary, secondary, enclosure, anchor,
                )
            },
        )
        .map_err(|error| anyhow::anyhow!("refinement failed: {error:?}"))?;
    match outcome {
        NaturalConjunctionOutcome::Verified(verified) => {
            // The independent witness is computed from the object epoch on both
            // paths, so this gate is the real backstop and it must still hold
            // when the scan was anchored.
            ensure!(
                verified.verify_digest(),
                "verified conjunction digest failed"
            );
            ensure!(
                verified.primary_position_residual_km() <= WITNESS_POSITION_KM
                    && verified.secondary_position_residual_km() <= WITNESS_POSITION_KM,
                "independent witness position residual escaped its budget: {} / {} km",
                verified.primary_position_residual_km(),
                verified.secondary_position_residual_km()
            );
            // The gate above is only worth something if the residual is a real
            // measurement. An identically-zero residual would satisfy it while
            // proving nothing, so require the witness to actually disagree with
            // the refined state by a representable amount.
            ensure!(
                verified.primary_position_residual_km() >= WITNESS_LIVENESS_FLOOR_KM
                    && verified.secondary_position_residual_km() >= WITNESS_LIVENESS_FLOOR_KM,
                "independent witness residual is {} / {} km, at or below the \
                 {WITNESS_LIVENESS_FLOOR_KM} km liveness floor -- the witness is not \
                 an independent measurement and the {WITNESS_POSITION_KM} km gate is \
                 vacuous",
                verified.primary_position_residual_km(),
                verified.secondary_position_residual_km()
            );
            Ok(Some(Decision {
                verified: true,
                offset_s: verified.refined_offset_s(),
                miss_km: verified.miss_distance_km(),
            }))
        }
        NaturalConjunctionOutcome::CandidateInfeasible(infeasible) => Ok(Some(Decision {
            verified: false,
            offset_s: infeasible.closest_offset_s(),
            miss_km: infeasible.miss_distance_km(),
        })),
        // Not a verdict. The sealed authority cannot propagate this pair across
        // this span at all -- an eclipse boundary the event scanner reports as
        // chatter -- so there is no answer here to compare against. Production
        // drops such a pair; this test moves to the next slab and the counters
        // at the end make sure it did not run out of slabs to compare.
        NaturalConjunctionOutcome::CandidatePropagationInfeasible(_) => Ok(None),
        // Production taints such a pair and moves on. This fixture is chosen to
        // sit far inside the witness gate, so a taint HERE means the fixture or
        // the integrator moved -- fail loudly instead of skipping the slab,
        // which would let the suite pass by comparing nothing.
        NaturalConjunctionOutcome::CandidateWitnessResidual(residual) => Err(anyhow::anyhow!(
            "witness residual escaped the {WITNESS_POSITION_KM} km gate on the anchor \
             fixture: {} / {} km position, {} / {} km/s velocity",
            residual.primary_position_residual_km(),
            residual.secondary_position_residual_km(),
            residual.primary_velocity_residual_km_s(),
            residual.secondary_velocity_residual_km_s(),
        )),
    }
}

/// Production dense arcs for both objects. `natural_dense_ephemeris_arc` may
/// return a SHORT arc, so the caller reads the length each actually reached.
fn dense_arcs(
    session: &TransferPostprocessSessionCore,
    primary: &NaturalObjectInput,
    secondary: &NaturalObjectInput,
) -> Result<(Vec<[f64; 6]>, Vec<[f64; 6]>)> {
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "f64 has no fallible integer conversion; the sealed horizon \
                  divides the node step exactly"
    )]
    let horizon_nodes = (HORIZON_S / NODE_STEP_S) as usize;
    let node_count = horizon_nodes.saturating_add(1);
    let arc = |object| {
        session
            .natural_dense_ephemeris_arc(object, NODE_STEP_S, node_count)
            .map_err(|error| anyhow::anyhow!("dense screening arc failed: {error:?}"))
    };
    Ok((arc(primary)?, arc(secondary)?))
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "usize has no infallible f64 conversion; the node index is bounded \
              by the sealed arc, far under 2^53"
)]
fn node_offset_s(node: usize) -> f64 {
    NODE_STEP_S * node as f64
}

/// The anchored scan must reach the from-epoch scan's answer, at offsets across
/// the whole authorized arc and on both sides of the acceptance threshold.
///
/// The slabs are not hand-picked instants: for each probed region the tightest
/// and the loosest 120 s slab on the cache are selected, so the pair exercises
/// an accepted conjunction and a rejected one at the same offset scale.
#[test]
#[ignore = "full fourteen-day anchored-scan qualification; run the exact command in docs/TESTING.md"]
fn anchored_scan_reaches_the_from_epoch_verdict_across_the_authorized_arc() -> Result<()> {
    let session = strict_session();
    let (primary, secondary) = conjunction_pair();
    let (primary_arc, secondary_arc) = dense_arcs(&session, &primary, &secondary)?;
    let usable = primary_arc.len().min(secondary_arc.len());
    ensure!(
        usable > 20_000,
        "fixture arcs reached only {usable} nodes; the far-end comparison this \
         test exists for cannot run"
    );

    // Probe a day in, a week in, and at the far end of the arc, because the
    // cache's deviation from the from-epoch authority grows with the arc.
    // Three probes: one day, seven days, fourteen days.
    //
    // The middle one was dropped once to save wall time, justified by "the
    // disagreement grows with arc length, so the far probe dominates". That
    // justification does not hold. Two samples cannot establish monotonicity --
    // x1 < x2 is equally consistent with the interior spiking and coming back
    // down -- and the quantity that growth was measured on (anchor error) is
    // not the quantity this test gates (verdict agreement, miss and root
    // deltas). Interior arc lengths need interior coverage; nothing about the
    // endpoints implies them.
    //
    // The growth check below is kept because it is informative and cheap, but
    // it is NOT what licenses the probe list. Do not thin these again without
    // an argument that covers the interior.
    let probes = [1_440_usize, 10_080, usable.saturating_sub(200)];
    let mut compared = 0_usize;
    let mut verified_seen = 0_usize;
    let mut rejected_seen = 0_usize;
    let mut worst_miss_km = 0.0_f64;
    let mut worst_root_km = 0.0_f64;
    let mut worst_anchor_km = 0.0_f64;
    // Per-probe worst anchor error, in probe order. Dropping the interior probe
    // is only sound if the disagreement really does grow with arc length, so
    // that growth is ASSERTED here rather than taken on trust from an external
    // measurement. If it ever stops holding, the interior probe is needed again
    // and this test says so instead of silently covering less.
    let mut anchor_error_by_probe: Vec<f64> = Vec::new();
    let mut unpropagatable = 0_usize;

    for centre in probes {
        // This probe's OWN worst, not an increment to the running max. Using
        // the running max would make the sample `max(x1,x2) - x1`, so the
        // comparison below would be testing `x2 > 2*x1` instead of `x2 > x1`,
        // and would pass or fail on the magnitudes rather than on the order.
        let mut probe_worst_anchor_km = 0.0_f64;
        // One orbit of slabs around the probe, which contains several nodal
        // crossings and several antinodes.
        let first = centre.saturating_sub(46).saturating_sub(centre % 2);
        let last = centre.saturating_add(46).min(usable.saturating_sub(3));
        let mut slabs: Vec<(f64, usize)> = Vec::new();
        let mut node = first;
        while node <= last {
            let tightest = (0..3)
                .map(|step| {
                    separation_km(
                        *primary_arc
                            .get(node.saturating_add(step))
                            .unwrap_or(&[f64::NAN; 6]),
                        *secondary_arc
                            .get(node.saturating_add(step))
                            .unwrap_or(&[f64::NAN; 6]),
                    )
                })
                .fold(f64::INFINITY, f64::min);
            ensure!(tightest.is_finite(), "slab at node {node} left the arc");
            slabs.push((tightest, node));
            node = node.saturating_add(2);
        }
        ensure!(slabs.len() > 10, "probe {centre} produced too few slabs");
        slabs.sort_by(|left, right| left.0.total_cmp(&right.0));
        let tight = *slabs.first().expect("non-empty slab list");
        let loose = *slabs.last().expect("non-empty slab list");
        ensure!(
            tight.0 < 1.0 && loose.0 > 1.0,
            "probe {centre} did not straddle the acceptance threshold: \
             tightest {:.6} km, loosest {:.6} km",
            tight.0,
            loose.0
        );

        // A slab the from-epoch path cannot propagate at all yields no verdict
        // to compare, so walk in from each end until one does. The attempt cap
        // keeps a pathological probe from silently turning into a long run.
        let tight_slabs = slabs.iter().take(MAX_SLAB_ATTEMPTS).copied();
        let loose_slabs = slabs.iter().rev().take(MAX_SLAB_ATTEMPTS).copied();
        let mut compared_here = 0_usize;
        for (side, ranked) in [
            ("tight", tight_slabs.collect::<Vec<_>>()),
            ("loose", loose_slabs.collect::<Vec<_>>()),
        ] {
            let mut decided = false;
            for (_, node) in ranked {
                let lower_offset_s = node_offset_s(node);
                let enclosure =
                    NaturalConjunctionEnclosure::new(lower_offset_s, lower_offset_s + SLAB_S);
                let anchor = NaturalConjunctionScanAnchor::new(
                    lower_offset_s,
                    *primary_arc.get(node).expect("probed node is on the arc"),
                    *secondary_arc.get(node).expect("probed node is on the arc"),
                )
                .map_err(|error| anyhow::anyhow!("cache node rejected as an anchor: {error:?}"))?;

                let Some(from_epoch) = decide(&session, &primary, &secondary, enclosure, None)?
                else {
                    unpropagatable = unpropagatable.saturating_add(1);
                    continue;
                };
                let anchored = decide(&session, &primary, &secondary, enclosure, Some(anchor))?
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "node {node}: the from-epoch path returned a verdict but the \
                         anchored path could not propagate at all"
                        )
                    })?;

                ensure!(
                    from_epoch.verified == anchored.verified,
                    "node {node}: verdicts disagree -- from-epoch verified {}, \
                 anchored verified {}",
                    from_epoch.verified,
                    anchored.verified
                );
                let miss_delta_km = (from_epoch.miss_km - anchored.miss_km).abs();
                let root_delta_s = (from_epoch.offset_s - anchored.offset_s).abs();
                let root_delta_km = root_delta_s * RELATIVE_SPEED_KM_S;
                ensure!(
                    miss_delta_km <= AGREEMENT_KM,
                    "node {node}: miss distances differ by {miss_delta_km:.9} km, \
                 ceiling {AGREEMENT_KM} km (from-epoch {:.9}, anchored {:.9})",
                    from_epoch.miss_km,
                    anchored.miss_km
                );
                ensure!(
                    root_delta_km <= AGREEMENT_KM,
                    "node {node}: refined roots differ by {root_delta_s:.9} s, which is \
                 {root_delta_km:.9} km of relative travel, ceiling {AGREEMENT_KM} km"
                );

                // What the anchor itself costs: the cache node against the
                // from-epoch authority at the same instant. This is the term the
                // 25 m witness budget has to absorb, measured rather than inherited.
                let authority = session
                    .natural_dense_ephemeris_grid(&primary, &[lower_offset_s])
                    .map_err(|error| anyhow::anyhow!("authority propagation failed: {error:?}"))?;
                let anchor_error_km = separation_km(
                    *primary_arc.get(node).expect("probed node is on the arc"),
                    *authority
                        .first()
                        .expect("single-offset grid returns a state"),
                );
                ensure!(
                    anchor_error_km < WITNESS_POSITION_KM,
                    "node {node}: anchor sits {anchor_error_km:.9} km from the \
                 from-epoch authority, already outside the {WITNESS_POSITION_KM} km \
                 witness budget before refinement runs"
                );

                worst_miss_km = worst_miss_km.max(miss_delta_km);
                worst_root_km = worst_root_km.max(root_delta_km);
                worst_anchor_km = worst_anchor_km.max(anchor_error_km);
                probe_worst_anchor_km = probe_worst_anchor_km.max(anchor_error_km);
                compared = compared.saturating_add(1);
                compared_here = compared_here.saturating_add(1);
                if from_epoch.verified {
                    verified_seen = verified_seen.saturating_add(1);
                } else {
                    rejected_seen = rejected_seen.saturating_add(1);
                }
                decided = true;
                break;
            }
            ensure!(
                decided,
                "probe {centre} {side} side: no slab in {MAX_SLAB_ATTEMPTS} attempts \
                 produced a from-epoch verdict to compare against"
            );
        }
        ensure!(
            compared_here == 2,
            "probe {centre} compared {compared_here} slabs"
        );
        anchor_error_by_probe.push(probe_worst_anchor_km);
    }

    // THE ASSUMPTION THAT LETS THE INTERIOR PROBE GO, CHECKED.
    //
    // Dropping the 7-day probe is only sound if the anchored-vs-from-epoch
    // disagreement really is worst at the far end of the arc -- otherwise the
    // 14-day probe does not dominate and an interior arc could disagree by more
    // than either endpoint tested. So require the later probe to be strictly
    // worse than the earlier one. If that ever stops holding, this fails and
    // tells us the interior coverage is needed again, instead of the test
    // quietly covering less than it claims.
    // `windows(2)` over a one-element list yields NOTHING, so a probe list that
    // shrank to a single probe would skip the comparison entirely and still
    // pass -- the interior coverage would be gone with nothing left to notice.
    // Require at least two probes and one comparison, explicitly.
    ensure!(
        probes.len() >= 2,
        "the arc-length comparison needs at least two probes, got {}",
        probes.len()
    );
    ensure!(
        anchor_error_by_probe.len() == probes.len(),
        "expected one anchor-error sample per probe, got {}",
        anchor_error_by_probe.len()
    );
    let mut growth_comparisons = 0_usize;
    for pair in anchor_error_by_probe.windows(2) {
        let [earlier, later] = pair else {
            anyhow::bail!("probe window is not a pair");
        };
        ensure!(
            later > earlier,
            "anchor error did not grow with arc length ({earlier:.3e} km then \
             {later:.3e} km); the far probe no longer dominates, so the interior \
             probe this test dropped is needed again"
        );
        growth_comparisons = growth_comparisons.saturating_add(1);
    }
    ensure!(
        growth_comparisons == probes.len().saturating_sub(1),
        "made {growth_comparisons} arc-length comparisons across {} probes",
        probes.len()
    );

    // A comparison that silently compared nothing, or only one verdict, would
    // pass every assertion above.
    ensure!(
        compared == probes.len().saturating_mul(2),
        "compared {compared} slabs, expected {}",
        probes.len().saturating_mul(2)
    );
    ensure!(
        verified_seen >= probes.len() && rejected_seen >= probes.len(),
        "the comparison did not cover both verdicts: {verified_seen} verified, \
         {rejected_seen} rejected"
    );
    eprintln!(
        "anchored-vs-from-epoch over {compared} slabs ({unpropagatable} skipped, \
         from-epoch propagation infeasible): worst miss delta \
         {worst_miss_km:.3e} km, worst root delta {worst_root_km:.3e} km, \
         worst anchor error {worst_anchor_km:.3e} km, witness budget \
         {WITNESS_POSITION_KM} km"
    );
    Ok(())
}

/// An anchor the accuracy argument does not cover must be refused, not
/// silently honoured.
///
/// The argument is that an anchored scan integrates at most one slab of lead
/// plus one enclosure. An anchor opening after the enclosure would ask the
/// propagator to run backwards; one opening too far before it would hide an
/// arbitrarily long integration behind a cached state.
#[test]
fn scan_anchor_fails_closed_outside_the_span_it_is_argued_over() -> Result<()> {
    let session = strict_session();
    let (primary, secondary) = conjunction_pair();
    let lower_offset_s = 86_400.0_f64;
    let enclosure = NaturalConjunctionEnclosure::new(lower_offset_s, lower_offset_s + SLAB_S);
    let state = primary.state();
    let other = secondary.state();

    let anchored = |offset_s: f64| {
        session.refine_natural_conjunction_from_scan_anchor(
            &primary,
            &secondary,
            enclosure,
            NaturalConjunctionScanAnchor::new(offset_s, state, other)
                .expect("finite anchor constructs"),
        )
    };

    // Opens after the enclosure.
    ensure!(
        matches!(
            anchored(lower_offset_s + 60.0),
            Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Anchor
            ))
        ),
        "an anchor after the enclosure was accepted"
    );
    // Leads the enclosure by more than one slab.
    ensure!(
        matches!(
            anchored(lower_offset_s - SLAB_S - 1.0),
            Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Anchor
            ))
        ),
        "an anchor leading by more than one slab was accepted"
    );
    // The boundary itself is inside the argument, so it must be admitted --
    // otherwise the gate above would pass for the wrong reason.
    ensure!(
        !matches!(
            anchored(lower_offset_s - SLAB_S),
            Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Anchor
            ))
        ),
        "a lead of exactly one slab was refused, so the gate is not the \
         boundary it is documented as"
    );

    // Non-finite anchors never reach the session at all.
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        ensure!(
            matches!(
                NaturalConjunctionScanAnchor::new(bad, state, other),
                Err(NaturalConjunctionFatalError::InvalidInput(
                    NaturalConjunctionInputError::Anchor
                ))
            ),
            "anchor offset {bad} was accepted"
        );
    }
    let mut poisoned = state;
    poisoned[2] = f64::NAN;
    ensure!(
        matches!(
            NaturalConjunctionScanAnchor::new(0.0, poisoned, other),
            Err(NaturalConjunctionFatalError::InvalidInput(
                NaturalConjunctionInputError::Anchor
            ))
        ),
        "a non-finite anchor state was accepted"
    );
    Ok(())
}
