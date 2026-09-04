//! Dense fixed-grid strict-HF ephemeris across the sealed Part A v3 arc.
//!
//! The screening cache is not the acceptance authority, so what has to be
//! pinned is not bit identity but a bounded, measured discrepancy against the
//! from-epoch propagation that IS the authority. Every assertion here is a
//! cardinality or a distance; none is a wall-clock or memory reading.

use anyhow::{ensure, Context as _, Result};
use two_phase_transfer_rs::types::{BodyForceConfig, BodyRole};
use two_phase_transfer_rs::{
    NaturalConjunctionFatalError, NaturalConjunctionInputError, NaturalObjectInput,
    NATURAL_DENSE_ARC_AUTHORITY_CEILING_KM as AUTHORITY_DISCREPANCY_CEILING_KM,
};

mod support;
use support::{strict_session, T0_JD_UTC};

const HORIZON_S: f64 = 14.0 * 86_400.0;
const NODE_STEP_S: f64 = 60.0;

#[expect(
    clippy::expect_used,
    reason = "test-only setup helper: a refused Part A authority must abort \
              the test loudly; clippy's allow-expect-in-tests covers \
              `#[test]` fns, not free helpers"
)]
fn object(index: u64) -> NaturalObjectInput {
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "u64 has no infallible f64 conversion; the index is a small \
                  loop counter"
    )]
    let step = index as f64;
    let radius = 6_878.0 + step * 137.0;
    let inclination = 0.2 + step * 0.11;
    let speed = (398_600.441_8_f64 / radius).sqrt();
    let mut source = [0x11_u8; 32];
    let mut body = [0x12_u8; 32];
    source[0] = u8::try_from(index % 251).unwrap_or(1).saturating_add(1);
    body[0] = u8::try_from(index % 241).unwrap_or(1).saturating_add(1);
    NaturalObjectInput::new(
        40_000_u64.saturating_add(index),
        source,
        body,
        T0_JD_UTC,
        [
            radius,
            0.0,
            0.0,
            0.0,
            speed * inclination.cos(),
            speed * inclination.sin(),
        ],
        BodyForceConfig::high_fidelity(BodyRole::DiagnosticTarget, 0.01, 2.2, 1.3),
    )
    .expect("valid natural-object authority")
}

fn position_error_km(left: [f64; 6], right: [f64; 6]) -> f64 {
    (left[0] - right[0])
        .hypot(left[1] - right[1])
        .hypot(left[2] - right[2])
}

/// The cache must track the from-epoch authority at every node it claims, not
/// just at the far endpoint. A one-day arc is 1,441 nodes across 144 chained
/// segments, so the segment-restart discrepancy has fully accumulated.
#[test]
fn dense_arc_tracks_the_from_epoch_authority_at_every_probed_node() -> Result<()> {
    let session = strict_session();
    let node_count = 1_441_usize;
    for index in 0..2_u64 {
        let object = object(index);
        let arc = session
            .natural_dense_ephemeris_arc(&object, NODE_STEP_S, node_count)
            .map_err(|error| anyhow::anyhow!("dense arc failed: {error:?}"))?;
        ensure!(
            arc.len() == node_count,
            "dense arc returned {} nodes, expected {node_count}",
            arc.len()
        );
        ensure!(
            arc.iter().all(|state| state.iter().all(|v| v.is_finite())),
            "dense arc left the finite domain"
        );

        // Node zero is the object's own epoch state, by definition.
        ensure!(
            position_error_km(
                *arc.first().context("dense arc returned no nodes")?,
                object_state(&object)
            ) < 1.0e-9,
            "dense arc node zero is not the epoch state"
        );

        for node in [360_usize, 720, 1_080, 1_440] {
            let offset_s = NODE_STEP_S * f64::from(u32::try_from(node)?);
            let authority = session
                .natural_dense_ephemeris_grid(&object, &[offset_s])
                .map_err(|error| anyhow::anyhow!("authority propagation failed: {error:?}"))?;
            let error_km = position_error_km(
                *arc.get(node)
                    .context("dense arc is shorter than the probed node")?,
                *authority
                    .first()
                    .context("authority grid returned no state")?,
            );
            ensure!(
                error_km <= AUTHORITY_DISCREPANCY_CEILING_KM,
                "object {index} node {node}: cache is {error_km:.9} km from the authority, \
                 ceiling {AUTHORITY_DISCREPANCY_CEILING_KM}"
            );
        }
    }
    Ok(())
}

/// The final node sits exactly on T0 + 14 d, the last instant the JB2008
/// persistence manifest authorizes. One node beyond it must fail closed rather
/// than silently read outside the arc.
#[test]
fn dense_arc_is_bounded_by_the_authorized_persistence_arc() -> Result<()> {
    let session = strict_session();
    let object = object(1);
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "f64 has no fallible integer conversion; the sealed horizon \
                  divides the node step exactly, and the count is asserted below"
    )]
    let horizon_nodes = (HORIZON_S / NODE_STEP_S) as usize;
    let authorized_nodes = horizon_nodes.saturating_add(1);
    ensure!(authorized_nodes == 20_161, "sealed node count changed");

    let error = session
        .natural_dense_ephemeris_arc(&object, NODE_STEP_S, authorized_nodes + 1)
        .expect_err("a node past the authorized arc was accepted");
    ensure!(
        matches!(
            error,
            NaturalConjunctionFatalError::InvalidInput(NaturalConjunctionInputError::Enclosure)
        ),
        "past-arc grid failed for the wrong reason: {error:?}"
    );
    Ok(())
}

#[test]
fn dense_arc_rejects_a_grid_it_cannot_honour() -> Result<()> {
    let session = strict_session();
    let object = object(0);
    for (label, step_s, count) in [
        ("zero nodes", NODE_STEP_S, 0_usize),
        ("non-finite step", f64::NAN, 8),
        ("negative step", -60.0, 8),
        ("zero step", 0.0, 8),
    ] {
        ensure!(
            session
                .natural_dense_ephemeris_arc(&object, step_s, count)
                .is_err(),
            "{label} was accepted"
        );
    }
    Ok(())
}

/// Full sealed arc. Kept out of the default run because it is 20,161 nodes per
/// object; it is the measurement the narrowphase margin is derived from.
#[test]
#[ignore = "full fourteen-day arc"]
fn full_sealed_arc_discrepancy_is_measured() -> Result<()> {
    let session = strict_session();
    let mut worst_km = 0.0_f64;
    for index in 0..8_u64 {
        let object = object(index);
        let arc = session
            .natural_dense_ephemeris_arc(&object, NODE_STEP_S, 20_161)
            .map_err(|error| anyhow::anyhow!("dense arc failed: {error:?}"))?;
        ensure!(arc.len() == 20_161, "sealed arc node count differs");
        let authority = session
            .natural_dense_ephemeris_grid(&object, &[HORIZON_S])
            .map_err(|error| anyhow::anyhow!("authority propagation failed: {error:?}"))?;
        let error_km = position_error_km(
            *arc.get(20_160)
                .context("sealed arc is shorter than its last node")?,
            *authority
                .first()
                .context("authority grid returned no state")?,
        );
        worst_km = worst_km.max(error_km);
        println!("object {index}: {error_km:.9} km at T0 + 14 d");
    }
    println!("worst full-arc authority discrepancy: {worst_km:.9} km");
    ensure!(
        worst_km <= AUTHORITY_DISCREPANCY_CEILING_KM,
        "full-arc discrepancy {worst_km:.9} km exceeds the ceiling"
    );
    Ok(())
}

const fn object_state(object: &NaturalObjectInput) -> [f64; 6] {
    object.state()
}
