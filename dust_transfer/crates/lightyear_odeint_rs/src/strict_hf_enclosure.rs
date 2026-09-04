use crate::precomputed_ephem::{
    embedded_catalogue_sha256_hex, part_a_ephemeris_authority, AllPrecomputedEphemeris, Body,
};
use crate::rhs::{jb2008_driver_authority, Jb2008DriverAuthority};
use crate::types::{ForceConfig, ForceFlags, StepperMethod};
use jb_rs::drivers::Jb2008Drivers;
use satpy_core::frame_time::authority::frame_authority;
use satpy_core::PackedGravityCoeffs;
use sha2::{Digest, Sha256};
use std::fmt;
use std::sync::OnceLock;

const EMBEDDED_DIR_R6_D15: &[u8] =
    include_bytes!("../../two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentityKind {
    Epoch,
    Force,
    Science,
    Gravity,
    Ephemeris,
    Atmosphere,
    Frame,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrictHfAuthorityError {
    MissingAsset(IdentityKind),
    InvalidAsset(IdentityKind),
    IdentityMismatch(IdentityKind),
}

impl fmt::Display for StrictHfAuthorityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingAsset(kind) => write!(formatter, "strict-HF {kind:?} asset is missing"),
            Self::InvalidAsset(kind) => write!(formatter, "strict-HF {kind:?} asset is invalid"),
            Self::IdentityMismatch(kind) => {
                write!(
                    formatter,
                    "strict-HF {kind:?} asset identity is noncanonical"
                )
            }
        }
    }
}

impl std::error::Error for StrictHfAuthorityError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AssetIdentities {
    force: [u8; 32],
    science: [u8; 32],
    gravity: [u8; 32],
    ephemeris: [u8; 32],
    atmosphere: [u8; 32],
    frame: [u8; 32],
}

/// Capability proving exact Part A force/science plus validated loaded assets.
///
/// Crate-private type, private field, no constructor. Only the two
/// `issue_for_rhs*` entry points below
/// can create it, after independently matching the compiled body class,
/// binding the two-part anchor epoch, checking loaded coverage at that epoch,
/// and recomputing all six asset identities. Full-arc coverage remains each
/// propagation request's responsibility because an RHS constructor has no horizon.
pub struct StrictHfEnclosureAuthority {
    /// Which body class the enclosure actually matched.
    ///
    /// Named `_body` while it was write-only, which is what let two guards be
    /// written that INFERRED the class from a predicate instead of reading it --
    /// both vacuous, both passing against a carve-out that had already been
    /// half-fixed. It is read now, so the underscore would be a lie.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the class this enclosure matched, interrogated by the \
                      carve-out tests. Production reaches the same fact through \
                      `has_canonical_force_signature`; storing it is what lets a \
                      test READ the answer instead of inferring it from a \
                      predicate, which is how two vacuous guards got written"
        )
    )]
    body: CanonicalBody,
    _utc_jd1_bits: u64,
    _utc_jd2_bits: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CanonicalBody {
    Transfer,
    Dust,
    BodySpecificObject,
}

pub fn issue_for_rhs(
    config: &ForceConfig,
    gravity: &PackedGravityCoeffs,
    ephemeris: Option<&AllPrecomputedEphemeris>,
    atmosphere: Option<&Jb2008Drivers>,
    utc_jd1: f64,
    utc_jd2: f64,
) -> Result<Option<StrictHfEnclosureAuthority>, StrictHfAuthorityError> {
    let science = nd_config::CompiledPartAScienceV1::part_a_v1();
    science
        .require_production_hybrid_authority()
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Science))?;
    let mut matched = None;
    for body in [CanonicalBody::Transfer, CanonicalBody::Dust] {
        let canonical = canonical_force_config_for_body(science, body)?;
        if has_canonical_force_signature(config, &canonical) {
            matched = Some((canonical, body));
            break;
        }
    }
    if matched.is_none() {
        let mut canonical = canonical_force_config_for_body(science, CanonicalBody::Transfer)?;
        if has_canonical_runtime_signature(config, &canonical)
            && body_coefficients_are_valid(config)
        {
            canonical.am_ratio = config.am_ratio;
            canonical.cd = config.cd;
            canonical.cr = config.cr;
            // `r_obj_m` belongs here for the same reason the three above do: it
            // is a per-object property, not a science knob. It was omitted, and
            // because `science_identity` HASHES it the carve-out could never
            // match -- the canonical config kept the Transfer body's 0.0 while
            // the flown config carried the real radius. Measured on the two
            // fixture targets: 0.3 vs 0 and 0.8 vs 0, with `am_ratio`, `cd`,
            // `cr` and the force identity all equal.
            //
            // The consequence was total: `issue_for_rhs` returned
            // `IdentityMismatch(Science)`, `LightyearRHS::try_new` propagated
            // it, `eclipse_coordinator` flattened it to `EclipseError::Geometry`
            // with `map_err(|_| ..)`, and `mass_solver` read that as
            // `MissAtZeroHfIntegrateFailure`. So EVERY strict-HF deterministic
            // mass row on EVERY v3 event failed with det_mass NaN and became a
            // frozen terminal -- an authority refusal wearing the costume of a
            // propagation failure.
            canonical.r_obj_m = config.r_obj_m;
            matched = Some((canonical, CanonicalBody::BodySpecificObject));
        }
    }
    let Some((canonical, body)) = matched else {
        return Ok(None);
    };
    let expected = canonical_identities(science, &canonical)?;
    // Preserve the authority's historical first-failure order: science and
    // force configuration are checked before the loaded gravity bytes. The
    // retained-Arc cache changes only how the digest is obtained at this exact
    // old validation point.
    let gravity_identity = gravity
        .authority_sha256()
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Gravity))?;
    let actual = loaded_identities(science, config, gravity_identity, ephemeris, atmosphere)?;
    verify_identities(actual, expected)?;
    validate_arc_coverage(config, ephemeris, atmosphere, utc_jd1, utc_jd2, 0.0, 0.0)?;
    Ok(Some(StrictHfEnclosureAuthority {
        body,
        _utc_jd1_bits: utc_jd1.to_bits(),
        _utc_jd2_bits: utc_jd2.to_bits(),
    }))
}

pub fn validate_arc_coverage(
    config: &ForceConfig,
    ephemeris: Option<&AllPrecomputedEphemeris>,
    atmosphere: Option<&Jb2008Drivers>,
    utc_jd1: f64,
    utc_jd2: f64,
    elapsed_start_s: f64,
    elapsed_end_s: f64,
) -> Result<(), StrictHfAuthorityError> {
    let utc_jd = utc_jd1 + utc_jd2;
    if !(utc_jd1.is_finite()
        && utc_jd2.is_finite()
        && utc_jd.is_finite()
        && elapsed_start_s.is_finite()
        && elapsed_end_s.is_finite()
        && elapsed_start_s <= elapsed_end_s)
    {
        return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch));
    }
    let anchor_tai_s = satpy_core::frame_time::authority::tai_seconds_from_utc_jd(utc_jd1, utc_jd2)
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Frame))?;
    let start_tai_s = anchor_tai_s + elapsed_start_s;
    let end_tai_s = anchor_tai_s + elapsed_end_s;
    if !(start_tai_s.is_finite() && end_tai_s.is_finite()) {
        return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch));
    }
    for tai_s in [start_tai_s, end_tai_s] {
        frame_authority()
            .segment_index(tai_s)
            .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Frame))?;
    }

    let (start_utc1, start_utc2) = utc_at_elapsed(utc_jd1, utc_jd2, elapsed_start_s)?;
    let (end_utc1, end_utc2) = utc_at_elapsed(utc_jd1, utc_jd2, elapsed_end_s)?;
    let start_utc_jd = start_utc1 + start_utc2;
    let end_utc_jd = end_utc1 + end_utc2;
    if !(start_utc_jd.is_finite() && end_utc_jd.is_finite() && start_utc_jd <= end_utc_jd) {
        return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch));
    }
    ephemeris
        .ok_or(StrictHfAuthorityError::MissingAsset(
            IdentityKind::Ephemeris,
        ))?
        .validate_dynamic_arc(config.dynamic_ephemeris_flags, start_utc_jd, end_utc_jd)
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Ephemeris))?;
    let start_utc = jb_rs::drivers::UtcJulianDay::new(start_utc_jd)
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch))?;
    let end_utc = jb_rs::drivers::UtcJulianDay::new(end_utc_jd)
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch))?;
    atmosphere
        .ok_or(StrictHfAuthorityError::MissingAsset(
            IdentityKind::Atmosphere,
        ))?
        .validate_utc_arc(start_utc, end_utc)
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Atmosphere))
}

fn utc_at_elapsed(
    utc_jd1: f64,
    utc_jd2: f64,
    elapsed_s: f64,
) -> Result<(f64, f64), StrictHfAuthorityError> {
    let (status, tai1, tai2) = satpy_core::frame_time::timescale::utctai(utc_jd1, utc_jd2);
    if status < 0 {
        return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch));
    }
    let shifted_tai2 = tai2 + elapsed_s / 86_400.0;
    if !shifted_tai2.is_finite() {
        return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch));
    }
    let (status, shifted_utc1, shifted_utc2) =
        satpy_core::frame_time::timescale::taiutc(tai1, shifted_tai2);
    if status < 0 || !(shifted_utc1.is_finite() && shifted_utc2.is_finite()) {
        return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch));
    }
    Ok((shifted_utc1, shifted_utc2))
}

fn has_canonical_force_signature(config: &ForceConfig, canonical: &ForceConfig) -> bool {
    has_canonical_runtime_signature(config, canonical)
        && config.am_ratio.to_bits() == canonical.am_ratio.to_bits()
        && config.cd.to_bits() == canonical.cd.to_bits()
        && config.cr.to_bits() == canonical.cr.to_bits()
        && config.r_obj_m.to_bits() == canonical.r_obj_m.to_bits()
}

fn has_canonical_runtime_signature(config: &ForceConfig, canonical: &ForceConfig) -> bool {
    config.sph_order == canonical.sph_order
        && config.force_flags == canonical.force_flags
        && config.subtract_first_order == canonical.subtract_first_order
        && config.atm_model == canonical.atm_model
        && config.target_propagation_mode == canonical.target_propagation_mode
        && config.dynamic_ephemeris_flags == canonical.dynamic_ephemeris_flags
        && config.dt_max.to_bits() == canonical.dt_max.to_bits()
        && config.eps.to_bits() == canonical.eps.to_bits()
        && config.integrator_method == canonical.integrator_method
}

fn body_coefficients_are_valid(config: &ForceConfig) -> bool {
    config.am_ratio.is_finite()
        && config.am_ratio > 0.0
        && config.cd.is_finite()
        && config.cd > 0.0
        && config.cr.is_finite()
        && config.cr > 0.0
        && config.r_obj_m.is_finite()
        && config.r_obj_m >= 0.0
}

fn verify_identities(
    actual: AssetIdentities,
    expected: AssetIdentities,
) -> Result<(), StrictHfAuthorityError> {
    for (kind, actual, expected) in [
        (IdentityKind::Force, actual.force, expected.force),
        (IdentityKind::Science, actual.science, expected.science),
        (IdentityKind::Gravity, actual.gravity, expected.gravity),
        (
            IdentityKind::Ephemeris,
            actual.ephemeris,
            expected.ephemeris,
        ),
        (
            IdentityKind::Atmosphere,
            actual.atmosphere,
            expected.atmosphere,
        ),
        (IdentityKind::Frame, actual.frame, expected.frame),
    ] {
        if actual != expected {
            return Err(StrictHfAuthorityError::IdentityMismatch(kind));
        }
    }
    Ok(())
}

#[cfg(test)]
fn canonical_force_config(
    science: &nd_config::CompiledPartAScienceV1,
) -> Result<ForceConfig, StrictHfAuthorityError> {
    canonical_force_config_for_body(science, CanonicalBody::Transfer)
}

fn canonical_force_config_for_body(
    science: &nd_config::CompiledPartAScienceV1,
    body: CanonicalBody,
) -> Result<ForceConfig, StrictHfAuthorityError> {
    let hybrid = science.hybrid();
    let runtime = science.strict_hf_runtime();
    let flag = |enabled: bool, flag: i32| if enabled { flag } else { 0 };
    let force_flags = flag(hybrid.force_drag, ForceFlags::DRAG)
        | flag(hybrid.force_srp, ForceFlags::SRP)
        | flag(hybrid.force_sun, ForceFlags::SUN_GRAVITY)
        | flag(hybrid.force_moon, ForceFlags::MOON_GRAVITY);
    let dynamic_ephemeris_flags = flag(runtime.dynamic_sun_ephemeris(), ForceFlags::SUN_GRAVITY)
        | flag(runtime.dynamic_moon_ephemeris(), ForceFlags::MOON_GRAVITY);
    let integrator_method = match hybrid.integrator_method {
        "vern7" => StepperMethod::Vern7,
        _ => return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Science)),
    };
    let target_propagation_mode = match runtime.target_propagation_authority() {
        "strict-hf-v3-fixed-ic" => 0,
        _ => return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Science)),
    };
    let (am_ratio, cd, cr) = match body {
        CanonicalBody::Transfer => (
            hybrid.transfer_am_ratio,
            hybrid.transfer_cd,
            hybrid.transfer_cr,
        ),
        CanonicalBody::Dust => (hybrid.dust_am_ratio, hybrid.dust_cd, hybrid.dust_cr),
        CanonicalBody::BodySpecificObject => {
            return Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Force));
        }
    };
    Ok(ForceConfig {
        sph_order: hybrid.gravity_order,
        force_flags,
        subtract_first_order: runtime.subtract_first_order(),
        atm_model: hybrid.atmosphere_model,
        am_ratio,
        cd,
        cr,
        target_propagation_mode,
        dynamic_ephemeris_flags,
        dt_max: hybrid.dt_max_s,
        eps: hybrid.tolerance,
        integrator_method,
        ..ForceConfig::default()
    })
}

/// The compiled science digest, taken from the process-wide cache when `science`
/// IS the sealed `part_a_v1` authority.
///
/// `sha256` serializes the whole authority to JSON before hashing, and this is
/// on the per-RHS path: a dense ephemeris arc builds one RHS per 600 s segment,
/// so a 14-day arc pays it about 4,000 times per object. Pointer identity is the
/// test, not value equality -- any other instance hashes its own bytes through
/// the uncached path, so the returned digest is what `sha256` would have
/// returned either way.
fn science_digest(science: &nd_config::CompiledPartAScienceV1) -> [u8; 32] {
    if std::ptr::eq(science, nd_config::CompiledPartAScienceV1::part_a_v1()) {
        nd_config::CompiledPartAScienceV1::part_a_v1_sha256()
    } else {
        science.sha256()
    }
}

fn canonical_identities(
    science: &nd_config::CompiledPartAScienceV1,
    canonical: &ForceConfig,
) -> Result<AssetIdentities, StrictHfAuthorityError> {
    static GRAVITY_IDENTITY: OnceLock<Result<[u8; 32], StrictHfAuthorityError>> = OnceLock::new();
    let gravity = *GRAVITY_IDENTITY
        .get_or_init(|| {
            crate::packed_constants_from_bytes(EMBEDDED_DIR_R6_D15, canonical.sph_order)
                .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Gravity))?
                .authority_sha256()
                .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Gravity))
        })
        .as_ref()
        .map_err(|error| *error)?;

    let ephemeris = expected_ephemeris_identity(canonical.dynamic_ephemeris_flags)?;
    let atmosphere = expected_atmosphere_identity(canonical.atm_model)?;
    Ok(AssetIdentities {
        force: force_identity(canonical),
        science: science_identity(science_digest(science), canonical),
        gravity,
        ephemeris,
        atmosphere,
        frame: science.strict_hf_runtime().frame_authority_sha256(),
    })
}

fn loaded_identities(
    science: &nd_config::CompiledPartAScienceV1,
    config: &ForceConfig,
    gravity: [u8; 32],
    ephemeris: Option<&AllPrecomputedEphemeris>,
    atmosphere: Option<&Jb2008Drivers>,
) -> Result<AssetIdentities, StrictHfAuthorityError> {
    let ephemeris = loaded_ephemeris_identity(
        ephemeris.ok_or(StrictHfAuthorityError::MissingAsset(
            IdentityKind::Ephemeris,
        ))?,
        config.dynamic_ephemeris_flags,
    )?;
    let atmosphere = loaded_atmosphere_identity(
        config.atm_model,
        atmosphere.ok_or(StrictHfAuthorityError::MissingAsset(
            IdentityKind::Atmosphere,
        ))?,
    )?;
    Ok(AssetIdentities {
        force: force_identity(config),
        science: science_identity(science_digest(science), config),
        gravity,
        ephemeris,
        atmosphere,
        frame: frame_authority().authority_sha256(),
    })
}

fn force_identity(config: &ForceConfig) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"nasa-dust/strict-hf-force-authority/v1\0");
    hasher.update(config.sph_order.to_le_bytes());
    hasher.update(config.force_flags.to_le_bytes());
    hasher.update([u8::from(config.subtract_first_order)]);
    hasher.update(config.atm_model.to_le_bytes());
    for value in [
        config.am_ratio,
        config.cd,
        config.cr,
        config.dt_max,
        config.eps,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.update([stepper_tag(config.integrator_method)]);
    hasher.finalize().into()
}

fn science_identity(science_sha256: [u8; 32], config: &ForceConfig) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"nasa-dust/strict-hf-science-authority/v1\0");
    hasher.update(science_sha256);
    hasher.update(force_identity(config));
    hasher.update([config.target_propagation_mode]);
    hasher.update(config.dynamic_ephemeris_flags.to_le_bytes());
    for value in [
        config.qm_ratio,
        config.r_obj_m,
        config.omega_earth,
        config.p_sun,
        config.mu_sun,
        config.mu_moon,
        config.mu_jupiter,
        config.mu_venus,
        config.mu_mars,
        config.mu_saturn,
        config.earth_radius,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hash_static_position(&mut hasher, config, ForceFlags::SUN_GRAVITY, config.sun_pos);
    hash_static_position(
        &mut hasher,
        config,
        ForceFlags::MOON_GRAVITY,
        config.moon_pos,
    );
    hash_static_position(
        &mut hasher,
        config,
        ForceFlags::JUPITER_GRAVITY,
        config.jupiter_pos,
    );
    hash_static_position(
        &mut hasher,
        config,
        ForceFlags::VENUS_GRAVITY,
        config.venus_pos,
    );
    hash_static_position(
        &mut hasher,
        config,
        ForceFlags::MARS_GRAVITY,
        config.mars_pos,
    );
    hash_static_position(
        &mut hasher,
        config,
        ForceFlags::SATURN_GRAVITY,
        config.saturn_pos,
    );
    hasher.finalize().into()
}

fn hash_static_position(
    hasher: &mut Sha256,
    config: &ForceConfig,
    flag: i32,
    position: Option<[f64; 3]>,
) {
    if (config.dynamic_ephemeris_flags & flag) != 0 {
        hasher.update([2]);
        return;
    }
    match position {
        None => hasher.update([0]),
        Some(position) => {
            hasher.update([1]);
            for component in position {
                hasher.update(component.to_bits().to_le_bytes());
            }
        }
    }
}

const fn stepper_tag(stepper: StepperMethod) -> u8 {
    match stepper {
        StepperMethod::Dopri5Compat => 0,
        StepperMethod::Tsit5 => 1,
        StepperMethod::Dop853 => 2,
        StepperMethod::Rkv98 => 3,
        StepperMethod::Vern7 => 4,
        StepperMethod::Vern9 => 5,
        StepperMethod::Esdirk43 => 6,
        StepperMethod::Auto => 7,
    }
}

fn loaded_ephemeris_identity(
    ephemeris: &AllPrecomputedEphemeris,
    required_flags: i32,
) -> Result<[u8; 32], StrictHfAuthorityError> {
    let authority = part_a_ephemeris_authority()
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Ephemeris))?;
    let mut bundle = Sha256::new();
    for body in Body::DEFAULT {
        if (required_flags & body.force_flag()) == 0 {
            continue;
        }
        let table = ephemeris
            .get(body)
            .ok_or(StrictHfAuthorityError::MissingAsset(
                IdentityKind::Ephemeris,
            ))?;
        bundle.update(body.name().as_bytes());
        bundle.update(b"=");
        bundle.update(table.content_sha256_hex().as_bytes());
        bundle.update(b"\n");
    }
    ephemeris_identity(authority.manifest_sha256(), &hex_digest(bundle.finalize()))
}

fn expected_ephemeris_identity(required_flags: i32) -> Result<[u8; 32], StrictHfAuthorityError> {
    let authority = part_a_ephemeris_authority()
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Ephemeris))?;
    let mut bundle = Sha256::new();
    for body in Body::DEFAULT {
        if (required_flags & body.force_flag()) == 0 {
            continue;
        }
        let identity = embedded_catalogue_sha256_hex(body).ok_or(
            StrictHfAuthorityError::MissingAsset(IdentityKind::Ephemeris),
        )?;
        bundle.update(body.name().as_bytes());
        bundle.update(b"=");
        bundle.update(identity.as_bytes());
        bundle.update(b"\n");
    }
    ephemeris_identity(authority.manifest_sha256(), &hex_digest(bundle.finalize()))
}

fn ephemeris_identity(
    manifest_sha256: &str,
    bundle_sha256: &str,
) -> Result<[u8; 32], StrictHfAuthorityError> {
    if manifest_sha256.len() != 64 || bundle_sha256.len() != 64 {
        return Err(StrictHfAuthorityError::InvalidAsset(
            IdentityKind::Ephemeris,
        ));
    }
    Ok(hash_records(
        b"nasa-dust/strict-hf-ephemeris-authority/v1\0",
        &[manifest_sha256, bundle_sha256],
    ))
}

fn loaded_atmosphere_identity_v2(drivers: &Jb2008Drivers) -> [u8; 32] {
    let identity = drivers.identity();
    hash_records(
        b"nasa-dust/strict-hf-atmosphere-authority/v1\0",
        &[
            jb_rs::jb2008::JB2008_KERNEL_NAME,
            jb_rs::jb2008::JB2008_KERNEL_VERSION,
            &identity.manifest_sha256_hex(),
            &identity.solfsmy_sha256_hex(),
            &identity.dtcfile_sha256_hex(),
            &identity.license_sha256_hex(),
            identity.solfsmy_release_header.trim_start_matches("# "),
        ],
    )
}

fn loaded_atmosphere_identity(
    atm_model: i32,
    drivers: &Jb2008Drivers,
) -> Result<[u8; 32], StrictHfAuthorityError> {
    let authority = jb2008_driver_authority(atm_model).ok_or(
        StrictHfAuthorityError::InvalidAsset(IdentityKind::Atmosphere),
    )?;
    match authority {
        Jb2008DriverAuthority::CompiledSetV2 => Ok(loaded_atmosphere_identity_v2(drivers)),
        Jb2008DriverAuthority::PartAV3PersistenceV1 => {
            let scenario = jb_rs::drivers::compiled_part_a_v3_identity()
                .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Atmosphere))?;
            Ok(part_a_v3_atmosphere_identity(drivers, &scenario))
        }
    }
}

fn expected_atmosphere_identity(atm_model: i32) -> Result<[u8; 32], StrictHfAuthorityError> {
    let authority = jb2008_driver_authority(atm_model).ok_or(
        StrictHfAuthorityError::InvalidAsset(IdentityKind::Atmosphere),
    )?;
    let drivers = authority
        .load()
        .map_err(|_| StrictHfAuthorityError::InvalidAsset(IdentityKind::Atmosphere))?;
    loaded_atmosphere_identity(atm_model, &drivers)
}

fn part_a_v3_atmosphere_identity(
    drivers: &Jb2008Drivers,
    scenario: &jb_rs::drivers::PartAV3Jb2008Identity,
) -> [u8; 32] {
    let loaded = drivers.identity();
    let loaded_manifest_sha256 = loaded.manifest_sha256_hex();
    let loaded_solfsmy_sha256 = loaded.solfsmy_sha256_hex();
    let loaded_dtcfile_sha256 = loaded.dtcfile_sha256_hex();
    let loaded_license_sha256 = loaded.license_sha256_hex();
    let mut hasher = Sha256::new();
    hasher.update(b"nasa-dust/strict-hf-atmosphere-authority/part-a-v3-persistence-v1\0");
    for record in [
        jb_rs::jb2008::JB2008_KERNEL_NAME,
        jb_rs::jb2008::JB2008_KERNEL_VERSION,
        scenario.authority_id,
        scenario.claim,
        scenario.policy,
        &scenario.manifest_sha256,
        &scenario.parent_manifest_sha256,
        &scenario.parent_solfsmy_sha256,
        &scenario.parent_dtcfile_sha256,
        &scenario.parent_license_sha256,
        scenario.observed_cutoff_utc_date,
        scenario.t0_utc,
        scenario.authorized_start_utc,
        scenario.authorized_end_utc,
        scenario.solar_support_first_utc_date,
        scenario.solar_support_last_utc_date,
        scenario.dtc_support_first_utc_date,
        scenario.dtc_support_last_utc_date,
        &loaded_manifest_sha256,
        &loaded_solfsmy_sha256,
        &loaded_dtcfile_sha256,
        &loaded_license_sha256,
        &loaded.solfsmy_release_header,
        &loaded.license_local_file,
    ] {
        hasher.update(record.len().to_le_bytes());
        hasher.update(record.as_bytes());
    }
    for value in [
        scenario.t0_utc_jd,
        scenario.authorized_start_utc_jd,
        scenario.authorized_end_utc_jd,
        loaded.solfsmy_coverage_start_jd,
        loaded.solfsmy_coverage_end_jd,
        loaded.dtc_coverage_start_jd,
        loaded.dtc_coverage_end_jd,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    for bits in scenario.source_solar_fields_bits {
        hasher.update(bits.to_le_bytes());
    }
    hasher.update(scenario.source_dtc_value.to_le_bytes());
    for value in [
        loaded.solfsmy_source_size_bytes,
        loaded.dtcfile_source_size_bytes,
        loaded.source_declared_record_count,
        loaded.solfsmy_parsed_record_count,
        loaded.dtcfile_parsed_record_count,
    ] {
        hasher.update(u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
    }
    hasher.update([u8::from(loaded.license_acknowledged)]);
    hasher.finalize().into()
}

fn hash_records(domain: &[u8], records: &[&str]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for record in records {
        hasher.update(record.len().to_le_bytes());
        hasher.update(record.as_bytes());
    }
    hasher.finalize().into()
}

fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = digest.as_ref();
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let high = HEX.get(usize::from(byte >> 4)).copied().unwrap_or(b'0');
        let low = HEX.get(usize::from(byte & 0x0f)).copied().unwrap_or(b'0');
        output.push(char::from(high));
        output.push(char::from(low));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_force_config, issue_for_rhs, validate_arc_coverage, IdentityKind,
        StrictHfAuthorityError, EMBEDDED_DIR_R6_D15,
    };
    use crate::precomputed_ephem::{get_precomputed_ephemeris, Body, PrecomputedEphemeris};
    use crate::types::ForceConfig;

    fn part_a_v3_test_jd() -> f64 {
        jb_rs::drivers::compiled_part_a_v3_identity()
            .expect("compiled Part A v3 identity")
            .t0_utc_jd
    }

    fn canonical_assets() -> (
        ForceConfig,
        std::sync::Arc<crate::precomputed_ephem::AllPrecomputedEphemeris>,
        std::sync::Arc<jb_rs::drivers::Jb2008Drivers>,
        std::sync::Arc<satpy_core::PackedGravityCoeffs>,
    ) {
        let science = nd_config::CompiledPartAScienceV1::part_a_v1();
        let config = canonical_force_config(science).expect("compiled Part A force config");
        crate::precomputed_ephem::load_precomputed_ephemeris(config.dynamic_ephemeris_flags)
            .expect("compiled Part A ephemeris must load");
        let ephemeris = get_precomputed_ephemeris().expect("loaded ephemeris");
        let atmosphere = jb_rs::drivers::compiled_part_a_v3_drivers()
            .expect("compiled Part A v3 JB2008 drivers");
        let gravity = crate::packed_constants_from_bytes(EMBEDDED_DIR_R6_D15, config.sph_order)
            .expect("canonical gravity pack");
        (config, ephemeris, atmosphere, gravity)
    }

    #[test]
    fn retained_gravity_digest_is_stable_across_authority_issuance() {
        let (config, ephemeris, atmosphere, gravity) = canonical_assets();
        let first = gravity
            .authority_sha256()
            .expect("first gravity validation");
        let same = gravity
            .authority_sha256()
            .expect("same-Arc gravity validation");
        assert_eq!(first, same);

        let equal_allocation =
            crate::packed_constants_from_bytes(EMBEDDED_DIR_R6_D15, config.sph_order)
                .expect("equal gravity pack");
        assert!(!std::sync::Arc::ptr_eq(&gravity, &equal_allocation));
        let equal = equal_allocation
            .authority_sha256()
            .expect("equal distinct-Arc gravity validation");
        assert_eq!(first, equal);
        assert!(issue_for_rhs(
            &config,
            gravity.as_ref(),
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
        )
        .expect("validated gravity token must issue authority")
        .is_some());
    }

    #[test]
    fn invalid_force_precedes_noncanonical_retained_gravity() {
        let (config, ephemeris, atmosphere, _) = canonical_assets();
        let noncanonical_gravity =
            crate::packed_constants_from_bytes(EMBEDDED_DIR_R6_D15, config.sph_order - 1)
                .expect("valid lower-order gravity pack");
        let mut invalid_force = config;
        invalid_force.force_flags ^= 1 << 29;

        assert!(issue_for_rhs(
            &invalid_force,
            noncanonical_gravity.as_ref(),
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
        )
        .expect("noncanonical force must fail before loaded-asset identity checks")
        .is_none());
        let Err(error) = issue_for_rhs(
            &config,
            noncanonical_gravity.as_ref(),
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
        ) else {
            panic!("noncanonical retained gravity must not issue authority");
        };
        assert!(matches!(
            error,
            StrictHfAuthorityError::IdentityMismatch(IdentityKind::Gravity)
        ));
    }

    /// A body-specific object must issue authority, radius included.
    ///
    /// The `BodySpecificObject` carve-out copies the caller's per-object
    /// coefficients into the canonical config so a real target can fly the
    /// sealed force model. It copied `am_ratio`, `cd` and `cr` but NOT
    /// `r_obj_m` -- which `science_identity` hashes -- so the canonical config
    /// kept the Transfer body's 0.0 while the flown config carried the real
    /// radius, and the identity could never match.
    ///
    /// Nothing covered the carve-out at all, which is how it shipped. The
    /// failure was total and silent: `issue_for_rhs` returned
    /// `IdentityMismatch(Science)`, `eclipse_coordinator` flattened it with
    /// `map_err(|_| EclipseError::Geometry)`, and `mass_solver` read that as
    /// `MissAtZeroHfIntegrateFailure` -- so every strict-HF deterministic mass
    /// row on every v3 event returned NaN and became a frozen terminal. An
    /// authority refusal wearing the costume of a propagation failure.
    ///
    /// The radius here is deliberately NOT the canonical 0.0. A zero would pass
    /// against the un-carved-out code and prove nothing.
    #[test]
    fn strict_hf_enclosure_issues_for_a_body_specific_object_with_a_real_radius() {
        let (canonical, ephemeris, atmosphere, gravity) = canonical_assets();
        let mut config = canonical;
        // Values from the two production fixture targets.
        config.am_ratio = 0.025;
        config.cd = 2.2;
        config.cr = 1.2;
        config.r_obj_m = 0.3;
        assert!(
            config.r_obj_m.to_bits() != canonical.r_obj_m.to_bits(),
            "the test radius equals the canonical one, so this proves nothing"
        );

        let authority = issue_for_rhs(
            &config,
            &gravity,
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
        )
        .expect("a body-specific object must not be refused by the strict-HF enclosure");
        // WHICH BRANCH, exactly. Asserting `!has_canonical_force_signature`
        // against ONE canonical config proves nothing: `issue_for_rhs` loops
        // over [Transfer, Dust], so a config failing to match the one this test
        // holds can still match the other and never reach the carve-out. The
        // authority records the body it matched, so read that instead of
        // inferring it.
        let authority =
            authority.expect("the body-specific carve-out did not match a body-specific object");
        assert_eq!(
            authority.body,
            super::CanonicalBody::BodySpecificObject,
            "this arm matched a canonical body directly, so it never exercised the carve-out"
        );

        // THE HARDER ARM, and the one the first version of this fix missed.
        // Above, the coefficients differ from canonical, so
        // `has_canonical_force_signature` fails and the carve-out runs. Here
        // they are IDENTICAL to canonical and only the radius differs -- which is
        // the common case, not an exotic one, because `cd` 2.2 and `cr` 1.2 are
        // shared by construction. That matched the canonical body directly,
        // taking its radius verbatim, and was then refused by the very identity
        // check it was supposed to satisfy. The carve-out never ran, because a
        // match had already been recorded.
        let mut radius_only = canonical;
        radius_only.r_obj_m = 0.8;
        // NON-VACUITY. `radius_only` is a copy of `canonical` with one field
        // changed, so asserting the other three are equal is a tautology -- it
        // cannot fail and guards nothing. The two things that CAN fail are the
        // ones that matter:
        //
        // (a) the radius really differs. If `canonical.r_obj_m` ever became
        //     0.8, this arm would be the canonical config itself, match early
        //     for the right reason, and pass while testing nothing.
        assert!(
            radius_only.r_obj_m.to_bits() != canonical.r_obj_m.to_bits(),
            "the canonical radius is already {}, so this arm is not a radius-only difference",
            canonical.r_obj_m
        );

        let radius_only_authority = issue_for_rhs(
            &radius_only,
            &gravity,
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
        )
        .expect("an object differing from canonical only in radius must not be refused");
        // (b) it goes down the OTHER branch from the first arm. Before the fix
        //     this config matched a canonical body early, took that body's
        //     radius verbatim, and was refused. After it, the radius mismatch
        //     pushes it into the carve-out -- so the branch it proves is that
        //     the early match no longer swallows a radius-only difference.
        let radius_only_authority = radius_only_authority
            .expect("a radius-only difference must not be refused by the enclosure");
        assert_eq!(
            radius_only_authority.body,
            super::CanonicalBody::BodySpecificObject,
            "a radius-only difference matched a canonical body directly, which means it took \
             that body's radius and the science identity cannot bind the real one"
        );
    }

    #[test]
    fn strict_hf_enclosure_canonical_loaded_assets_issue_authority() {
        let (config, ephemeris, atmosphere, gravity) = canonical_assets();

        assert!(issue_for_rhs(
            &config,
            &gravity,
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
        )
        .expect("canonical strict-HF assets must issue")
        .is_some());
    }

    #[test]
    fn strict_hf_enclosure_canonical_dust_body_issues_authority() {
        let science = nd_config::CompiledPartAScienceV1::part_a_v1();
        let mut config = canonical_force_config(science).expect("compiled Part A force config");
        let hybrid = science.hybrid();
        config.am_ratio = hybrid.dust_am_ratio;
        config.cd = hybrid.dust_cd;
        config.cr = hybrid.dust_cr;
        crate::precomputed_ephem::load_precomputed_ephemeris(config.dynamic_ephemeris_flags)
            .expect("compiled Part A ephemeris must load");
        let ephemeris = get_precomputed_ephemeris().expect("loaded ephemeris");
        let atmosphere = jb_rs::drivers::compiled_part_a_v3_drivers()
            .expect("compiled Part A v3 JB2008 drivers");
        let gravity = crate::packed_constants_from_bytes(EMBEDDED_DIR_R6_D15, config.sph_order)
            .expect("canonical gravity pack");

        assert!(issue_for_rhs(
            &config,
            &gravity,
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
        )
        .expect("canonical dust strict-HF assets must issue")
        .is_some());
    }

    #[test]
    fn strict_hf_enclosure_body_specific_object_issues_authority() {
        let (mut config, ephemeris, atmosphere, gravity) = canonical_assets();
        config.am_ratio = 0.012;
        config.cd = 2.1;
        config.cr = 1.2;

        assert!(issue_for_rhs(
            &config,
            &gravity,
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
        )
        .expect("body-specific strict-HF assets must validate")
        .is_some());
    }

    #[test]
    fn strict_hf_enclosure_rejects_loaded_noncanonical_gravity() {
        let (config, ephemeris, atmosphere, gravity) = canonical_assets();
        let hostile = gravity
            .truncated_to(config.sph_order - 1)
            .expect("lower-order gravity pack");
        assert!(matches!(
            issue_for_rhs(
                &config,
                &hostile,
                Some(&ephemeris),
                Some(&atmosphere),
                part_a_v3_test_jd(),
                0.0,
            ),
            Err(StrictHfAuthorityError::IdentityMismatch(
                IdentityKind::Gravity
            ))
        ));
    }

    #[test]
    fn strict_hf_enclosure_rejects_loaded_mutated_ephemeris_bytes() {
        let (config, ephemeris, atmosphere, gravity) = canonical_assets();
        let mut hostile = (*ephemeris).clone();
        let mut bytes = include_bytes!("../data/ephemeris/sun.bin").to_vec();
        *bytes.last_mut().expect("nonempty Sun table") ^= 1;
        let path = std::env::temp_dir().join(format!("nd-mutated-sun-{}.bin", std::process::id()));
        std::fs::write(&path, bytes).expect("write hostile Sun table");
        let mutated = PrecomputedEphemeris::load(&path).expect("parse mutated Sun table");
        std::fs::remove_file(&path).expect("remove hostile Sun table");
        hostile.set(Body::Sun, mutated);
        assert!(matches!(
            issue_for_rhs(
                &config,
                &gravity,
                Some(&hostile),
                Some(&atmosphere),
                part_a_v3_test_jd(),
                0.0,
            ),
            Err(StrictHfAuthorityError::IdentityMismatch(
                IdentityKind::Ephemeris
            ))
        ));
    }

    #[test]
    fn strict_hf_enclosure_rejects_loaded_unapproved_atmosphere() {
        let (config, ephemeris, _atmosphere, gravity) = canonical_assets();
        let hostile = jb_rs::drivers::Jb2008Drivers::from_set_bytes(
            include_bytes!("../../jb_rs/data/jb2008/SOLFSMY.TXT"),
            include_bytes!("../../jb_rs/data/jb2008/DTCFILE.TXT"),
        )
        .expect("parse unapproved driver bytes");
        assert!(matches!(
            issue_for_rhs(
                &config,
                &gravity,
                Some(&ephemeris),
                Some(&hostile),
                part_a_v3_test_jd(),
                0.0,
            ),
            Err(StrictHfAuthorityError::IdentityMismatch(
                IdentityKind::Atmosphere
            ))
        ));
    }

    #[test]
    fn part_a_v3_enclosure_rejects_historical_v2_atmosphere_identity() {
        let (config, ephemeris, _atmosphere, gravity) = canonical_assets();
        assert_eq!(config.atm_model, 8);
        let v2 = jb_rs::drivers::compiled_drivers().expect("compiled v2 SET drivers");

        assert!(matches!(
            issue_for_rhs(
                &config,
                &gravity,
                Some(&ephemeris),
                Some(&v2),
                part_a_v3_test_jd(),
                0.0,
            ),
            Err(StrictHfAuthorityError::IdentityMismatch(
                IdentityKind::Atmosphere
            ))
        ));
    }

    #[test]
    fn strict_hf_enclosure_noncanonical_force_gets_no_authority() {
        let science = nd_config::CompiledPartAScienceV1::part_a_v1();
        let mut config = canonical_force_config(science).expect("compiled Part A force config");
        config.atm_model = 6;
        let gravity = crate::packed_constants_from_bytes(EMBEDDED_DIR_R6_D15, config.sph_order)
            .expect("canonical gravity pack");

        assert!(issue_for_rhs(&config, &gravity, None, None, f64::NAN, 0.0)
            .expect("noncanonical force is outside enclosure authority")
            .is_none());
    }

    #[test]
    fn strict_hf_enclosure_nonfinite_two_part_epoch_fails_closed() {
        // Supply the real ephemeris and atmosphere. With `None` the
        // missing-asset arm fires first and this never reaches the epoch
        // check it is named for -- it passed on the reason it did not mean.
        let (config, ephemeris, atmosphere, gravity) = canonical_assets();

        assert!(matches!(
            issue_for_rhs(
                &config,
                &gravity,
                Some(&ephemeris),
                Some(&atmosphere),
                part_a_v3_test_jd(),
                f64::NAN,
            ),
            Err(super::StrictHfAuthorityError::InvalidAsset(
                super::IdentityKind::Epoch
            ))
        ));
    }

    #[test]
    fn strict_hf_enclosure_epoch_outside_frame_span_fails_closed() {
        // Supply the real ephemeris and atmosphere. With `None` this returned
        // MissingAsset(Ephemeris) and never reached the frame-span check.
        let (config, ephemeris, atmosphere, gravity) = canonical_assets();

        assert!(matches!(
            issue_for_rhs(
                &config,
                &gravity,
                Some(&ephemeris),
                Some(&atmosphere),
                9_000_000.5,
                0.0,
            ),
            Err(super::StrictHfAuthorityError::InvalidAsset(
                super::IdentityKind::Frame
            ))
        ));
    }

    #[test]
    fn strict_hf_enclosure_canonical_fourteen_day_arc_has_loaded_coverage() {
        let (config, ephemeris, atmosphere, _gravity) = canonical_assets();

        validate_arc_coverage(
            &config,
            Some(&ephemeris),
            Some(&atmosphere),
            part_a_v3_test_jd(),
            0.0,
            0.0,
            14.0 * 86_400.0,
        )
        .expect("canonical strict-HF assets must cover the closed v3 horizon");
    }

    #[test]
    fn strict_hf_enclosure_arc_rejects_reversed_or_uncovered_time() {
        let (config, ephemeris, atmosphere, _gravity) = canonical_assets();

        assert_eq!(
            validate_arc_coverage(
                &config,
                Some(&ephemeris),
                Some(&atmosphere),
                part_a_v3_test_jd(),
                0.0,
                1.0,
                0.0,
            ),
            Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Epoch))
        );
        assert_eq!(
            validate_arc_coverage(
                &config,
                Some(&ephemeris),
                Some(&atmosphere),
                part_a_v3_test_jd(),
                0.0,
                0.0,
                1.0e12,
            ),
            Err(StrictHfAuthorityError::InvalidAsset(IdentityKind::Frame))
        );
    }
}
