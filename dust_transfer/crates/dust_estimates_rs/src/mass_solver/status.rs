#[derive(Clone, Copy, Debug)]
pub struct MfJ2MassSolveResult {
    pub root_mass_kg: f64,
    pub miss_at_root_km: f64,
    pub miss_at_zero_km: f64,
    pub miss_at_upper_km: f64,
    pub iterations: usize,
    pub status: MfJ2MassSolveStatusCode,
}

#[inline]
pub(super) const fn write_mf_j2_result(
    result: MfJ2MassSolveResult,
    mass_out: &mut f64,
    status_out: &mut MfJ2MassSolveStatusCode,
    miss_zero_out: &mut f64,
    miss_root_out: &mut f64,
    miss_upper_out: &mut f64,
    iterations_out: &mut usize,
) {
    *mass_out = result.root_mass_kg;
    *status_out = result.status;
    *miss_zero_out = result.miss_at_zero_km;
    *miss_root_out = result.miss_at_root_km;
    *miss_upper_out = result.miss_at_upper_km;
    *iterations_out = result.iterations;
}

/// Declare the status enum and every table derived from it, from ONE list.
///
/// `code()`, `from_code()`, `as_str()` and the test-side variant list were four
/// hand-maintained copies of the same information, and they drifted:
/// `HfAuthorityRefused = 22` reached `code()` and `as_str()` without reaching
/// `from_code()`, so it could be sealed into qualification evidence that the
/// same path then refused to decode.
///
/// EIGHT successive guards were written against that drift and every one had a
/// silent bypass, because each compared two things that could drift together,
/// or parsed this file's text and missed a shape Rust accepts -- an implicit
/// discriminant, a same-line attribute, two variants on one line, a doc comment
/// containing a brace. Generating the tables removes the drift instead of
/// detecting it: there is one list, and nothing left to keep in step with it.
///
/// Parameterized over the enum name, the compact-code integer type, and the
/// test-side variant-list name, so every status enum in this module is
/// declared through the same one-list shape rather than growing a fresh pair
/// of hand tables.
macro_rules! mass_solve_status_codes {
    (
        $(#[doc = $enum_doc:literal])*
        $enum_name:ident($code_ty:ty), all = $all_name:ident;
        $($(#[doc = $doc:literal])* $name:ident = $code:literal => $text:literal,)+
    ) => {
        $(#[doc = $enum_doc])*
        #[repr(i32)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $enum_name {
            $($(#[doc = $doc])* $name = $code,)+
        }

        impl $enum_name {
            /// Stable compact code for sealed qualification evidence.
            #[must_use]
            pub const fn code(self) -> $code_ty {
                match self { $(Self::$name => $code,)+ }
            }

            /// Inverse of [`Self::code`]. Total by construction.
            #[must_use]
            pub const fn from_code(code: $code_ty) -> Option<Self> {
                match code { $($code => Some(Self::$name),)+ _ => None }
            }

            #[inline]
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$name => $text,)+ }
            }
        }

        /// Every variant, generated from the same list. Cannot omit one.
        ///
        /// `#[cfg(test)]` because production never enumerates statuses -- it
        /// encodes and decodes them individually. Without the gate this is a
        /// dead constant in every normal build, and the warning was missed
        /// because the check filtered for errors only.
        #[cfg(test)]
        const $all_name: &[$enum_name] = &[$($enum_name::$name,)+];
    };
}

mass_solve_status_codes! {
    /// Deterministic mass-solver terminal status.
    ///
    /// Status codes are intentionally stable because downstream diagnostics map these
    /// numeric codes to human-readable failure reasons.
    MassSolveStatusCode(u8), all = ALL_MASS_SOLVE_STATUSES;
    /// Root solve converged (or validate-only path accepted a corrected mass).
    Converged = 0 => "converged",
    /// Event is already safe at mass=0.
    SafeByDefault = 1 => "safe_by_default",
    /// Strict HF mode requested, but HF context preparation failed.
    HfStrictPrepFailed = 2 => "hf_strict_prep_failed",
    /// Initial miss-distance evaluation at mass=0 was non-finite.
    MissAtZeroNonFinite = 3 => "miss_at_zero_nonfinite",
    /// Could not find a finite upper bracket mass.
    UpperBoundNonFinite = 4 => "upper_bound_nonfinite",
    /// Even the finite upper bracket mass cannot satisfy miss-distance target.
    PhysicsLimitedUpperBound = 5 => "physics_limited_upper_bound",
    /// Non-finite residual encountered during Brent iterations.
    NonFiniteDuringBrent = 6 => "nonfinite_during_brent",
    /// Brent iteration budget exhausted; returning best approximation.
    MaxIterReached = 7 => "max_iter_reached",
    /// Validate-only mode fell back to LF seed because HF context was incomplete.
    HfValidateFallbackLfSeed = 8 => "hf_validate_fallback_lf_seed",
    /// Validate-only mode could not compute a finite LF seed.
    HfValidateLfSeedInvalid = 9 => "hf_validate_lf_seed_invalid",
    /// Validate-only mode failed due strict HF preparation failure.
    HfValidateStrictPrepFailed = 10 => "hf_validate_strict_prep_failed",
    /// Validate-only mode initial HF miss evaluation was non-finite.
    HfValidateInitialMissNonFinite = 11 => "hf_validate_initial_miss_nonfinite",
    /// Validate-only correction/repair encountered non-finite HF miss.
    HfValidateRepairMissNonFinite = 12 => "hf_validate_repair_miss_nonfinite",
    /// Validate-only mode reached upper mass bound without meeting target.
    HfValidatePhysicsLimited = 13 => "hf_validate_physics_limited",
    /// Validate-only Brent refinement hit non-finite residual.
    HfValidateBrentNonFinite = 14 => "hf_validate_brent_nonfinite",
    /// Validate-only HF call budget exhausted; returning best approximation.
    HfValidateBudgetExhausted = 15 => "hf_validate_budget_exhausted",
    /// Root solve reached a non-positive boundary mass.
    ConvergedNonPositive = 16 => "converged_nonpositive",
    /// miss(m=0) failed because velocity state was non-finite.
    MissAtZeroInvalidVelocity = 17 => "miss_at_zero_invalid_velocity",
    /// miss(m=0) failed during ECI->equinoctial conversion.
    MissAtZeroInvalidOrbit = 18 => "miss_at_zero_invalid_orbit",
    /// miss(m=0) failed because HF integration returned no final state.
    MissAtZeroHfIntegrateFailure = 19 => "miss_at_zero_hf_integrate_failure",
    /// miss(m=0) failed because propagated state became non-finite.
    MissAtZeroPropagateNonFinite = 20 => "miss_at_zero_propagate_nonfinite",
    /// Authoritative HF trajectory violates protected Earth radius.
    HfTrajectoryPhysicallyInfeasible = 21 => "hf_trajectory_physically_infeasible",
    /// The strict-HF enclosure REFUSED the configuration handed to it.
    ///
    /// Not an integration failure: nothing was integrated. The inputs did not
    /// match the sealed authority, so no trajectory exists to judge. This
    /// previously collapsed into `MissAtZeroHfIntegrateFailure` via a catch-all
    /// `Err(_)` arm, which reported a propagation failure that never happened
    /// and cost a three-layer instrumentation pass to trace back to a missing
    /// `r_obj_m` in the strict-HF carve-out.
    HfAuthorityRefused = 22 => "hf_authority_refused",
}

mass_solve_status_codes! {
    /// MF/J2 deterministic mass-solver terminal status.
    ///
    /// Discriminants carry a retired gap at 2; the explicit `= code` literals
    /// preserve it verbatim, and the round-trip test below pins it open.
    MfJ2MassSolveStatusCode(i32), all = ALL_MF_J2_MASS_SOLVE_STATUSES;
    /// Root solve converged.
    Converged = 0 => "converged",
    /// Event is already safe at mass=0.
    SafeByDefault = 1 => "safe_by_default",
    /// Initial miss-distance evaluation at mass=0 was non-finite.
    MissAtZeroNonFinite = 3 => "miss_at_zero_nonfinite",
    /// Could not find a finite upper bracket mass.
    UpperNonFinite = 4 => "upper_nonfinite",
    /// Even the finite upper bracket mass cannot satisfy miss-distance target.
    PhysicsLimited = 5 => "physics_limited",
    /// Non-finite residual encountered at a bisection midpoint.
    MidNonFinite = 6 => "mid_nonfinite",
    /// Iteration budget exhausted; returning best approximation.
    MaxIterReached = 7 => "max_iter_reached",
    /// The physically valid release-mass domain is empty: with no dust released
    /// the target is already unbound, or its perigee is already at or below the
    /// reentry interface, so no release mass leaves it in a usable orbit.
    ///
    /// Reserved for exactly that condition. A restricted bisection that runs and
    /// still fails reports its own reason instead, because an iteration-budget
    /// or sampling failure is not an atmospheric verdict.
    ///
    /// Reported as a NaN mass, so it routes exactly as the other non-finite
    /// statuses do — through `LABEL_OTHER` and the deterministic-mass validity
    /// gate to `REASON_DETERMINISTIC_MASS_INVALID`. The distinct code exists so
    /// the condition is nameable in diagnostics, not to open a new downstream path.
    AtmosphericLimited = 8 => "atmospheric_limited",
}

#[inline]
pub(super) fn converged_status_for_mass(mass: f64) -> MassSolveStatusCode {
    if mass <= 0.0 {
        MassSolveStatusCode::ConvergedNonPositive
    } else {
        MassSolveStatusCode::Converged
    }
}

#[cfg(test)]
mod tests {
    use super::{MassSolveStatusCode, MfJ2MassSolveStatusCode};

    #[test]
    fn mf_j2_status_codes_are_closed_and_round_trip() {
        for status in super::ALL_MF_J2_MASS_SOLVE_STATUSES.iter().copied() {
            assert_eq!(
                MfJ2MassSolveStatusCode::from_code(status.code()),
                Some(status)
            );
        }

        // The discriminants are NOT contiguous: 2 is a retired gap and must
        // stay undecodable, so the contiguity/past-the-end closure proof used
        // for MassSolveStatusCode does not transfer. Closure here is an
        // exhaustive scan of a superset range: the decoder knows exactly as
        // many codes as the generated list names, so a variant reachable from
        // `from_code` but missing from the list fails the count.
        assert_eq!(MfJ2MassSolveStatusCode::from_code(2), None);
        let decodable = (-1..=64)
            .filter(|&code| MfJ2MassSolveStatusCode::from_code(code).is_some())
            .count();
        assert_eq!(decodable, super::ALL_MF_J2_MASS_SOLVE_STATUSES.len());
    }

    #[test]
    fn qualification_status_codes_are_closed_and_round_trip() {
        for status in super::ALL_MASS_SOLVE_STATUSES.iter().copied() {
            assert_eq!(MassSolveStatusCode::from_code(status.code()), Some(status));
        }

        // CLOSURE, derived from the list rather than written as a literal.
        //
        // This was `from_code(22)`, and that literal is what let
        // `HfAuthorityRefused = 22` be added to `code()` and `as_str()` while
        // being omitted from `from_code()`: the status was emitted into sealed
        // qualification evidence that could not be decoded, and this test still
        // passed because 22 really was unknown to the decoder.
        //
        // Deriving the bound from `super::ALL_MASS_SOLVE_STATUSES.len()` makes the list
        // and the decoder agree on COUNT, not just on the entries the list
        // happens to name. A variant present in `from_code` but missing from the
        // list now fails here -- which the literal could not catch, because
        // removing the tail entry keeps every other assertion true.
        let past_the_end =
            u8::try_from(super::ALL_MASS_SOLVE_STATUSES.len()).expect("status count must fit u8");
        assert_eq!(
            MassSolveStatusCode::from_code(past_the_end),
            None,
            "the decoder knows a code past the end of ALL_MASS_SOLVE_STATUSES, so a variant is \
             missing from the list"
        );
        assert_eq!(MassSolveStatusCode::from_code(u8::MAX), None);

        // Discriminants are contiguous from zero, so the list must be in order.
        // Without this the closure bound above could be satisfied by a list of
        // the right length naming the wrong variants.
        for (index, status) in super::ALL_MASS_SOLVE_STATUSES.iter().enumerate() {
            assert_eq!(
                usize::from(status.code()),
                index,
                "ALL_MASS_SOLVE_STATUSES is not in discriminant order at index {index}"
            );
        }
    }
}
