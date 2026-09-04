#[derive(Debug)]
pub struct Tableau {
    pub stages: usize,
    pub c: &'static [f64],
    pub a: &'static [&'static [f64]],
    pub b: &'static [f64],
    pub b_hat: Option<&'static [f64]>,
    pub err: Option<&'static [f64]>,
    pub err3: Option<&'static [f64]>,
    /// Smallest `eps` at which this tableau's THIRD-order embedded estimate is
    /// trustworthy enough to blend into the error norm, under
    /// `ErrorControl::Absolute`.
    ///
    /// `None` means never blend, and is the correct value for every tableau
    /// with `err3: None`. A tableau that supplies `err3` must supply this too;
    /// `err3_blend_threshold_is_declared_by_every_tableau_that_has_err3`
    /// enforces the pairing.
    ///
    /// # Why this lives on the tableau
    ///
    /// It was previously a literal `eps >= 1e-6` in the solver's error-norm
    /// branch. That made the DEFINITION of the error norm change
    /// discontinuously partway through any tolerance sweep that crossed 1e-6 —
    /// `max5` on one side, `sqrt(max5^2 + 0.01*max3^2)` on the other — for
    /// reasons that have nothing to do with the tolerance and everything to do
    /// with one specific tableau's coefficients. A property of DOP853 was being
    /// expressed as a property of the solver.
    ///
    /// The threshold itself is unchanged; only its home is. See
    /// `dop853.rs` for the stability justification.
    pub err3_min_eps: Option<f64>,
    /// The method's advertised order. The solver never reads it — it steps
    /// from `b`/`err` and controls from `order_err` — so the only consumers
    /// are `order_conditions`, which checks the coefficients actually deliver
    /// this order, and the per-tableau spot checks. That is the point: a
    /// declared order nothing verifies is worse than none.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read only by the order-condition checks")
    )]
    pub order: u8,
    pub order_err: u8,
    pub fsal: bool,
}

#[cfg(test)]
mod order_conditions;

#[cfg(test)]
mod err3_pairing {
    use super::*;

    /// The pinned corpus from `order_conditions`: exact count and membership
    /// are asserted against the `Method` enum there, so this module cannot
    /// silently miss a tableau the way its former hand-copied list could.
    fn all() -> Vec<(&'static str, &'static Tableau)> {
        super::order_conditions::all_tableaus()
    }

    /// `err3` and `err3_min_eps` must be declared together.
    ///
    /// The solver folds the third-order estimate in only when BOTH are present,
    /// so a tableau supplying `err3` without a threshold would silently never
    /// use it — a coefficient set carried at full cost and never read. The
    /// reverse pairing is equally wrong: a threshold without coefficients is a
    /// claim about an estimate that does not exist.
    ///
    /// This is the invariant that made moving the threshold out of the solver
    /// safe. While it was a literal `eps >= 1e-6` in the error-norm branch,
    /// nothing connected it to the tableau it described, and a new tableau with
    /// an `err3` would have silently inherited DOP853's stability threshold.
    #[test]
    fn err3_blend_threshold_is_declared_by_every_tableau_that_has_err3() {
        for (name, tableau) in all() {
            assert_eq!(
                tableau.err3.is_some(),
                tableau.err3_min_eps.is_some(),
                "{name}: err3 and err3_min_eps must be declared together \
                 (err3={:?}, err3_min_eps={:?})",
                tableau.err3.map(<[f64]>::len),
                tableau.err3_min_eps,
            );
            if let Some(min_eps) = tableau.err3_min_eps {
                assert!(
                    min_eps > 0.0 && min_eps.is_finite(),
                    "{name}: err3_min_eps must be a positive finite tolerance, got {min_eps}"
                );
            }
        }
    }

    /// DOP853 is the only tableau carrying a third-order estimate.
    ///
    /// Pinned so that adding one elsewhere forces a decision about its own
    /// stability threshold rather than inheriting DOP853's 1e-6 by accident.
    #[test]
    fn dop853_is_the_only_tableau_with_a_third_order_estimate() {
        let with_err3: Vec<&str> = all()
            .into_iter()
            .filter(|(_, t)| t.err3.is_some())
            .map(|(name, _)| name)
            .collect();
        assert_eq!(with_err3, vec!["dop853"]);
        assert_eq!(dop853::tableau().err3_min_eps, Some(1e-6));
    }
}

pub mod dop853;
pub mod dopri5;
pub mod rkv98;
pub mod tsit5;
pub mod vern7;
pub mod vern9;
