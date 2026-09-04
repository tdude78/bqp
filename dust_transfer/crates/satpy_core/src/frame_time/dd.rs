//! Minimal double-double arithmetic, transliterated from the sealed 4AF
//! generator (`crates/satpy_core/oracle/ErfaFrameTimeVectors.c`).
//!
//! The generator raises only EOP interpolation, ERA, outer matrix composition,
//! and the conditioned five-point stencil to double-double; the same primitive
//! is reproduced bit-for-bit here so the production chain matches the sealed
//! fixture. Written parentheses and evaluation order are load-bearing: do not
//! reassociate.

/// A double-double number `hi + lo` with `|lo| <= 0.5 ulp(hi)`.
#[derive(Clone, Copy, Debug)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

#[inline]
#[must_use]
pub const fn dd(hi: f64, lo: f64) -> Dd {
    Dd { hi, lo }
}

#[inline]
#[must_use]
pub const fn from(x: f64) -> Dd {
    Dd { hi: x, lo: 0.0 }
}

#[inline]
fn normalize(a: f64, b: f64) -> Dd {
    let s = a + b;
    let e = b - (s - a);
    dd(s, e)
}

// The double-double methods keep explicit names matching the sealed generator
// and a fixed evaluation order. Standard arithmetic traits would invite
// reassociation at call sites.
impl Dd {
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    #[inline]
    #[must_use]
    pub fn add_dd(self, b: Self) -> Self {
        let s = self.hi + b.hi;
        let v = s - self.hi;
        let e = (self.hi - (s - v)) + (b.hi - v) + self.lo + b.lo;
        normalize(s, e)
    }

    #[inline]
    #[must_use]
    pub fn neg_dd(self) -> Self {
        dd(-self.hi, -self.lo)
    }

    #[inline]
    #[must_use]
    pub fn sub_dd(self, b: Self) -> Self {
        self.add_dd(b.neg_dd())
    }

    #[inline]
    #[must_use]
    pub fn mul_dd(self, b: Self) -> Self {
        let p = self.hi * b.hi;
        let e = self.hi.mul_add(b.hi, -p) + self.hi * b.lo + self.lo * b.hi;
        normalize(p, e)
    }

    #[inline]
    #[must_use]
    pub fn scale(self, b: f64) -> Self {
        self.mul_dd(from(b))
    }

    #[inline]
    #[must_use]
    pub fn div_dd(self, b: Self) -> Self {
        let q1 = self.hi / b.hi;
        let remainder = self.sub_dd(b.mul_dd(from(q1)));
        let q2 = (remainder.hi + remainder.lo) / b.hi;
        from(q1).add_dd(from(q2))
    }
}

/// First-order double-double sine/cosine: apply the angle low part as a linear
/// correction to the binary64 `sin`/`cos` of the high part.
#[inline]
#[must_use]
pub fn sincos(x: Dd) -> (Dd, Dd) {
    let s = x.hi.sin();
    let c = x.hi.cos();
    let sine = from(s).add_dd(from(c * x.lo));
    let cosine = from(c).sub_dd(from(s * x.lo));
    (sine, cosine)
}
