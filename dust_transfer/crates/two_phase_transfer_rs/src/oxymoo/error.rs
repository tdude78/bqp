use std::fmt;

/// Source marker for an `OxyMOO` arithmetic or representable-capacity overflow.
///
/// `OxyMOO`'s public result remains [`anyhow::Result`]. Consumers that need a
/// domain-specific response can downcast an error chain to this marker.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArithmeticOverflow;

impl fmt::Display for ArithmeticOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OxyMOO arithmetic overflow")
    }
}

impl std::error::Error for ArithmeticOverflow {}
