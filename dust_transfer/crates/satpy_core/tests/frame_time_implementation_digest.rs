//! Forces `CHAIN_VERSION` to move when the frame/time ALGORITHM moves.
//!
//! `frame_authority_sha256` binds EOP bytes, derived leap intervals, constants
//! and `CHAIN_VERSION`, and strict-HF enclosure treats that digest as SEMANTIC
//! identity for the frame chain. It does not bind the implementation. So fixing
//! the leap conversion, the angular-velocity extraction, or the interpolation
//! leaves the same semantic SHA unless someone remembers to bump a string by
//! hand -- and a digest that means "this is the same frame chain" while the
//! frame chain changed underneath it is worse than no digest.
//!
//! This is the reminder, made mechanical. It does not compute frame identity
//! and is not consumed by any receipt: it fails when the sources move without
//! the version, and the fix is to bump both in the same commit.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// Source files whose contents define what the frame chain COMPUTES.
///
/// `eop_table.bin` is deliberately absent: EOP bytes are already bound by
/// `frame_authority_sha256` directly, and a data refresh is not an algorithm
/// change.
const FRAME_TIME_ALGORITHM_FILES: [&str; 10] = [
    "authority.rs",
    "chain.rs",
    "cio.rs",
    "dd.rs",
    "eop.rs",
    "era.rs",
    "fund_args.rs",
    "iau2006.rs",
    "mod.rs",
    "tables.rs",
];

/// Recorded digest of the files above, taken together with `CHAIN_VERSION`.
///
/// Re-pin on ANY change to those files, copying the value the failure message
/// prints rather than re-deriving it. The digest covers whole files, comments
/// and test modules included, so an edit that changes nothing semantic still
/// moves it -- deliberately: a gate that tried to hash only the "real" code
/// would have to slice each file at a `cfg(test)` marker, and a slice is
/// exactly the shape that stops seeing the code it was meant to cover.
///
/// The question the failure asks is whether `CHAIN_VERSION` must move WITH it.
/// It must whenever the frame chain now computes something different, because
/// `frame_authority_sha256` claims semantic identity and does not read these
/// bytes. A comment or test-only edit re-pins the digest alone.
const FRAME_TIME_ALGORITHM_SHA256: &str =
    "4bc5c097fc719ff572ebb95a774155cade67fbf51b03dc218e73a5c808a2e1e2";

#[test]
fn frame_time_algorithm_digest_moves_with_the_chain_version() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/frame_time");
    let mut hasher = Sha256::new();
    hasher.update(b"nasa-dust/satpy-core/frame-time-algorithm/v1\0");
    hasher.update(satpy_core::frame_time::authority::CHAIN_VERSION.as_bytes());
    hasher.update([0_u8]);
    for name in FRAME_TIME_ALGORITHM_FILES {
        let path = dir.join(name);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        hasher.update(name.as_bytes());
        hasher.update([0_u8]);
        hasher.update(
            u64::try_from(bytes.len())
                .expect("file length fits u64")
                .to_be_bytes(),
        );
        hasher.update(&bytes);
    }
    let digest = format!("{:x}", hasher.finalize());
    assert_eq!(
        digest, FRAME_TIME_ALGORITHM_SHA256,
        "the frame/time implementation changed.\n\
         `frame_authority_sha256` claims semantic identity for the frame chain \
         and does not read these bytes, so decide explicitly: if the chain now \
         COMPUTES something different, bump CHAIN_VERSION in this same commit. \
         Either way re-pin this digest to {digest}."
    );
}
