# Third-party notices

The evaluation license in `LICENSE.md` covers Truman DeWalch's own code. The
third-party components and data below are redistributed with the bundle under
their own terms, which continue to apply. Full texts are in
`third_party_licenses/` and in the locations named in each section.

## Orekit (Apache License 2.0)

`dust_transfer/crates/jb_rs/src/jb2008.rs` is an adapted, standalone Rust
implementation derived from Orekit 13.1.2 `JB2008.java` and
`AbstractJacchiaBowmanModel.java`, Copyright 2002-2025 CS GROUP, licensed
under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
The file has been modified: it was translated to Rust and restructured. Orekit
credits Bruce R. Bowman (HQ AFSPC) for the original 2008 FORTRAN routine, and
Pascal Parraud, Fabien Maussion and Bryan Cazabonne for the Java translations.
The pinned upstream hashes are recorded in
`dust_transfer/crates/jb_rs/THIRD_PARTY_NOTICES.md`.

## JB2008 solar and geomagnetic indices (Space Environment Technologies)

`dust_transfer/crates/jb_rs/data/jb2008/SOLFSMY.TXT` and `DTCFILE.TXT` are
unmodified data products distributed by Space Environment Technologies (SET)
under the SET Software License and Warranty Agreement
(`third_party_licenses/SET_JB2008_License.html`). As that license requires:
these indices are distributed and manufactured by Space Environment
Technologies, and additional information is available at
http://sol.spacenvironment.net/~JB2006/ and http://sol.spacenvironment.net/~JB2008/ .
No SET source code is used or adapted in this bundle.

## GOCE DIR-R6 gravity field (ICGEM / GFZ, CC BY 4.0)

`dust_transfer/crates/two_phase_transfer_rs/data/spher_const/GO_CONS_GCF_2_DIR_R6_d15.txt`
is a degree and order 15 truncation of the model
`GO_CONS_EGM_GOC_2__20091009T000000_20131020T235959_0201` (GOCE DIR Release 6),
obtained from ICGEM, https://icgem.gfz.de/ , DOI 10.5880/ICGEM.2019.004, and
licensed under Creative Commons Attribution 4.0 International
(`third_party_licenses/CC-BY-4.0.txt`). The truncation is the only change.

## ERFA and SOFA-derived frame and time routines (BSD-style)

The GCRS to ITRS frame and time chain in
`dust_transfer/crates/satpy_core/src/frame_time/` and its embedded
`eop_table.bin` are derived from ERFA 2.0.1 (Copyright (C) 2013-2021, NumFOCUS
Foundation), which is itself derived with permission from the IAU SOFA
library, under the terms in `third_party_licenses/ERFA_LICENSE.txt`. This is a
library derived from SOFA, not SOFA itself. The Earth-orientation data
(`dust_transfer/assets/reference/frame_time/finals2000A.all` and `tai-utc.dat`)
are public IERS and USNO products.

## Planetary ephemerides (JPL, via astropy, BSD 3-Clause)

`dust_transfer/crates/lightyear_odeint_rs/data/ephemeris/*.bin` are tables of
Sun, Moon, Jupiter and Venus positions sampled with astropy 7.2.0
(`get_body_barycentric`, BSD 3-Clause, https://www.astropy.org/) from the JPL
planetary ephemeris, a public-domain product of NASA/JPL.

## Rust and Python dependencies

The compiled Rust crates (nalgebra, rayon, pyo3, numpy, serde and others) are
used under the MIT, Apache-2.0 or BSD licenses declared on crates.io and are
recorded in `dust_transfer/Cargo.lock`. NumPy (BSD 3-Clause) is a runtime
dependency and is not redistributed.
