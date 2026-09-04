# Third-party notices

## Orekit JB2008 implementation

`src/jb2008.rs` contains an adapted, standalone f64 implementation derived
from Orekit 13.1.2 `JB2008.java` and
`AbstractJacchiaBowmanModel.java`.

Copyright 2002-2025 CS GROUP.

Orekit credits Bruce R. Bowman (HQ AFSPC, Space Analysis Division) for original
2008 FORTRAN routine, Pascal Parraud for Java translation, Fabien Maussion for
base-model Java translation, and Bryan Cazabonne for field-element translation.
Licensed under Apache License, Version 2.0.

Pinned upstream:

- Orekit tag `13.1.2`, peeled commit
  `1841b163be0c2cdeb8b69f3ee1dd4f46e3cb3797`.
- `JB2008.java` SHA-256:
  `60803446e936284efc458c558484ae056d0e13e8b7e9ab5d396e3f7bc9628e6c`.
- `AbstractJacchiaBowmanModel.java` SHA-256:
  `50d757faab7b85d464da412b620a3e3ab701cf3efad09d6c73bff380020cb879`.
- Maven `orekit-13.1.2.jar` SHA-256:
  `89c2060c60dbe194a87dddcf3bb8343ebd16733958efe4dcc996cebbbeed655d`.
- Maven `hipparchus-core-4.0.2.jar` SHA-256:
  `7c56992f3af64429d871c33c00808ee5db5d9ed56b395b5d3d31319c4ef7ba0a`.
- Maven `hipparchus-geometry-4.0.2.jar` SHA-256:
  `4e8eede49aabd4fb71f08dd0b8b87297a9e78ed36f05c3caa4e63de5f469cceb`.
- Maven `hipparchus-ode-4.0.2.jar` SHA-256:
  `9d6fc6bc8d41068d5bc2a0880e31366e203890df8a49a9d16a39eea6595e6f7b`.
- Maven `hipparchus-fitting-4.0.2.jar` SHA-256:
  `4610b8ecfb6c083fd08b8698a022bb65afc1ceeeeec375eeee5f25441aade3ba`.
- Maven `hipparchus-optim-4.0.2.jar` SHA-256:
  `21fdb9ed87b14d7ae5c254e2d9db66b2b7984ab8c9123a26148c98f3e57716c4`.
- Maven `hipparchus-filtering-4.0.2.jar` SHA-256:
  `7f7d97b5141218f8d96765fbc96a87ed2064c48f80ccd48cc1286eff41fec21b`.
- Maven `hipparchus-stat-4.0.2.jar` SHA-256:
  `1cae30333c2d6c9658a8bb0fd9381a27dcc03465c7b4bcefa49af9213ba4dd92`.

Boundary-vector oracle:

- Generator: `oracle/OrekitJb2008Vectors.java`, SHA-256
  `ef9b974dff0c8a02278381a32be336333c6dd2f8f14c9ec2a72f67136d806beb`.
- Orekit-data commit:
  `315cce51de50b277c8885eeb3877267a882d2740`.
- Orekit-data commit archive SHA-256:
  `8a9a46176a739344236e3688d844aca572ff3c1509103662f6afa826e8495159`.
- `tai-utc.dat` SHA-256:
  `3524e1ae34d67e858873a89e59983bbc5bd100221da898e796c1b36036a310c3`.
- Runtime: Amazon Corretto OpenJDK 11.0.30.7.1.
- Exact 91 km result: `2.79456652601987750e-06 kg/m^3`,
  IEEE-754 bits `0x3ec7714931e9f622`.
- Exact 35,000 km result: `4.00863613385024900e-18 kg/m^3`,
  IEEE-754 bits `0x3c527c8fee504f59`.

No SET source code was used or adapted.

## Orekit JB2008 synthetic-frame mapping oracle

`assets/reference/orekit_jb2008` redistributes the fixed dependencies used
only to regenerate the Part A synthetic-frame JB2008 mapping oracle. Orekit
13.1.2 and Hipparchus core and geometry 4.0.2 are licensed under Apache
License, Version 2.0. Their unmodified embedded `LICENSE.txt` and `NOTICE.txt`
files are preserved below that asset root.

The sealed reduced runtime contains only:

- `orekit-13.1.2.jar` SHA-256
  `89c2060c60dbe194a87dddcf3bb8343ebd16733958efe4dcc996cebbbeed655d`;
- `hipparchus-core-4.0.2.jar` SHA-256
  `7c56992f3af64429d871c33c00808ee5db5d9ed56b395b5d3d31319c4ef7ba0a`;
- `hipparchus-geometry-4.0.2.jar` SHA-256
  `4e8eede49aabd4fb71f08dd0b8b87297a9e78ed36f05c3caa4e63de5f469cceb`.

The canonical three-JAR aggregate SHA-256 is
`7e3b504bfd38b0d6713b959085e7fcfba8a6ae635bf4b769006d816d6b7e7d24`.
The exact Maven POM receipts, source URLs, byte sizes, individual hashes, and
legal-entry origins are recorded in
`assets/reference/orekit_jb2008/manifest.json`, whose raw SHA-256 is
`6d77ddb18ad82e7b2f3c6a319d6c03c7b214ce751602f368d0fa7dec64c42d48`
and semantic SHA-256 is
`53889bcd31fbd1eaae141a5dd46179ef379afa3b909dc2a6453d772690cb0096`.

The separately identified repository-local generator
`crates/lightyear_odeint_rs/oracle/OrekitJbEciAdapterVectors.java` has SHA-256
`ce5fc9f0123a5ca54bc07b4bb89ba0cb8c1a4ab3abb62c9fe2cfbe4a994a5883`.
Its generated fixture has raw SHA-256
`928ffe14784be8f3db114f4b3ea4a06e4b84ae95d0d73227e214fd30263adade`
and semantic SHA-256
`7321f742c8f41afa9a81b1e7e9a866f6413f7f97e72ca9faf0df3dc8c44c1eb9`.
No Orekit-data archive or runtime external data is included.

This receipt supports only an Orekit synthetic-frame mapping oracle plus Rust
primitive-kernel conformance. It is not GCRF/ITRF validation, independent
physical validation, or production Rust adapter conformance.
