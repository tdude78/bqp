# Sealed frame/time reference assets

These repo-local bytes support Task 4A characterization only. They do not
assign a frame or time scale to legacy B500 states and do not change production
authority.

## ERFA source transport

The source archive was reconstructed from the cached pyerfa 2.0.1.5 source
distribution. `ERFA_CONFIGURE_AC` declares ERFA 2.0.1 and SOFA 20231011;
`PYERFA_PKG_INFO` declares pyerfa 2.0.1.5. This is therefore labeled
`ERFA 2.0.1 / SOFA 20231011-derived`, not an exact SOFA distribution.

The gzip archive contains exactly 255 regular `erfa/src/*.c` or
`erfa/src/*.h` members: 251 C files and 4 headers. It contains no directory,
link, device, or other member. Member paths are sorted bytewise. USTAR metadata
is normalized to uid/gid 0, owner/group `root`, mode 0644, and
`2023-10-11T00:00:00Z`; gzip omits original name and time.

The canonical source aggregate uses paths relative to the recovered `erfa`
root. For every regular `src/*.c` or `src/*.h` file in bytewise path order, it
hashes:

`path || NUL || decimal_byte_length || NUL || contents`

There is no separator after contents. The 255-file aggregate is
`0155ec199de5e4d0279ab9655a9b980ac6f731be6c994ba795858e93a1204d1a`.
The archive container hash is separately recorded in `manifest.json`.
`t_erfa_c.c` and `t_erfa_c_extra.c` remain preserved in the archive but are
excluded from oracle compilation because they define upstream test drivers.

## USNO data

The three USNO payloads are exact official bytes retrieved on 2026-07-23.
`USNO_PUBLIC_RELEASE_RECEIPT.md` records their direct URLs, hashes, format
source, and public-release/copying notices.

Every generator and test must consume these checked-in bytes only. Mutable
cache, sibling-checkout, and network paths are forbidden.
