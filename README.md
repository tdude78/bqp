# Evaluation bundle for BQP, from Truman DeWalch (MVP 0.1.0, 2026-09-03)

Everything in this directory is for BQP's internal evaluation only. The terms
are the PolyForm Strict License 1.0.0 plus the additional permission in
`LICENSE.md`, which runs until 2026-12-31. You may read and run the code. You
may not change it, pass it on, or use it commercially. Copyright stays with the
author.

The code is a minimal viable subset of a larger private system that I wrote
during doctoral research at Virginia Tech, supported by NASA grant
80NSSC23K1502. Nothing outside this bundle is licensed. Third-party components
are listed in `THIRD_PARTY_NOTICES.md`, and `MANIFEST.sha256` records the exact
contents.

| Item | Where |
|---|---|
| Draft dissertation (PDF). A draft: please do not distribute or cite it. | `dissertation_dewalch_draft.pdf` |
| Dissertation LaTeX source | `dissertation_source/` (build with `./build_pdf.sh`) |
| License and notices | `LICENSE.md`, `THIRD_PARTY_NOTICES.md`, `third_party_licenses/` |
| Transfer optimizer and high-fidelity propagator, Rust with a Python API. Builds from source in about a minute. | `dust_transfer/`, starting at `dust_transfer/README.md` |

To run the code:

```bash
cd dust_transfer
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
python examples/quickstart.py
```
