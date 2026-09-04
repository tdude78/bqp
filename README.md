# Evaluation bundle for BQP — Truman DeWalch (MVP 0.1.0, 2026-09-03)

**Terms:** everything in this directory is provided for BQP's internal
evaluation only, under the PolyForm Strict License 1.0.0 plus the additional
permission in `LICENSE.md` (valid until 2026-12-31). No changes, no
redistribution, no commercial use. Copyright remains with the author. The code
is a minimal viable subset of a larger private system written by the author
during doctoral research at Virginia Tech supported by NASA grant 80NSSC23K1502; nothing outside this bundle is
licensed. Third-party components: `THIRD_PARTY_NOTICES.md`. Exact contents:
`MANIFEST.sha256`.

| Item | Where |
|---|---|
| Draft dissertation (PDF) — draft, not for distribution or citation | `dissertation_dewalch_draft.pdf` |
| Dissertation LaTeX source | `dissertation_source/` (build with `./build_pdf.sh`) |
| License / notices | `LICENSE.md`, `THIRD_PARTY_NOTICES.md`, `third_party_licenses/` |
| Transfer optimizer + high-fidelity propagator, Rust with a Python API (build from source, ~1 min) | `dust_transfer/` — start at `dust_transfer/README.md` |

Quick run of the code:

```bash
cd dust_transfer
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
python examples/quickstart.py
```
