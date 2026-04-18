# ECGtizer Improvement Roadmap

## Phase 1 — Foundation (P0 fixes)
- [x] Add proper `.gitignore` + untrack `.DS_Store`
- [x] Fix deprecated `cElementTree` → `ElementTree` in `PDF2XML_mod.py`
- [x] Fix dependency list in `setup.py` (add torch, opencv, matplotlib, etc.; remove mxnet, wurlitzer)
- [x] Fix Python version conflict (bump to 3.9+, remove pytesseract, remove hardcoded prefix)
- [x] Fix broken import + syntax error in `Generate_Database.py`
- [x] Add comprehensive test suite (137 tests: unit + integration)

## Phase 2 — Code Quality (P1)
- [x] Replace `print()` with `logging` module across codebase
- [x] Remove dead code (commented Pytesseract, unused imports: io, base64, re, pandas)
- [x] Fix bare except clauses in `helper_functions.py`
- [x] Fix wildcard import in `PDF2XML.py` → explicit imports
- [x] Fix boolean comparison anti-patterns (`== True`, `!= False`, `!= None`, `type() ==`)
- [x] Clean up `PDF2XML_mod.py` inconsistent variable naming

## Phase 3 — Performance & Style (P2)
- [x] Extract magic numbers into named constants
- [x] Vectorize pixel-level loops with NumPy in `PDF2XML.py`
- [x] Add type hints to public APIs
- [x] Refactor monolithic functions in `PDF2XML.py` (text_extraction, tracks_extraction)

## Phase 4 — Best Practices (P3)
- [x] Migrate to `pyproject.toml`
- [x] Set up CI/CD (GitHub Actions)
- [x] Add pre-commit hooks (black, flake8, mypy)
- [x] Add LICENSE file
- [x] Generate API docs with Sphinx

## Phase 6 — Security Hardening (P0/P1) — 2026-04-17

Findings from full codebase security audit. Issues filed on UMMISCO/ecgtizer.

### Critical
- [x] **C1** (#5) — `torch.load` without `weights_only=True` allows arbitrary code exec via malicious `.pth` (`ecgtizer/completion.py:368`)
- [ ] **C2** (#6) — `pickle.load` on vendored `translation.pkl` / `styles.pkl` (`Create_database/ecg_image_generator/HandwrittenText/generate.py:182,207`)
- [x] **C3** (#7) — `requests==2.21.0` → bump `>=2.32.3` (CVE-2023-32681, CVE-2024-35195)
- [x] **C4** (#8) — `tensorflow==2.14.0` + `keras==2.14.0` → bump `tensorflow>=2.18`, `keras>=3.8` (CVE-2025-1550 Keras Lambda RCE)

### High
- [x] **H1** (#9) — XML parsing hardening via `disable_entities=True` + pin `xmltodict>=0.13`
- [x] **H2-H6** (#10) — Bumped scikit-learn, validators, opencv-python, scipy, spacy
- [ ] **H7** — Migrate off abandoned `imgaug` to `albumentations` (separate issue)

### Medium
- [ ] **M1** (#11) — `anonymisation.py:89` only masks top-left 200×200 region; misses footer/margin patient IDs. **Does not meet HIPAA/GDPR.** Needs OCR-based masking + audit log + lossless output
- [x] **M2** (#12) — PDF page/DPI caps added in `PDF2XML.py` (MAX_PDF_PAGES=5, MAX_DPI=1200)
- [x] **M3** (#13) — SSRF hardened: https-only, timeout=10, no redirects, raise_for_status

### Low
- [x] **L1** (#14) — Pinned `Pillow>=12.2.0` in `pyproject.toml`

### Already verified clean
- No `eval`/`exec`/`shell=True`/hardcoded secrets/SQL/`tempfile.mktemp`
- `yaml.safe_load` correctly used in `helper_functions.py:22`
- `re` usage in `ecgtizer/` clean (no ReDoS)

---

## Phase 5 — Documentation (P4)
- [x] Rewrite README.md with architecture diagram, usage examples, format tables
- [x] Add module-level docstrings to all 9 modules + `__all__` in `__init__.py`
- [x] Add NumPy-style docstrings to core modules (ecgtizer.py, PDF2XML_mod.py, anonymisation.py)
- [x] Add NumPy-style docstrings to analyses.py, completion.py, extraction_functions.py
- [x] Add NumPy-style docstrings to XML2PDF.py (fix stale module docstring)
- [x] Set up Sphinx documentation framework (docs/ directory, conf.py, 8 API RST pages)
- [x] Add `[docs]` optional dependency group to pyproject.toml
- [x] Add Sphinx docs build step to CI/CD workflow
- [x] Add functional documentation with real examples for all 8 modules
- [x] Add tests for anonymisation.py and XML2PDF.xml_to_pdf (+17 tests, fix xml_to_pdf type1 bug)
- [x] Remove legacy setup.py (replaced by pyproject.toml)
- [x] Add pipeline vignette notebook (docs/vignette_pipeline.ipynb) — 12-stage visual walkthrough

---

## Backlog (untracked local scripts — refactor before committing)

Four diagnostic scripts sitting untracked in `scripts/` with hardcoded `~/Desktop/ecg_test/10 ECGs_testing` paths. Useful tooling but not portable as-is:

- [ ] `scripts/batch_diagnostics.py` (479 LOC) — batch diagnostic panels + round-trip per ECG
- [ ] `scripts/diagnose_quality.py` (225 LOC) — step-by-step pipeline diagnostic, saves intermediate images
- [ ] `scripts/ecg_diagnostic_report.py` (945 LOC) — polished publication-quality diagnostic panels
- [ ] `scripts/roundtrip_test.py` (84 LOC) — PDF → XML → PDF → XML scatter comparison

**Refactor task:** take input/output dirs via argparse (`--input-dir`, `--output-dir`), drop hardcoded `os.path.expanduser('~/Desktop/...')`, add usage examples in docstrings. Then commit. ~30 min per script.

---

## Test Coverage Summary
| Module | Unit Tests | Integration Tests | Total |
|--------|-----------|-------------------|-------|
| extraction_functions.py | 19 | 2 | 21 |
| PDF2XML.py | 14 | - | 14 |
| PDF2XML_mod.py | 18 | 4 | 22 |
| completion.py | 22 | 6 | 28 |
| analyses.py | 18 | 1 | 19 |
| XML2PDF.py | 21 | 9 | 30 |
| anonymisation.py | 5 | 4 | 9 |
| Integration (e2e) | - | 11 | 11 |
| **Total** | **117** | **37** | **154** |

---

## Progress Log
| Date | Branch | What was done |
|------|--------|---------------|
| 2026-02-27 | fix/foundation-cleanup | Phase 1 complete: .gitignore, cElementTree fix, setup.py deps, Python 3.9+ bump, Generate_Database fix, 137 tests added |
| 2026-02-27 | fix/foundation-cleanup | Phase 2 (5/6): logging module, dead code removal, bare excepts, wildcard imports, boolean anti-patterns |
| 2026-02-27 | fix/foundation-cleanup | Phase 2 (6/6): PDF2XML_mod.py variable naming cleanup |
| 2026-02-27 | fix/foundation-cleanup | Phase 3 complete: magic number constants, NumPy vectorization (5 loops), type hints on all public APIs, refactored monolithic functions |
| 2026-02-27 | fix/foundation-cleanup | Phase 4 (4/5): pyproject.toml, GitHub Actions CI, pre-commit hooks, LICENSE file |
| 2026-02-27 | fix/foundation-cleanup | Phase 4 (5/5) + Phase 5 complete: Sphinx docs, README rewrite, NumPy-style docstrings on all 72 public symbols |
| 2026-02-27 | fix/foundation-cleanup | CI docs build, 17 new tests (anonymisation + XML2PDF), xml_to_pdf type1 bug fix, setup.py removal, functional docs with real examples |
| 2026-02-27 | fix/foundation-cleanup | Pipeline vignette notebook: 12-stage visual walkthrough with real ECG data |
| 2026-04-17 | fix/foundation-cleanup | Phase 6 security: full codebase audit, 10 GH issues filed, 8 fixed (C1, C3, C4, H1, H2-H6, M2, M3, L1); open for follow-up: C2 pickle verification (#6), H7 imgaug migration (#15), M1 anonymisation rewrite (#11) |
| 2026-04-18 | fix/foundation-cleanup | Bump min Python to 3.10 (Pillow 12.2.0 requirement); full suite 159 tests green on py3.12; filed upstream at alphanumericslab/ecg-image-kit (issue #22, PR #23 for SSRF fix) |
