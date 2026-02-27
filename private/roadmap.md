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
- [ ] Migrate to `pyproject.toml`
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Add pre-commit hooks (black, flake8, mypy)
- [ ] Add LICENSE file
- [ ] Generate API docs with Sphinx

---

## Test Coverage Summary
| Module | Unit Tests | Integration Tests | Total |
|--------|-----------|-------------------|-------|
| extraction_functions.py | 19 | 2 | 21 |
| PDF2XML.py | 14 | - | 14 |
| PDF2XML_mod.py | 18 | 4 | 22 |
| completion.py | 22 | 6 | 28 |
| analyses.py | 18 | 1 | 19 |
| XML2PDF.py | 17 | 5 | 22 |
| Integration (e2e) | - | 11 | 11 |
| **Total** | **108** | **29** | **137** |

---

## Progress Log
| Date | Branch | What was done |
|------|--------|---------------|
| 2026-02-27 | fix/foundation-cleanup | Phase 1 complete: .gitignore, cElementTree fix, setup.py deps, Python 3.9+ bump, Generate_Database fix, 137 tests added |
| 2026-02-27 | fix/foundation-cleanup | Phase 2 (5/6): logging module, dead code removal, bare excepts, wildcard imports, boolean anti-patterns |
| 2026-02-27 | fix/foundation-cleanup | Phase 2 (6/6): PDF2XML_mod.py variable naming cleanup |
| 2026-02-27 | fix/foundation-cleanup | Phase 3 complete: magic number constants, NumPy vectorization (5 loops), type hints on all public APIs, refactored monolithic functions |
