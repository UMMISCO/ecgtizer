# ECGtizer Improvement Roadmap

## Phase 1 — Foundation (P0 fixes)
- [ ] Add proper `.gitignore`
- [ ] Fix deprecated `cElementTree` → `ElementTree` in `PDF2XML_mod.py`
- [ ] Fix dependency list in `setup.py` (add torch, remove mxnet, add missing deps)
- [ ] Fix Python version conflict (env file vs numpy requirements)
- [ ] Fix broken import in `Generate_Database.py`

## Phase 2 — Code Quality (P1)
- [ ] Replace `print()` with `logging` module across codebase
- [ ] Remove dead code (commented Pytesseract, unused imports)
- [ ] Fix bare except clauses
- [ ] Fix wildcard imports
- [ ] Fix boolean comparison anti-patterns
- [ ] Add basic unit tests for extraction functions

## Phase 3 — Performance & Style (P2)
- [ ] Extract magic numbers into named constants
- [ ] Vectorize pixel-level loops with NumPy
- [ ] Add type hints to public APIs
- [ ] Refactor monolithic functions in `PDF2XML.py`

## Phase 4 — Best Practices (P3)
- [ ] Migrate to `pyproject.toml`
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Add pre-commit hooks (black, flake8, mypy)
- [ ] Add LICENSE file
- [ ] Generate API docs with Sphinx

---

## Progress Log
| Date | Branch | What was done |
|------|--------|---------------|
| | | |
