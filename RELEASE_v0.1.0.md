# BioScope Studio v0.1.0 (Initial Open Source Release)

## Summary

This release turns BioScope Studio from a private prototype into a public, maintainable open-source project.

## Included in this release

- Open-source governance assets added:
  - `LICENSE` (MIT)
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
- GitHub collaboration templates added:
  - Issue templates for bug reports and feature requests
  - Pull request template
- Continuous integration added:
  - GitHub Actions workflow for Python 3.10/3.11
  - Syntax validation and automated test execution
- Developer onboarding improved:
  - Updated `README.md` and `README.zh-CN.md`
  - Standardized environment setup with `.env.example`
- Runtime startup flow hardened:
  - Better fallback behavior in `run_demo.sh`
  - Actionable startup error messages

## Validation

- `python -m compileall app.py bioclip_model.py small_target_optimizer.py vector_store.py services tests`
- `python -m pytest` (38 passed)

## Upgrade notes

- Existing users should copy fresh environment defaults:

```bash
cp .env.example .env
```

- Set `DASHSCOPE_API_KEY` before starting the app.

## Known caveats

- The launch script supports Conda-first runtime and local Python fallback.
- If `streamlit` is missing in the selected runtime, the script now exits with an explicit install hint.
