# Contributing to BioScope Studio

Thanks for your interest in improving BioScope Studio.

## Development setup

1. Fork and clone the repository.
2. Create a virtual environment.
3. Install runtime and test dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
```

4. Copy environment template and set required values.

```bash
cp .env.example .env
```

At minimum, set `DASHSCOPE_API_KEY` before running the app.

## Run locally

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Run tests

```bash
python -m pytest
```

## Pull request checklist

- Keep changes focused and scoped to one problem.
- Add or update tests when behavior changes.
- Ensure tests pass locally before opening a PR.
- Update documentation when setup, behavior, or configuration changes.
- Use clear PR descriptions with motivation and verification steps.

## Coding guidelines

- Follow existing project style and naming conventions.
- Avoid broad refactors in feature PRs.
- Keep public behavior backward compatible unless explicitly discussed.

## Reporting issues

- Use issue templates for bug reports and feature requests.
- For security issues, follow `SECURITY.md` instead of opening a public issue.
