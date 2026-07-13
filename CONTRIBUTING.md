# Contributing to Mighty

Thank you for your interest in contributing to Mighty! This document covers how to set up your environment, run tests, submit changes, and get support.

## Setting Up a Development Environment

We recommend [uv](https://github.com/astral-sh/uv) for environment management.

```bash
uv venv --python=3.11
source .venv/bin/activate
make install-dev   # installs Mighty in editable mode with all dev dependencies
```

Optional environment backends:

```bash
uv pip install -e ".[carl]"       # CARL environments
uv pip install -e ".[dacbench]"   # DACBench environments
uv pip install -e ".[pufferlib]"  # Pufferlib environments
```

## Running Tests

```bash
make test                                      # full test suite with coverage
uv run pytest test/test_cli.py -v              # single file
uv run pytest test/ -k "test_name" -v         # single test by name
```

## Linting and Formatting

```bash
make check    # check formatting (ruff format --check + ruff check)
make format   # auto-fix formatting (isort + ruff format + ruff check --fix)
make typing   # run mypy type checks
```

Please ensure `make check` passes before submitting a pull request.

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`.
2. Make your changes, including tests for any new functionality.
3. Ensure `make test` and `make check` both pass locally.
4. Open a pull request and fill in the PR template, including references to any related issues.

We review pull requests on a best-effort basis. We appreciate your patience — if a review is slow it is usually because the reviewers are busy, not because the contribution is unwelcome.

## Reporting Issues

Please open a [GitHub issue](https://github.com/automl/Mighty/issues) with:
- A minimal reproducible example
- Your Python and Mighty version (`python --version`, `pip show mighty-rl`)
- The full error traceback

Check existing issues before opening a new one.

## Getting Support

- **GitHub Issues** — for bugs and feature requests
- **GitHub Discussions** — for questions and general usage help
- **Email** — for anything else: a.mohan@ai.uni-hannover.de

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
