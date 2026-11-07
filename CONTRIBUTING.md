# Contributing to TMU-GPT-2

Thanks for your interest in contributing!

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

## Code Style

- Keep functions small and readable.
- Add docstrings for public APIs.
- Avoid breaking existing TMU plugin behavior.
- Prefer CPU-compatible tests. GPU-specific features should be optional.

## Pull Request Checklist

- [ ] Tests added or updated
- [ ] Docs updated if behavior changed
- [ ] No secrets or large model weights committed
- [ ] CHANGELOG updated for user-facing changes

## Reporting Bugs

Please include:

- OS and Python version
- PyTorch and Transformers version
- GPU model, if relevant
- Minimal reproducible command
- Full traceback