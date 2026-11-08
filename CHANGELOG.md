## [0.2.0] - 2026-11-08

### Added
- **Additional fusion layer** (TMU Layer v2): `src/tmu/fusion.py` me `TemporalFusionLayer` aur `TemporalFusionStack`.
  - Gated residual MLP over the TMU time embedding (`x + gate * delta`).
  - Zero-init output projection → layer init pe exact identity, pretrained behaviour safe.
  - External + internal embeddings ab combine hone ke baad fusion stack se guzarte hain
    (`TimeManagementUnit.encode_time(..., extra_embed=...)`).
  - CLI flags: `--fusion-layers`, `--fusion-expansion`, `--fusion-dropout`.
- **Transplant / export support**: `src/tmu/transplant.py`.
  - `export_tmu_layer()` / `load_tmu_layer()` — portable TMU layer file (config + state dict + metadata).
  - `mount_tmu_on_gpt2()` — trained TMU layer ko plain GPT-2 par mount karta hai.
  - `GPT2WithTMU` — base model + TMU wrapper with time-aware `generate()`.
  - `evaluate_transplant()` — baseline vs transplant loss/perplexity + time-shift logit delta.
- New scripts: `scripts/train_tmu_layer.py` (train + export + test) aur `scripts/test_transplant.py`
  (exported layer ko GPT-2 par transplant karke test).
- New docs: `docs/fusion_layer.md`, `docs/transplant.md`.
- New tests: `tests/test_fusion.py`, `tests/test_transplant.py`.

### Changed
- `tmu.encode_time()` ab optional `extra_embed` support karta hai (advanced path me internal
  clock embedding combine hoke fusion layer se guzarti hai).
- `load_tmu_state_dict()` helper — purane checkpoints (fusion params ke bina) ab bhi load hote hain.
- `pyproject.toml`: version `0.2.0`, ruff/black config, project URLs.
- README: additional layer + transplant sections, architecture diagram update.

## [0.1.1] - 2026-11-07

### Fixed
- **Critical**: `src/tmu/genrate.py` → `src/tmu/generate.py` — `import tmu` was broken (ImportError) since `__init__.py` imports from `.generate`.
- `src/tmu/looping_advans.py` → `src/tmu/looping_advanced.py` to match the public import name `tmu.looping_advanced` used by tests.

### Changed
- Repo made production-ready: root typo files renamed (`.gitignore`, `LICENSE`, `requirements.txt`, `.dockerignore`, `Dockerfile`, `Makefile`), docs/examples/scripts/tests typos fixed, 16 root research prototypes moved into `research/`, CI workflow moved to root `.github/workflows/ci.yml`, README cleaned (duplicate diagram removed, structure section added).

# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-11-07

### Added
- TMU core module for time-aware modulation.
- `TMUPlugin` wrapper for PyTorch models, especially GPT-2.
- External world-time cyclic features:
  - second
  - minute
  - hour
  - day
  - week
  - month/year approximation
- Internal AI clock features:
  - CPU nanosecond timer
  - CUDA event timing
  - GPU SM clock detection
  - estimated GPU cycles
  - per-layer timing
  - traffic/QoS monitor
  - system thermal/power/memory monitor
- Basic capability heads:
  - rhythm head
  - pace head
- Advanced capability heads:
  - rhythm head
  - pace head
  - layer fraction head
  - token fraction head
- Advanced internal adapter for 24-D internal clock features.
- Pace controllers:
  - basic pace controller
  - advanced traffic/thermal/memory-aware pace controller
- Generation modes:
  - static cache-safe layer skipping
  - dynamic no-cache layer skipping
- Confidence/entropy-based early stopping in dynamic mode.
- Data loaders:
  - JSONL
  - JSON
  - CSV
- Timestamp parsing:
  - Unix seconds
  - ISO datetime
  - common datetime formats
- Optional latency-label training:
  - `latency_ms`
  - `tokens_used`
  - `desired_layers`
  - `desired_pace`
  - `quality`
- Time coherence evaluation:
  - logit MAD
  - cosine similarity
  - argmax agreement
  - daily/weekly/yearly coherence metrics
- CLI entrypoint:
  - `python -m tmu.cli`
- Optional true CUDA cycle timer:
  - `clock64()`
  - `%globaltimer`
- Tests for core TMU behavior.