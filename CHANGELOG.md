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