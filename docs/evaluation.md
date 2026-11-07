# Evaluation Guide

## Time Coherence

The evaluator runs the same text at multiple timestamps:

```text
t
t + 24 hours
t + 7 days
t + 1 year
t + 12 hours as cross-period
```

It compares logits and measures:

- MAD: mean absolute difference
- COS: cosine similarity
- AGREE: argmax next-token agreement

## Expected Behavior

If TMU learns periodic world rhythm:

- `t` vs `t + 24h` should be relatively stable
- `t` vs `t + 7d` should be relatively stable
- `t` vs `t + 1y` may be moderately stable
- `t` vs `t + 12h` may show more difference

## Command

```bash
python -m tmu.cli \
  --mode advanced \
  --epochs 1 \
  --eval-samples 2
```

## Interpretation

High cosine similarity means outputs are stable across same periodic time.

Low MAD means logits do not explode when time changes.

High agreement means the model chooses similar next tokens at equivalent time phases.