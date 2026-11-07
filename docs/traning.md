# Training Guide

## What is trained?

Only these components are trained:

- TMU module
- capability heads
- advanced internal adapter

The base GPT-2 model remains frozen.

## Losses

### Basic training

```text
loss =
  w_lm * language_modeling_loss
+ w_rhythm * rhythm_prediction_loss
+ w_pace * pace_prediction_loss
+ budget_penalty * over_budget_penalty
```

### Advanced training

```text
loss =
  w_lm * language_modeling_loss
+ w_rhythm * rhythm_prediction_loss
+ w_pace * pace_prediction_loss
+ w_layer * layer_fraction_loss
+ w_token * token_fraction_loss
+ budget_penalty * over_budget_penalty
```

## Targets

### Rhythm target

Future cyclic time features, example: 10 minutes ahead.

### Pace target

Can come from:

- `desired_pace`
- `desired_layers / num_layers`
- `tokens_used / max_label_tokens`
- heuristic fallback

### Layer target

Comes from:

- `desired_layers / num_layers`
- fallback to pace target

### Token target

Comes from:

- `tokens_used / max_label_tokens`
- fallback to pace target

## Recommended small run

```bash
python -m tmu.cli \
  --mode advanced \
  --epochs 1 \
  --batch-size 2 \
  --seq-len 32 \
  --samples 64 \
  --lr 1e-3
```

## Notes

- Training is intentionally small-scale.
- For real pacing behavior, provide real latency labels.
- GPU events make timing more accurate but slower.