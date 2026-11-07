# API Overview

## Core Classes

### `TimeManagementUnit`

Encodes time into embeddings and applies modulation to activations.

```python
from tmu import TimeManagementUnit

tmu = TimeManagementUnit(hidden_dim=768)
embed = tmu.encode_time(time_stamp=1767229200.0, batch_size=1)
x = tmu.apply_to(x, embed)
```

### `TMUPlugin`

Wraps a PyTorch model and registers hooks.

```python
from tmu import TMUPlugin

plugin = TMUPlugin(
    base_model=model,
    hidden_dim=768,
    mode="hooks",
    target_modules=["GPT2Block", "block"],
)

out = plugin(input_ids, time_stamp=1767229200.0)
```

## Heads

### Basic heads

```python
from tmu import TmuCapabilityHeads

heads = TmuCapabilityHeads(hidden_dim=768)
rhythm_pred, pace, pace_logit = heads(time_embed, budget)
```

### Advanced heads

```python
from tmu import AdvancedTmuCapabilityHeads

heads = AdvancedTmuCapabilityHeads(hidden_dim=768, num_layers=12)
rhythm_pred, pace, layer_frac, token_frac, pace_logit = heads(time_embed, budget)
```

## Internal Clock

```python
from tmu import AdvancedInternalClock, SystemMonitor, TrafficMonitor

system_monitor = SystemMonitor("cuda")
traffic_monitor = TrafficMonitor()

clock = AdvancedInternalClock(
    device="cuda",
    max_steps=32,
    budget_s=1.0,
    use_gpu_events=False,
    system_monitor=system_monitor,
    traffic_monitor=traffic_monitor,
)

snapshot = clock.tick(tokens=16, prompt_difficulty=0.5, confidence=0.5)
```

## Generation

### Static cache-safe

```python
from tmu import generate_advanced_static

text, plan, snapshot = generate_advanced_static(
    plugin=plugin,
    heads=heads,
    adapter=adapter,
    tokenizer=tokenizer,
    prompt="Hello",
    timestamp=1767229200.0,
    budget=0.5,
    budget_s=1.0,
    max_new_tokens=32,
)
```

### Dynamic no-cache

```python
from tmu import generate_advanced_dynamic

text, plan, snapshot, metrics = generate_advanced_dynamic(
    plugin=plugin,
    heads=heads,
    adapter=adapter,
    tokenizer=tokenizer,
    prompt="Hello",
    timestamp=1767229200.0,
    budget=0.5,
    budget_s=1.0,
    max_new_tokens=16,
    enable_per_layer_timing=True,
)
```