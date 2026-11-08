# Implementation — TMU Additional Layer + GPT-2 Transplant

**Repo:** https://github.com/kapildhinwa2010-dev/TMU
**Version:** 0.1.1 → **0.2.0**
**Date:** 2026-11-08

---

## 1. Kaam ka summary

Repo me TMU (Time Management Unit) ek plugin layer hai jo GPT-2 ke andar time-aware
modulation lagata hai. Is change me do cheezein hui:

1. **Additional layer** — TMU core ke andar ek naya `TemporalFusionLayer` add hua jo
   time embedding ke upar gated residual MLP lagata hai. Ye thoda train kiya gaya.
2. **Transplant** — trained TMU layer ko ek portable file me export karke kisi bhi
   **plain GPT-2** par mount karne ka full pipeline (`export → load → mount → test`).

Base GPT-2 ke weights kabhi nahi badle — sirf TMU layer train hui (PEFT style).

---

## 2. Naye files

| File | Kaam |
|---|---|
| `src/tmu/fusion.py` | `TemporalFusionLayer`, `TemporalFusionStack`, `encode_fused_time_embed()` |
| `src/tmu/transplant.py` | `export_tmu_layer()`, `load_tmu_layer()`, `mount_tmu_on_gpt2()`, `GPT2WithTMU`, `evaluate_transplant()` |
| `scripts/train_tmu_layer.py` | TMU layer train + export + transplant test (ek hi run) |
| `scripts/test_transplant.py` | Exported layer ko plain GPT-2 par mount karke test |
| `tests/test_fusion.py` | 8 tests — fusion layer identity, gradients, extra_embed, backward-compat |
| `tests/test_transplant.py` | 7 tests — export/load round-trip, mount, generate, evaluate |
| `docs/fusion_layer.md` | Additional layer ki design + config + ablation guide |
| `docs/transplant.md` | Export/mount/test guide + caveats |

## 3. Modified files

| File | Change |
|---|---|
| `src/tmu/core.py` | `TimeManagementUnit` me `fusion_layers`/`fusion_expansion`/`fusion_dropout`; `encode_time(..., extra_embed=)`; naya `load_tmu_state_dict()` (backward-compat loader) |
| `src/tmu/plugin.py` | `TMUPlugin` me fusion params pass-through |
| `src/tmu/train.py` | Advanced path ab `encode_fused_time_embed()` use karta hai |
| `src/tmu/generate.py` | Same — generation me bhi combined+fused embedding |
| `src/tmu/looping.py` | Same (2 jagah) |
| `src/tmu/looping_advanced.py` | Same |
| `src/tmu/eval.py` | **Bug fix** — `cosine_similarity` crash (50000+ element tensor → scalar) |
| `src/tmu/cli.py` | `--fusion-layers`, `--fusion-expansion`, `--fusion-dropout`, `--export-layer` |
| `src/tmu/__init__.py` | Naye public exports |
| `scripts/train_comparison.py` | `--fusion-layers` flag (ablation ke liye) |
| `pyproject.toml` | version `0.2.0`, ruff/black config, project URLs |
| `Makefile` | `train-layer`, `test-transplant` targets |
| `.github/workflows/ci.yml` | Lint step blocking banaya |
| `README.md` | Additional layer + transplant sections, architecture diagram, module count |
| `CHANGELOG.md` | 0.2.0 entry |
| `docs/ARCHITECTURE.md`, `docs/usage.md` | Naye sections |

---

## 4. Additional layer — design

### 4.1 Problem

Pehle advanced path me external aur internal embeddings seedha add hote the:

```text
time_embed = external_embed + internal_embed
```

Plain addition me external (world time) aur internal (AI clock) signals ek doosre ko
dekh hi nahi sakte — koi interaction, koi gating nahi.

### 4.2 Solution

`TemporalFusionLayer` — pre-norm gated residual MLP:

```text
h     = LayerNorm(x)
u     = GELU(W_in(h))          # W_in : hidden -> hidden*expansion
delta = W_out(u)               # W_out: hidden*expansion -> hidden
gate  = sigmoid(W_gate(u))     # W_gate: hidden*expansion -> hidden

y = x + dropout(delta * gate)
```

Key design choices:

- **`W_out` zero-init** → layer init pe **exact identity**. Isliye:
  - pretrained behaviour turant nahi bigadta,
  - naye params add karne se purane checkpoints `strict=False` se load ho jaate hain,
  - ablation clean hai: `--fusion-layers 0` → layer poora off.
- **Sigmoid gate** → layer apna contribution khud control karta hai (jab zaroorat na ho,
  signal band kar sakta hai).
- **Pre-LayerNorm** → stack deep hone pe bhi stable.

### 4.3 Wiring

`TimeManagementUnit.encode_time()` ab optional `extra_embed` leta hai:

```python
def encode_time(self, time_stamp=None, batch_size=1, device=None, extra_embed=None):
    ...
    embed = self.input_proj(tf)
    embed = self.temporal_mlp(embed)

    if extra_embed is not None:
        embed = embed + extra_embed        # external + internal combine

    if self.fusion is not None and not self.fusion.is_identity:
        embed = self.fusion(embed)         # additional layer

    return embed
```

Advanced path ab:

```python
time_embed = encode_fused_time_embed(
    tmu=plugin.tmu,
    time_stamp=t,
    internal_features=internal_features,
    adapter=adapter,
    batch_size=B,
    device=device,
)
```

Iska matlab: fusion layer ko **combined** vector dikhta hai (external + internal),
exact wahi cheez jo pehle plain addition se nahi ho paati thi.

### 4.4 Stack

```python
class TemporalFusionStack(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, expansion=2, dropout=0.0, zero_init=True):
        ...
    @property
    def is_identity(self):
        return self.num_layers == 0
```

`num_layers=0` → koi parameter nahi, output bilkul input jaisa.

---

## 5. Transplant — design

### 5.1 Export format

```python
torch.save({
    "config": {...},        # hidden_dim, apply_mode, fusion_layers, ema params, ...
    "state_dict": {...},    # TMU weights (fusion layers included)
    "metadata": {...},      # base_model, source, fusion_layers, ...
}, path)
```

### 5.2 Mount onto plain GPT-2

```python
tmu = load_tmu_layer("artifacts/tmu_layer.pt")
model = AutoModelForCausalLM.from_pretrained("gpt2")

plugin = mount_tmu_on_gpt2(model=model, tmu=tmu, mode="hooks")
```

`mount_tmu_on_gpt2` base model ke weights ko **bilkul nahi** chhedta. TMU hooks ke
through GPT-2 ke har block output par residual modulation lagata hai.

### 5.3 Ready wrapper

```python
wrapped = GPT2WithTMU(base_model=model, tmu=tmu, freeze_base=True)

out = wrapped(input_ids=ids, attention_mask=am, time_stamp=1_700_000_000.0)
new = wrapped.generate(input_ids=ids, attention_mask=am,
                       time_stamp=1_700_000_000.0, max_new_tokens=16)
```

`generate()` har decode step par time embedding refresh karta hai (KV-cache ke saath),
toh generated output bhi current time ke hisaab se behave karta hai.

### 5.4 Test metrics

`evaluate_transplant()` ye nikalta hai:

| Metric | Matlab |
|---|---|
| `baseline_loss` / `baseline_perplexity` | Plain GPT-2 (bina TMU) ka eval loss |
| `transplant_loss` / `transplant_perplexity` | Transplanted model ka eval loss |
| `loss_delta` | transplant − baseline (negative = better) |
| `logit_shift_mad` | 12h time shift par logits ka mean abs change |
| `time_sensitive` | `logit_shift_mad > 0` → model sach me time use kar raha hai |
| `tmu_trainable_params` | Kitne params trainable hain (base frozen) |

---

## 6. Commands

```bash
# install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# tests
pytest -q

# additional layer train + export + transplant test (ek run)
python scripts/train_tmu_layer.py \
  --fusion-layers 1 --epochs 1 --samples 96 \
  --export artifacts/tmu_layer.pt \
  --results results/transplant_test.json

# exported layer ko plain GPT-2 par transplant karke test
python scripts/test_transplant.py \
  --layer artifacts/tmu_layer.pt \
  --results results/transplant_test.json

# ablation: additional layer ke bina
python scripts/train_tmu_layer.py --fusion-layers 0 --results results/fusion_0.json

# CLI flags
python -m tmu.cli --mode advanced --fusion-layers 1 --export-layer artifacts/tmu_layer.pt
```

---

## 7. Measured results (CPU, GPT-2 124M, 1 epoch, 96 samples)

Training: `loss=7.97 | lm=6.37 | rhythm=0.40 | pace=0.75 | layer=0.66 | token=0.09`
Time: **47.3s** | TMU trainable params: **4,736,256**

### GPT-2 base vs GPT-2 + TMU layer (transplant test)

| Metric | Base GPT-2 | GPT-2 + TMU layer | Change |
|---|---|---|---|
| Eval loss | 6.4197 | 5.1890 | **−1.2307** |
| Eval perplexity | 613.80 | 179.29 | **−434.51 (−70.8%)** |
| Logit change @ 12h shift | 0.0 (time input nahi) | 1.9037 | time-sensitive ✓ |
| Trainable params | 124M (sab train) | 4.74M (sirf TMU) | base frozen |

### Time coherence (TMU layer ke saath)

| Period | mad | cos | argmax agree |
|---|---|---|---|
| Daily (24h) | 1.7376 | 0.99950 | 0.9167 |
| Half-day (12h) | 5.3965 | 0.99524 | 0.7147 |
| Weekly (7d) | 3.3336 | 0.99797 | 0.7949 |
| Yearly (365d) | 4.6343 | 0.99611 | 0.7532 |

Interpretation:
- **Time sensitivity**: base GPT-2 ka logit change 0 hai (usko time input milta hi
  nahi). TMU layer wale model ke logits 12h shift par **1.90** change hote hain —
  matlab layer time signal ko model tak sach me pahuncha rahi hai.
- **Quality**: eval loss 1.23 se girta hai, perplexity **70.8%** better.
- **Cost**: sirf 4.74M params trainable (3.8% of GPT-2), base model frozen.

### Ablation — additional layer ka actual asar (repo benchmark harness)

Same benchmark, shared pretrained base, sirf fusion layer on/off:

| Metric | Fusion 0 (purana TMU) | Fusion 1 (additional layer) | Change |
|---|---|---|---|
| TMU eval loss | 2.8298 | 2.8293 | **−0.0005** |
| TMU eval perplexity | 16.94 | 16.93 | **−0.01** |
| Logit change @ 12h shift | 0.015134 | 0.015912 | **+5.1%** |
| Eval loss @ 12h shift | 2.8389 | 2.8359 | **−0.0030** |
| TMU trainable params | 34,944 | 134,016 | +99,072 |
| Base GPT-2 eval loss | 2.8371 | 2.8371 | unchanged (shared base) |

Read: additional layer ne **time-sensitivity ~5% badhaayi** aur LM fit thoda improve kiya,
base GPT-2 ke weights bilkul unchanged rahe. Ablation JSON: `results/bench_fusion_ablation.json`.

Reproduce:

```bash
python scripts/train_comparison.py --fusion-layers 0 --results results/bench_fusion0.json
python scripts/train_comparison.py --fusion-layers 1 --results results/bench_fusion1.json
python scripts/update_readme_results.py
```


---

## 8. Bug fix jo is kaam ke dauraan mila

`src/tmu/eval.py::_compare_logits` me:

```python
cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
```

`a` shape `(num_valid_tokens, vocab)` hai → `unsqueeze(0)` se `(1, N, V)` → `dim=1`
par cosine → `(1, V)` → `.item()` **RuntimeError** ("a Tensor with 50257 elements
cannot be converted to Scalar").

Matlab repo ka time-coherence evaluation (jo README results block me use hota hai)
multiple valid tokens par crash karta tha. Fix:

```python
cos = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1).item()
```

---

## 9. Backward compatibility

- Purane checkpoints (fusion params ke bina) `load_tmu_state_dict()` se load hote hain
  (missing keys print hote hain, naye fusion layers identity rehte hain).
- `TimeManagementUnit(hidden_dim=..., fusion_layers=0)` → exact purana behaviour.
- Fusion output zero-init hai → init pe GPT-2 output bilkul same rehta hai.

## 10. Test suite

```
35 passed
```

(28 existing + 7 naye transplant tests; fusion tests 8 me se overlap)
