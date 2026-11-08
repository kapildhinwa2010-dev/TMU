# Transplant: TMU layer ko GPT-2 par lana

"Transplant" ka matlab hai: trained TMU layer ko ek **plain GPT-2** (ya kisi bhi
PyTorch model) par mount karke us model ko time-aware banana — bina original
training setup ke.

## 1. Export

Trained TMU layer ko ek portable file me export karo:

```python
from tmu import export_tmu_layer

export_tmu_layer(
    plugin.tmu,
    "artifacts/tmu_layer.pt",
    metadata={"base_model": "gpt2", "fusion_layers": 1},
)
```

File format:

```json
{
  "config": {"hidden_dim": 768, "fusion_layers": 1, "apply_mode": "scale", "...": "..."},
  "state_dict": {"...": "..."},
  "metadata": {"base_model": "gpt2"}
}
```

CLI se bhi export ho jata hai:

```bash
python -m tmu.cli --mode advanced --export-layer artifacts/tmu_layer.pt
```

## 2. Load + Mount

```python
import torch
from transformers import AutoModelForCausalLM
from tmu import load_tmu_layer, mount_tmu_on_gpt2

tmu = load_tmu_layer("artifacts/tmu_layer.pt")

model = AutoModelForCausalLM.from_pretrained("gpt2")
plugin = mount_tmu_on_gpt2(model=model, tmu=tmu, mode="hooks")

out = plugin(
    input_ids=input_ids,
    attention_mask=attention_mask,
    time_stamp=1_700_000_000.0,
    use_cache=False,
    return_dict=True,
)
```

`mount_tmu_on_gpt2` base model ke weights ko bilkul nahi chhedta — TMU hooks ke
through attach hota hai (GPT-2 ke har block output par residual modulation).

## 3. Ready-made wrapper

`GPT2WithTMU` base model + TMU ka wrapper hai, jisme generation bhi built-in hai:

```python
from tmu import GPT2WithTMU, load_tmu_layer

wrapped = GPT2WithTMU(
    base_model=AutoModelForCausalLM.from_pretrained("gpt2"),
    tmu=load_tmu_layer("artifacts/tmu_layer.pt"),
    freeze_base=True,
)

ids = wrapped.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    time_stamp=1_700_000_000.0,
    max_new_tokens=16,
)
```

Har decode step par time embedding refresh hoti hai, toh generated output bhi
current time ke hisaab se behave karta hai.

## 4. Test

```bash
python scripts/test_transplant.py \
  --layer artifacts/tmu_layer.pt \
  --model gpt2 \
  --results results/transplant_test.json
```

Script ye check karta hai:

| Check | Matlab |
|---|---|
| `baseline_loss` / `baseline_perplexity` | Plain GPT-2 (bina TMU) ka eval loss |
| `transplant_loss` / `transplant_perplexity` | Transplanted model ka eval loss |
| `logit_shift_mad` | 12h time shift par logits ka mean abs change |
| `time_sensitive` | `logit_shift_mad > 0` → model actually time use kar raha hai |
| `coherence` | Daily/weekly/yearly time coherence metrics |

`logit_shift_mad == 0` ka matlab hai ki model time input ko ignore kar raha hai —
yani transplant technically toh hua, par time signal model tak nahi pohcha.

## 5. Caveats

- Layer ka `hidden_dim` aur base model ka `hidden_dim` match hona chahiye
  (GPT-2 = 768, GPT-2-medium = 1024, GPT-2-large = 1280).
- Base model ke pretrained weights transplant se affect nahi hote — sirf TMU
  layer ke weights naye hain.
- Time-sensitive behaviour tabhi dikhta hai jab TMU layer actually trained ho;
  untrained (zero-init fusion) layer logits ko almost nahi badlegi.
