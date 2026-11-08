# Additional Fusion Layer (TMU Layer v2)

TMU core me ab ek **additional layer** aa gaya hai — `TemporalFusionLayer`.
Ye layer TMU ke time embedding ke upar ek gated residual MLP lagata hai.

## 1. Kyun?

Pehle TMU ka flow ye tha:

```text
cyclic time features -> input_proj -> temporal_mlp -> time_embed
internal clock features -> AdvancedInternalAdapter -> internal_embed

time_embed = external_embed + internal_embed      # plain addition
```

Plain addition me koi interaction nahi hai — external aur internal signals ek doosre
ko dekh nahi sakte. Fusion layer ek chhota residual block add karta hai jo:

- combined embedding ko normalize karta hai,
- use mein expansion karta hai,
- ek **learned gate** se apna contribution control karta hai,
- residual add karta hai.

## 2. Math

```text
h      = LayerNorm(x)
u      = GELU(W_in(h))
delta  = W_out(u)
gate   = sigmoid(W_gate(u))

y = x + dropout(delta * gate)
```

`W_out` zero-init hota hai, isliye layer **init pe exact identity** hai.
Isse:

- naye params add karne se pretrained behaviour turant nahi bigadta,
- purane checkpoints `strict=False` se load ho jaate hain (naye fusion keys missing),
- ablation clean hota hai: `--fusion-layers 0` se layer poori tarah off.

## 3. Stack

`TemporalFusionStack` ek ya zyada `TemporalFusionLayer` blocks ko sequentially
lagaata hai. `num_layers=0` pe stack ke koi parameters nahi hote aur output
bilkul input jaisa rehta hai.

## 4. Kahan lagta hai?

`TimeManagementUnit.encode_time()` ab `extra_embed` accept karta hai:

```python
time_embed = tmu.encode_time(
    time_stamp=t,
    batch_size=B,
    device=device,
    extra_embed=internal_embed,   # internal clock adapter ka output
)
```

Advanced path pe external aur internal embedding pehle combine hote hain, phir
fusion stack unpe lagta hai. Isliye additional layer ko **combined** signal
dikhta hai — exact wahi cheez jo pehle plain addition se nahi ho paati thi.

Helper ke liye: `tmu.fusion.encode_fused_time_embed(...)`.

## 5. Config

CLI flags:

| Flag | Default | Meaning |
|---|---|---|
| `--fusion-layers` | `1` | Additional fusion layers ki sankhya (0 = off) |
| `--fusion-expansion` | `2` | Hidden expansion factor |
| `--fusion-dropout` | `0.0` | Fusion layer dropout |

Library level:

```python
from tmu import TimeManagementUnit, TMUPlugin

tmu = TimeManagementUnit(hidden_dim=768, fusion_layers=2)

plugin = TMUPlugin(
    base_model=gpt2,
    fusion_layers=2,
    fusion_expansion=2,
)
```

## 6. Training

Fusion layer ke params TMU ke trainable set me automatically aa jaate hain
(`freeze_base_and_enable_tmu`), isliye base GPT-2 frozen rehta hai aur sirf
TMU (including fusion layers) train hota hai.

```bash
python scripts/train_tmu_layer.py \
  --fusion-layers 1 \
  --epochs 1 \
  --samples 96 \
  --export artifacts/tmu_layer.pt
```

## 7. Ablation

Same setup me `--fusion-layers 0` vs `--fusion-layers 1` compare karke pata
chalta hai ki additional layer ka actual contribution kitna hai.

```bash
python scripts/train_tmu_layer.py --fusion-layers 0 --results results/fusion_0.json
python scripts/train_tmu_layer.py --fusion-layers 1 --results results/fusion_1.json
```
