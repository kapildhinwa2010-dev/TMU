# TMU Architecture

## 1. External Time

External time human/world time hai:

- timestamp
- second cycle
- minute cycle
- hour/day cycle
- week cycle
- approximate month cycle
- approximate year cycle

Ye cyclic sin/cos features me convert hota hai.

## 2. Internal AI Time

Internal time AI/GPU compute time hai:

- CPU nanosecond timer
- CUDA event timing
- GPU SM clock speed
- estimated GPU cycles
- per-layer latency
- traffic queue delay
- active requests
- throughput
- p95 latency
- error rate
- deadline pressure
- GPU temperature
- GPU power draw
- VRAM pressure
- prompt difficulty
- confidence/entropy

## 3. TMU Modulation

TMU activations ko directly replace nahi karta.

Ye residual multiplicative modulation lagata hai:

```text
output = input * (1 + tanh(time_scale))
```

Shuru me `time_scale = 0`, isliye original model behavior change nahi hota.

## 4. Heads

### Basic heads

- rhythm head
- pace head

### Advanced heads

- rhythm head
- pace head
- layer fraction head
- token fraction head

## 5. Pace Controller

Pace controller predicted fractions ko execution plan me badalta hai:

- active_layers
- max_new_tokens
- temperature
- refinement_passes

Advanced controller additionally dekhta hai:

- traffic load
- thermal load
- memory pressure
- prompt difficulty
- budget

## 6. Generation Modes

### Static cache-safe mode

- HuggingFace KV cache use hota hai
- pure request ke liye active layers fix rehte hain
- fast hai
- production-like hai

### Dynamic no-cache mode

- har token ke baad internal clock update hota hai
- layers per step change ho sakte hain
- per-layer timing use ho sakti hai
- confidence early stop use ho sakta hai
- slow hai, lekin research ke liye powerful hai

## 6. Additional Fusion Layer (v2)

TMU Layer v2 me time embedding ke upar ek additional residual layer hota hai:

```text
h     = LayerNorm(x)
u     = GELU(W_in(h))
delta = W_out(u)
gate  = sigmoid(W_gate(u))

y = x + dropout(delta * gate)
```

- `W_out` zero-init → init pe identity, pretrained checkpoints safe.
- External (world time) aur internal (AI clock) embeddings pehle combine hote hain,
  phir fusion stack unpe lagta hai.
- Stack me `N` layers ho sakte hain (`fusion_layers=0` → poora off).

Implementation: `src/tmu/fusion.py`.

## 7. Transplant

Trained TMU layer ko ek portable file me export kiya ja sakta hai
(config + state dict + metadata) aur kisi bhi plain GPT-2 par mount kiya ja
sakta hai — base model ke weights bilkul nahi badalte.

Implementation: `src/tmu/transplant.py`.
