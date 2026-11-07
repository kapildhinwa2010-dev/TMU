{"text": "Morning meeting started. Please summarize quickly.", "timestamp": "2026-01-01T09:30:00", "budget": 0.25, "latency_ms": 120, "tokens_used": 12, "desired_layers": 4, "desired_pace": 0.33, "quality": 0.82}
{"text": "Explain in detail why transformer attention works.", "timestamp": "2026-01-01T14:10:00", "budget": 0.85, "latency_ms": 650, "tokens_used": 48, "desired_layers": 10, "desired_pace": 0.83, "quality": 0.94}
{"text": "Night planning session for tomorrow.", "timestamp": 1767229200, "budget": 0.45, "latency_ms": 260, "tokens_used": 20, "desired_layers": 6, "desired_pace": 0.50, "quality": 0.88}text,timestamp,budget,latency_ms,tokens_used,desired_layers,desired_pace,quality
"Morning meeting started. Please summarize quickly.","2026-01-01T09:30:00",0.25,120,12,4,0.33,0.82
"Explain in detail why transformer attention works.","2026-01-01T14:10:00",0.85,650,48,10,0.83,0.94
"Night planning session for tomorrow.",1767229200,0.45,260,20,6,0.50,0.88python gpt2_tmu_advanced.pypython gpt2_tmu_advanced.py \
  --epochs 1 \
  --batch-size 2 \
  --seq-len 32 \
  --samples 96 \
  --max-gen-tokens 16python gpt2_tmu_advanced.py \
  --data labels.jsonl \
  --dynamic \
  --per-layer-timing \
  --use-gpu-events \
  --max-gen-tokens 12python gpt2_tmu_advanced.py \
  --data labels.jsonl \
  --dynamic \
  --enable-cuda-cycles \
  --max-gen-tokens 12python gpt2_tmu_advanced.py --data labels.jsonl{
  "pace": 0.34,
  "layer_frac": 0.29,
  "token_frac": 0.31,
  "active_layers": 4,
  "max_new_tokens": 12,
  "temperature": 0.68,
  "refinement_passes": 1,
  "traffic_load": 0.12,
  "thermal_load": 0.22,
  "load_multiplier": 0.78,
  "difficulty": 0.61,
  "budget": 0.25,
  "generated_tokens": 12,
  "last_delta_us": 980.4,
  "last_gpu_delta_us": 910.2,
  "sm_clock_hz": 1500000000.0
}{
  "active_layers": 10,
  "max_new_tokens": 48,
  "temperature": 0.92,
  "refinement_passes": 3
}Human/World Timestamp
        │
        ▼
External TMU Encoder ─────────────
                                  │
GPU/CPU Internal Clock            │
Traffic Monitor                   ├──► Combined Time Embed
System Monitor                    │
Per-Layer Timer                   │
Prompt Difficulty                 │
Confidence/Entropy                │
        │                         │
        ▼                         ▼
Advanced Internal Adapter ──► GPT-2 Activations Modulation
                                  │
                                  ▼
                      Advanced Heads:
                      rhythm / pace / layers / tokens
                                  │
                                  ▼
                      Advanced Pace Controller
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
     Static KV-cache layer skip             Dynamic no-cache layer skip
     fast, safe                             true per-step adaptive