┌──────────────────────┐
Human/World Time ───────▶│ External TMU Encoder │──┐
                         └──────────────────────┘  │
                                                   ▼
                                             Combined Time Embed
                                                   ▲
                         ┌──────────────────────┐  │
GPU/CPU Internal Clock ─▶│ InternalClockAdapter │──┘
                         └──────────────────────┘
                                   │
                                   ▼
                         GPT-2 activations modulation
                                   │
                                   ▼
                         Pace / Layer Skipping / Token Budget