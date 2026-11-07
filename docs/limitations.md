# Limitations

## TMU is not consciousness

TMU does not give the model a subjective feeling of time.

It converts time measurements into numerical features and uses them to modulate activations and control compute pacing.

## Internal clock is approximate

The internal clock uses:

- CPU nanosecond timer
- CUDA events
- nvidia-smi GPU clock queries
- estimated GPU cycles

It is not the same as reading per-thread GPU cycle counters inside every kernel unless the optional CUDA extension is used.

## Dynamic layer skipping is slow

Dynamic no-cache layer skipping recomputes the full sequence every token.

It is useful for research and control experiments, not fast production serving.

## Static layer skipping is safer

Static cache-safe layer skipping fixes active layers for the whole request.

It is faster and compatible with KV cache.

## Training labels matter

If you want real reasoning-speed control, provide labels such as:

- latency_ms
- tokens_used
- desired_layers
- desired_pace
- quality

Synthetic demo labels only teach toy behavior.