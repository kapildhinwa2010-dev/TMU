@torch.no_grad()
def generate_fast_with_internal_plan(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    adapter: InternalClockAdapter,
    tokenizer: Any,
    prompt: str,
    timestamp: TimeLike,
    budget: float,
    budget_s: Optional[float] = None,
    max_new_tokens: int = 32,
    pace_config: Optional[PaceConfig] = None,
    use_gpu_events: bool = False,
    do_sample: bool = False,
    top_k: int = 50,
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, Any], ClockSnapshot]:
    """
    Fast path:
      - internal clock se initial pace predict karo
      - plan banao
      - HuggingFace generate() use karo

    Isme true per-token layer skipping nahi hota,
    lekin max_new_tokens aur temperature internal clock se control hote hain.
    """
    model = plugin.base_model
    model.eval()
    heads.eval()
    adapter.eval()

    device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(int(seed))

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    prompt_len = int(input_ids.shape[1])

    clock = InternalClock(
        device=device,
        max_steps=int(max_new_tokens),
        budget_s=budget_s,
        use_gpu_events=use_gpu_events,
        query_gpu_clock=True,
    )

    snapshot = clock.tick(
        tokens=prompt_len,
        sync=bool(use_gpu_events),
    )

    internal_features = snapshot_to_internal_tensor(
        snapshot,
        batch_size=1,
        device=device,
    )

    pace, time_embed = predict_pace_internal(
        plugin=plugin,
        heads=heads,
        adapter=adapter,
        timestamp=timestamp,
        budget=budget,
        batch_size=1,
        device=device,
        internal_features=internal_features,
    )

    num_layers = len(model.transformer.h)
    controller = PaceController(num_layers, pace_config)
    plan = controller.from_pace(float(pace[0].item()), float(budget))

    max_new = min(int(max_new_tokens), int(plan["max_new_tokens"]))
    temp = max(1e-3, float(plan["temperature"]))

    old_current = plugin._current_time_embed
    old_cache = None

    if hasattr(model, "config"):
        old_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True

    # Generation ke dauraan TMU hooks ko ye combined time_embed milega
    plugin._current_time_embed = time_embed

    try:
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_k"] = top_k

        output_ids = model.generate(**gen_kwargs)

    finally:
        plugin._current_time_embed = old_current
        if hasattr(model, "config") and old_cache is not None:
            model.config.use_cache = old_cache

    text = tokenizer.decode(
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
    )

    plan_out = dict(plan)
    plan_out["pace"] = float(pace[0].item())
    plan_out["generated_tokens"] = int(output_ids.shape[1] - prompt_len)
    plan_out["last_delta_ms"] = snapshot.delta_ms
    plan_out["last_delta_us"] = snapshot.delta_us
    plan_out["last_gpu_delta_us"] = snapshot.gpu_delta_us
    plan_out["last_cycles"] = snapshot.cycles
    plan_out["sm_clock_hz"] = snapshot.sm_clock_hz

    return text, plan_out, snapshot