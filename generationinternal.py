@torch.no_grad()
def generate_with_internal_clock(
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
    True internal-clock adaptive generation.

    Ye custom GPT-2 forward use karta hai:
      - layer skipping possible
      - har token ke baad internal clock update
      - pace ke hisab se active layers aur max tokens change

    Note:
      Ye KV cache use nahi karta, isliye slow ho sakta hai.
      Lekin pacing/layer-control ke liye sabse useful hai.
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

    # First tick: prompt processing time
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

    max_allowed = int(max_new_tokens)
    target_tokens = min(max_allowed, int(plan["max_new_tokens"]))

    generated = input_ids.clone()
    steps_done = 0

    while steps_done < target_tokens:
        attn = torch.ones_like(generated)

        out = custom_gpt2_forward_with_tmu(
            model=model,
            tmu=plugin.tmu,
            input_ids=generated,
            attention_mask=attn,
            time_embed=time_embed,
            active_layers=int(plan["active_layers"]),
            modulate_blocks=True,
        )

        logits = out["logits"][:, -1, :]

        temp = max(1e-3, float(plan["temperature"]))

        if do_sample:
            scaled = logits / temp

            if top_k is None or top_k <= 0:
                k = scaled.size(-1)
            else:
                k = min(int(top_k), scaled.size(-1))

            topk_logits, topk_indices = torch.topk(scaled, k)
            probs = F.softmax(topk_logits, dim=-1)
            selected = torch.multinomial(probs, 1)
            next_token = topk_indices.gather(-1, selected)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)
        steps_done += 1

        if (
            tokenizer.eos_token_id is not None
            and int(next_token.item()) == int(tokenizer.eos_token_id)
        ):
            break

        # -----------------------------
        # Internal clock update after token
        # -----------------------------
        snapshot = clock.tick(
            tokens=1,
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

        plan = controller.from_pace(float(pace[0].item()), float(budget))

        # Dynamic token budget:
        # agar pace/budget pressure badla, to remaining tokens adjust ho sakte hain
        target_tokens = min(
            max_allowed,
            steps_done + int(plan["max_new_tokens"]),
        )

        # Agar time budget khatam hone ke kareeb hai, jaldi rok do
        if snapshot.pressure >= 0.98:
            break

    text = tokenizer.decode(
        generated[0][prompt_len:],
        skip_special_tokens=True,
    )

    plan_out = dict(plan)
    plan_out["pace"] = float(pace[0].item())
    plan_out["generated_tokens"] = steps_done
    plan_out["last_delta_ms"] = snapshot.delta_ms
    plan_out["last_delta_us"] = snapshot.delta_us
    plan_out["last_gpu_delta_us"] = snapshot.gpu_delta_us
    plan_out["last_cycles"] = snapshot.cycles
    plan_out["sm_clock_hz"] = snapshot.sm_clock_hz

    return text, plan_out, snapshot