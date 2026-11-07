def train_gpt2_tmu_internal(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    adapter: InternalClockAdapter,
    train_loader: DataLoader,
    device: str = "cuda",
    clock: Optional[InternalClock] = None,
    epochs: int = 2,
    lr: float = 1e-3,
    w_lm: float = 1.0,
    w_rhythm: float = 1.0,
    w_pace: float = 0.7,
    budget_penalty: float = 2.0,
    grad_clip: float = 1.0,
    use_gpu_events: bool = False,
    budget_s: Optional[float] = None,
    verbose: bool = True,
):
    """
    GPT-2 + TMU + Internal Clock training.

    Base GPT-2 freeze rehta hai.
    Sirf TMU, heads, aur InternalClockAdapter train hote hain.
    """
    plugin.to(device)
    heads.to(device)
    adapter.to(device)

    if hasattr(plugin.base_model, "config"):
        plugin.base_model.config.use_cache = False

    if clock is None:
        clock = InternalClock(
            device=device,
            max_steps=64,
            budget_s=budget_s,
            use_gpu_events=use_gpu_events,
            query_gpu_clock=True,
        )

    tmu_params = freeze_base_and_enable_tmu(plugin)

    params = (
        list(tmu_params)
        + list(heads.parameters())
        + list(adapter.parameters())
    )

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)

    history = []

    for epoch in range(int(epochs)):
        plugin.base_model.eval()
        plugin.tmu.train()
        heads.train()
        adapter.train()

        total_loss = 0.0
        total_lm = 0.0
        total_rhythm = 0.0
        total_pace = 0.0
        total_over = 0.0
        total_delta_us = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            t = batch["t"].to(device, non_blocking=True)
            budget = batch["budget"].to(device, non_blocking=True)

            y_rhythm = batch["y_rhythm"].to(device, non_blocking=True)
            y_pace = batch["y_pace"].to(device, non_blocking=True)

            B = int(input_ids.shape[0])

            # -----------------------------
            # AI internal clock tick
            # -----------------------------
            snapshot = clock.tick(
                tokens=int(input_ids.numel()),
                sync=bool(use_gpu_events),
            )

            internal_features = snapshot_to_internal_tensor(
                snapshot,
                batch_size=B,
                device=device,
            )

            # -----------------------------
            # External time + internal time
            # -----------------------------
            external_embed = plugin.tmu.encode_time(
                time_stamp=t,
                batch_size=B,
                device=device,
            )

            internal_embed = adapter(internal_features)
            time_embed = external_embed + internal_embed

            # -----------------------------
            # GPT-2 forward with TMU hooks
            # -----------------------------
            outputs = plugin(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                time_embed=time_embed,
                use_cache=False,
                return_dict=True,
            )

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss_lm = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs and outputs["loss"] is not None:
                loss_lm = outputs["loss"]
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
                loss_lm = outputs[0]
            else:
                raise RuntimeError("GPT-2 output se language modeling loss nahi mil paya.")

            # -----------------------------
            # Rhythm + pace heads
            # -----------------------------
            rhythm_pred, pace_pred, _ = heads(time_embed, budget)

            loss_rhythm = F.mse_loss(rhythm_pred, y_rhythm)

            pace_safe = pace_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_pace_bce = F.binary_cross_entropy(pace_safe, y_pace)

            over_budget = F.relu(pace_pred - budget).mean()
            loss_pace = loss_pace_bce + float(budget_penalty) * over_budget

            loss = (
                float(w_lm) * loss_lm
                + float(w_rhythm) * loss_rhythm
                + float(w_pace) * loss_pace
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))

            optimizer.step()

            total_loss += float(loss.item())
            total_lm += float(loss_lm.item())
            total_rhythm += float(loss_rhythm.item())
            total_pace += float(loss_pace.item())
            total_over += float(over_budget.item())
            total_delta_us += float(snapshot.delta_us)
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_lm = total_lm / max(1, num_batches)
        avg_rhythm = total_rhythm / max(1, num_batches)
        avg_pace = total_pace / max(1, num_batches)
        avg_over = total_over / max(1, num_batches)
        avg_delta_us = total_delta_us / max(1, num_batches)

        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "loss_lm": avg_lm,
                "loss_rhythm": avg_rhythm,
                "loss_pace": avg_pace,
                "over_budget": avg_over,
                "avg_delta_us": avg_delta_us,
            }
        )

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"lm={avg_lm:.4f} | "
                f"rhythm={avg_rhythm:.4f} | "
                f"pace={avg_pace:.4f} | "
                f"over={avg_over:.4f} | "
                f"delta_us={avg_delta_us:.2f}"
            )

    return history