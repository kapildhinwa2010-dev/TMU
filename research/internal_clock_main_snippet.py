adapter = InternalClockAdapter(hidden_dim=hidden_dim).to(device)
clock = InternalClock(
    device=device,
    max_steps=64,
    budget_s=2.0,
    use_gpu_events=False,  # training fast rakhta hai
    query_gpu_clock=True,
)

history = train_gpt2_tmu_internal(
    plugin=plugin,
    heads=heads,
    adapter=adapter,
    train_loader=loader,
    device=device,
    clock=clock,
    epochs=2,
    lr=1e-3,
    use_gpu_events=False,
    budget_s=2.0,
    verbose=True,
)

save_system_internal(plugin, heads, adapter, "gpt2_tmu_internal.pt")