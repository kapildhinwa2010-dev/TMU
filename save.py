def save_system_internal(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    adapter: InternalClockAdapter,
    path: str,
) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    torch.save(
        {
            "tmu_state_dict": plugin.tmu.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
        },
        path,
    )


def load_system_internal(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    adapter: InternalClockAdapter,
    path: str,
    device: str = "cpu",
):
    ckpt = torch.load(path, map_location=device)

    plugin.tmu.load_state_dict(ckpt["tmu_state_dict"])
    heads.load_state_dict(ckpt["heads_state_dict"])
    adapter.load_state_dict(ckpt["adapter_state_dict"])

    return plugin, heads, adapter