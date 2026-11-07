import torch

from tmu.clock import AdvancedInternalClock, advanced_snapshot_to_tensor
from tmu.utils import ADV_INTERNAL_DIM


def test_advanced_clock_snapshot_features():
    clock = AdvancedInternalClock(
        device="cpu",
        max_steps=8,
        budget_s=1.0,
        use_gpu_events=False,
        system_monitor=None,
        traffic_monitor=None,
    )

    snap = clock.tick(
        tokens=4,
        prompt_difficulty=0.5,
        confidence=0.7,
        per_layer_times=[1.0, 2.0, 1.5],
    )

    assert len(snap.feature_list) == ADV_INTERNAL_DIM
    assert snap.delta_s >= 0.0
    assert snap.pressure >= 0.0

    tensor = advanced_snapshot_to_tensor(snap, batch_size=2, device=torch.device("cpu"))
    assert tensor.shape == (2, ADV_INTERNAL_DIM)