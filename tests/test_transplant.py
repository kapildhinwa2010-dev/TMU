from __future__ import annotations

import os
import tempfile

import torch

from tmu.core import TimeManagementUnit
from tmu.transplant import (
    GPT2WithTMU,
    TmuLayerConfig,
    evaluate_transplant,
    export_tmu_layer,
    format_transplant_report,
    load_tmu_layer,
    load_tmu_layer_metadata,
    mount_tmu_on_gpt2,
    write_transplant_report,
)


def _tiny_gpt2():
    """Network-free tiny GPT-2 (tests ke liye)."""

    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _samples(n: int = 2, seq_len: int = 8):
    out = []
    for i in range(n):
        input_ids = torch.randint(0, 64, (seq_len,))
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        out.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
                "t": torch.tensor(1_700_000_000.0 + i * 3600.0),
            }
        )
    return out


def test_export_and_load_round_trip():
    tmu = TimeManagementUnit(hidden_dim=32, fusion_layers=2)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "layer.pt")

        export_tmu_layer(tmu, path, metadata={"note": "unit-test"})

        assert os.path.exists(path)

        meta = load_tmu_layer_metadata(path)
        assert meta["note"] == "unit-test"
        assert meta["config"]["fusion_layers"] == 2

        loaded = load_tmu_layer(path)

        assert isinstance(loaded, TimeManagementUnit)
        assert loaded.fusion_layers == 2

        x = torch.randn(2, 32)
        te = loaded.encode_time(time_stamp=1_700_000_000.0, batch_size=2)

        assert torch.allclose(loaded.apply_to(x, te), x, atol=1e-6)


def test_config_round_trip():
    tmu = TimeManagementUnit(hidden_dim=32, fusion_layers=1, apply_mode="add")

    cfg = TmuLayerConfig.from_tmu(tmu)
    rebuilt = cfg.to_tmu()

    assert rebuilt.hidden_dim == 32
    assert rebuilt.apply_mode == "add"
    assert rebuilt.fusion_layers == 1


def test_mount_tmu_on_gpt2_keeps_base_model_unchanged():
    model = _tiny_gpt2()
    tmu = TimeManagementUnit(hidden_dim=32, fusion_layers=1)

    before = {k: v.clone() for k, v in model.state_dict().items()}

    plugin = mount_tmu_on_gpt2(model=model, tmu=tmu, mode="hooks")

    assert plugin.tmu.fusion_layers == 1

    for k, v in model.state_dict().items():
        assert torch.equal(before[k], v)


def test_gpt2_with_tmu_forward_and_generate():
    model = _tiny_gpt2()
    tmu = TimeManagementUnit(hidden_dim=32, fusion_layers=1)

    wrapped = GPT2WithTMU(base_model=model, tmu=tmu, freeze_base=True)

    input_ids = torch.randint(0, 64, (1, 6))
    attention_mask = torch.ones_like(input_ids)

    out = wrapped(
        input_ids=input_ids,
        attention_mask=attention_mask,
        time_stamp=1_700_000_000.0,
        use_cache=False,
        return_dict=True,
    )

    assert out.logits.shape == (1, 6, 64)

    gen = wrapped.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        time_stamp=1_700_000_000.0,
        max_new_tokens=3,
        do_sample=False,
    )

    assert gen.shape == (1, 9)


def test_evaluate_transplant_reports_time_sensitivity():
    model = _tiny_gpt2()
    tmu = TimeManagementUnit(hidden_dim=32, fusion_layers=1)

    # Fusion layer ko randomize karo taaki time signal actually logits tak jaaye.
    for p in tmu.fusion.parameters():
        if p.dim() > 0:
            p.data.normal_(0.0, 0.05)

    plugin = mount_tmu_on_gpt2(model=model, tmu=tmu, mode="hooks")

    metrics = evaluate_transplant(
        base_model=model,
        plugin=plugin,
        samples=_samples(2),
        device="cpu",
        time_shift_seconds=43200.0,
    )

    assert metrics["num_samples"] == 2
    assert metrics["baseline_loss"] > 0
    assert metrics["transplant_loss"] > 0
    assert metrics["tmu_trainable_params"] > 0


def test_write_and_format_report():
    metrics = {
        "num_samples": 4,
        "baseline_loss": 3.0,
        "baseline_perplexity": 20.09,
        "transplant_loss": 2.9,
        "transplant_perplexity": 18.17,
        "time_shift_seconds": 43200.0,
        "logit_shift_mad": 0.01,
        "time_sensitive": True,
        "tmu_trainable_params": 1234,
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out", "report.json")
        write_transplant_report(metrics, path)

        assert os.path.exists(path)

    md = format_transplant_report(metrics)
    assert "Baseline GPT-2" in md
    assert "time-sensitive" in md
