from types import SimpleNamespace

import torch

from tmu.looping_advanced import (
    ADV_TRAFFIC_DIM,
    AdvancedTrafficMonitor,
    AnchorScheduler,
    LayerImportanceScorer,
    build_advanced_traffic_feature_list,
    make_advanced_loop_plan,
)


def make_dummy_snapshot():
    base_features = [0.1] * 24
    traffic = {
        "active_norm": 0.3,
        "queue_delay_norm": 0.1,
        "throughput_norm": 0.4,
        "p95_norm": 0.2,
        "error_rate": 0.0,
        "deadline_pressure": 0.1,
        "latency_trend_norm": 0.15,
        "burstiness_norm": 0.25,
        "backlog_norm": 0.2,
        "sla_violation_norm": 0.0,
        "queue_growth_norm": 0.1,
        "cost_per_token_norm": 0.18,
        "deadline_slack_norm": 0.7,
        "throughput_trend_norm": 0.12,
    }
    system = {
        "gpu_util_norm": 0.5,
        "mem_pressure": 0.4,
        "temp_norm": 0.35,
        "power_norm": 0.3,
        "throttle_risk": 0.1,
        "freq_norm": 0.8,
    }

    return SimpleNamespace(
        feature_list=base_features,
        traffic=traffic,
        system=system,
    )


def test_advanced_traffic_feature_dim():
    snap = make_dummy_snapshot()
    feats = build_advanced_traffic_feature_list(
        snap,
        prompt_info={"ppl_norm": 0.4, "entropy_norm": 0.6},
    )
    assert len(feats) == ADV_TRAFFIC_DIM


def test_advanced_traffic_monitor_snapshot_keys():
    monitor = AdvancedTrafficMonitor(window_s=10.0, concurrency_limit=2)

    monitor.request_start(
        request_id="r1",
        prompt_tokens=4,
        priority=0.5,
        deadline_s=1.0,
        expected_tokens=8,
    )

    monitor.request_end(
        request_id="r1",
        generated_tokens=8,
        success=True,
    )

    snap = monitor.snapshot()

    for key in (
        "burstiness_norm",
        "backlog_norm",
        "latency_trend_norm",
        "cost_per_token_norm",
        "sla_violation_norm",
        "deadline_slack_norm",
        "queue_growth_norm",
        "throughput_trend_norm",
    ):
        assert key in snap
        assert 0.0 <= float(snap[key]) <= 1.0


def test_anchor_scheduler_range():
    sched = AnchorScheduler(mode="u", min_strength=0.1, max_strength=1.0)

    for step in range(10):
        s = sched.strength(
            step=step,
            total_steps=10,
            pressure=0.4,
            confidence=0.6,
            difficulty=0.8,
        )
        assert 0.0 <= s <= 1.0


def test_layer_importance_scores_sum_to_one():
    scorer = LayerImportanceScorer(num_layers=4)
    scorer.observe(0, 1.0)
    scorer.observe(1, 2.0)
    scorer.observe(2, 0.5)
    scorer.observe(3, 4.0)

    scores = scorer.scores()
    assert len(scores) == 4
    assert abs(sum(scores) - 1.0) < 1e-5


def test_make_advanced_loop_plan_easy_prompt():
    plan = make_advanced_loop_plan(
        num_layers=12,
        loop_frac=0.0,
        difficulty=0.1,
        budget=0.2,
        traffic_load=0.0,
        thermal_load=0.0,
    )

    assert plan["loop_scalar"] == 0
    assert all(l == 0 for l in plan["layer_loops"])
    assert plan["final_refine_passes"] == 0


def test_make_advanced_loop_plan_hard_prompt_with_importance():
    scores = [0.01] * 12
    scores[10] = 0.8
    scores[11] = 0.19

    plan = make_advanced_loop_plan(
        num_layers=12,
        loop_frac=0.9,
        difficulty=0.95,
        budget=0.9,
        traffic_load=0.0,
        thermal_load=0.0,
        layer_scores=scores,
    )

    assert plan["loop_scalar"] > 0
    assert max(plan["layer_loops"]) > 0
    assert plan["layer_loops"][10] >= plan["layer_loops"][0]