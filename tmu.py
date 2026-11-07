# train_tmu.py
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tmu import (
    TMUPlugin,
    TimeManagementUnit,
    freeze_base_and_enable_tmu,
    cyclic_time_features,
)


TIME_FEATURE_DIM = getattr(TimeManagementUnit, "FEATURE_DIM", 12)


def _first_tensor(obj: Any) -> torch.Tensor:
    """
    विभिन्न output formats से पहला torch.Tensor निकालता है।
    """
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (tuple, list)):
        for item in obj:
            if torch.is_tensor(item):
                return item

    if isinstance(obj, dict):
        for value in obj.values():
            if torch.is_tensor(value):
                return value

    if hasattr(obj, "last_hidden_state") and torch.is_tensor(obj.last_hidden_state):
        return obj.last_hidden_state

    raise TypeError("Model output से torch.Tensor नहीं मिल पाया।")


class TmuCapabilityHeads(nn.Module):
    """
    TMU के साथ training heads:
    1. rhythm_head  -> future/cyclic time pattern predict करना
    2. pace_head    -> reasoning speed / compute budget predict करना
    """

    def __init__(self, hidden_dim: int, budget_dim: int = 1):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.budget_dim = int(budget_dim)

        # Budget embedding: initially zero, ताकि शुरू में model behavior न बदले
        self.budget_proj = nn.Linear(self.budget_dim, self.hidden_dim)
        nn.init.zeros_(self.budget_proj.weight)
        nn.init.zeros_(self.budget_proj.bias)

        # World rhythm prediction head
        self.rhythm_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, TIME_FEATURE_DIM),
        )

        # Thinking speed / pace prediction head
        pace_hidden = max(8, self.hidden_dim // 2)
        self.pace_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, pace_hidden),
            nn.GELU(),
            nn.Linear(pace_hidden, 1),
        )

        # Pace initially neutral ~0.5
        nn.init.zeros_(self.pace_head[-1].weight)
        nn.init.constant_(self.pace_head[-1].bias, 0.0)

    def forward(
        self,
        time_embed: torch.Tensor,
        budget: torch.Tensor | float,
    ):
        """
        time_embed: (B, hidden_dim)
        budget:     (B,) या scalar, normalized 0..1
        returns:
            rhythm_pred: (B, TIME_FEATURE_DIM)
            pace:        (B,) in [0, 1]
            pace_logit:  (B,)
        """
        if not torch.is_tensor(budget):
            budget = torch.tensor(float(budget), device=time_embed.device)

        budget = budget.to(device=time_embed.device, dtype=time_embed.dtype)

        if budget.dim() == 0:
            budget = budget.unsqueeze(0)

        if budget.dim() == 1:
            budget = budget.unsqueeze(-1)

        if budget.shape[0] != time_embed.shape[0]:
            if budget.shape[0] == 1:
                budget = budget.expand(time_embed.shape[0], -1)
            elif time_embed.shape[0] == 1:
                time_embed = time_embed.expand(budget.shape[0], -1)
            else:
                raise ValueError(
                    f"Budget batch {budget.shape[0]} != time_embed batch {time_embed.shape[0]}"
                )

        conditioned = time_embed + self.budget_proj(budget)

        rhythm_pred = self.rhythm_head(conditioned)
        pace_logit = self.pace_head(conditioned).squeeze(-1)
        pace = torch.sigmoid(pace_logit)

        return rhythm_pred, pace, pace_logit


class SyntheticTimeDataset(Dataset):
    """
    Demo dataset:
    - x: input token ids
    - t: timestamp
    - budget: allowed normalized time 0..1
    - y_task: time-dependent classification label
    - y_rhythm: future cyclic time features
    - y_pace: desired reasoning speed, capped by budget
    """

    def __init__(
        self,
        num_samples: int = 2048,
        seq_len: int = 16,
        vocab_size: int = 128,
        base_time: float = 1_700_000_000.0,
        seed: int = 0,
    ):
        super().__init__()

        g = torch.Generator()
        g.manual_seed(int(seed))

        self.num_samples = int(num_samples)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)

        # Random input tokens
        self.x = torch.randint(
            0,
            self.vocab_size,
            (self.num_samples, self.seq_len),
            generator=g,
        )

        # Increasing timestamps with random gaps
        deltas = torch.rand(self.num_samples, generator=g) * 3600.0 + 10.0
        self.t = base_time + torch.cumsum(deltas, dim=0)

        # Normalized time budget: 0.1 to 1.0
        self.budget = torch.rand(self.num_samples, generator=g) * 0.9 + 0.1

        # Day phase: 0..1
        day_phase = torch.remainder(self.t, 86400.0) / 86400.0

        # Task label depends on time + input parity
        time_label = (day_phase * 4.0).long() % 4
        input_label = self.x.sum(dim=1) % 2
        self.y_task = (time_label + input_label) % 4

        # Rhythm target: cyclic features 10 minutes future में
        self.y_rhythm = cyclic_time_features(self.t + 600.0)

        # Desired pace depends on input difficulty and time-of-day activity
        difficulty = (
            self.x.float().std(dim=1, unbiased=False) / float(max(1, self.vocab_size - 1))
        ).clamp(0.0, 1.0)

        time_need = 0.25 + 0.45 * torch.abs(torch.sin(2.0 * math.pi * day_phase))

        desired_pace = (0.5 * difficulty + 0.5 * time_need).clamp(0.0, 1.0)

        # Model कभी budget से ज्यादा pace नहीं लेना चाहिए
        self.y_pace = torch.minimum(desired_pace, self.budget)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "t": self.t[idx],
            "budget": self.budget[idx],
            "y_task": self.y_task[idx],
            "y_rhythm": self.y_rhythm[idx],
            "y_pace": self.y_pace[idx],
        }


def train_tmu_capabilities(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    train_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 3,
    lr: float = 2e-3,
    w_task: float = 1.0,
    w_rhythm: float = 1.0,
    w_pace: float = 0.7,
    budget_penalty: float = 2.0,
    grad_clip: float = 1.0,
    verbose: bool = True,
):
    """
    सिर्फ TMU + capability heads train करते हैं।
    Base model freeze रहता है, इसलिए 10-minute style toy training possible है।
    """
    plugin.to(device)
    heads.to(device)

    # Base model freeze, TMU trainable
    tmu_params = freeze_base_and_enable_tmu(plugin)
    params = list(tmu_params) + list(heads.parameters())

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)

    history = []

    for epoch in range(int(epochs)):
        plugin.train()
        heads.train()

        total_loss = 0.0
        total_task = 0.0
        total_rhythm = 0.0
        total_pace = 0.0
        total_over = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            t = batch["t"].to(device, non_blocking=True)
            budget = batch["budget"].to(device, non_blocking=True)

            y_task = batch["y_task"].to(device, non_blocking=True)
            y_rhythm = batch["y_rhythm"].to(device, non_blocking=True)
            y_pace = batch["y_pace"].to(device, non_blocking=True)

            # ---------- 1. Main task loss ----------
            raw_out = plugin(x, time_stamp=t)
            logits = _first_tensor(raw_out)

            if logits.dim() == 3:
                # (B, L, V) -> sequence classification demo
                cls_logits = logits.mean(dim=1)
            elif logits.dim() == 2:
                cls_logits = logits
            else:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

            loss_task = F.cross_entropy(cls_logits, y_task)

            # ---------- 2. TMU time embedding ----------
            time_embed = plugin.tmu.encode_time(
                time_stamp=t,
                batch_size=int(x.shape[0]),
                device=device,
            )

            # ---------- 3. Rhythm + pace heads ----------
            rhythm_pred, pace_pred, _ = heads(time_embed, budget)

            loss_rhythm = F.mse_loss(rhythm_pred, y_rhythm)

            pace_safe = pace_pred.clamp(1e-6, 1.0 - 1e-6)
            loss_pace_bce = F.binary_cross_entropy(pace_safe, y_pace)

            # Budget exceed penalty: pace > budget होने पर सजा
            over_budget = F.relu(pace_pred - budget).mean()
            loss_pace = loss_pace_bce + float(budget_penalty) * over_budget

            loss = (
                float(w_task) * loss_task
                + float(w_rhythm) * loss_rhythm
                + float(w_pace) * loss_pace
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))

            optimizer.step()

            total_loss += float(loss.item())
            total_task += float(loss_task.item())
            total_rhythm += float(loss_rhythm.item())
            total_pace += float(loss_pace.item())
            total_over += float(over_budget.item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_task = total_task / max(1, num_batches)
        avg_rhythm = total_rhythm / max(1, num_batches)
        avg_pace = total_pace / max(1, num_batches)
        avg_over = total_over / max(1, num_batches)

        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "loss_task": avg_task,
                "loss_rhythm": avg_rhythm,
                "loss_pace": avg_pace,
                "over_budget": avg_over,
            }
        )

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"task={avg_task:.4f} | "
                f"rhythm={avg_rhythm:.4f} | "
                f"pace={avg_pace:.4f} | "
                f"over={avg_over:.4f}"
            )

    return history


@torch.no_grad()
def tmu_time_recommendation(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    sample_tensor: torch.Tensor,
    time_stamp: torch.Tensor | float,
    budget: torch.Tensor | float,
    min_steps: int = 1,
    max_steps: int = 8,
):
    """
    TMU + heads से recommend करता है कि कितने reasoning steps / compute use करने चाहिए।

    Actual generation में आप इसे ऐसे use कर सकते हो:
        max_new_tokens = recommended_steps * tokens_per_step
    या dynamic layer skipping में:
        active_layers = recommended_steps
    """
    plugin.eval()
    heads.eval()

    device = sample_tensor.device
    batch_size = int(sample_tensor.shape[0])

    time_embed = plugin.tmu.encode_time(
        time_stamp=time_stamp,
        batch_size=batch_size,
        device=device,
    )

    _, pace, _ = heads(time_embed, budget)

    min_steps = float(min_steps)
    max_steps = float(max_steps)

    steps_float = min_steps + pace * (max_steps - min_steps)
    steps = torch.round(steps_float).clamp(min=min_steps, max=max_steps).long()

    return pace.cpu(), steps.cpu()


def save_tmu_capabilities(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    path: str,
):
    torch.save(
        {
            "tmu_state_dict": plugin.tmu.state_dict(),
            "heads_state_dict": heads.state_dict(),
        },
        path,
    )


def load_tmu_capabilities(
    plugin: TMUPlugin,
    heads: TmuCapabilityHeads,
    path: str,
    device: str = "cpu",
):
    ckpt = torch.load(path, map_location=device)
    plugin.tmu.load_state_dict(ckpt["tmu_state_dict"])
    heads.load_state_dict(ckpt["heads_state_dict"])
    return plugin, heads


# ---------------- Demo tiny model ----------------
class TinyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 64, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    d_model = 64

    # Demo base model। अपने FluidLayer/model को यहां replace करें।
    base_model = TinyLanguageModel(vocab_size=128, d_model=d_model, num_layers=2)

    plugin = TMUPlugin(
        base_model=base_model,
        hidden_dim=d_model,
        mode="hooks",
        target_modules=["block", "layer", "fluid"],
        ema_momentum=0.1,
        dropout=0.0,
        use_ema=True,
        apply_mode="scale",
    )

    heads = TmuCapabilityHeads(hidden_dim=d_model)

    dataset = SyntheticTimeDataset(
        num_samples=1024,
        seq_len=12,
        vocab_size=128,
        base_time=1_700_000_000.0,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=False,
    )

    print("\n=== Training TMU capabilities ===")
    history = train_tmu_capabilities(
        plugin=plugin,
        heads=heads,
        train_loader=loader,
        device=device,
        epochs=3,
        lr=2e-3,
        w_task=1.0,
        w_rhythm=1.0,
        w_pace=0.7,
        budget_penalty=2.0,
        verbose=True,
    )

    save_tmu_capabilities(plugin, heads, "tmu_capabilities_demo.pt")
    print("\nSaved: tmu_capabilities_demo.pt")

    print("\n=== Time/pace recommendation demo ===")
    sample_x = dataset.x[:4].to(device)
    sample_t = dataset.t[:4].to(device)
    sample_budget = dataset.budget[:4].to(device)

    pace, steps = tmu_time_recommendation(
        plugin=plugin,
        heads=heads,
        sample_tensor=sample_x,
        time_stamp=sample_t,
        budget=sample_budget,
        min_steps=1,
        max_steps=8,
    )

    print("Budgets:", sample_budget.cpu().tolist())
    print("Predicted pace:", pace.tolist())
    print("Recommended reasoning steps:", steps.tolist())