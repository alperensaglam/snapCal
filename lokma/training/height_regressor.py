"""Phase 9: non-LiDAR volume regressor (the RGB sibling of the fill-density head).

On a LiDAR-less device the depth-integration tier can't run, so `AutoStrategy`
falls back to a coarse area×height static prior. This head learns the missing
piece directly from appearance: it predicts the food's **volume V (cm³)** from
the masked RGB crop alone, so the app can still compute ``mass = V_pred · D_pred``
(D from the fill-density head, fed V_pred + a coverage proxy).

It deliberately mirrors :mod:`lokma.training.fill_regressor` — same 96×96 masked
crop, same tiny CNN+MLP, same training loop and manifest — with two differences
the no-depth scenario forces:

  * inputs drop the two depth-derived scalars (``log1p(volume)``, ``coverage``);
    only ``mask_area_fraction`` survives (it needs no depth).
  * the target is ``log1p(calculated_volume_cm3)`` — log-space keeps the
    regression well-conditioned across the ~10–1000 cm³ range; inference applies
    ``expm1`` to recover cm³.

torch is imported lazily (only this module needs it) so the rest of
``lokma.training`` and the ingest path stay torch-free. Run via
``scripts/train_height_head.py`` in ``lokma_env``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from lokma.config import AppConfig
from lokma.training.fill_regressor import IMG_SIZE, _crop_to_mask

_N_SCALARS = 1  # [mask_area_fraction] — the only input available without depth


class VolumeDataset(Dataset):
    """Reads dataset_manifest.json → (masked RGB crop, scalars, log1p(volume) target)."""

    def __init__(self, manifest_path: Path, img_size: int = IMG_SIZE, limit: int | None = None):
        rows = json.loads(Path(manifest_path).read_text())
        self.rows = rows[:limit] if limit else rows
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(r["mask_path"]).convert("L"), dtype=np.float32) / 255.0
        if mask.shape != rgb.shape[:2]:
            mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0])), dtype=np.float32) / 255.0
        masked = rgb * mask[..., None]                       # focus on the food
        crop = _crop_to_mask(masked, mask, self.img_size)
        img = torch.from_numpy(crop.transpose(2, 0, 1)).float()

        area_frac = float(mask.mean())
        scalars = torch.tensor([area_frac], dtype=torch.float32)
        target = torch.tensor([math.log1p(float(r["calculated_volume_cm3"]))], dtype=torch.float32)
        return img, scalars, target


class VolumeRegressor(nn.Module):
    """Tiny CNN over the food crop + an MLP fusing mask_area_fraction → 1 scalar.

    Identical topology to :class:`FillRegressor` (so the CoreML export + on-device
    crop pipeline are shared), only the scalar fan-in differs.
    """

    def __init__(self, n_scalars: int = _N_SCALARS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + n_scalars, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, img: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(img).flatten(1)
        return self.head(torch.cat([feat, scalars], dim=1)).squeeze(1)


def run_training(cfg: AppConfig | None = None, *, epochs: int | None = None,
                 batch: int | None = None, lr: float | None = None,
                 limit: int | None = None) -> Path:
    """Train the volume head on the manifest; save best weights. Returns the path."""
    cfg = cfg or AppConfig()
    epochs = epochs or cfg.volume_head_epochs
    batch = batch or cfg.volume_head_batch
    lr = lr or cfg.volume_head_lr
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    ds = VolumeDataset(cfg.n5k_processed_dir / "dataset_manifest.json", limit=limit)
    n_val = max(1, int(len(ds) * 0.2))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch)

    model = VolumeRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss()
    out_dir = Path("runs/train/height_head")
    out_dir.mkdir(parents=True, exist_ok=True)
    best = math.inf
    best_path = out_dir / "height_head.pt"

    for ep in range(epochs):
        model.train()
        for img, sc, y in train_dl:
            img, sc, y = img.to(device), sc.to(device), y.to(device).squeeze(1)
            opt.zero_grad()
            loss = loss_fn(model(img, sc), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = [loss_fn(model(img.to(device), sc.to(device)), y.to(device).squeeze(1)).item()
                     for img, sc, y in val_dl]
        vmean = float(np.mean(vloss)) if vloss else math.inf
        print(f"epoch {ep + 1}/{epochs}  val_huber={vmean:.5f}")
        if vmean < best:
            best = vmean
            torch.save(model.state_dict(), best_path)

    (out_dir / "height_head_meta.json").write_text(json.dumps(
        {"n_scalars": _N_SCALARS, "best_val_huber": best, "target": "log1p_volume_cm3"}, indent=2))
    print(f"Saved best (val_huber={best:.5f}) -> {best_path}")
    return best_path
