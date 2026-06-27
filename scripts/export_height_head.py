#!/usr/bin/env python3
"""Export the trained non-LiDAR volume head to an Apple CoreML ``.mlpackage`` (Phase 9).

Turns ``runs/train/height_head/height_head.pt`` (the ``VolumeRegressor``) into
``ios/Lokma/Resources/HeightHead.mlpackage`` + a metadata sidecar that pins the
on-device **feature contract** the Swift side must reproduce exactly. The app feeds:

  • image   : a *masked* 96×96 RGB crop of the food (0–255; the /255 is baked in here)
  • scalars : [mask_area_fraction]                     (RGB only — no depth needed)

and reads ``volume_log1p``; the Swift wrapper applies ``expm1`` to recover V (cm³),
so the LiDAR-less path computes ``mass = V_pred · D_pred`` (D from the fill head,
fed V_pred + a coverage=1.0 proxy).

Run in the ML env (torch + coremltools):

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_height_head.py --verify

``--verify`` runs torch vs the converted CoreML model on a random input and asserts
they agree within 5e-3 — the export's de-risking step. (The on-device *crop*
pipeline must separately match the documented preprocessing; validate on device.)

The tolerance is looser than the fill head's 1e-3 by design: the check compares the
raw ``volume_log1p`` output (before expm1), whose magnitude is ~11.5, so the model's
fp16 quantization (mlprogram default) shows up as a larger *absolute* delta for the
*same* relative error (~2e-4). 5e-3 is still below one fp16 ULP at that magnitude
(~0.008), so a genuinely broken conversion is still caught.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402

DEFAULT_WEIGHTS = AppConfig().project_root / "runs" / "train" / "height_head" / "height_head.pt"
DEFAULT_OUT = AppConfig().project_root / "ios" / "Lokma" / "Resources" / "HeightHead.mlpackage"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the volume head -> CoreML .mlpackage")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--verify", action="store_true", help="assert CoreML matches torch within 5e-3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}\nTrain first: python scripts/train_height_head.py")

    import numpy as np
    import torch
    import coremltools as ct
    from PIL import Image

    from lokma.training.height_regressor import IMG_SIZE, _N_SCALARS, VolumeRegressor

    model = VolumeRegressor()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    img = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    scalars = torch.rand(1, _N_SCALARS)
    traced = torch.jit.trace(model, (img, scalars))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(name="image", shape=(1, 3, IMG_SIZE, IMG_SIZE),
                         scale=1 / 255.0, color_layout=ct.colorlayout.RGB),
            ct.TensorType(name="scalars", shape=(1, _N_SCALARS)),
        ],
        outputs=[ct.TensorType(name="volume_log1p")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        import shutil
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    mlmodel.save(str(out))

    sidecar = out.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps({
        "model": "volume_head",
        "img_size": IMG_SIZE,
        "input_image": "masked food crop, RGB 0-255 (/255 baked into the model)",
        "input_scalars": ["mask_area_fraction"],
        "output": "volume_log1p",
        "output_postprocess": "expm1(volume_log1p) -> volume_cm3 (applied on-device)",
        "preprocessing": "crop to mask bbox -> resize to img_size -> multiply by mask -> RGB",
        "usage": "LiDAR-less fallback; mass = V_pred * D_pred (D from fill head, coverage=1.0 proxy)",
        "source_weights": str(args.weights),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"CoreML package -> {out}")
    print(f"Sidecar        -> {sidecar}")

    if args.verify:
        img255 = (img * 255.0).round().clamp(0, 255)
        pil = Image.fromarray(img255[0].permute(1, 2, 0).byte().numpy(), mode="RGB")
        with torch.no_grad():
            torch_out = float(model(img, scalars).item())
        ml = mlmodel.predict({"image": pil, "scalars": scalars.numpy().astype(np.float32)})
        ml_out = float(np.asarray(ml["volume_log1p"]).reshape(-1)[0])
        delta = abs(ml_out - torch_out)
        # Looser than the fill head's 1e-3: volume_log1p magnitude (~11.5) makes fp16
        # quantization show up as a larger absolute delta for the same relative error
        # (~2e-4). 5e-3 is still sub-ULP at that magnitude, so it catches real breakage.
        tol = 5e-3
        print(f"verify: torch={torch_out:.6f}  coreml={ml_out:.6f}  |Δ|={delta:.2e}")
        if delta > tol:
            raise SystemExit(f"CoreML/torch mismatch {delta:.2e} > {tol:.0e} — export is not faithful")
        print(f"verify: OK (CoreML matches torch within {tol:.0e})")

    print("\nNext: `cd ios && xcodegen generate` (if needed), then build & run in Xcode.")


if __name__ == "__main__":
    main()
