#!/usr/bin/env python3
"""Export the trained fill-density head to an Apple CoreML ``.mlpackage`` (Phase 8B).

Turns ``runs/train/fill_head/fill_head.pt`` (the Phase-7 ``FillRegressor``) into
``ios/Lokma/Resources/FillHead.mlpackage`` + a metadata sidecar that pins the
on-device **feature contract** the Swift side must reproduce exactly. The app feeds:

  • image   : a *masked* 96×96 RGB crop of the food (0–255; the /255 is baked in here)
  • scalars : [log1p(volume_cm3), coverage, mask_area_fraction]   (depth-derived)

and reads ``fill_density`` D, so mass = V·D on-device (Phase-8 Stage A formula).

Run in the ML env (torch + coremltools):

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_fill_head.py --verify

``--verify`` runs torch vs the converted CoreML model on a random input and asserts
they agree within 1e-3 — the export's de-risking step. (The on-device *crop*
pipeline must separately match the documented preprocessing; validate on device.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lokma.config import AppConfig  # noqa: E402

DEFAULT_WEIGHTS = AppConfig().project_root / "runs" / "train" / "fill_head" / "fill_head.pt"
DEFAULT_OUT = AppConfig().project_root / "ios" / "Lokma" / "Resources" / "FillHead.mlpackage"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the fill-density head -> CoreML .mlpackage")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--verify", action="store_true", help="assert CoreML matches torch within 1e-3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}\nTrain first: python scripts/train_fill_head.py")

    import numpy as np
    import torch
    import coremltools as ct
    from PIL import Image

    from lokma.training.fill_regressor import IMG_SIZE, _N_SCALARS, FillRegressor

    model = FillRegressor()
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
        outputs=[ct.TensorType(name="fill_density")],
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
        "model": "fill_density_head",
        "img_size": IMG_SIZE,
        "input_image": "masked food crop, RGB 0-255 (/255 baked into the model)",
        "input_scalars": ["log1p(volume_cm3)", "coverage", "mask_area_fraction"],
        "output": "fill_density",
        "preprocessing": "crop to mask bbox -> resize to img_size -> multiply by mask -> RGB",
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
        ml_out = float(np.asarray(ml["fill_density"]).reshape(-1)[0])
        delta = abs(ml_out - torch_out)
        print(f"verify: torch={torch_out:.6f}  coreml={ml_out:.6f}  |Δ|={delta:.2e}")
        if delta > 1e-3:
            raise SystemExit(f"CoreML/torch mismatch {delta:.2e} > 1e-3 — export is not faithful")
        print("verify: OK (CoreML matches torch within 1e-3)")

    print("\nNext: `cd ios && xcodegen generate` (if needed), then build & run in Xcode.")


if __name__ == "__main__":
    main()
