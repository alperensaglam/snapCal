"""Taxonomy-driven YOLO-seg dataset prep for the v2 (Altın Liste) retrain.

Reads ``GOLDEN_LIST`` from the taxonomy (the single source of truth for class
order/names), ingests a user-provided ``data/raw/lokma_v2/<slug>/`` image tree,
auto-labels with the pretrained seg model (class-agnostic pseudo-masks), splits
train/val, and writes ``data.yaml``. ``ultralytics`` is imported lazily so this
module — and ``write_data_yaml`` — stay testable without it.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

from lokma.config import AppConfig
from lokma.knowledge.taxonomy import GOLDEN_LIST

logger = logging.getLogger(__name__)

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png")


def write_data_yaml(out_dir: Path | str, classes: tuple[str, ...]) -> Path:
    """Write a YOLO ``data.yaml`` with ``names`` in the given class order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(classes)],
    ]
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def auto_label(img_path: Path, model, class_id: int, conf: float = 0.25) -> list[str]:
    """Pseudo-label an image: every seg mask the model finds -> this class_id."""
    results = model.predict(str(img_path), conf=conf, verbose=False)
    lines: list[str] = []
    if results and results[0].masks is not None:
        for polygon in results[0].masks.xyn:
            points = " ".join(f"{coord:.6f}" for pair in polygon for coord in pair)
            lines.append(f"{class_id} {points}")
    return lines


def build_v2_dataset(config: AppConfig | None = None, train_ratio: float = 0.8, conf: float = 0.25) -> dict[str, int]:
    """Build the v2 dataset from ``config.raw_v2_dir`` -> ``config.yolo_v2_dataset_dir``.

    Returns a per-slug image count (0 where the user has not supplied images yet).
    """
    config = config or AppConfig()
    from ultralytics import YOLO  # lazy — heavy

    out = Path(config.yolo_v2_dataset_dir)
    raw = Path(config.raw_v2_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    model = YOLO(str(config.pretrained_seg_model))
    counts: dict[str, int] = {}

    for class_id, slug in enumerate(GOLDEN_LIST):
        class_dir = raw / slug
        images: list[Path] = []
        if class_dir.exists():
            for pattern in IMAGE_GLOBS:
                images.extend(sorted(class_dir.glob(pattern)))
        if not images:
            logger.warning("no images for '%s' (%s)", slug, class_dir)
            counts[slug] = 0
            continue

        random.shuffle(images)
        cut = int(len(images) * train_ratio)
        for split, group in (("train", images[:cut]), ("val", images[cut:])):
            for img in group:
                new_name = f"{slug}_{img.name}"
                dest_img = out / "images" / split / new_name
                shutil.copy2(img, dest_img)
                lines = auto_label(dest_img, model, class_id, conf)
                (out / "labels" / split / f"{Path(new_name).stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
        counts[slug] = len(images)
        logger.info("%-18s %d images (class_id=%d)", slug, len(images), class_id)

    write_data_yaml(out, GOLDEN_LIST)
    return counts
