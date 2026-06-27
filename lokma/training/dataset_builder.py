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
from typing import Callable

from lokma.config import AppConfig
from lokma.knowledge.taxonomy import GOLDEN_LIST

logger = logging.getLogger(__name__)

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png")

#: A raw dataset label -> canonical taxonomy slug (or None to skip). Supplied by
#: the data-pipeline Vocabulary Alignment Bridge; see :mod:`lokma.data_pipeline`.
LabelResolver = Callable[[str], "str | None"]


def resolve_seg_dirs(
    raw_dir: Path | str, label_resolver: LabelResolver | None = None
) -> list[tuple[int, str, Path]]:
    """Plan the ``(class_id, slug, image_dir)`` tuples to ingest.

    Default (``label_resolver=None``) reproduces the ordinal ``GOLDEN_LIST``
    behavior exactly: one entry per slug, reading ``raw_dir/<slug>/``. With a
    resolver, every *subdirectory name* is treated as an external dataset label,
    remapped to a canonical slug (unmapped -> skipped, no contamination) and
    assigned its ``GOLDEN_LIST`` ``class_id`` so the trained class order stays
    stable across heterogeneous sources.
    """
    raw_dir = Path(raw_dir)
    if label_resolver is None:
        return [(class_id, slug, raw_dir / slug) for class_id, slug in enumerate(GOLDEN_LIST)]
    plan: list[tuple[int, str, Path]] = []
    if not raw_dir.exists():
        return plan
    for child in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        slug = label_resolver(child.name)
        if slug is None:
            continue  # unmapped label — skipped by the bridge's report
        if slug not in GOLDEN_LIST:
            logger.warning("dir %r resolved to %r, not in GOLDEN_LIST — skipping", child.name, slug)
            continue
        plan.append((GOLDEN_LIST.index(slug), slug, child))
    return plan


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


def build_v2_dataset(
    config: AppConfig | None = None,
    train_ratio: float = 0.8,
    conf: float = 0.25,
    label_resolver: LabelResolver | None = None,
) -> dict[str, int]:
    """Build the v2 dataset from ``config.raw_v2_dir`` -> ``config.yolo_v2_dataset_dir``.

    Returns a per-slug image count (0 where the user has not supplied images yet).
    ``label_resolver`` (from the data-pipeline Vocabulary Alignment Bridge) remaps
    external label directories to canonical slugs; the default (``None``) preserves
    the ordinal ``GOLDEN_LIST`` behavior byte-for-byte.
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

    for class_id, slug, class_dir in resolve_seg_dirs(raw, label_resolver):
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
