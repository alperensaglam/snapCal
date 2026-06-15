"""VolumeEngineService — turns a real-world footprint area into a volume.

Pure, stateless geometry. Each food has a shape prior (prism / cylinder /
paraboloid / flat); the volume is the footprint area extruded by an assumed
height according to that prior. Height comes from calibration/depth upstream;
here it is a parameter.
"""

from __future__ import annotations

from lokma.core.models import VolumeEstimate


class VolumeEngineService:
    """Computes volume (cm³) from a real-world area (cm²) and a shape prior."""

    #: Assumed thickness for a thin/flat layer (e.g. pizza, carpaccio), in cm.
    FLAT_LAYER_CM: float = 0.8

    def estimate_volume(self, real_area_cm2: float, shape: str, height_cm: float) -> VolumeEstimate:
        shape_key = (shape or "prism").lower()

        if shape_key in ("prism", "cylinder"):
            volume = real_area_cm2 * height_cm
            effective_height = height_cm
        elif shape_key == "paraboloid":
            volume = 0.5 * real_area_cm2 * height_cm
            effective_height = height_cm
        elif shape_key == "flat":
            volume = real_area_cm2 * self.FLAT_LAYER_CM
            effective_height = self.FLAT_LAYER_CM
        else:  # unknown shape -> simple extrusion fallback
            volume = real_area_cm2 * height_cm
            effective_height = height_cm

        return VolumeEstimate(
            volume_cm3=volume,
            shape=shape_key,
            height_cm=effective_height,
            real_area_cm2=real_area_cm2,
            method=f"geometric:{shape_key}",
        )
