"""Tests for the geometric volume priors."""

from lokma.geometry.volume_engine import VolumeEngineService


def test_prism_is_simple_extrusion():
    result = VolumeEngineService().estimate_volume(10.0, "prism", 2.0)
    assert result.volume_cm3 == 20.0
    assert result.shape == "prism"


def test_cylinder_matches_prism():
    assert VolumeEngineService().estimate_volume(10.0, "cylinder", 2.0).volume_cm3 == 20.0


def test_paraboloid_is_half():
    assert VolumeEngineService().estimate_volume(10.0, "paraboloid", 2.0).volume_cm3 == 10.0


def test_flat_uses_thin_layer():
    result = VolumeEngineService().estimate_volume(10.0, "flat", 99.0)
    assert result.volume_cm3 == 10.0 * VolumeEngineService.FLAT_LAYER_CM
    assert result.height_cm == VolumeEngineService.FLAT_LAYER_CM


def test_unknown_shape_falls_back_to_extrusion():
    assert VolumeEngineService().estimate_volume(5.0, "weird", 3.0).volume_cm3 == 15.0
