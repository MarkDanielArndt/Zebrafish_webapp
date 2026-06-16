import sys

from zebrafish_analysis.slicer_extension.ZebrafishAnalysis.ZebrafishAnalysisLib.dependency_installer import _is_importable


def test_is_importable_finds_numpy():
    assert _is_importable("numpy") is True


def test_is_importable_misses_nonexistent():
    assert _is_importable("this_package_does_not_exist_xyz_000") is False


def test_is_importable_does_not_trigger_import():
    """find_spec must not cause the module to appear in sys.modules."""
    pkg = "skimage"
    was_present = pkg in sys.modules
    sys.modules.pop(pkg, None)  # ensure not already loaded
    _is_importable("scikit-image")
    after = pkg in sys.modules
    assert after is False, "find_spec must not import the package"
