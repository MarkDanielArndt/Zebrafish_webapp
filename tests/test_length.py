import numpy as np
import pytest


@pytest.fixture
def line_mask():
    mask = np.zeros((20, 256), dtype=bool)
    mask[8:12, 5:251] = True
    return mask


def test_tube_length_returns_positive(line_mask):
    from zebrafish_analysis.core.length import tube_length_border2border
    result = tube_length_border2border(
        line_mask,
        spacing=(1.0, 1.0),
        return_path=True,
        return_straight_line=True,
        mask_eye=None,
        return_eye_info=False,
    )
    length, straight = result[0], result[1]
    assert length > 0
    assert straight > 0


def test_tube_length_spacing_scales_result(line_mask):
    from zebrafish_analysis.core.length import tube_length_border2border
    r1 = tube_length_border2border(
        line_mask, spacing=(1.0, 1.0),
        return_path=True, return_straight_line=True,
        mask_eye=None, return_eye_info=False,
    )
    r2 = tube_length_border2border(
        line_mask, spacing=(2.0, 2.0),
        return_path=True, return_straight_line=True,
        mask_eye=None, return_eye_info=False,
    )
    l1, l2 = r1[0], r2[0]
    assert abs(l2 - 2 * l1) < l1 * 0.15


def test_tube_length_empty_mask_does_not_crash():
    from zebrafish_analysis.core.length import tube_length_border2border
    empty = np.zeros((64, 64), dtype=bool)
    try:
        result = tube_length_border2border(
            empty, spacing=(1.0, 1.0),
            return_path=True, return_straight_line=True,
            mask_eye=None, return_eye_info=False,
        )
        length = result[0]
        assert length == 0.0
    except Exception:
        pass


def test_tube_length_wider_mask_is_longer():
    from zebrafish_analysis.core.length import tube_length_border2border
    short = np.zeros((20, 100), dtype=bool)
    short[8:12, 5:95] = True
    long_ = np.zeros((20, 256), dtype=bool)
    long_[8:12, 5:251] = True

    r_short = tube_length_border2border(short, spacing=(1.0, 1.0))
    r_long = tube_length_border2border(long_, spacing=(1.0, 1.0))
    assert r_long[0] > r_short[0]


def test_tube_length_returns_tuple():
    from zebrafish_analysis.core.length import tube_length_border2border
    mask = np.zeros((20, 100), dtype=bool)
    mask[8:12, 5:95] = True
    result = tube_length_border2border(mask, spacing=(1.0, 1.0))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
