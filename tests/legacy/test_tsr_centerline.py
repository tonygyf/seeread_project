"""Recognition-centerline regressions; no weights, network or GPU required."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.models.rectify import rectify_strip
from seeread.models.tsr import tsr_decode


def _maps(kind):
    y, x = np.mgrid[:160, :160]
    mid = 80 + 14 * np.sin(np.pi * (x - 20) / 120) if kind == "curved" else np.full_like(x, 80)
    body = (x >= 20) & (x <= 140)
    text = (body & (abs(y - mid) <= 10)).astype(np.float32)
    center = (body & (abs(y - mid) <= 3)).astype(np.float32)
    geo = np.zeros((3, 160, 160), np.float32)
    geo[0] = np.log1p(20)
    slope = 14 * np.pi / 120 * np.cos(np.pi * (x - 20) / 120) if kind == "curved" else np.zeros_like(x)
    geo[1] = 1 / np.sqrt(1 + slope ** 2)
    geo[2] = slope / np.sqrt(1 + slope ** 2)
    if kind == "vertical":
        text, center = text.T.copy(), center.T.copy()
        geo = geo.transpose(0, 2, 1).copy()
        geo[[1, 2]] = geo[[2, 1]]
    return text, center, geo


def _decode(kind, mode):
    return tsr_decode(*_maps(kind), stride=4, centerline_mode=mode)[0]


def test_sliced_horizontal_centerline_has_no_band_edge_zigzag():
    d = _decode("horizontal", "sliced")
    c = d["centers"] / 4
    assert np.allclose(c[:, 1], 80)
    assert np.all(np.diff(c[:, 0]) > 0)
    assert np.allclose(c[[0, -1], 0], [20, 140])
    assert np.allclose(d["heights"], 80)
    arc = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
    assert abs(arc - 120) < 1e-4
    old = _decode("horizontal", "legacy")["centers"] / 4
    assert np.linalg.norm(np.diff(old, axis=0), axis=1).sum() > 1.25 * arc


def test_sliced_vertical_centerline_follows_reading_direction():
    c = _decode("vertical", "sliced")["centers"] / 4
    assert np.allclose(c[:, 0], 80)
    assert np.all(np.diff(c[:, 1]) > 0)
    assert np.allclose(c[[0, -1], 1], [20, 140])


def test_sliced_curved_centerline_preserves_curve_instead_of_flattening():
    d = _decode("curved", "sliced")
    c = d["centers"] / 4
    expected = 80 + 14 * np.sin(np.pi * (c[:, 0] - 20) / 120)
    residual = np.abs(c[:, 1] - expected).mean()
    old = _decode("curved", "legacy")["centers"] / 4
    old_expected = 80 + 14 * np.sin(np.pi * (old[:, 0] - 20) / 120)
    assert residual < 0.6, residual
    assert np.abs(old[:, 1] - old_expected).mean() > 3 * residual
    assert np.ptp(c[:, 1]) > 12
    assert np.all(np.diff(c[:, 0]) > 0)


def test_recognition_mode_keeps_detection_polygons_and_scores_identical():
    for kind in ("horizontal", "vertical", "curved"):
        maps = _maps(kind)
        default = tsr_decode(*maps)
        legacy = tsr_decode(*maps, centerline_mode="legacy")
        sliced = tsr_decode(*maps, centerline_mode="sliced")
        assert len(default) == len(legacy) == len(sliced) == 1
        assert default[0]["score"] == legacy[0]["score"] == sliced[0]["score"]
        for key in ("poly", "centers", "heights"):
            assert np.array_equal(default[0][key], legacy[0][key])
        assert np.array_equal(default[0]["poly"], sliced[0]["poly"])


def test_perfect_mask_sliced_strip_recovers_sampling_coordinates():
    # Encode original pixel coordinates in the image, so crop error measures
    # geometry itself and does not depend on a particular font or recognizer.
    y, x = np.mgrid[:640, :640]
    image = torch.tensor(np.stack([x, y, np.zeros_like(x)]), dtype=torch.float32)[None] / 639
    centers = torch.tensor([[80., 320.], [560., 320.]])
    heights = torch.full((2,), 80.)
    target = rectify_strip(image, centers, heights)
    errors = {}
    for mode in ("legacy", "sliced"):
        d = _decode("horizontal", mode)
        crop = rectify_strip(image, torch.from_numpy(d["centers"]), torch.from_numpy(d["heights"]))
        errors[mode] = float((crop[:, :2] - target[:, :2]).abs().mean() * 639)
    assert errors["legacy"] > 5, errors
    assert errors["sliced"] < 1e-3, errors


def test_unknown_centerline_mode_is_rejected():
    try:
        tsr_decode(*_maps("horizontal"), centerline_mode="typo")
    except ValueError as exc:
        assert "centerline_mode" in str(exc)
    else:
        raise AssertionError("Unknown centerline mode silently accepted")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
