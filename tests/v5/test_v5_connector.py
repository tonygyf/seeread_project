"""CPU geometry regressions for explicit Seeing-to-Reading crop candidates."""
import numpy as np
import pytest
from shapely import intersects_xy
from shapely.geometry import Polygon
import torch

from seeread.v5.spotting import polygon_crop, crop_exposure


def test_crop_exposure_uses_pixels_preserves_rgb_and_bounds_gain():
    crop = torch.full((1, 3, 16, 16), .1)
    actual, gain = crop_exposure(crop, 'p95')
    assert gain == pytest.approx(7.)
    assert torch.allclose(actual, torch.full_like(crop, .7))
    bright = torch.ones_like(crop)
    assert torch.equal(crop_exposure(bright, 'p95')[0], bright)
    black = torch.zeros_like(crop)
    actual, gain = crop_exposure(black, 'p95')
    assert gain == 8. and torch.equal(actual, black)
    color = crop * torch.tensor([1., .5, .25])[None, :, None, None]
    actual, gain = crop_exposure(color, 'p95')
    assert torch.allclose(actual, color * gain)
    assert crop_exposure(crop)[0] is crop


def test_exposure_keeps_instance_mapping_and_invalid_crop_in_actual_connector():
    from tests.v5.test_v5_spotting import spotter
    from seeread.v5.spotting import V5Spotter
    model, detector, _ = spotter()
    adapted = V5Spotter(detector, model.reading, exposure='p95')
    valid = [[20, 20], [100, 20], [100, 50], [20, 50]]
    invalid = [[-1, 0], [10, 0], [10, 10]]
    records = adapted.recognize_polygons(torch.full((3, 100, 160), .1), [valid, invalid, valid])
    assert [r['instance_id'] for r in records] == [0, 1, 2]
    assert records[0]['crop_exposure_gain'] == pytest.approx(7.)
    assert records[2]['crop_exposure_gain'] == pytest.approx(7.)
    assert records[1]['crop_exposure_gain'] is None
    assert records[1]['text'] == '' and records[1]['crop_error'] == 'polygon_outside_image'
    assert not detector.calls


def coordinate_image():
    y, x = torch.meshgrid(torch.arange(240), torch.arange(400), indexing='ij')
    return torch.stack([x / 399, y / 239, torch.zeros_like(x).float()])


@pytest.mark.parametrize('polygon', [
    [[40,20],[200,160],[360,20],[360,60],[200,200],[40,60]],
    [[40,60],[360,60],[360,61.2],[40,61.2]],
])
def test_boundary_candidate_samples_inside_bent_and_thin_word(polygon):
    image = coordinate_image()
    source = Polygon(polygon)
    legacy = polygon_crop(image, polygon)
    assert legacy.reason is None
    legacy_xy = legacy.crop[0, :2].numpy() * np.array([399, 239])[:, None, None]
    assert (~intersects_xy(source.buffer(1e-3), legacy_xy[0], legacy_xy[1])).mean() > .01
    result = polygon_crop(image, polygon, sampling='polygon_boundary')
    assert result.reason is None
    xy = result.crop[0, :2].numpy() * np.array([399, 239])[:, None, None]
    assert intersects_xy(source.buffer(1e-3), xy[0], xy[1]).all()


def test_original_crop_remains_default_and_boundary_preserves_straight_word():
    image = coordinate_image()
    polygon = [[40,60],[360,60],[360,100],[40,100]]
    original = polygon_crop(image, polygon)
    explicit = polygon_crop(image, polygon, sampling='local_normal')
    assert torch.equal(original.crop, explicit.crop)
    boundary = polygon_crop(image, polygon, sampling='polygon_boundary')
    assert boundary.reason is None
    assert torch.allclose(original.crop, boundary.crop, atol=1e-5)


def test_crop_comparison_uses_one_detection_pass_and_keeps_failed_gt():
    from tests.v5.test_v5_spotting import spotter, coordinate_image as image
    from tools.v5_eval_connector import compare_image
    poly = [[40, 40], [360, 40], [360, 80], [40, 80]]
    model, detector, _ = spotter([poly])
    gt = [dict(poly=poly, text='word'), dict(poly=[[-1, 0], [10, 0], [10, 10]], text='bad')]
    cases = compare_image(image(120, 400), gt, detector, model.reading,
                          dict(detector_hw=[160, 320], decoder={}, crop={}))
    assert len(detector.calls) == 1
    assert list(cases) == ['local_normal', 'polygon_boundary']
    a, b = cases.values()
    assert len(a['predictions']) == len(b['predictions']) == 1
    assert np.array_equal(a['predictions'][0]['poly'], b['predictions'][0]['poly'])
    assert a['predictions'][0]['score'] == b['predictions'][0]['score']
    for case in cases.values():
        assert len(case['gt_crops']) == 2
        assert case['gt_crops'][1]['prediction'] == ''
        assert case['gt_crops'][1]['crop_error'] == 'polygon_outside_image'


def test_white_region_resampling_preserves_reading_rgb_range():
    # Real detector outline from img524: bilinear roundoff produced 1+float32 eps.
    polygon = [[243.000015,0],[243.000015,48.4],[259.2,60.5],[259.2,145.2],
               [275.4,157.3],[275.4,193.6],[421.2,193.6],[437.4,181.5],
               [502.2,181.5],[518.4,169.4],[567,169.4],[567,0]]
    result = polygon_crop(torch.ones(3,200,600), polygon)
    assert result.reason is None
    from seeread.v5.enhancers import _validate_input
    _validate_input(result.crop)


def test_supported_boundary_strategy_records_geometric_fallback_only():
    polygon = [[243.000015,0],[243.000015,48.4],[259.2,60.5],[259.2,145.2],
               [275.4,157.3],[275.4,193.6],[421.2,193.6],[437.4,181.5],
               [502.2,181.5],[518.4,169.4],[567,169.4],[567,0]]
    image = torch.ones(3,200,600)
    assert polygon_crop(image, polygon, sampling='polygon_boundary').reason == 'multiple_cross_sections'
    result = polygon_crop(image, polygon, sampling='boundary_when_supported')
    assert result.reason is None and result.sampling == 'local_normal'
    assert result.fallback_reason == 'multiple_cross_sections'
    assert torch.equal(result.crop, polygon_crop(image, polygon).crop)
    straight = [[20,20],[200,20],[200,50],[20,50]]
    good = polygon_crop(image, straight, sampling='boundary_when_supported')
    assert good.sampling == 'polygon_boundary' and good.fallback_reason is None
    assert torch.equal(good.crop, polygon_crop(image, straight, sampling='polygon_boundary').crop)
    bad = polygon_crop(image, [[-1,0],[10,0],[10,10]], sampling='boundary_when_supported')
    assert bad.reason == 'polygon_outside_image' and bad.fallback_reason is None
