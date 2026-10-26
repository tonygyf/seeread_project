"""CPU polygon/geometry/loss contract tests for the V5 detector."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.v5.seeing_data import targets_from_polygons
from seeread.v5.seeing_losses import detection_loss
from seeread.v5.seeing_decode import decode, farthest_point_sampling


def _prediction(target):
    logits = lambda mask: np.stack([np.where(mask, -10., 10.), np.where(mask, 10., -10.)]).astype(np.float32)
    return dict(text=logits(target['text']), center=logits(target['center']),
                height=target['height'][None], angle=target['angle'][None])


def _iou(a, b):
    a, b = Polygon(a), Polygon(b)
    return a.intersection(b).area / a.union(b).area


def _roundtrip(poly, image_hw, feature_hw, threshold=.65):
    target = targets_from_polygons([{'poly': poly}], image_hw, feature_hw)
    predictions = decode(_prediction(target), image_hw=image_hw)
    assert len(predictions) == 1
    overlap = _iou(poly, predictions[0]['poly'])
    assert overlap > threshold, (overlap, predictions[0]['poly'])
    return target, predictions[0]


def test_perfect_maps_roundtrip_horizontal_rotated_and_odd_vertex_curve():
    rectangle = np.array([[40, 60], [360, 60], [360, 100], [40, 100]], np.float32)
    target, _ = _roundtrip(rectangle, (240, 400), (60, 100))
    assert np.allclose(target['height'][target['center'] > 0], 10)
    assert np.allclose(target['angle'][target['center'] > 0], 0)
    rotated = cv2.boxPoints(((200., 120.), (240., 44.), 30.))
    rt, _ = _roundtrip(rotated, (240, 400), (60, 100))
    assert abs(np.median(rt['angle'][rt['center'] > 0]) - np.pi / 6) < .01
    top_x, bottom_x = np.linspace(40, 360, 9), np.linspace(40, 360, 8)
    curve_y = lambda x: 80 + 30 * np.sin(np.pi * (x - 40) / 320)
    odd_curve = np.concatenate([np.stack([top_x, curve_y(top_x)], 1),
                                np.stack([bottom_x[::-1], curve_y(bottom_x[::-1]) + 40], 1)]).astype(np.float32)
    assert len(odd_curve) % 2 == 1
    ct, _ = _roundtrip(odd_curve, (240, 400), (60, 100))
    # This interior bounding-box pixel lies above the curved top boundary;
    # replacing the polygon with minAreaRect would wrongly label it positive.
    assert ct['text'][22, 50] == 0 and ct['text'][32, 50] == 1
    assert np.ptp(ct['angle'][ct['center'] > 0]) > .4


def test_non_square_inverse_scale_and_explicit_stride_agree():
    polygon = np.array([[60, 40], [540, 40], [540, 140], [60, 140]], np.float32)
    target, prediction = _roundtrip(polygon, (200, 600), (50, 100))
    by_stride = decode(_prediction(target), stride=(4, 6))[0]
    assert np.array_equal(prediction['poly'], by_stride['poly'])
    assert np.allclose(target['height'][target['center'] > 0], 25.)
    assert np.max(prediction['poly'][:, 0]) > 500


def test_adjacent_instances_stay_separate_and_fps_is_deterministic():
    polygons = [np.array([[20, 30], [180, 30], [180, 70], [20, 70]], np.float32),
                np.array([[220, 30], [380, 30], [380, 70], [220, 70]], np.float32)]
    target = targets_from_polygons([{'poly': p} for p in polygons], (120, 400), (30, 100))
    a, b = decode(_prediction(target), image_hw=(120, 400)), decode(_prediction(target), image_hw=(120, 400))
    assert len(a) == len(b) == 2
    assert all(np.array_equal(x['poly'], y['poly']) and x['score'] == y['score'] for x, y in zip(a, b))
    assert all(max(_iou(p, d['poly']) for d in a) > .65 for p in polygons)
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    assert len(set(farthest_point_sampling(pts, 4).tolist())) == 4


def test_optional_all_contours_retains_disconnected_reconstruction_fragments():
    mask = np.zeros((24, 40), np.int64)
    mask[12, 4:36] = 1
    target = dict(text=mask, center=mask, height=np.full(mask.shape, 6., np.float32),
                  angle=np.zeros(mask.shape, np.float32))
    kwargs = dict(image_hw=mask.shape, rectangle_width=2., sample_step=20., max_samples=3,
                  close_kernel=0)
    original = decode(_prediction(target), **kwargs)
    candidate = decode(_prediction(target), keep_all_contours=True, **kwargs)
    assert len(original) == 1 and len(candidate) == 3
    assert any(np.array_equal(original[0]['poly'], p['poly']) for p in candidate)
    assert len(decode(_prediction(target), **kwargs)) == 1


def test_vertical_and_polygon_vertex_start_winding_do_not_change_targets():
    polygon = np.array([[100, 20], [140, 20], [140, 220], [100, 220]], np.float32)
    target, _ = _roundtrip(polygon, (240, 400), (60, 100))
    assert np.allclose(target['height'][target['center'] > 0], 10)
    assert np.allclose(abs(target['angle'][target['center'] > 0]), np.pi / 2)
    for variant in (polygon[::-1], np.roll(polygon, 1, axis=0)):
        other = targets_from_polygons([{'poly': variant}], (240, 400), (60, 100))
        assert all(np.array_equal(target[key], other[key]) for key in ('text', 'center', 'valid'))
        assert np.allclose(target['height'], other['height'])
        assert np.allclose(target['angle'], other['angle'])


def _loss_prediction(size=(16, 16)):
    h, w = size
    return dict(text=torch.randn(1, 2, h, w, requires_grad=True),
                center=torch.randn(1, 2, h, w, requires_grad=True),
                height=torch.ones(1, 1, h, w, requires_grad=True),
                angle=torch.zeros(1, 1, h, w, requires_grad=True),
                scm_mask=torch.rand(1, 2, 8, 8, requires_grad=True),
                sem_main=torch.randn(1, 4, 8, 8, requires_grad=True),
                scm_sem=torch.randn(1, 4, 8, 8, requires_grad=True))


def test_ignore_masks_every_loss_and_auxiliary_semantic_target_is_detached():
    target = targets_from_polygons([{'poly': [[0, 0], [63, 0], [63, 63], [0, 63]], 'ignore': True}],
                                   (64, 64), (16, 16))
    pred = _loss_prediction()
    losses = detection_loss(pred, target)
    assert all(torch.isfinite(value) and value.item() == 0 for value in losses.values())
    sum(losses.values()).backward()
    assert pred['scm_sem'].grad is None
    assert all(value.grad is not None and value.grad.abs().sum() == 0
               for key, value in pred.items() if key != 'scm_sem')
    target = targets_from_polygons([], (64, 64), (16, 16))
    pred = _loss_prediction()
    detection_loss(pred, target)['scm_sem'].backward()
    assert pred['sem_main'].grad.abs().sum() > 0
    assert pred['scm_sem'].grad is None


def test_angle_is_pi_periodic_and_empty_center_has_zero_geometry_loss():
    target = targets_from_polygons([{'poly': [[4, 20], [60, 20], [60, 44], [4, 44]]}], (64, 64), (16, 16))
    pred = _loss_prediction()
    pred['height'] = torch.tensor(target['height'])[None, None].requires_grad_()
    pred['angle'] = (torch.tensor(target['angle'])[None, None] + np.pi).requires_grad_()
    losses = detection_loss(pred, target)
    assert losses['height'].item() == 0 and losses['angle'].item() < 1e-12
    empty = targets_from_polygons([], (64, 64), (16, 16))
    no_geometry = detection_loss(_loss_prediction(), empty)
    assert no_geometry['height'].item() == no_geometry['angle'].item() == 0
    assert decode(_prediction(empty), image_hw=(64, 64)) == []


def test_ignore_does_not_dilute_valid_loss_or_change_instance_order_semantics():
    polygon = [[4, 20], [60, 20], [60, 44], [4, 44]]
    ignore = [[16, 16], [48, 16], [48, 48], [16, 48]]
    first = targets_from_polygons([{'poly': polygon}, {'poly': ignore, 'ignore': True}], (64, 64), (16, 16))
    second = targets_from_polygons([{'poly': ignore, 'ignore': True}, {'poly': polygon}], (64, 64), (16, 16))
    assert all(np.array_equal(first[key], second[key]) for key in first)
    pred = _loss_prediction()
    before = detection_loss(pred, first)
    valid = torch.tensor(first['valid'])
    with torch.no_grad():
        for key in ('text', 'center', 'height', 'angle'):
            pred[key][..., ~valid] += 1000
        small_valid = F.interpolate(valid[None, None].float(), size=(8, 8), mode='nearest')[0, 0].bool()
        for key in ('scm_mask', 'sem_main', 'scm_sem'):
            pred[key][..., ~small_valid] += 1000
    after = detection_loss(pred, first)
    assert all(torch.allclose(before[key], after[key], atol=1e-7) for key in before)
    ce = F.cross_entropy(pred['text'], torch.tensor(first['text'])[None], reduction='none')[0]
    assert torch.allclose(after['text'], ce[valid].mean())


def test_single_ignored_pixel_cannot_disappear_at_scm_resolution():
    target = targets_from_polygons([], (64, 64), (16, 16))
    target['valid'][1, 1] = False  # nearest resize to 8x8 would miss this pixel
    pred = _loss_prediction()
    before = detection_loss(pred, target)
    with torch.no_grad():
        for key in ('scm_mask', 'sem_main'):
            pred[key][..., 0, 0] += 1000
    after = detection_loss(pred, target)
    assert torch.allclose(before['scm_mask'], after['scm_mask'])
    assert torch.allclose(before['scm_sem'], after['scm_sem'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('PASS', name)
