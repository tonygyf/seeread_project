"""CPU contracts for the unbenchmarked V5 composition; no pretrained downloads."""
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch
import pytest
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.v5.recognition import ReadingPipeline
from seeread.v5.seeing import SeeingDetector
from seeread.v5.seeing_data import targets_from_polygons
from seeread.v5.spotting import V5Spotter, polygon_crop


def coordinate_image(h=240, w=400):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    return torch.stack([x / (w - 1), y / (h - 1), torch.zeros_like(x).float()])


def test_horizontal_crop_is_direct_square_and_preserves_pixel_coordinates():
    image = coordinate_image()
    polygon = np.array([[40, 60], [360, 60], [360, 100], [40, 100]], np.float32)
    result = polygon_crop(image, polygon)
    assert result.reason is None and result.crop.shape == (1, 3, 384, 384)
    expected_x = torch.linspace(40 / 399, 360 / 399, 384)
    expected_y = torch.linspace(60 / 239, 100 / 239, 384)
    assert torch.allclose(result.crop[0, 0], expected_x[None].expand(384, -1), atol=2e-6)
    assert torch.allclose(result.crop[0, 1], expected_y[:, None].expand(-1, 384), atol=2e-6)
    assert np.allclose(result.heights, 40)


def test_rotated_vertical_and_odd_curve_crops_do_not_use_legacy_geometry():
    image = coordinate_image()
    rotated = cv2.boxPoints(((200., 120.), (220., 40.), 25.))
    vertical = np.array([[100, 20], [140, 20], [140, 220], [100, 220]], np.float32)
    top_x, bottom_x = np.linspace(40, 360, 9), np.linspace(40, 360, 8)
    curve = lambda x: 70 + 35 * np.sin(np.pi * (x - 40) / 320)
    odd = np.r_[np.c_[top_x, curve(top_x)], np.c_[bottom_x[::-1], curve(bottom_x[::-1]) + 40]].astype(np.float32)
    assert len(odd) == 17
    with patch('seeread.data.geometry.split_polygon', side_effect=AssertionError('legacy geometry used')), \
         patch('cv2.minAreaRect', side_effect=AssertionError('polygon replaced by rectangle')):
        for polygon in (rotated, vertical, odd):
            result = polygon_crop(image, polygon)
            assert result.reason is None, result.reason
            assert torch.isfinite(result.crop).all()
            assert result.crop.shape == (1, 3, 384, 384)
        curved = polygon_crop(image, odd)
        assert np.ptp(curved.centers[:, 1]) > 25
        upright = polygon_crop(image, vertical)
        assert np.all(np.diff(upright.centers[:, 1]) > 0)


class ImageOnlyDetector(SeeingDetector):
    def __init__(self, polygons=None, original_hw=(120, 400), feature_hw=(40, 80)):
        nn.Module.__init__(self)
        self.polygons, self.original_hw, self.feature_hw = polygons or [], original_hw, feature_hw
        self.calls = []
        self.provenance = {'implementation': 'V5 CPU contract test double'}

    def forward(self, image, **kwargs):
        self.calls.append((tuple(image.shape), kwargs))
        assert not kwargs, 'GT mask/SCM forwarded into inference'
        target = targets_from_polygons([{'poly': p} for p in self.polygons], self.original_hw, self.feature_hw)
        def logits(mask):
            pos = torch.tensor(mask).bool()
            return torch.stack([torch.where(pos, -10., 10.), torch.where(pos, 10., -10.)])[None]
        return dict(text=logits(target['text']), center=logits(target['center']),
                    height=torch.tensor(target['height'])[None, None],
                    angle=torch.tensor(target['angle'])[None, None])


class RecordingRecognizer(nn.Module):
    def __init__(self, mode='ok'):
        super().__init__()
        self.batches, self.mode = [], mode
        self.provenance = {'public_pretrained': 'CPU test double, not benchmark weights'}

    def predict(self, crops):
        self.batches.append(tuple(crops.shape))
        if self.mode == 'raise':
            raise RuntimeError('intentional test error')
        if self.mode == 'mismatch':
            return []
        return ['word'] * len(crops)


def spotter(polygons=None, rec_chunk=8, mode='ok'):
    detector, rec = ImageOnlyDetector(polygons), RecordingRecognizer(mode)
    pipeline = ReadingPipeline(nn.Identity(), rec)
    return V5Spotter(detector, pipeline, detector_hw=(160, 320), rec_chunk=rec_chunk), detector, rec


def test_non_square_detector_resize_maps_back_once_without_gt_mask():
    polygon = np.array([[40, 40], [360, 40], [360, 80], [40, 80]], np.float32)
    model, detector, rec = spotter([polygon])
    predictions = model.predict(coordinate_image(120, 400))
    assert len(predictions) == 1 and predictions[0]['text'] == 'word'
    assert predictions[0]['crop_error'] is None
    assert detector.calls == [((1, 3, 160, 320), {})]
    # Decoder returns original coordinates, despite detector stretch 400x120 -> 320x160.
    assert predictions[0]['poly'][:, 0].max() > 350
    assert predictions[0]['poly'][:, 1].max() < 100
    assert rec.batches == [(1, 3, 384, 384)]


def test_gt_and_prediction_paths_share_crops_and_chunking_preserves_all_boxes():
    image = coordinate_image(120, 400)
    model, detector, rec = spotter(rec_chunk=8)
    good = np.array([[40, 40], [360, 40], [360, 80], [40, 80]], np.float32)
    invalid = good.copy(); invalid[0, 0] = -1
    detections = [dict(poly=good, score=.9)] * 17 + [dict(poly=invalid, score=.8)]
    with patch('seeread.v5.spotting.decode', return_value=detections):
        predicted = model.predict(image)
    assert len(predicted) == 18
    assert [shape[0] for shape in rec.batches] == [8, 8, 1]
    assert all(shape[1:] == (3, 384, 384) for shape in rec.batches)
    assert predicted[-1]['text'] == '' and predicted[-1]['crop_error'] == 'polygon_outside_image'
    assert np.array_equal(predicted[-1]['poly'], invalid) and predicted[-1]['score'] == .8
    calls = len(detector.calls)
    gt = model.recognize_polygons(image, [good, invalid])
    assert len(detector.calls) == calls  # GT diagnostic never invokes detector.
    assert [p['text'] for p in gt] == ['word', '']
    assert gt[-1]['crop_error'] == predicted[-1]['crop_error']


def test_recognizer_failures_propagate_and_invalid_fold_is_reported():
    good = np.array([[40, 40], [360, 40], [360, 80], [40, 80]], np.float32)
    image = coordinate_image(120, 400)
    for mode, reason in (('raise', 'intentional test error'), ('mismatch', 'number of predictions')):
        model, _, _ = spotter(mode=mode)
        with patch('seeread.v5.spotting.decode', return_value=[dict(poly=good, score=.9)]):
            with pytest.raises((RuntimeError, ValueError), match=reason):
                model(image)
    # A folded C shape has two transverse intervals at some principal-axis
    # locations. It must report ambiguity instead of choosing a branch silently.
    folded = np.array([[20, 20], [180, 20], [180, 50], [60, 50],
                       [60, 80], [180, 80], [180, 110], [20, 110]], np.float32)
    result = polygon_crop(image, folded)
    assert result.reason == 'multiple_cross_sections', result.reason
    for error in (RuntimeError('sampling failed'), torch.OutOfMemoryError('out of memory')):
        with patch('seeread.v5.spotting.rectify_strip', side_effect=error):
            with pytest.raises(type(error), match=str(error)):
                polygon_crop(image, good)


def test_chunk_limit_and_legacy_component_rejection():
    detector, rec = ImageOnlyDetector(), ReadingPipeline(nn.Identity(), RecordingRecognizer())
    for chunk in (0, 9):
        try:
            V5Spotter(detector, rec, rec_chunk=chunk)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe recognition chunk accepted')
    try:
        V5Spotter(nn.Identity(), rec)
    except TypeError:
        pass
    else:
        raise AssertionError('Legacy/arbitrary detector accepted')


def test_placement_cases_share_weights_and_raw_crops_and_keep_failed_gt():
    from tools.v5_eval_spotting import evaluate_cases
    from seeread.v5.reading import metrics

    class PixelRecognizer(RecordingRecognizer):
        def __init__(self):
            super().__init__()
            self.means = []

        def predict(self, crops):
            self.means.append(float(crops.mean()))
            return super().predict(crops)

    class HalfEnhancer(nn.Module):
        def forward(self, image):
            return image * .5

    detector, recognizer = ImageOnlyDetector(), PixelRecognizer()
    image = coordinate_image(120, 400)
    polygon = np.array([[40, 40], [360, 40], [360, 80], [40, 80]], np.float32)
    changed = polygon.copy(); changed[:, 0] += 10
    invalid = polygon.copy(); invalid[0, 0] = -1
    gt = [dict(poly=p, text='word', ignore=False) for p in (polygon, invalid)]
    seen_means = []
    original_forward = detector.forward

    def forward(canvas):
        seen_means.append(float(canvas.mean()))
        return original_forward(canvas)

    with patch.object(detector, 'forward', side_effect=forward), \
         patch('tools.v5_eval_spotting.decode', side_effect=[
             [dict(poly=polygon, score=.9)], [dict(poly=changed, score=.7)]]):
        result = evaluate_cases(image, gt, detector, recognizer, HalfEnhancer(),
                                detector_hw=(160, 320), decoder={}, crop={})
    assert len(detector.calls) == 2  # B reuses A's detector output, not another forward.
    assert seen_means[1] == pytest.approx(.5 * seen_means[0])
    assert np.array_equal(result['A']['predictions'][0]['poly'], result['B']['predictions'][0]['poly'])
    assert result['A']['predictions'][0]['score'] == result['B']['predictions'][0]['score'] == .9
    assert np.array_equal(result['C']['predictions'][0]['poly'], changed)
    assert result['C']['predictions'][0]['score'] == .7
    # GT-A, GT-B, predicted-A, predicted-B, predicted-C. The only enhanced
    # recognizer inputs are B; C's different polygon still samples raw pixels.
    assert recognizer.means[1] == pytest.approx(.5 * recognizer.means[0])
    assert recognizer.means[3] == pytest.approx(.5 * recognizer.means[2])
    expected = polygon_crop(image, changed).crop.mean().item()
    assert recognizer.means[4] == pytest.approx(expected)
    assert len(result['A']['gt_crops']) == 2
    assert result['A']['gt_crops'][1]['prediction'] == ''
    assert result['A']['gt_crops'][1]['crop_error'] == 'polygon_outside_image'
    assert metrics(result['A']['gt_crops'])['cer'] == .5
    assert result['C']['gt_crops'] is result['A']['gt_crops']


def test_placement_preflight_and_pairing_reject_missing_weights_and_test_sources(tmp_path):
    import json
    from tools.v5_eval_spotting import CASES, check_inputs, paired_rows

    image = 'raw/TotalText/Images/Train/img1.jpg'
    row = dict(image=image, source_id=image, instances=[], meta=dict(split='dev'))
    bright, dark = tmp_path / 'bright.jsonl', tmp_path / 'dark.jsonl'
    for path in (bright, dark):
        path.write_text(json.dumps(row) + '\n')
    protocol = tmp_path / 'protocol.json'; protocol.write_text('{}')
    config = dict(cases=CASES, manifests=dict(bright=str(bright), dark=str(dark)),
                  protocol=str(protocol), weights=dict(detector=None, reading=None))
    with patch('torch.cuda.is_available', side_effect=AssertionError('Preflight accessed CUDA')), \
         patch('torch.load', side_effect=AssertionError('Preflight loaded weights')):
        assert check_inputs(config)['missing'] == {'detector': 'unset', 'reading': 'unset'}
        assert len(paired_rows(config['manifests'])['bright']) == 1
    row['source_id'] = 'raw/TotalText/Images/Test/img1.jpg'
    for path in (bright, dark):
        path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='Official test'):
        paired_rows(config['manifests'])


def test_placement_protocol_detects_changed_manifests_and_pixels(tmp_path):
    import json
    from tools.v5_eval_spotting import verified_rows
    from seeread.v5.reading import sha256

    bright_image, dark_image = 'bright.png', 'dark.png'
    for filename in (bright_image, dark_image):
        cv2.imwrite(str(tmp_path / filename), np.zeros((10, 10, 3), np.uint8))
    manifests = {key: str(tmp_path / (key + '.jsonl')) for key in ('bright', 'dark')}
    for key, image in (('bright', bright_image), ('dark', dark_image)):
        Path(manifests[key]).write_text(json.dumps(dict(image=image, source_id=bright_image, instances=[])) + '\n')
    protocol = dict(samples=1, outputs={key: dict(sha256=sha256(path)) for key, path in manifests.items()},
                    pairs=[dict(bright=bright_image, dark=dark_image,
                                bright_sha256=sha256(tmp_path / bright_image), dark_sha256=sha256(tmp_path / dark_image))])
    path = tmp_path / 'protocol.json'; path.write_text(json.dumps(protocol))
    config = dict(data_root=str(tmp_path), manifests=manifests, protocol=str(path))
    assert len(verified_rows(config)['dark']) == 1
    Path(manifests['dark']).write_text(Path(manifests['dark']).read_text() + '\n')
    with pytest.raises(ValueError, match='Manifest SHA256'):
        verified_rows(config)
    Path(manifests['dark']).write_text(Path(manifests['dark']).read_text().rstrip() + '\n')
    cv2.imwrite(str(tmp_path / dark_image), np.ones((10, 10, 3), np.uint8))
    with pytest.raises(ValueError, match='image bytes'):
        verified_rows(config)


if __name__ == '__main__':
    torch.set_num_threads(2)
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('PASS', name)


def test_distinct_instances_keep_identity_across_streamed_chunks_and_invalid_crop():
    model, _, rec = spotter(rec_chunk=2)
    image = coordinate_image(120, 400)
    polygons = [np.array([[x, 30], [x+10, 30], [x+10, 60], [x, 60]], np.float32)
                for x in (20, 50, 90, 140, 200)]
    invalid = polygons[0].copy(); invalid[0, 0] = -1
    polygons.insert(1, invalid)
    from seeread.v5 import spotting
    crop = spotting._polygon_crop
    calls, batch_calls = [], []
    def counted(*args, **kwargs):
        calls.append(1)
        return crop(*args, **kwargs)
    def identify(batch):
        batch_calls.append(len(calls))
        return [str(round(float(c[0].mean()) * 399)) for c in batch]
    with patch.object(spotting, '_polygon_crop', side_effect=counted), patch.object(rec, 'predict', side_effect=identify):
        result = model.recognize_polygons(image, polygons)
    assert [p['instance_id'] for p in result] == list(range(6))
    assert [p['text'] for p in result] == ['25', '', '55', '95', '145', '205']
    assert batch_calls == [3, 5, 6]  # Recognition starts before all crops are materialized.


def test_association_diagnostics_separates_merge_fragment_and_ignored_neighbors():
    from tools.v5_eval_spotting import association_diagnostics
    def box(x0, x1): return [[x0, 0], [x1, 0], [x1, 10], [x0, 10]]
    gt = [dict(poly=box(0, 10)), dict(poly=box(20, 30)), dict(poly=box(40, 50), ignore=True)]
    merged = association_diagnostics([dict(poly=box(0, 50))], gt)
    assert merged['merged_prediction_candidates'] == [0]
    assert {link['gt_index'] for link in merged['links']} == {0, 1}
    split = association_diagnostics([dict(poly=box(0, 5)), dict(poly=box(5, 10)), dict(poly=box(20, 30))], gt)
    assert split['fragmented_gt_candidates'] == [0]
    assert split['merged_prediction_candidates'] == []
