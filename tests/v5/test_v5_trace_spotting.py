import cv2
import numpy as np
import pytest

from tools.v5_trace_spotting import crop_atlas, page, check_manifest


def test_trace_atlas_preserves_pixels_order_and_chunks(tmp_path):
    raw=np.full((384,384,3),[11,22,33],np.uint8)
    enhanced=np.full_like(raw,[70,80,90])
    files=crop_atlas(tmp_path,'pred',[(raw,enhanced)]*17)
    assert files==[('pred_0000.png',0,16),('pred_0016.png',16,1)]
    pixels=cv2.cvtColor(cv2.imread(str(tmp_path/files[0][0])),cv2.COLOR_BGR2RGB)
    assert pixels.shape==(16*384,768,3)
    assert np.array_equal(pixels[:384,:384],raw)
    assert np.array_equal(pixels[-384:,384:],enhanced)
    with pytest.raises(ValueError,match='exact 384x384'):
        crop_atlas(tmp_path,'bad',[(raw[:10],enhanced)])


def test_trace_title_escapes_source_text(tmp_path):
    page(tmp_path/'index.html','<script>bad</script>','<p>content</p>')
    text=(tmp_path/'index.html').read_text()
    assert '<script>' not in text and '&lt;script&gt;' in text


def test_trace_rejects_changed_gt_with_same_image_path(tmp_path):
    import hashlib
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text('{"image":"same.png","instances":[{"text":"original"}]}\n')
    parent={'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    check_manifest(manifest,parent)
    manifest.write_text('{"image":"same.png","instances":[{"text":"changed"}]}\n')
    with pytest.raises(ValueError,match='GT manifest'):
        check_manifest(manifest,parent)
