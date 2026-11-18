"""Image-only adapter for the pinned upstream DarkText VGG detector.

Run in the upstream environment with its source root on PYTHONPATH.  This file
does not import or inspect ground-truth annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_detector(checkpoint, device):
    from src.models import textnet
    from src.detection.mask_detector import TextDetector

    # The full checkpoint supplies the backbone; do not download a second one.
    original_vgg = textnet.VggNet
    textnet.VggNet = lambda name, pretrain=True: original_vgg(name=name, pretrain=False)
    try:
        model = textnet.TextNet(backbone='vgg', is_training=False)
    finally:
        textnet.VggNet = original_vgg
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = {key: value for key, value in saved['model'].items() if 'gcn_model' not in key}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return TextDetector(model), saved.get('epoch')


def infer(detector, image, device, size=640):
    height, width = image.shape[:2]
    resized = cv2.resize(image, (size, size)).astype(np.float32) / 255.0
    resized = (resized - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        contours, _ = detector.detect(tensor, None)
    scale = np.array([width / size, height / size], np.float32)
    # Upstream returns a pair of contours per instance and no confidence score.
    return [{'poly': (np.asarray(item[0], np.float32) * scale).tolist(), 'score': None}
            for item in contours]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--synthetic-smoke', action='store_true')
    args = parser.parse_args()
    if bool(args.image) == bool(args.synthetic_smoke):
        parser.error('Choose exactly one of --image or --synthetic-smoke')
    detector, epoch = load_detector(args.checkpoint, torch.device(args.device))
    if args.synthetic_smoke:
        image = np.zeros((64, 64, 3), np.uint8)
        size = 64
    else:
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None:
            raise FileNotFoundError(args.image)
        image, size = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 640
    result = dict(source='DarkText upstream VGG baseline; no PDE', checkpoint=str(args.checkpoint),
                  checkpoint_sha256=sha256(args.checkpoint), epoch=epoch,
                  image=str(args.image) if args.image else 'synthetic_black_64',
                  predictions=infer(detector, image, torch.device(args.device), size=size))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
