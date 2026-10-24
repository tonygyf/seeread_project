"""TextNet / TextSnake-style detector ported from DarkText-private (Reading-in-the-dark 24' line).

INVERENCE-ONLY, self-contained: VGG16 backbone + FPN + 1x1 8-ch predict head, plus the mask-based
TextSnake decoder (detect_contours1). Removed vs upstream: GCN / Graph_RPN / kornia morphology
(all unused by the mask detector's `forward_test` path), the `erode*/diale*` train-only morphology
layers (skipped via strict=False on load), and the matplotlib heatmap visualisation.

Param names intentionally mirror upstream so the checkpoint loads via torch.load(...)['model']:
  fpn.backbone.stage1..5.*, fpn.deconv5.*, fpn.merge4/3/2/1.{conv1x1,conv3x3,deconv}.*, predict.0.*
Output of forward_test = raw 8-ch logits (stride 1, input resolution):
  [0:2]=TR softmax, [2:4]=TCL softmax, [4]=sin, [5]=cos, [6:8]=top/bot distance (radii).
"""
from __future__ import annotations
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VggNet(nn.Module):
    """vgg16 features split into the 5 stages that the upstream FPN consumes (no BN variant)."""

    def __init__(self, name="vgg16", pretrain=False):
        super().__init__()
        assert name == "vgg16", "only vgg16 (no BN) matches the textgraph_vgg checkpoint"
        base_net = models.vgg16(weights=None)
        self.stage1 = nn.Sequential(*[base_net.features[i] for i in range(0, 5)])
        self.stage2 = nn.Sequential(*[base_net.features[i] for i in range(5, 10)])
        self.stage3 = nn.Sequential(*[base_net.features[i] for i in range(10, 17)])
        self.stage4 = nn.Sequential(*[base_net.features[i] for i in range(17, 24)])
        self.stage5 = nn.Sequential(*[base_net.features[i] for i in range(24, 31)])

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return C1, C2, C3, C4, C5


class UpBlok(nn.Module):
    """FPN upsample+merge block (upstream verbatim: conv1x1 -> relu -> conv3x3 -> relu -> deconv 2x)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = F.relu(self.conv1x1(x))
        x = F.relu(self.conv3x3(x))
        return self.deconv(x)


class FPN(nn.Module):
    def __init__(self, backbone="vgg", is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        assert backbone in ("vgg", "vgg_bn")
        self.backbone = VggNet(name="vgg16" if backbone == "vgg" else "vgg16_bn", pretrain=False)
        self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.merge4 = UpBlok(512 + 256, 128)
        self.merge3 = UpBlok(256 + 128, 64)
        self.merge2 = UpBlok(128 + 64, 32)
        self.merge1 = UpBlok(64 + 32, 32)

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = F.relu(self.deconv5(C5))
        up4 = F.relu(self.merge4(up5, C4))
        up3 = F.relu(self.merge3(up4, C3))
        up2 = F.relu(self.merge2(up3, C2))
        up1 = self.merge1(up2, C1)                # no relu after merge1 (upstream verbatim)
        return up1, up2, up3, up4, up5


class TextNet(nn.Module):
    """Inference-time TextNet: FPN + 1x1 predict (8ch). No GCN, no train-only morphology."""

    def __init__(self, backbone="vgg", is_training=False):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(backbone, is_training)
        self.out_channel = 8
        self.predict = nn.Sequential(nn.Conv2d(32, self.out_channel, kernel_size=1, stride=1, padding=0))

    def forward_test(self, img):
        up1, up2, up3, up4, up5 = self.fpn(img)
        return self.predict(up1)

    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        new_state = {k: v for k, v in state_dict["model"].items() if "gcn_model" not in k}
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        # erode1-4 / diale1-4 are the only expected missing keys (train-only morphology).
        return missing, unexpected


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_KERNEL_17x17 = np.ones((17, 17), np.uint8)


def regularize_sin_cos(sin, cos):
    scale = np.sqrt(1.0 / (sin ** 2 + cos ** 2))
    return sin * scale, cos * scale


class MaskTextDetector:
    """TextSnake mask-decoder, ported verbatim from DarkText-private mask_detector.detect_contours1.

    `detect(image_tensor)` takes (1,3,S,S) normalised tensor and returns list[dict(poly, score)] in
    input-pixel coordinates (stride-1), matching upstream's contour output."""

    def __init__(self, model):
        self.model = model
        self.expend = 0.9
        model.eval()

    @staticmethod
    def farthest_point_sampling(point, n_sample, start_idx=0):
        n_points = point.shape[0]
        if n_points <= n_sample:
            return np.arange(n_points)
        if start_idx is None or start_idx < 0 or start_idx >= n_points:
            start_idx = np.random.randint(0, n_points)
        sampled_indices = [start_idx]
        min_distances = np.full(n_points, np.inf)
        for _ in range(n_sample - 1):
            current_point = point[sampled_indices[-1]]
            dist_to_current = np.linalg.norm(point - current_point, axis=1)
            min_distances = np.minimum(min_distances, dist_to_current)
            farthest_idx = np.argmax(min_distances)
            sampled_indices.append(farthest_idx)
        return np.array(sampled_indices)

    def bbox_transfor_inv(self, radius_map, sin_map, cos_map, score_map, wclip=(2, 8)):
        xy_text = np.argwhere(score_map > 0)
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        origin = xy_text
        radius = radius_map[xy_text[:, 0], xy_text[:, 1], :]
        sin = sin_map[xy_text[:, 0], xy_text[:, 1]]
        cos = cos_map[xy_text[:, 0], xy_text[:, 1]]
        dtx = radius[:, 0] * cos * self.expend
        dty = radius[:, 0] * sin * self.expend
        ddx = radius[:, 1] * cos * self.expend
        ddy = radius[:, 1] * sin * self.expend
        topp = origin + np.stack([dty, dtx], axis=-1)
        botp = origin - np.stack([ddy, ddx], axis=-1)
        width = (radius[:, 0] + radius[:, 1]) // 3
        width = np.clip(width, wclip[0], wclip[1])

        top1 = topp - np.stack([width * cos, -width * sin], axis=-1)
        top2 = topp + np.stack([width * cos, -width * sin], axis=-1)
        bot1 = botp - np.stack([width * cos, -width * sin], axis=-1)
        bot2 = botp + np.stack([width * cos, -width * sin], axis=-1)

        bbox = np.stack([top1, top2, bot2, bot1], axis=1)[:, :, ::-1]
        bboxs = np.zeros((bbox.shape[0], 9), dtype=np.float32)
        bboxs[:, :8] = bbox.reshape((-1, 8))
        bboxs[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        return bboxs

    def detect_contours1(self, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred):
        tr_pred_mask = tr_pred > 0.8
        tcl_pred_mask = tcl_pred > 0.36
        tcl_mask = tcl_pred_mask.astype(np.uint8)
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        tcl_contours, _hierarchy = cv2.findContours(tcl_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros_like(tcl_mask)
        H, W = tcl_mask.shape

        blank_images, boxes_per_contour = [], []
        for cont in tcl_contours:
            deal_map = mask.copy()
            cv2.drawContours(deal_map, [cont], -1, 1, -1)
            deal_sum = int(deal_map.sum())
            if deal_sum <= 140 or deal_sum >= 100000:
                continue
            text_map = tr_pred * deal_map
            bboxs = self.bbox_transfor_inv(radii_pred, sin_pred, cos_pred, text_map, wclip=(5, 5))
            boxes = bboxs[:, :8].reshape((-1, 4, 2)).astype(np.int32)
            if boxes.shape[0] > 100:
                keep_idx = self.farthest_point_sampling(boxes[:, :1, :].squeeze(), 100)
                boxes = boxes[keep_idx]
            blank = np.zeros((H, W), dtype=np.uint8)
            for box in boxes:
                cv2.drawContours(blank, [box], -1, (255, 255, 255), 1)
            blank_images.append(blank)
            boxes_per_contour.append(boxes)

        if not blank_images:
            return []

        bbox_contours = []
        for blank in blank_images:
            dilated = cv2.dilate(blank, _KERNEL_17x17, iterations=1)
            eroded = cv2.erode(dilated, _KERNEL_17x17, iterations=1)
            boundary_pts, _hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            for cnt in boundary_pts:
                epsilon = 0.004 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                boundary_point = np.squeeze(np.array(approx))
                if boundary_point is None or len(boundary_point.shape) != 2:
                    continue
                recon_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(recon_mask, [boundary_point], -1, 1, -1)
                recon_sum = int(recon_mask.sum())
                if recon_sum <= 140:
                    continue
                overlap = (recon_mask * tr_pred_mask).sum()
                if overlap < recon_sum * 0.5:
                    continue
                bbox_contours.append([boundary_point, boundary_point])
        return bbox_contours

    def detect(self, image):
        with torch.no_grad():
            output = self.model.forward_test(image)
        tr_pred = output[0, 0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = output[0, 2:4].softmax(dim=0).data.cpu().numpy()
        sin_pred = output[0, 4].data.cpu().numpy()
        cos_pred = output[0, 5].data.cpu().numpy()
        radii_pred = output[0, 6:8].permute(1, 2, 0).contiguous().data.cpu().numpy()

        contours = self.detect_contours1(tr_pred[1], tcl_pred[1], sin_pred, cos_pred, radii_pred)
        polys = []
        for (cont, _lines) in contours:
            # score = mean text probability inside the reconstructed polygon (upstream has no score).
            m = np.zeros(tr_pred[1].shape, np.uint8)
            cv2.drawContours(m, [cont], -1, 1, -1)
            score = float(tr_pred[1][m > 0].mean()) if (m > 0).any() else 0.0
            polys.append(dict(poly=np.asarray(cont, np.float32), score=score))
        return polys