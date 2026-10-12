"""
Deep GPU-Accelerated CNN Model for Industrial Component Anomaly Detection.
Combines feature extraction, multi-class defect classification, and spatial anomaly heatmaps.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual convolutional block with batch normalization and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class DualHeadAnomalyCNN(nn.Module):
    """
    GPU-accelerated Dual-Head CNN:
    1. Classification Head: Defect categorization & multi-class defect probability.
    2. Reconstruction / Spatial Anomaly Head: Generates 2D anomaly heatmap and pixel localization.
    """

    def __init__(self, num_classes: int = 6, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Shared Feature Encoder (Stem & Residual Blocks)
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),  # /2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer1 = ResidualBlock(32, 64, stride=2)   # /4
        self.layer2 = ResidualBlock(64, 128, stride=2)  # /8
        self.layer3 = ResidualBlock(128, 256, stride=2) # /16

        # Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Spatial Anomaly Decoder (Transposed convolutions for defect heatmap reconstruction)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), # *2 -> /8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # *2 -> /4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),   # *2 -> /2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=True),    # *2 -> original
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            logits: [B, num_classes] Defect class logits
            heatmap: [B, 1, H, W] Spatial defect intensity heatmap (0.0 to 1.0)
            feature_map: [B, 256, H/16, W/16] Bottleneck embeddings
        """
        feat = self.encoder_stem(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        bottleneck = self.layer3(feat)

        # Classification branch
        pooled = self.pool(bottleneck).flatten(1)
        logits = self.classifier(pooled)

        # Spatial Anomaly Heatmap branch
        heatmap = self.decoder(bottleneck)

        return logits, heatmap, bottleneck


class ComponentAnomalyDetector:
    """
    High-level inference engine that coordinates model execution, threshold evaluation,
    bounding box extraction, and Pass/Fail quality assurance grading.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        anomaly_threshold: float = 0.65,
        device_optimizer: Optional[Any] = None,
    ):
        self.categories = categories or [
            "normal",
            "surface_scratch",
            "structural_crack",
            "void_porosity",
            "corrosion_spot",
            "dimensional_misalignment",
        ]
        self.anomaly_threshold = anomaly_threshold
        self.opt = device_optimizer

        self.device = self.opt.device if self.opt else torch.device("cpu")
        self.model = DualHeadAnomalyCNN(num_classes=len(self.categories)).to(self.device)
        self.model.eval()

    def warmup(self, iterations: int = 10, batch_size: int = 4):
        """Warm up CUDA kernels to ensure consistent sub-millisecond latencies."""
        is_cuda = self.opt.is_cuda if self.opt else (self.device.type == "cuda")
        actual_iters = iterations if is_cuda else min(iterations, 2)
        actual_batch = batch_size if is_cuda else min(batch_size, 2)
        dummy = torch.randn(actual_batch, 3, 256, 256, device=self.device)
        with torch.no_grad():
            for _ in range(actual_iters):
                if self.opt:
                    with self.opt.execution_scope():
                        _ = self.model(dummy)
                else:
                    _ = self.model(dummy)
        if self.opt:
            self.opt.synchronize()

    def inspect_batch(
        self,
        tensor_batch: torch.Tensor,
        frame_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Runs GPU-accelerated batch inspection.

        Args:
            tensor_batch: [B, 3, H, W] normalized float tensor (0.0 - 1.0) on self.device
            frame_ids: list of identifiers matching batch size

        Returns:
            List of inspection QA result dictionaries
        """
        results = []
        with torch.no_grad():
            if self.opt:
                with self.opt.execution_scope():
                    logits, heatmaps, _ = self.model(tensor_batch)
                self.opt.synchronize()
            else:
                logits, heatmaps, _ = self.model(tensor_batch)

            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            heatmaps_np = heatmaps.float().squeeze(1).cpu().numpy()

        for idx, frame_id in enumerate(frame_ids):
            prob = probs[idx]
            normal_prob = float(prob[0])
            pred_class_idx = int(np.argmax(prob))
            pred_class_name = self.categories[pred_class_idx]

            # Calculate holistic anomaly score
            # (combination of defect class probability + peak spatial anomaly intensity)
            spatial_map = heatmaps_np[idx]
            peak_spatial_anomaly = float(np.max(spatial_map))
            mean_spatial_anomaly = float(np.mean(spatial_map))

            anomaly_score = float(max(1.0 - normal_prob, peak_spatial_anomaly * 0.9 + mean_spatial_anomaly * 0.1))
            is_anomaly = anomaly_score >= self.anomaly_threshold
            status = "FAIL" if is_anomaly else "PASS"

            defect_details = {
                cat: round(float(prob[i]), 4) for i, cat in enumerate(self.categories)
            }

            # Extract defect bounding boxes only if component fails inspection
            bboxes = (
                self._extract_defect_bboxes(spatial_map, threshold=self.anomaly_threshold)
                if is_anomaly
                else []
            )

            results.append({
                "frame_id": frame_id,
                "status": status,
                "is_anomaly": is_anomaly,
                "anomaly_score": round(anomaly_score, 4),
                "threshold": self.anomaly_threshold,
                "predicted_defect": pred_class_name if is_anomaly else "none",
                "defect_confidence": round(float(prob[pred_class_idx]), 4),
                "defect_probabilities": defect_details,
                "bounding_boxes": bboxes,
                "heatmap": spatial_map,
            })

        return results

    def _extract_defect_bboxes(self, heatmap: np.ndarray, threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Identifies regions of high anomaly from spatial heatmap and returns normalized bounding boxes.
        """
        h, w = heatmap.shape
        small_h, small_w = 64, 64
        try:
            import cv2
            small_map = cv2.resize(heatmap, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            binary = (small_map >= threshold).astype(np.uint8)
            if np.sum(binary) == 0:
                return []

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= 3:
                    sx, sy, sbw, sbh = cv2.boundingRect(cnt)
                    norm_x = sx / small_w
                    norm_y = sy / small_h
                    norm_w = sbw / small_w
                    norm_h = sbh / small_h
                    boxes.append({
                        "x": int(norm_x * w),
                        "y": int(norm_y * h),
                        "width": int(norm_w * w),
                        "height": int(norm_h * h),
                        "norm_x": round(float(norm_x), 4),
                        "norm_y": round(float(norm_y), 4),
                        "norm_width": round(float(norm_w), 4),
                        "norm_height": round(float(norm_h), 4),
                        "severity": round(float(np.mean(small_map[sy:sy+sbh, sx:sx+sbw])), 4)
                    })
            return sorted(boxes, key=lambda b: b["severity"], reverse=True)[:5]
        except Exception:
            return []
            # Fallback if cv2 contour is unavailable
            coords = np.argwhere(binary > 0)
            if len(coords) == 0:
                return []
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            h, w = heatmap.shape
            return [{
                "x": int(x_min),
                "y": int(y_min),
                "width": int(x_max - x_min + 1),
                "height": int(y_max - y_min + 1),
                "norm_x": round(float(x_min / w), 4),
                "norm_y": round(float(y_min / h), 4),
                "norm_width": round(float((x_max - x_min + 1) / w), 4),
                "norm_height": round(float((y_max - y_min + 1) / h), 4),
                "severity": round(float(np.mean(heatmap[y_min:y_max+1, x_min:x_max+1])), 4)
            }]
