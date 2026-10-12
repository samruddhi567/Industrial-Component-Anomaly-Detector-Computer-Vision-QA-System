"""
Unit tests for Dual-Head CNN Anomaly Detector and PyTorch optimization.
"""

import pytest
import torch
import numpy as np

from src.models.cnn_detector import DualHeadAnomalyCNN, ComponentAnomalyDetector
from src.models.cuda_optimizer import DeviceOptimizer


def test_dual_head_cnn_shapes():
    model = DualHeadAnomalyCNN(num_classes=6, in_channels=3)
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits, heatmap, bottleneck = model(dummy_input)

    assert logits.shape == (2, 6), f"Expected logits [2, 6], got {logits.shape}"
    assert heatmap.shape == (2, 1, 256, 256), f"Expected heatmap [2, 1, 256, 256], got {heatmap.shape}"
    assert bottleneck.shape == (2, 256, 16, 16), f"Expected bottleneck [2, 256, 16, 16], got {bottleneck.shape}"
    assert (heatmap >= 0.0).all() and (heatmap <= 1.0).all(), "Heatmap values must be strictly normalized [0, 1]"


def test_anomaly_detector_verdict():
    opt = DeviceOptimizer(requested_device="cpu")
    detector = ComponentAnomalyDetector(anomaly_threshold=0.5, device_optimizer=opt)

    dummy_batch = torch.randn(2, 3, 256, 256)
    results = detector.inspect_batch(dummy_batch, ["FRAME-1", "FRAME-2"])

    assert len(results) == 2
    for res in results:
        assert "status" in res
        assert res["status"] in ["PASS", "FAIL"]
        assert "anomaly_score" in res
        assert 0.0 <= res["anomaly_score"] <= 1.0
        assert "bounding_boxes" in res
        assert "heatmap" in res
