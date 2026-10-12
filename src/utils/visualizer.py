"""
Visualizer and Heads-Up Display (HUD) Rendering Engine.
Generates industrial QA overlays, colorized anomaly heatmaps,
bounding boxes, and Base64/JPEG streaming representations.
"""

import base64
from typing import Dict, Any, Optional
import numpy as np
import cv2


class QAVisualizer:
    """
    Renders production-grade inspection overlays on industrial component frames:
    - Thermal / JET anomaly heatmaps blended with original camera feed
    - Defect bounding boxes with severity and label tags
    - Production HUD (Status badge, FPS, latency, QA Alert banner)
    """

    @staticmethod
    def render_overlay(
        frame: np.ndarray,
        result: Dict[str, Any],
        show_heatmap: bool = True,
        heatmap_alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Combines raw frame, heatmap, bounding boxes, and QA telemetry into an annotated image.
        """
        h, w = frame.shape[:2]
        output = frame.copy()

        # 1. Overlay heatmap if requested and present
        heatmap = result.get("heatmap")
        if show_heatmap and heatmap is not None and result.get("is_anomaly", False):
            # Resize heatmap to match image resolution
            hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
            # Clip and scale to 0-255 uint8
            hm_uint8 = np.clip(hm_resized * 255.0, 0, 255).astype(np.uint8)
            # Apply color map (JET provides high visual contrast for thermal defect representation)
            colored_map = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

            # Mask only significant anomaly regions to avoid washing out normal surfaces
            threshold_mask = (hm_resized > (result.get("threshold", 0.65) * 0.5)).astype(np.float32)[:, :, None]
            blended = (colored_map * heatmap_alpha + output * (1.0 - heatmap_alpha)).astype(np.uint8)
            output = (blended * threshold_mask + output * (1.0 - threshold_mask)).astype(np.uint8)

        # 2. Draw Defect Bounding Boxes
        bboxes = result.get("bounding_boxes", [])
        is_anomaly = result.get("is_anomaly", False)
        status = result.get("status", "PASS")

        for box in bboxes:
            # Denormalize coords
            bx = int(box.get("norm_x", 0) * w)
            by = int(box.get("norm_y", 0) * h)
            bw = int(box.get("norm_width", 0) * w)
            bh = int(box.get("norm_height", 0) * h)
            severity = box.get("severity", 0.0)

            # Red box for defects
            color = (40, 40, 235)  # Crimson red BGR
            cv2.rectangle(output, (bx, by), (bx + bw, by + bh), color, 2)

            # Label banner
            label = f"DEFECT {severity:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output, (bx, max(0, by - lh - 6)), (bx + lw + 8, by), color, -1)
            cv2.putText(
                output,
                label,
                (bx + 4, max(lh, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Heads-Up Display (HUD)
        # Top banner background
        hud_height = 54
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (20, 20, 24), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

        # Status Badge (PASS = Vibrant Green, FAIL = Bold Red)
        badge_color = (46, 184, 87) if status == "PASS" else (40, 40, 235)
        badge_text = f"QA: {status}"
        cv2.rectangle(output, (10, 10), (130, 44), badge_color, -1)
        cv2.putText(
            output,
            badge_text,
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Frame ID & Component
        frame_id = result.get("frame_id", "FRAME-0000")
        comp_type = result.get("component_type", "PART").upper()
        cv2.putText(
            output,
            f"{frame_id} | {comp_type}",
            (145, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Anomaly Score Gauge & Defect name
        anomaly_score = result.get("anomaly_score", 0.0)
        pred_defect = result.get("predicted_defect", "none")
        score_color = (80, 220, 120) if anomaly_score < 0.65 else (50, 80, 240)
        cv2.putText(
            output,
            f"Score: {anomaly_score:.3f} | Pred: {pred_defect}",
            (145, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            score_color,
            1,
            cv2.LINE_AA,
        )

        # Latency & Worker ID (Top Right)
        latency = result.get("latency_ms", 0.0)
        worker_id = result.get("worker_id", 0)
        perf_text = f"{latency:.1f}ms (W{worker_id})"
        (tw, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            output,
            perf_text,
            (w - tw - 15, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # QA Alert Ribbon if continuous failure triggers alert
        if result.get("qa_alert", False):
            cv2.rectangle(output, (0, h - 35), (w, h), (30, 30, 210), -1)
            alert_msg = "!! PRODUCTION LINE ALERT: CONSECUTIVE DEFECT CRITERIA EXCEEDED !!"
            (aw, _), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
            cv2.putText(
                output,
                alert_msg,
                (max(10, (w - aw) // 2), h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output

    @staticmethod
    def encode_to_base64_jpeg(image: np.ndarray, quality: int = 85) -> str:
        """Encodes an image to a base64 data URI string for WebSocket/REST."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        if not success:
            return ""
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def encode_to_jpeg_bytes(image: np.ndarray, quality: int = 80) -> bytes:
        """Encodes an image to raw JPEG bytes for MJPEG streaming."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        if not success:
            return b""
        return buffer.tobytes()
