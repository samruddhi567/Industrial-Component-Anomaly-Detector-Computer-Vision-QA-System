"""
FastAPI Routes for Industrial Computer Vision QA System.
Exposes REST endpoints, MJPEG video streaming, metrics, benchmarks, and WebSockets.
"""

import asyncio
import time
from typing import Dict, Any, Optional
import numpy as np
import cv2
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse

from .schemas import (
    InspectionResponse,
    StreamStatusResponse,
    MetricResponse,
    BenchmarkResponse,
)
from ..pipeline.async_pipeline import AsyncQAPipeline
from ..models.cnn_detector import ComponentAnomalyDetector
from ..models.cuda_optimizer import DeviceOptimizer
from ..utils.visualizer import QAVisualizer
from ..benchmark.latency_evaluator import BenchmarkRunner

router = APIRouter(prefix="/api/v1")

# Global pipeline reference set during app startup
pipeline_instance: Optional[AsyncQAPipeline] = None
detector_instance: Optional[ComponentAnomalyDetector] = None
optimizer_instance: Optional[DeviceOptimizer] = None


def set_pipeline_context(pipeline: AsyncQAPipeline, detector: ComponentAnomalyDetector, opt: DeviceOptimizer):
    global pipeline_instance, detector_instance, optimizer_instance
    pipeline_instance = pipeline
    detector_instance = detector
    optimizer_instance = opt


@router.get("/health")
async def health_check():
    """Returns system status, active device, and worker health."""
    dev_info = optimizer_instance.get_device_info() if optimizer_instance else {}
    return {
        "status": "healthy",
        "service": "Industrial Component Anomaly Detector - CV QA System",
        "timestamp": time.time(),
        "hardware": dev_info,
        "streaming_active": pipeline_instance.is_running if pipeline_instance else False,
    }


@router.post("/inspect/upload", response_model=InspectionResponse)
async def inspect_uploaded_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = None,
):
    """
    Synchronously inspects an uploaded industrial component image file.
    Runs GPU-accelerated inference and returns defect classification,
    anomaly heatmap coordinates, bounding boxes, and pass/fail verdict.
    """
    if detector_instance is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    t0 = time.perf_counter()

    # Preprocess
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(detector_instance.device)
    )

    frame_id = f"UPLOAD-{int(time.time()*1000)}"

    # Temporarily update threshold if provided
    old_th = detector_instance.anomaly_threshold
    if threshold is not None:
        detector_instance.anomaly_threshold = threshold

    inf_t0 = time.perf_counter()
    results = detector_instance.inspect_batch(tensor, [frame_id])
    inf_ms = (time.perf_counter() - inf_t0) * 1000.0

    detector_instance.anomaly_threshold = old_th

    res = results[0]
    total_ms = (time.perf_counter() - t0) * 1000.0
    res["latency_ms"] = round(total_ms, 3)
    res["inference_time_ms"] = round(inf_ms, 3)
    res["component_type"] = "uploaded_component"
    res["qa_alert"] = False

    # Render annotated image overlay
    annotated = QAVisualizer.render_overlay(image, res, show_heatmap=True)
    res["annotated_image_base64"] = QAVisualizer.encode_to_base64_jpeg(annotated)

    return res


@router.post("/stream/start")
async def start_stream():
    """Starts the real-time hardware frame stream and async worker queues."""
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    if pipeline_instance.is_running:
        return {"message": "Stream pipeline is already active.", "status": "running"}

    await pipeline_instance.start()
    return {"message": "Stream pipeline started successfully.", "status": "running"}


@router.post("/stream/stop")
async def stop_stream():
    """Stops the real-time hardware frame stream."""
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    if not pipeline_instance.is_running:
        return {"message": "Stream pipeline is already stopped.", "status": "stopped"}

    await pipeline_instance.stop()
    return {"message": "Stream pipeline stopped successfully.", "status": "stopped"}


@router.get("/stream/status", response_model=StreamStatusResponse)
async def get_stream_status():
    """Retrieves current camera streaming and worker queue status."""
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    return {
        "is_running": pipeline_instance.is_running,
        "camera_type": pipeline_instance.stream.camera_type,
        "target_fps": pipeline_instance.stream.target_fps,
        "worker_concurrency": pipeline_instance.worker_concurrency,
        "queue_depth": pipeline_instance.queue.qsize(),
        "total_inspected": pipeline_instance.metrics.total_inspected,
    }


@router.get("/stream/latest")
async def get_latest_frame():
    """Returns latest inspected frame metadata and base64 rendered overlay."""
    if pipeline_instance is None or pipeline_instance.latest_frame_data is None:
        return {"message": "No frame inspected yet."}

    data = pipeline_instance.latest_frame_data
    raw = data.get("raw_frame")
    overlay_b64 = ""
    if raw is not None:
        annotated = QAVisualizer.render_overlay(raw, data, show_heatmap=True)
        overlay_b64 = QAVisualizer.encode_to_base64_jpeg(annotated)

    # Return clean serializable dict without raw numpy arrays
    clean_res = {k: v for k, v in data.items() if k not in ["raw_frame", "heatmap"]}
    clean_res["annotated_image_base64"] = overlay_b64
    return clean_res


@router.get("/stream/video_feed")
async def mjpeg_video_feed():
    """
    MJPEG multipart streaming endpoint.
    Embed directly into HTML: <img src="/api/v1/stream/video_feed">
    """
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    async def frame_generator():
        while True:
            if pipeline_instance and pipeline_instance.latest_frame_data:
                data = pipeline_instance.latest_frame_data
                raw = data.get("raw_frame")
                if raw is not None:
                    annotated = QAVisualizer.render_overlay(raw, data, show_heatmap=True)
                    jpeg_bytes = QAVisualizer.encode_to_jpeg_bytes(annotated, quality=75)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )
            await asyncio.sleep(0.033)  # ~30 FPS stream display

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/metrics", response_model=MetricResponse)
async def get_metrics():
    """Fetches comprehensive pipeline latency, queue depth, defect distribution, and hardware telemetry."""
    if pipeline_instance is None or optimizer_instance is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    return {
        "pipeline_metrics": pipeline_instance.metrics.get_summary(),
        "queue_metrics": pipeline_instance.queue.get_metrics(),
        "hardware_info": optimizer_instance.get_device_info(),
    }


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(test_frames: int = 50):
    """
    Runs automated comparative benchmark between baseline sequential inspection
    and the asynchronous accelerated pipeline, reproducing the ~92% latency reduction.
    """
    loop = asyncio.get_running_loop()
    runner = BenchmarkRunner(num_test_frames=test_frames)
    # Run in executor thread so API remains responsive
    result = await loop.run_in_executor(None, runner.execute_full_benchmark)
    return result


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    High-frequency WebSocket channel pushing annotated frames and live telemetry to web dashboards.
    """
    await websocket.accept()
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=16)

    def on_frame(result: Dict[str, Any]):
        try:
            if not queue.full():
                queue.put_nowait(result)
        except Exception:
            pass

    if pipeline_instance:
        pipeline_instance.subscribe(on_frame)

    try:
        while True:
            item = await queue.get()
            raw = item.get("raw_frame")
            image_b64 = ""
            if raw is not None:
                annotated = QAVisualizer.render_overlay(raw, item, show_heatmap=True)
                image_b64 = QAVisualizer.encode_to_base64_jpeg(annotated, quality=75)

            payload = {
                "frame_id": item.get("frame_id"),
                "status": item.get("status"),
                "is_anomaly": item.get("is_anomaly"),
                "anomaly_score": item.get("anomaly_score"),
                "predicted_defect": item.get("predicted_defect"),
                "latency_ms": item.get("latency_ms"),
                "throughput_fps": pipeline_instance.metrics.get_summary()["throughput_fps"] if pipeline_instance else 0,
                "consecutive_defects": pipeline_instance.metrics.consecutive_defects if pipeline_instance else 0,
                "qa_alert": item.get("qa_alert", False),
                "image_base64": image_b64,
            }
            await websocket.send_json(payload)
            queue.task_done()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        if pipeline_instance:
            pipeline_instance.unsubscribe(on_frame)
