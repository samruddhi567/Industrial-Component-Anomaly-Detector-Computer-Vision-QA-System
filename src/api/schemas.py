"""
Pydantic Schemas for Industrial QA REST APIs and Telemetry.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x: int
    y: int
    width: int
    height: int
    norm_x: float
    norm_y: float
    norm_width: float
    norm_height: float
    severity: float


class InspectionResponse(BaseModel):
    frame_id: str
    status: str = Field(description="QA verdict: PASS or FAIL")
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    predicted_defect: str
    defect_confidence: float
    defect_probabilities: Dict[str, float]
    bounding_boxes: List[BoundingBoxSchema]
    latency_ms: float
    inference_time_ms: float
    qa_alert: bool = False
    annotated_image_base64: Optional[str] = None


class StreamStatusResponse(BaseModel):
    is_running: bool
    camera_type: str
    target_fps: int
    worker_concurrency: int
    queue_depth: int
    total_inspected: int


class MetricResponse(BaseModel):
    pipeline_metrics: Dict[str, Any]
    queue_metrics: Dict[str, Any]
    hardware_info: Dict[str, Any]


class BenchmarkResponse(BaseModel):
    summary: Dict[str, Any]
    baseline: Dict[str, Any]
    accelerated: Dict[str, Any]
