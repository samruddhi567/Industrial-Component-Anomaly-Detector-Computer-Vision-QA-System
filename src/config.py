"""
System configuration management for Industrial Component Anomaly Detector.
"""

from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field

CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"


class SystemConfig(BaseModel):
    app_name: str = "Industrial Component Anomaly Detector - CV QA System"
    version: str = "2025.1.0"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


class HardwareConfig(BaseModel):
    camera_type: str = "simulator"
    camera_id: int = 0
    rtsp_url: str = ""
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 60
    buffer_size: int = 128


class InferenceConfig(BaseModel):
    device: str = "cuda"
    use_amp: bool = True
    model_type: str = "dual_head_cnn"
    anomaly_threshold: float = 0.65
    batch_size: int = 8
    max_batch_wait_ms: float = 5.0
    warmup_iterations: int = 10
    defect_categories: List[str] = [
        "normal",
        "surface_scratch",
        "structural_crack",
        "void_porosity",
        "corrosion_spot",
        "dimensional_misalignment",
    ]


class PipelineConfig(BaseModel):
    worker_concurrency: int = 4
    queue_max_size: int = 256
    drop_policy: str = "drop_oldest"
    metrics_window_seconds: int = 60
    alert_on_consecutive_defects: int = 3


class AppSettings(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def load_from_yaml(cls, path: Optional[Path] = None) -> "AppSettings":
        target = path or CONFIG_FILE_PATH
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()


settings = AppSettings.load_from_yaml()
