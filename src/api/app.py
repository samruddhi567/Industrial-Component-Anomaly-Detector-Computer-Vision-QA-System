"""
FastAPI Application Entrypoint for Industrial CV QA System.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ..config import settings
from ..hardware.frame_stream import HardwareFrameStream
from ..models.cnn_detector import ComponentAnomalyDetector
from ..models.cuda_optimizer import DeviceOptimizer
from ..pipeline.async_pipeline import AsyncQAPipeline
from .routes import router, set_pipeline_context

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes hardware devices, GPU models, and asynchronous pipeline."""
    logger.info("Initializing Industrial Component Anomaly Detector QA System...")

    # Initialize Hardware Optimization (CUDA / AMP)
    device_optimizer = DeviceOptimizer(
        requested_device=settings.inference.device,
        use_amp=settings.inference.use_amp,
    )

    # Initialize GPU Anomaly Detector
    detector = ComponentAnomalyDetector(
        categories=settings.inference.defect_categories,
        anomaly_threshold=settings.inference.anomaly_threshold,
        device_optimizer=device_optimizer,
    )

    # Initialize Camera Stream (Simulator or physical camera)
    frame_stream = HardwareFrameStream(
        camera_type=settings.hardware.camera_type,
        camera_id=settings.hardware.camera_id,
        rtsp_url=settings.hardware.rtsp_url,
        width=settings.hardware.frame_width,
        height=settings.hardware.frame_height,
        target_fps=settings.hardware.target_fps,
    )

    # Initialize Asynchronous QA Pipeline with bounded worker queue
    pipeline = AsyncQAPipeline(
        stream=frame_stream,
        detector=detector,
        device_optimizer=device_optimizer,
        queue_max_size=settings.pipeline.queue_max_size,
        batch_size=settings.inference.batch_size,
        max_batch_wait_ms=settings.inference.max_batch_wait_ms,
        worker_concurrency=settings.pipeline.worker_concurrency,
        alert_consecutive_threshold=settings.pipeline.alert_on_consecutive_defects,
    )

    # Inject context into API router
    set_pipeline_context(pipeline, detector, device_optimizer)

    # Start live processing by default
    await pipeline.start()

    logger.info("System fully operational. Ready for QA inspection.")
    yield

    logger.info("Shutting down pipeline and releasing hardware...")
    await pipeline.stop()


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    app = FastAPI(
        title=settings.system.app_name,
        version=settings.system.version,
        description="High-Throughput Asynchronous Computer Vision Quality Assurance System",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API router
    app.include_router(router)

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        async def root():
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "Industrial CV QA System Backend Online"}

    return app


app = create_app()
