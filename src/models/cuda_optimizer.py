"""
CUDA acceleration and hardware optimization engine for PyTorch models.
Provides CUDA stream management, AMP fp16 mixed precision, warmup, and device diagnostics.
"""

import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


class DeviceOptimizer:
    """
    Manages CUDA hardware streams, device selection, mixed precision, and performance profiling.
    """

    def __init__(self, requested_device: str = "cuda", use_amp: bool = True):
        self.requested_device = requested_device
        self.device = self._resolve_device(requested_device)
        self.is_cuda = self.device.type == "cuda"
        self.use_amp = use_amp and (self.is_cuda or hasattr(torch.amp, "autocast"))
        self._stream: Optional[torch.cuda.Stream] = None

        if self.is_cuda:
            self._stream = torch.cuda.Stream(device=self.device)
            # Enable cuDNN benchmark for fixed input tensor dimensions
            torch.backends.cudnn.benchmark = True
            logger.info(
                f"CUDA initialized on {torch.cuda.get_device_name(self.device)} "
                f"(Compute Capability: {torch.cuda.get_device_capability(self.device)}, "
                f"Total Memory: {torch.cuda.get_device_properties(self.device).total_memory / (1024**3):.2f} GB)"
            )
        else:
            logger.info("Operating in CPU execution mode (CUDA unavailable or disabled).")

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    @property
    def stream(self) -> Optional[torch.cuda.Stream]:
        return self._stream

    @contextmanager
    def execution_scope(self):
        """
        Executes inference inside a dedicated CUDA stream with automatic mixed precision (AMP).
        """
        if self.is_cuda and self._stream is not None:
            with torch.cuda.stream(self._stream):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    yield
            # Stream sync handled asynchronously or when tensor data is retrieved
        else:
            if self.use_amp and hasattr(torch.amp, "autocast"):
                with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=self.use_amp):
                    yield
            else:
                yield

    def synchronize(self):
        """
        Explicitly synchronize the stream or device if synchronous barrier is required.
        """
        if self.is_cuda and self._stream is not None:
            self._stream.synchronize()
        elif self.is_cuda:
            torch.cuda.synchronize()

    def get_device_info(self) -> Dict[str, Any]:
        """
        Returns hardware accelerator telemetry for REST API diagnostics.
        """
        info = {
            "device": str(self.device),
            "is_cuda": self.is_cuda,
            "amp_enabled": self.use_amp,
        }
        if self.is_cuda:
            props = torch.cuda.get_device_properties(self.device)
            allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
            info.update({
                "gpu_name": torch.cuda.get_device_name(self.device),
                "total_memory_mb": round(props.total_memory / (1024**2), 2),
                "allocated_memory_mb": round(allocated, 2),
                "reserved_memory_mb": round(reserved, 2),
                "multi_processor_count": props.multi_processor_count,
            })
        else:
            info.update({
                "gpu_name": "Host CPU",
                "cpu_threads": torch.get_num_threads(),
            })
        return info
