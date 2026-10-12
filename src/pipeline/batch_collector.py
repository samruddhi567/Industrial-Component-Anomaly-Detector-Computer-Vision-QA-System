"""
Dynamic Batch Collector for high-throughput GPU inference.
Aggregates arriving queue items into optimal tensor batches without inducing latency penalties.
"""

import asyncio
import time
from typing import List, Tuple, Any
import numpy as np
import cv2
import torch
from .worker_queue import AsyncWorkerQueue, FrameQueueItem


class DynamicBatchCollector:
    """
    Collects frames from the asynchronous worker queue dynamically:
    - Waits up to max_wait_ms for batch_size items.
    - Yields immediately if batch_size is reached, or when timeout expires.
    - Packs frames into a contiguous PyTorch tensor on the target device.
    """

    def __init__(
        self,
        queue: AsyncWorkerQueue,
        batch_size: int = 8,
        max_wait_ms: float = 5.0,
        max_batch_wait_ms: Optional[float] = None,
        target_size: Tuple[int, int] = (256, 256),
        device: torch.device = torch.device("cpu"),
    ):
        self.queue = queue
        self.batch_size = max(1, batch_size)
        effective_wait = max_batch_wait_ms if max_batch_wait_ms is not None else max_wait_ms
        self.max_wait_sec = effective_wait / 1000.0
        self.target_size = target_size
        self.device = device

    async def collect_batch(self) -> Tuple[torch.Tensor, List[FrameQueueItem]]:
        """
        Collects 1 to batch_size items and returns a batched PyTorch tensor and items.
        """
        items: List[FrameQueueItem] = []

        # Wait for at least one item
        first_item = await self.queue.get()
        items.append(first_item)

        deadline = time.perf_counter() + self.max_wait_sec

        while len(items) < self.batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break

            try:
                # Poll with remaining timeout
                item = await asyncio.wait_for(self.queue.get(), timeout=max(0.0005, remaining))
                items.append(item)
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break

        # Convert images into batched normalized tensor [B, 3, H, W]
        tensors = []
        for it in items:
            frame = it.frame
            # Resize and convert BGR -> RGB
            resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0.0, 1.0] and transpose to [3, H, W]
            arr = rgb.astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))

        batch_tensor = torch.stack(tensors).to(self.device, non_blocking=True)
        return batch_tensor, items
