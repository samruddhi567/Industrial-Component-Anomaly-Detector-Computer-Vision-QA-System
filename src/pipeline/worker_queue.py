"""
Asynchronous Worker Queue Engine with Dynamic Backpressure and Frame Dropping.
Guarantees real-time hardware frame stream processing without blocking ingestion threads.
"""

import asyncio
import time
import logging
from typing import Optional, Any, Dict
from collections import deque

logger = logging.getLogger(__name__)


class FrameQueueItem:
    """Encapsulates a frame stream item with enqueue timestamps and telemetry."""

    def __init__(self, frame: Any, frame_id: str, metadata: Dict[str, Any]):
        self.frame = frame
        self.frame_id = frame_id
        self.metadata = metadata
        self.enqueue_time = time.perf_counter()


class AsyncWorkerQueue:
    """
    High-throughput asynchronous queue designed for real-time video stream ingestion.
    Supports:
    - Bounded memory footprint (queue_max_size)
    - Zero-blocking frame ingestion
    - 'drop_oldest' backpressure policy to eliminate latency buffering
    - Real-time queue metrics (queue depth, drop rate, latency overhead)
    """

    def __init__(self, maxsize: int = 256, drop_policy: str = "drop_oldest"):
        self.maxsize = maxsize
        self.drop_policy = drop_policy
        self._queue: asyncio.Queue[FrameQueueItem] = asyncio.Queue(maxsize=maxsize)
        self._lock = asyncio.Lock()

        # Telemetry
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.total_dropped = 0
        self._recent_wait_times: deque = deque(maxlen=100)

    async def put(self, frame: Any, frame_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Asynchronously enqueues a frame. If queue is full, handles according to drop_policy.
        Never blocks the hardware ingestion thread.
        """
        item = FrameQueueItem(frame, frame_id, metadata)

        if self._queue.full():
            if self.drop_policy == "drop_oldest":
                try:
                    # Drop oldest uninspected frame to preserve real-time recency
                    _ = self._queue.get_nowait()
                    self._queue.task_done()
                    self.total_dropped += 1
                except (asyncio.QueueEmpty, ValueError):
                    pass
            else:
                # Reject incoming frame
                self.total_dropped += 1
                return False

        try:
            self._queue.put_nowait(item)
            self.total_enqueued += 1
            return True
        except asyncio.QueueFull:
            self.total_dropped += 1
            return False

    async def get(self) -> FrameQueueItem:
        """
        Retrieves next item from the queue and records queuing delay.
        """
        item = await self._queue.get()
        self.total_dequeued += 1
        wait_time_ms = (time.perf_counter() - item.enqueue_time) * 1000.0
        self._recent_wait_times.append(wait_time_ms)
        return item

    def task_done(self):
        """Signals completion of the task."""
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()

    def is_empty(self) -> bool:
        return self._queue.empty()

    def get_metrics(self) -> Dict[str, Any]:
        """Returns real-time queue performance metrics."""
        avg_wait = (
            sum(self._recent_wait_times) / len(self._recent_wait_times)
            if self._recent_wait_times
            else 0.0
        )
        return {
            "current_queue_depth": self._queue.qsize(),
            "max_queue_capacity": self.maxsize,
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "total_dropped": self.total_dropped,
            "drop_rate_pct": round(
                (self.total_dropped / max(self.total_enqueued + self.total_dropped, 1)) * 100, 2
            ),
            "avg_queue_wait_ms": round(avg_wait, 3),
            "drop_policy": self.drop_policy,
        }
