"""
Unit tests for Asynchronous Worker Queue, Dynamic Batch Collector, and Pipeline.
"""

import asyncio
import pytest
import numpy as np

from src.pipeline.worker_queue import AsyncWorkerQueue
from src.pipeline.batch_collector import DynamicBatchCollector


@pytest.mark.asyncio
async def test_async_worker_queue_backpressure():
    # Small capacity queue to test backpressure frame drop policy
    q = AsyncWorkerQueue(maxsize=3, drop_policy="drop_oldest")

    # Enqueue 5 items into capacity 3
    for i in range(5):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        success = await q.put(frame, f"F-{i}", {"index": i})
        assert success is True

    metrics = q.get_metrics()
    assert metrics["current_queue_depth"] == 3
    assert metrics["total_dropped"] == 2
    assert metrics["total_enqueued"] == 5

    # Oldest frames (0 and 1) dropped; next should be frame 2
    item = await q.get()
    assert item.frame_id == "F-2"


@pytest.mark.asyncio
async def test_dynamic_batch_collector():
    q = AsyncWorkerQueue(maxsize=10)
    collector = DynamicBatchCollector(queue=q, batch_size=4, max_wait_ms=10.0)

    # Put 2 items in queue (less than batch_size 4)
    for i in range(2):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        await q.put(frame, f"F-{i}", {})

    # Collector should timeout after max_wait_ms and return the 2 collected items
    batch_tensor, items = await collector.collect_batch()
    assert len(items) == 2
    assert batch_tensor.shape == (2, 3, 256, 256)
