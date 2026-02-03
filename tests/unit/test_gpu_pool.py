import threading
import time

import pytest

from utils.gpu_pool import GpuLeasePool


class TestGpuLeasePool:
    def test_prefers_preferred_gpu_when_idle(self) -> None:
        pool = GpuLeasePool(gpu_ids=[0, 1, 2, 3], preferred_gpu=0, capacity_per_gpu=1)

        with pool.lease() as lease1:
            assert lease1.gpu_id == 0

        with pool.lease() as lease2:
            assert lease2.gpu_id == 0

    def test_spills_to_next_gpu_when_preferred_busy(self) -> None:
        pool = GpuLeasePool(gpu_ids=[0, 1, 2, 3], preferred_gpu=0, capacity_per_gpu=1)

        with pool.lease() as lease1:
            assert lease1.gpu_id == 0

            with pool.lease(timeout_seconds=0.2) as lease2:
                assert lease2.gpu_id == 1

    def test_blocks_until_gpu_is_released(self) -> None:
        pool = GpuLeasePool(gpu_ids=[0, 1], preferred_gpu=0, capacity_per_gpu=1)

        start_evt = threading.Event()
        release_evt = threading.Event()
        result: dict[str, int] = {}

        def worker() -> None:
            start_evt.set()
            with pool.lease(timeout_seconds=2.0) as lease:
                result["gpu"] = lease.gpu_id
                release_evt.set()

        with pool.lease() as lease1:
            assert lease1.gpu_id == 0

            t = threading.Thread(target=worker, daemon=True)
            t.start()

            assert start_evt.wait(timeout=1.0)
            # Give the worker a moment to attempt acquisition.
            time.sleep(0.1)
            assert "gpu" not in result

        # After releasing lease1, worker can acquire.
        assert release_evt.wait(timeout=2.0)
        assert result["gpu"] == 0

    def test_timeout_when_all_gpus_busy(self) -> None:
        pool = GpuLeasePool(gpu_ids=[0], preferred_gpu=0, capacity_per_gpu=1)

        with pool.lease():
            with pytest.raises(TimeoutError):
                _ = pool.acquire(timeout_seconds=0.05)
