"""GPU lease pool utilities.

This module provides a small, thread-safe scheduler that hands out GPU "leases"
so callers can prefer a primary GPU (e.g. GPU 0) while still allowing work to
spill onto other GPUs when there are concurrent requests.

Design goals:
- Torch-free: unit tests can run on Windows hosts without CUDA.
- Fair enough: prefers the configured GPU first, then others in order.
- Safe: capacity per GPU can be enforced (default 1).
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Condition
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class GpuLease:
    """A lease representing exclusive (or capacity-limited) access to a GPU."""

    gpu_id: int


class _GpuLeaseContext(AbstractContextManager[GpuLease]):
    def __init__(self, pool: "GpuLeasePool", gpu_id: int) -> None:
        self._pool = pool
        self._gpu_id = gpu_id

    def __enter__(self) -> GpuLease:
        return GpuLease(gpu_id=self._gpu_id)

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self._pool.release(self._gpu_id)
        return None


class GpuLeasePool:
    """A thread-safe GPU scheduler with a preferred GPU.

    Args:
        gpu_ids: GPUs available for leasing (integers as seen by CUDA).
        preferred_gpu: GPU to try first when available.
        capacity_per_gpu: Maximum concurrent leases per GPU.

    Notes:
        This pool does not interact with CUDA or torch. It only coordinates
        selection across threads.
    """

    def __init__(
        self,
        *,
        gpu_ids: Iterable[int],
        preferred_gpu: Optional[int] = None,
        capacity_per_gpu: int = 1,
    ) -> None:
        ids = [int(x) for x in gpu_ids]
        if not ids:
            raise ValueError("gpu_ids must not be empty")
        if capacity_per_gpu < 1:
            raise ValueError("capacity_per_gpu must be >= 1")

        self._gpu_ids: List[int] = sorted(set(ids))
        self._preferred_gpu: int = (
            int(preferred_gpu)
            if preferred_gpu is not None and int(preferred_gpu) in self._gpu_ids
            else self._gpu_ids[0]
        )
        self._capacity_per_gpu = int(capacity_per_gpu)

        self._in_use = {gpu_id: 0 for gpu_id in self._gpu_ids}
        self._cv = Condition()

    @property
    def gpu_ids(self) -> List[int]:
        """Return the pool's GPU ids in ascending order."""

        return list(self._gpu_ids)

    @property
    def preferred_gpu(self) -> int:
        """Return the GPU id that is preferred when idle."""

        return self._preferred_gpu

    def lease(self, *, timeout_seconds: Optional[float] = None) -> _GpuLeaseContext:
        """Acquire a GPU lease.

        Args:
            timeout_seconds: Optional timeout to wait for a GPU to become
                available. If None, waits indefinitely.

        Returns:
            A context manager yielding a :class:`GpuLease`.

        Raises:
            TimeoutError: If timeout_seconds is set and no GPU became available.
        """

        gpu_id = self.acquire(timeout_seconds=timeout_seconds)
        return _GpuLeaseContext(self, gpu_id)

    def acquire(self, *, timeout_seconds: Optional[float] = None) -> int:
        """Acquire and return a GPU id."""

        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + float(timeout_seconds)

        with self._cv:
            while True:
                gpu_id = self._try_acquire_locked()
                if gpu_id is not None:
                    return gpu_id

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for a free GPU")
                    self._cv.wait(timeout=remaining)
                else:
                    self._cv.wait()

    def release(self, gpu_id: int) -> None:
        """Release a previously acquired GPU id."""

        gpu_id = int(gpu_id)
        with self._cv:
            if gpu_id not in self._in_use:
                raise ValueError(f"Unknown gpu_id: {gpu_id}")
            if self._in_use[gpu_id] <= 0:
                raise RuntimeError(f"GPU {gpu_id} released more times than acquired")

            self._in_use[gpu_id] -= 1
            self._cv.notify_all()

    def _try_acquire_locked(self) -> Optional[int]:
        # Preferred first, then remaining GPUs in ascending order.
        ordered = [self._preferred_gpu] + [g for g in self._gpu_ids if g != self._preferred_gpu]
        for gpu_id in ordered:
            if self._in_use[gpu_id] < self._capacity_per_gpu:
                self._in_use[gpu_id] += 1
                return gpu_id
        return None
