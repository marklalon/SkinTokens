"""Shared fixtures for the serve concurrency tests.

These tests exercise the real pipeline orchestration in :mod:`serve` (locks,
executors, semaphore, cancellation, RNG isolation) while stubbing out the heavy
stage workers (bpy parse/export, GPU inference, remote renamer) so no model,
GPU, or Blender runtime is required.
"""

import asyncio
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

import serve


@pytest.fixture(autouse=True)
def _quiet_logs():
    """Silence the per-progress INFO spam during tests."""
    previous = serve.logger.level
    serve.logger.setLevel(logging.WARNING)
    yield
    serve.logger.setLevel(previous)


@pytest.fixture(autouse=True)
def fresh_state():
    """Give every test pristine locks, semaphore, and single-worker executors.

    Locks/semaphores are recreated per test so a waiter future created in one
    test's event loop never leaks into another test's loop.
    """
    bpy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-bpy")
    gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-gpu")

    serve.state.bpy_executor = bpy_executor
    serve.state.gpu_executor = gpu_executor
    serve.state.bpy_lock = asyncio.Lock()
    serve.state.gpu_lock = asyncio.Lock()
    serve.state.renamer_sem = asyncio.Semaphore(serve.RENAMER_CONCURRENCY)
    serve.state.active_jobs = 0
    serve.state.ready = True

    yield

    bpy_executor.shutdown(wait=True, cancel_futures=True)
    gpu_executor.shutdown(wait=True, cancel_futures=True)


class ResourceTracker:
    """Thread-safe record of how many requests occupy each resource at once.

    The stage workers run on executor threads while the renamer runs on the
    event loop, so all bookkeeping is guarded by a plain threading lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.active = Counter()
        self.max_active = Counter()
        self.overlap_bpy_gpu = False
        self.events = []

    def enter(self, resource: str, request_id: str):
        with self._lock:
            self.active[resource] += 1
            self.max_active[resource] = max(
                self.max_active[resource], self.active[resource]
            )
            # A single request never holds bpy and gpu at the same time (its
            # stages are sequential), so concurrent bpy+gpu activity proves two
            # *different* requests are pipelining.
            if self.active["bpy"] > 0 and self.active["gpu"] > 0:
                self.overlap_bpy_gpu = True
            self.events.append((time.monotonic(), "enter", resource, request_id))

    def exit(self, resource: str, request_id: str):
        with self._lock:
            self.active[resource] -= 1
            self.events.append((time.monotonic(), "exit", resource, request_id))


@pytest.fixture
def tracker():
    return ResourceTracker()
