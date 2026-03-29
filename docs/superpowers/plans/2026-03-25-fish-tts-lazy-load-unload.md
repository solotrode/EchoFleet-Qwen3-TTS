# Fish TTS Lazy Start/Stop Container Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclaim Fish Audio VRAM by starting the `fish-sglang` container only when the first Fish request arrives, then stopping that container after a configurable idle period.

**Why this plan changed:** The previous version assumed `FishAudioService` owned an in-process SGLang subprocess. That is not how this stack works. Fish runs as a dedicated Docker Compose service (`fish-sglang`), and `FishAudioService` is only a remote client wrapper over that service.

**Architecture:**
1. Add a small Docker-backed controller that can inspect, start, stop, and health-check the `fish-sglang` container, with internal serialization for lifecycle transitions.
2. Call that controller from the synchronous Fish endpoint path before `FishAudioService.generate()`, but run the blocking start/readiness work off the FastAPI event loop.
3. Reuse the existing API-side Fish idle watcher to stop the container after inactivity.
4. Expose both app-layer Fish state and container state in diagnostics so idle/restart behavior is observable.
5. Keep `FishAudioService` focused on HTTP calls to the SGLang API, not process ownership.

**Non-goals:**
- Do not add subprocess management inside `FishAudioService`.
- Do not wire Fish lifecycle into the Qwen Redis worker loop unless Fish jobs are actually routed there.
- Do not rely on Redis queue depth alone to decide whether it is safe to unload.
- Do not try to infer pre-restart Fish idleness from process-local state alone; after an API restart, any already-running `fish-sglang` container must be reconciled explicitly.

> **Operational Note:** After an API restart, any pre-existing `fish-sglang` container will remain running until either: (a) a new Fish request triggers ownership, or (b) a manual unload is issued. This is intentional to avoid incorrectly stopping a container that may still be in use by other services.

**Primary tradeoff:** This plan requires the API container to control Docker. The recommended implementation uses the Docker Engine socket plus the Docker SDK for Python. If granting Docker socket access is unacceptable, use a separate supervisor service instead.

> **Security Note:** Mounting the Docker socket (`/var/run/docker.sock`) grants the container full Docker daemon access. This is a privileged operation—ensure only trusted services have access. Alternatively, use a narrow supervisor service that exposes `start fish`, `stop fish`, and `status fish` endpoints.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `inference/fish_sglang_controller.py` | Docker-based start/stop/readiness controller for `fish-sglang` |
| `inference/fish_audio_service.py` | Keep remote HTTP wrapper; optionally accept a lifecycle controller dependency |
| `api/main.py` | Ensure Fish container is running before sync generation without blocking the event loop; stop it from the idle watcher; expose lifecycle status |
| `config/settings.py` | Add Fish lifecycle config |
| `compose.yaml` | Mount Docker socket into API container, grant socket access to the non-root API user, and remove hard dependency on `fish-sglang` health at API boot |
| `Dockerfile` | Preserve non-root API execution while supporting the chosen Docker socket access strategy |
| `requirements.txt` | Add Docker SDK if not already present |
| `tests/unit/test_fish_sglang_controller.py` | Unit tests for container lifecycle control |
| `tests/unit/test_fish_audio_service.py` | Keep existing Fish service tests aligned with remote-wrapper behavior |
| `tests/integration/test_fish_audio.py` | Verify sync endpoint triggers container start before generation and surfaces container state |

---

## Chunk 1: Add a Container Lifecycle Controller

### Task 1: Create a Docker-backed FishSGLangController

**Files:**
- Add: `inference/fish_sglang_controller.py`
- Modify: `requirements.txt`
- Test: `tests/unit/test_fish_sglang_controller.py`

- [ ] **Step 1: Read current Fish deployment details**

```bash
Read compose.yaml fish-sglang section
Read inference/fish_audio_service.py
```

- [ ] **Step 2: Add Docker SDK dependency if missing**

```text
# requirements.txt
docker>=7.0.0,<8
```

- [ ] **Step 3: Write failing unit tests for controller behavior**

```python
# tests/unit/test_fish_sglang_controller.py
import pytest
from unittest.mock import MagicMock, patch

from inference.fish_sglang_controller import FishSGLangController


def test_ensure_running_starts_exited_container_and_waits_for_health():
    with patch("inference.fish_sglang_controller.docker.from_env") as mock_from_env:
        docker_client = MagicMock()
        container = MagicMock()
        mock_from_env.return_value = docker_client
        docker_client.containers.get.return_value = container

        container.status = "exited"
        container.attrs = {"State": {"Health": {"Status": "healthy"}}}

        controller = FishSGLangController(container_name="fish-sglang")
        controller.ensure_running()

        container.start.assert_called_once()


def test_stop_if_running_stops_running_container():
    with patch("inference.fish_sglang_controller.docker.from_env") as mock_from_env:
        docker_client = MagicMock()
        container = MagicMock()
        mock_from_env.return_value = docker_client
        docker_client.containers.get.return_value = container

        container.status = "running"

        controller = FishSGLangController(container_name="fish-sglang")
        assert controller.stop_if_running() is True

        container.stop.assert_called_once()


def test_ensure_running_noops_when_already_healthy():
    with patch("inference.fish_sglang_controller.docker.from_env") as mock_from_env:
        docker_client = MagicMock()
        container = MagicMock()
        mock_from_env.return_value = docker_client
        docker_client.containers.get.return_value = container

        container.status = "running"
        container.attrs = {"State": {"Health": {"Status": "healthy"}}}

        controller = FishSGLangController(container_name="fish-sglang")
        controller.ensure_running()

        container.start.assert_not_called()


def test_ensure_running_raises_when_container_not_found():
    from docker.errors import NotFound

    with patch("inference.fish_sglang_controller.docker.from_env") as mock_from_env:
        docker_client = MagicMock()
        mock_from_env.return_value = docker_client
        docker_client.containers.get.side_effect = NotFound("container not found")

        controller = FishSGLangController(container_name="fish-sglang")

        with pytest.raises(RuntimeError, match="Fish container not found"):
            controller.ensure_running()


def test_controller_serializes_lifecycle_transitions():
    """Verify that concurrent ensure_running and stop_if_running calls do not interleave."""
    import threading

    with patch("inference.fish_sglang_controller.docker.from_env") as mock_from_env:
        docker_client = MagicMock()
        container = MagicMock()
        mock_from_env.return_value = docker_client
        docker_client.containers.get.return_value = container

        call_log = []

        original_start = container.start
        original_stop = container.stop

        def tracked_start(*args, **kwargs):
            call_log.append("start")
            return original_start(*args, **kwargs)

        def tracked_stop(*args, **kwargs):
            call_log.append("stop")
            return original_stop(*args, **kwargs)

        container.start.side_effect = tracked_start
        container.stop.side_effect = tracked_stop

        # First call: container is exited — should trigger start
        container.status = "exited"
        container.attrs = {"State": {"Health": {"Status": "healthy"}}}

        controller = FishSGLangController(container_name="fish-sglang")

        controller.ensure_running()

        # Second call: container now running — stop_if_running should trigger stop
        container.status = "running"
        controller.stop_if_running()

        # Calls must appear in order with no interleaving
        assert call_log == ["start", "stop"], f"Expected ['start', 'stop'], got {call_log}"

        # Verify the lifecycle lock exists and is a threading.Lock-compatible type
        assert hasattr(controller, "_lifecycle_lock")
        assert hasattr(controller._lifecycle_lock, "acquire")

        # Verify concurrent calls do not raise (lock is reentrant-safe for sequential callers)
        errors = []

        def run_ensure():
            container.status = "running"
            container.attrs = {"State": {"Health": {"Status": "healthy"}}}
            try:
                controller.ensure_running()
            except Exception as e:
                errors.append(e)

        def run_stop():
            container.status = "running"
            try:
                controller.stop_if_running()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_ensure), threading.Thread(target=run_stop)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent lifecycle calls raised: {errors}"
```

- [ ] **Step 4: Implement controller**

Add `inference/fish_sglang_controller.py` with behavior like:

```python
from __future__ import annotations

from typing import Optional
import threading
import time

import docker
from docker.errors import DockerException, NotFound

from utils.logging import get_logger

logger = get_logger(__name__)


class FishSGLangController:
    """Manage the lifecycle of the fish-sglang Docker container."""

    def __init__(
        self,
        container_name: str,
        startup_timeout_seconds: int = 300,
        stop_timeout_seconds: int = 30,
        poll_interval_seconds: float = 2.0,
        docker_socket_path: Optional[str] = None,
        docker_client: Optional[docker.DockerClient] = None,
    ) -> None:
        self._container_name = container_name
        self._startup_timeout_seconds = startup_timeout_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        if docker_client is not None:
            self._docker = docker_client
        elif docker_socket_path:
            self._docker = docker.DockerClient(base_url=f"unix://{docker_socket_path}")
        else:
            self._docker = docker.from_env()
        self._lifecycle_lock = threading.Lock()

    def ensure_running(self) -> None:
        with self._lifecycle_lock:
            try:
                container = self._get_container()
                container.reload()

                if container.status != "running":
                    logger.info("Starting fish-sglang container", extra={"container": self._container_name})
                    container.start()

                self._wait_until_healthy(container)
            except TimeoutError:
                raise
            except DockerException as e:
                raise RuntimeError("Failed to start or inspect Fish container") from e

    def stop_if_running(self) -> bool:
        with self._lifecycle_lock:
            try:
                container = self._get_container()
                container.reload()

                if container.status != "running":
                    return False

                logger.info("Stopping fish-sglang container", extra={"container": self._container_name})
                container.stop(timeout=self._stop_timeout_seconds)
                return True
            except DockerException as e:
                raise RuntimeError("Failed to stop or inspect Fish container") from e

    def is_running(self) -> bool:
        try:
            container = self._get_container()
            container.reload()
            return container.status == "running"
        except DockerException as e:
            raise RuntimeError("Failed to inspect Fish container state") from e

    def _get_container(self):
        try:
            return self._docker.containers.get(self._container_name)
        except NotFound as e:
            raise RuntimeError(f"Fish container not found: {self._container_name}") from e
        except DockerException as e:
            raise RuntimeError("Failed to access Docker API for Fish container control") from e

    def _wait_until_healthy(self, container) -> None:
        deadline = time.monotonic() + self._startup_timeout_seconds
        while time.monotonic() < deadline:
            try:
                container.reload()
                health = ((container.attrs or {}).get("State") or {}).get("Health") or {}
                health_status = health.get("Status")

                if container.status == "running" and health_status in (None, "healthy"):
                    return
            except DockerException as e:
                raise RuntimeError("Failed while waiting for Fish container health") from e

            time.sleep(self._poll_interval_seconds)

        raise TimeoutError(f"Timed out waiting for Fish container {self._container_name} to become healthy")
```

Controller contract: `ensure_running()` and `stop_if_running()` should only surface `TimeoutError` or `RuntimeError` to callers. Wrap raw Docker SDK exceptions inside the controller so the endpoint can translate failures to 503 responses consistently. The controller should serialize start/stop transitions internally so concurrent requests and the idle watcher cannot interleave container lifecycle actions.

Important: the controller remains synchronous. Callers must invoke `ensure_running()` from a worker thread (for example `asyncio.to_thread(...)`) rather than directly on the FastAPI event loop.

- [ ] **Step 5: Run targeted tests**

```bash
pytest tests/unit/test_fish_sglang_controller.py -v
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt inference/fish_sglang_controller.py tests/unit/test_fish_sglang_controller.py
git commit -m "feat: add fish-sglang container lifecycle controller"
```

---

## Chunk 2: Start Fish On First Request

> **Pre-condition — single worker required:** The process-local state added in this chunk (`_FISH_REQUEST_COUNT`, `_FISH_SGLANG_CONTROLLER`) only works correctly when the API runs with a single Uvicorn worker. Multiple workers each have their own memory, so request counts and singleton state diverge across processes. The `--workers 1` enforcement is wired in Chunk 4, but if you test this chunk against a locally running server, ensure it is started with `--workers 1` or the in-flight counter will give false readings.

### Task 2: Ensure the sync Fish endpoint starts the container before calling the backend

**Files:**
- Modify: `api/main.py`
- Modify: `inference/fish_audio_service.py`
- Test: `tests/integration/test_fish_audio.py`

- [ ] **Step 1: Read the existing Fish sync path and idle watcher**

```bash
Read api/main.py get_sync_fish_audio_service, _run_fish_idle_watcher, and /v1/tts/s2-pro/sync
Read inference/fish_audio_service.py
```

- [ ] **Step 2: Write failing integration test for lazy container start**

```python
# tests/integration/test_fish_audio.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np


@pytest.mark.asyncio
async def test_s2_pro_sync_endpoint_ensures_container_running_before_generate():
    """Verify the sync endpoint calls ensure_running before generate, both in one to_thread call."""
    from api.main import app
    from httpx import AsyncClient

    mock_controller = MagicMock()
    mock_service = MagicMock()

    # Simulate the blocking helper returning audio bytes + sample rate
    fake_audio = np.zeros(48000, dtype=np.int16).tobytes()
    fake_result = (fake_audio, 24000)

    call_order = []

    def blocking_helper():
        # The implementation should call ensure_running then generate inside one thread
        mock_controller.ensure_running()
        call_order.append("ensure_running")
        result = mock_service.generate(text="Hello world")
        call_order.append("generate")
        return result

    mock_service.generate.return_value = fake_result

    # Patch asyncio.to_thread so we can intercept the blocking helper and run it
    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("api.main.get_fish_sglang_controller", return_value=mock_controller):
        with patch("api.main.get_sync_fish_audio_service", return_value=mock_service):
            with patch("api.main.asyncio.to_thread", side_effect=fake_to_thread):
                async with AsyncClient(app=app, base_url="http://test") as client:
                    response = await client.post("/v1/tts/s2-pro/sync", json={"text": "Hello world"})

    assert response.status_code == 200
    mock_controller.ensure_running.assert_called_once()
    # ensure_running must precede generate
    assert call_order.index("ensure_running") < call_order.index("generate")
```

> **Implementation note:** The test above reflects the consolidated single-`to_thread` approach recommended in Step 4. The endpoint should wrap both `controller.ensure_running()` and `svc.generate()` inside one blocking helper function passed to `asyncio.to_thread()`. Do not use `asyncio.get_event_loop().run_in_executor()` — that pattern is deprecated and must be removed as part of this chunk.

- [ ] **Step 3: Add a singleton accessor for the controller**

In `api/main.py`, add a singleton similar to the Fish service singleton:

```python
_FISH_SGLANG_CONTROLLER = None
_FISH_SGLANG_CONTROLLER_LOCK = threading.Lock()


def get_fish_sglang_controller() -> FishSGLangController:
    global _FISH_SGLANG_CONTROLLER
    if _FISH_SGLANG_CONTROLLER is None:
        with _FISH_SGLANG_CONTROLLER_LOCK:
            if _FISH_SGLANG_CONTROLLER is None:
                from inference.fish_sglang_controller import FishSGLangController
                import docker as _docker
                from docker.errors import DockerException as _DockerException

                controller = FishSGLangController(
                    container_name=settings.fish_sglang_container_name,
                    startup_timeout_seconds=settings.fish_startup_timeout_seconds,
                    stop_timeout_seconds=settings.fish_stop_timeout_seconds,
                    docker_socket_path=settings.fish_docker_socket_path,
                )

                # Probe Docker availability immediately so misconfiguration
                # surfaces at first use rather than silently at request time.
                try:
                    controller._docker.ping()
                    logger.info("Fish container controller: Docker API reachable")
                except _DockerException as e:
                    logger.error(
                        "Fish container controller: Docker API unreachable — "
                        "Fish lazy-start will fail until socket access is fixed",
                        extra={"error": str(e)},
                    )
                    # Do not raise: allow the API to boot; the endpoint will return 503
                    # when ensure_running() is called and the controller cannot reach Docker.

                _FISH_SGLANG_CONTROLLER = controller
    return _FISH_SGLANG_CONTROLLER
```

- [ ] **Step 4: Ensure the controller is invoked before generate without blocking the event loop**

In `/v1/tts/s2-pro/sync`:

```python
controller = get_fish_sglang_controller()
await asyncio.to_thread(controller.ensure_running)
svc = get_sync_fish_audio_service()
```

If the endpoint is later refactored into a blocking helper, it is also acceptable to run the entire blocking startup-plus-generate segment in one worker thread. What must not happen is calling `controller.ensure_running()` directly on the event loop thread.

> **Note:** The existing endpoint uses the deprecated `asyncio.get_event_loop().run_in_executor()`.
> While this plan only adds `asyncio.to_thread(controller.ensure_running)`, consider consolidating
> both the `ensure_running` call and the `svc.generate` call into a single `asyncio.to_thread()`
> invocation (wrapping both in one blocking helper function). This avoids two thread-pool round
> trips and removes the deprecated pattern in one step.

Keep `FishAudioService` as an HTTP client. Do not make it spawn processes.

- [ ] **Step 5: Add Fish in-flight request counter**

Add a lightweight counter to track active Fish requests and prevent container stop during requests.
Use a single lock for both the counter and the lifecycle-state decisions — a separate `_FISH_REQUEST_LOCK`
is not needed and would create a second lock that must always be acquired in the same order:

```python
import threading
from typing import Optional

_FISH_REQUEST_COUNT = 0
_FISH_LIFECYCLE_STATE_LOCK = threading.Lock()


def _increment_fish_request_count() -> None:
    global _FISH_REQUEST_COUNT
    with _FISH_LIFECYCLE_STATE_LOCK:
        _FISH_REQUEST_COUNT += 1


def _decrement_fish_request_count() -> None:
    global _FISH_REQUEST_COUNT
    with _FISH_LIFECYCLE_STATE_LOCK:
        _FISH_REQUEST_COUNT -= 1


def _is_fish_request_active() -> bool:
    # Callers that need an atomic read+decision must already hold _FISH_LIFECYCLE_STATE_LOCK.
    # This helper may also be called without the lock for logging-only purposes.
    return _FISH_REQUEST_COUNT > 0
```

Wrap the endpoint handler with try/finally to increment before and decrement after:
```python
def _handle_fish_sync_request(...):
    _increment_fish_request_count()
    try:
        await asyncio.to_thread(controller.ensure_running)
        svc = get_sync_fish_audio_service()
        # ... generate logic
    finally:
        _decrement_fish_request_count()
```

The request counter alone is not enough. Use `_FISH_LIFECYCLE_STATE_LOCK` when making the decision
that it is safe to unload/stop, so the watcher cannot observe `0` active requests at the same time
a new request is beginning. Do not introduce a second lock for the counter — all reads and writes
to `_FISH_REQUEST_COUNT` that affect lifecycle decisions must be guarded by `_FISH_LIFECYCLE_STATE_LOCK`.

- [ ] **Step 6: Keep FishAudioService remote-wrapper semantics explicit**

Update `FishAudioService` docstrings/comments as needed so they clearly state:
- it talks to a remote SGLang server
- lazy load means readiness/start coordination at the container layer
- `unload_model()` only clears the app-layer loaded flag

- [ ] **Step 7: Handle container start failures**

If `controller.ensure_running()` fails (timeout, container not found, Docker error), return a 503 Service Unavailable with a clear error message. Do not retry indefinitely—the client can retry the request.

This depends on the controller contract above: raw Docker SDK exceptions should already have been normalized to `RuntimeError` before they reach the endpoint.

```python
try:
    await asyncio.to_thread(controller.ensure_running)
except TimeoutError:
    return JSONResponse(
        status_code=503,
        content={"error": "Fish TTS service starting up, please retry"},
    )
except RuntimeError as e:
    logger.error("Fish container error", extra={"error": str(e)})
    return JSONResponse(
        status_code=503,
        content={"error": "Fish TTS service unavailable"},
    )
```

- [ ] **Step 8: Run Fish integration tests**

```bash
pytest tests/integration/test_fish_audio.py -v
```

- [ ] **Step 9: Commit**

```bash
git add api/main.py inference/fish_audio_service.py tests/integration/test_fish_audio.py
git commit -m "feat: start fish-sglang on first sync request"
```

---

## Chunk 3: Stop Fish After Idle

### Task 3: Use the existing API-side idle watcher to stop the container

**Files:**
- Modify: `api/main.py`
- Test: `tests/integration/test_fish_audio.py`

- [ ] **Step 1: Read existing idle watcher behavior**

```bash
Read api/main.py _run_fish_idle_watcher
Read inference/fish_audio_service.py unload_idle_models
```

> **Ordering note:** `_maybe_unload_idle_fish_backend` passes `fish_idle_unload_seconds` to the
> service, but that setting is only added in Chunk 4. Implement Chunk 4 Step 1 (settings fields)
> **before** wiring up this helper, or use `getattr(settings, "fish_idle_unload_seconds", 300)`
> as a safe fallback until the setting is committed.
>
> **Settings clarification:** There are two separate idle timeout settings:
> - `fish_idle_unload_seconds` (new) - controls Fish container stop after idle
> - `tts_unload_idle_seconds` (existing) - controls Qwen model unload after idle
>
> These are intentionally separate because Fish and Qwen have different lifecycle managers (Docker container vs in-process model). Both the watcher and `FishAudioService.unload_idle_models()` should use `fish_idle_unload_seconds` for Fish-specific behavior.

- [ ] **Step 2: Write failing test for idle stop**

```python
def test_fish_idle_watcher_stops_container_after_unload_signal():
    from api import main

    mock_service = MagicMock()
    mock_service.unload_idle_models.return_value = [("s2-pro", "remote")]

    mock_controller = MagicMock()

    with patch.object(main, "_SYNC_FISH_AUDIO_SERVICE", mock_service):
        with patch("api.main.get_fish_sglang_controller", return_value=mock_controller):
            # call a small extracted helper instead of spinning a real thread
            unloaded = main._maybe_unload_idle_fish_backend(idle_unload_seconds=1)

    assert unloaded == [("s2-pro", "remote")]
    mock_controller.stop_if_running.assert_called_once()
```

- [ ] **Step 3: Extract watcher logic into a helper**

In `api/main.py`, factor the idle behavior into a helper that:
1. checks `svc.unload_idle_models(...)`
2. if unload occurred, calls `controller.stop_if_running()`
3. logs both the app-layer unload and the container stop

Example shape:

```python
def _maybe_unload_idle_fish_backend(idle_unload_seconds: int):
    with _FISH_LIFECYCLE_STATE_LOCK:
        if _is_fish_request_active():
            return []

        svc = _SYNC_FISH_AUDIO_SERVICE
        if svc is None:
            return []

        unloaded = svc.unload_idle_models(idle_seconds=idle_unload_seconds)
        if unloaded:
            controller = get_fish_sglang_controller()
            controller.stop_if_running()
        return unloaded
```

Then make `_run_fish_idle_watcher()` call that helper.

> **Lock scope warning:** `_FISH_LIFECYCLE_STATE_LOCK` is held across both `unload_idle_models()` and `controller.stop_if_running()`. This means any new incoming Fish request will stall on the lock until the watcher finishes. Verify that `FishAudioService.unload_idle_models()` is a cheap, non-blocking, in-memory operation with no I/O before shipping this shape. If it does any I/O (HTTP, disk), factor the lock into two phases: a locked read-then-clear of app-layer state, followed by an unlocked Docker stop call, accepting that a very tight race with a new request may result in starting an already-stopping container (which the `ensure_running()` path already handles by checking container status).

Restart caveat: `_SYNC_FISH_AUDIO_SERVICE` only exists after this API process has handled at least one Fish request. Do not have the watcher stop a pre-existing running `fish-sglang` container after API restart unless you first add explicit reconciliation for last-activity state. Without that, the safe behavior is to leave the container alone until a new Fish request or a manual unload re-establishes ownership.

- [ ] **Step 4: Respect active sync work**

Use the in-flight counter from Chunk 2 in `_maybe_unload_idle_fish_backend()`:

```python
def _maybe_unload_idle_fish_backend(idle_unload_seconds: int):
    with _FISH_LIFECYCLE_STATE_LOCK:
        if _is_fish_request_active():
            return []  # Skip unload if request in flight or just starting

        svc = _SYNC_FISH_AUDIO_SERVICE
        if svc is None:
            return []

        unloaded = svc.unload_idle_models(idle_seconds=idle_unload_seconds)
        if unloaded:
            controller = get_fish_sglang_controller()
            controller.stop_if_running()
        return unloaded
```

Do not use Redis queue depth for this; Fish sync requests are not driven by the Qwen worker queue. The important requirement is that the safety decision (`no active request, so unload+stop is allowed`) is made under the same state lock that coordinates request begin/end bookkeeping.

- [ ] **Step 5: Run targeted tests**

```bash
pytest tests/integration/test_fish_audio.py -v -k "idle or s2_pro"
```

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/integration/test_fish_audio.py
git commit -m "feat: stop fish-sglang container after idle timeout"
```

---

## Chunk 4: Configuration and Compose Wiring

### Task 4: Add explicit Fish lifecycle configuration, Docker access, and observability

**Files:**
- Modify: `api/main.py`
- Modify: `config/settings.py`
- Modify: `compose.yaml`
- Modify: `Dockerfile`
- Test: `tests/integration/test_fish_audio.py`

- [ ] **Step 1: Add settings fields**

In `config/settings.py`, add snake_case settings consistent with the rest of the file:

```python
self.fish_idle_unload_seconds: int = int(os.getenv("FISH_IDLE_UNLOAD_SECONDS", "300"))
self.fish_startup_timeout_seconds: int = int(os.getenv("FISH_STARTUP_TIMEOUT_SECONDS", "300"))
self.fish_stop_timeout_seconds: int = int(os.getenv("FISH_STOP_TIMEOUT_SECONDS", "30"))
self.fish_sglang_container_name: str = os.getenv("FISH_SGLANG_CONTAINER_NAME", "fish-sglang")
self.fish_docker_socket_path: str = os.getenv("FISH_DOCKER_SOCKET_PATH", "/var/run/docker.sock")
```

Prefer using `fish_idle_unload_seconds` for Fish and leave `tts_unload_idle_seconds` untouched for Qwen unless you intentionally want a shared timeout.

- [ ] **Step 2: Update compose and image assumptions for API container Docker access**

The API image currently runs as a non-root user. Treat Docker socket access as an explicit implementation requirement, not a troubleshooting afterthought.

In `compose.yaml` for `echofleet-qwen3-tts`:
- mount `/var/run/docker.sock:/var/run/docker.sock`
- add env vars for the Fish lifecycle settings if you want explicit overrides
- add an explicit socket access strategy for the non-root API process, preferably a supplementary group via `group_add` and a configurable host Docker GID
- **Enforce single worker:** Add `--workers 1` to the API command override to ensure process-local request counting works correctly

> **Docker socket verification:** Before deploying, verify the API container can access Docker by running this test inside the container:
> ```bash
> docker --version && docker ps
> ```
> If this fails, the socket permissions need adjustment before the Fish lifecycle features can work.

In `Dockerfile` and/or Compose wiring:
- preserve non-root execution for the API process
- document how the container user gains access to the mounted socket
- fail fast with a clear log/error if Docker API access is unavailable even though Fish lifecycle control is enabled

Example:

```yaml
environment:
  - FISH_IDLE_UNLOAD_SECONDS=300
  - FISH_STARTUP_TIMEOUT_SECONDS=300
  - FISH_STOP_TIMEOUT_SECONDS=30
  - FISH_SGLANG_CONTAINER_NAME=fish-sglang
  - HOST_DOCKER_GID=${HOST_DOCKER_GID:-999}
group_add:
  - "${HOST_DOCKER_GID:-999}"
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
command: ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

If the deployment environment cannot guarantee correct socket permissions for the non-root API user, stop here and choose the supervisor-service alternative instead of shipping a half-working Docker-socket design.

- [ ] **Step 3: Relax API startup dependency on fish-sglang health**

Because `fish-sglang` may be intentionally stopped at API boot time, remove the strict API dependency on `fish-sglang` being healthy before the API starts.

> **Recommended:** Remove the `fish-sglang` entry from `echofleet-qwen3-tts.depends_on` entirely. The controller will start the container on first request.

Alternative (if ordering is needed):
- Keep only an order dependency (no health check) if Compose version/behavior requires it.

The API must be able to boot while Fish is stopped.

- [ ] **Step 4: Preserve fish-sglang definition itself**

Do not remove the `fish-sglang` service definition. The controller will start and stop that existing named container.

- [ ] **Step 5: Verify fish-sglang container definition exists**

Before deploying, verify the `fish-sglang` service is defined in `compose.yaml`:
```bash
docker compose config --services | grep fish-sglang
```
If missing, the controller cannot manage a non-existent container. The service definition must remain.

- [ ] **Step 6: Update model status to include container lifecycle state**

Extend `/v1/models/status` in `api/main.py` so it reports both:
- the existing app-layer Fish wrapper state
- container-level state from `FishSGLangController` such as container status, health status, and whether Docker control is available

This avoids ambiguity after API restarts, where the wrapper may be uninitialized while the container is still running.

Add an integration test that verifies the status payload includes a Fish container section when the controller is available.

- [ ] **Step 7: Run config sanity checks**

```bash
pytest tests/integration/test_fish_audio.py -v
```

- [ ] **Step 8: Commit**

```bash
git add api/main.py config/settings.py compose.yaml Dockerfile tests/integration/test_fish_audio.py
git commit -m "feat: configure fish-sglang on-demand container control"
```

---

## Chunk 5: Manual Unload and Operational Validation

### Task 5: Ensure manual unload can also stop the container

**Files:**
- Modify: `api/main.py`
- Test: `tests/integration/test_fish_audio.py`
- Optional docs: `docs/MODEL_UNLOAD.md`

- [ ] **Step 1: Update `/v1/models/unload` behavior for `s2-pro`**

Current behavior only clears the app-layer Fish handle. Extend it so that when unloading `s2-pro` and no Fish request is active, it also stops the `fish-sglang` container.

Run the unload decision through the same lifecycle-state lock used by the idle watcher so manual unload and request-start cannot race each other.

- [ ] **Step 2: Keep active-job safeguards**

Keep the existing running-job checks for Qwen-managed unloads, and add the Fish in-flight request guard so manual unload does not stop Fish mid-request.

Scope this carefully:
- For `model_type == "s2-pro"`, gate only on active Fish sync requests. Do not block on unrelated Qwen Redis jobs.
- For Qwen model unloads, keep the existing Redis running-job checks.
- For `model_type is None` (unload all), apply both guards because the request spans both backends.

- [ ] **Step 3: Add or update tests**

```python
@pytest.mark.asyncio
async def test_models_unload_s2_pro_stops_container_when_idle():
    ...
```

- [ ] **Step 4: Validate the end-to-end lifecycle manually**

```bash
# 1. Confirm fish-sglang is stopped
docker ps -a --filter name=fish-sglang

# 2. Call the sync Fish endpoint
curl -X POST http://localhost:8000/v1/tts/s2-pro/sync -H "Content-Type: application/json" -d '{"text":"hello"}'

# 3. Confirm fish-sglang is running
docker ps --filter name=fish-sglang

# 4. Wait past idle timeout and confirm fish-sglang stops
docker ps -a --filter name=fish-sglang
```

> **Note:** Replace `localhost:8000` with your actual API host/port if different.

- [ ] **Step 5: Run focused tests**

```bash
pytest tests/integration/test_fish_audio.py -v
```

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/integration/test_fish_audio.py docs/MODEL_UNLOAD.md
git commit -m "feat: stop fish-sglang on manual or idle unload"
```

---

## Acceptance Criteria

- [ ] The API can start successfully while `fish-sglang` is stopped.
- [ ] The first `POST /v1/tts/s2-pro/sync` starts `fish-sglang`, waits for health, and then succeeds.
- [ ] The cold-start readiness path does not block the FastAPI event loop; blocking container control runs in a worker thread.
- [ ] Subsequent Fish sync requests reuse the already running container.
- [ ] Controller start/stop transitions are serialized so concurrent requests, manual unload, and the idle watcher cannot interleave container lifecycle actions unsafely.
- [ ] After the configured idle timeout and with no Fish request in flight, the API clears the Fish app-layer state and stops the `fish-sglang` container.
- [ ] Manual unload of `s2-pro` also stops the container when safe.
- [ ] Docker SDK failures are normalized inside the controller so the API returns 503s instead of leaking raw Docker exceptions.
- [ ] Deployment steps explicitly handle Docker socket permissions for the non-root API user, or the plan falls back to a supervisor service instead.
- [ ] No Fish lifecycle logic is added to the Qwen Redis worker loop.
- [ ] Tests reflect the real architecture: remote Fish client plus Docker-managed container lifecycle.
- [ ] `/v1/models/status` distinguishes app-layer Fish state from container-level Fish state.
- [ ] The restart behavior is explicit: after API restart, an already-running `fish-sglang` container is not treated as idle unless the implementation adds explicit reconciliation beyond process-local state.
- [ ] The API runs with a single Uvicorn worker (`--workers 1`) as enforced in `compose.yaml`.
- [ ] Docker socket is verified accessible from the API container before deployment.

---

## Notes

- The cold-start path will be slower than the warm path because the container must boot, load weights, and pass its health check.
- The cold-start path must run off the event loop; otherwise a single slow container start can stall unrelated API traffic.
- Mounting the Docker socket into the API container is powerful and should be treated as a privileged operation.
- This plan intentionally does not stop a `fish-sglang` container that predates the current API process unless ownership/last-activity reconciliation is added explicitly.
- If Docker socket access is rejected, create a narrow supervisor service that exposes `start fish`, `stop fish`, and `status fish` endpoints, then swap the controller to call that service instead of Docker directly.
- The in-flight request counter by itself is insufficient. Use a shared lifecycle-state lock for request begin/end bookkeeping and unload decisions, and a controller-local lock for Docker start/stop transitions.
- The in-flight request counter uses process-local threading state. **REQUIRED:** The API must run with a single Uvicorn worker (`--workers 1`). This is enforced in `compose.yaml` via the command override. If multiple workers are needed, replace the process-local counter with Redis-backed counting using the existing Redis connection.
- The idle watcher runs as a daemon thread. Consider adding a `/v1/health` endpoint that checks the watcher thread is alive and logs a warning if it has exited unexpectedly.

---

## Troubleshooting

### Container fails to start on first request
- **Symptom:** 503 response, timeout error in logs
- **Check:** Verify `fish-sglang` container definition exists in Compose: `docker compose ps -a`
- **Check:** Verify Docker socket is mounted in API container: `/var/run/docker.sock:/var/run/docker.sock`
- **Check:** Verify container has healthcheck defined in Compose

### Container stops unexpectedly during idle
- **Symptom:** Fish requests fail after period of inactivity
- **Check:** Verify idle watcher is running: check API logs for `_run_fish_idle_watcher`
- **Check:** Verify `FISH_IDLE_UNLOAD_SECONDS` is set appropriately
- **Check:** Verify no active Fish requests when container stops (check in-flight counter)

### fish-sglang stays running after API restart
- **Symptom:** `fish-sglang` remains up even though no new Fish request has run since API restart
- **Cause:** The watcher only has process-local Fish activity state; it cannot safely infer pre-restart idleness
- **Fix:** Trigger a manual unload or add explicit reconciliation/persisted last-activity tracking before stopping pre-existing containers automatically

### API fails to start with fish-sglang dependency error
- **Symptom:** API container fails to boot
- **Fix:** Remove or relax `depends_on` in compose.yaml as described in Chunk 4, Step 3

### Docker socket permission denied
- **Symptom:** `PermissionError: [Errno 13] /var/run/docker.sock`
- **Fix:** Ensure API container runs with appropriate group or UID that has Docker socket access

### Idle watcher not running
- **Symptom:** Fish container never stops after idle timeout
- **Check:** Verify `_run_fish_idle_watcher` thread was started at API boot (check logs for "Starting Fish idle watcher")
- **Check:** Verify `FISH_IDLE_UNLOAD_SECONDS` is set > 0
- **Check:** Verify API is running with single worker (`--workers 1`)