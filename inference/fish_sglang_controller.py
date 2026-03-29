"""Fish SGLang container lifecycle controller.

This module provides a controller that manages the Docker container lifecycle
for the Fish Audio S2 Pro SGLang server.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from docker import DockerClient
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
        docker_client: Optional[DockerClient] = None,
    ) -> None:
        self._container_name = container_name
        self._startup_timeout_seconds = startup_timeout_seconds
        self._stop_timeout_seconds = stop_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        if docker_client is not None:
            self._docker = docker_client
        elif docker_socket_path:
            self._docker = DockerClient(base_url=f"unix://{docker_socket_path}")
        else:
            self._docker = DockerClient(base_url="unix:///var/run/docker.sock")
        self._lifecycle_lock = threading.Lock()

    def ensure_running(self) -> None:
        """Ensure the fish-sglang container is running and healthy.

        Raises:
            TimeoutError: If the container does not become healthy within the timeout.
            RuntimeError: If Docker is unreachable or the container cannot be started.
        """
        with self._lifecycle_lock:
            try:
                container = self._get_container()
                container.reload()

                if container.status != "running":
                    logger.info(
                        "Starting fish-sglang container", extra={"container": self._container_name}
                    )
                    container.start()

                self._wait_until_healthy(container)
            except TimeoutError:
                raise
            except DockerException as e:
                raise RuntimeError("Failed to start or inspect Fish container") from e

    def stop_if_running(self) -> bool:
        """Stop the fish-sglang container if it is running.

        Returns:
            True if the container was stopped, False if it was not running.

        Raises:
            RuntimeError: If Docker is unreachable or the container cannot be stopped.
        """
        with self._lifecycle_lock:
            try:
                container = self._get_container()
                container.reload()

                if container.status != "running":
                    return False

                logger.info(
                    "Stopping fish-sglang container", extra={"container": self._container_name}
                )
                container.stop(timeout=self._stop_timeout_seconds)
                return True
            except DockerException as e:
                raise RuntimeError("Failed to stop or inspect Fish container") from e

    def is_running(self) -> bool:
        """Check if the fish-sglang container is currently running.

        Returns:
            True if the container is running, False otherwise.

        Raises:
            RuntimeError: If Docker is unreachable.
        """
        try:
            container = self._get_container()
            container.reload()
            return container.status == "running"
        except DockerException as e:
            raise RuntimeError("Failed to inspect Fish container state") from e

    def get_container_info(self) -> Dict[str, Any]:
        """Get detailed information about the container.

        Returns:
            Dict containing container status, health, and other details.

        Raises:
            RuntimeError: If Docker is unreachable or container not found.
        """
        try:
            container = self._get_container()
            container.reload()
            state = container.attrs.get("State", {})
            health = state.get("Health", {})

            return {
                "container_name": self._container_name,
                "status": container.status,
                "health_status": health.get("Status") if health else None,
                "started_at": state.get("StartedAt"),
                "finished_at": state.get("FinishedAt"),
                "exit_code": state.get("ExitCode"),
            }
        except NotFound:
            return {
                "container_name": self._container_name,
                "status": "not_found",
                "error": f"Container {self._container_name} not found",
            }
        except DockerException as e:
            raise RuntimeError("Failed to get container info") from e

    def _get_container(self):
        """Get the Docker container object.

        Returns:
            The Docker container object.

        Raises:
            RuntimeError: If the container is not found or Docker is unreachable.
        """
        try:
            return self._docker.containers.get(self._container_name)
        except NotFound as e:
            raise RuntimeError(f"Fish container not found: {self._container_name}") from e
        except DockerException as e:
            raise RuntimeError("Failed to access Docker API for Fish container control") from e

    def _wait_until_healthy(self, container) -> None:
        """Wait for the container to become healthy.

        Args:
            container: The Docker container object.

        Raises:
            TimeoutError: If the container does not become healthy within the timeout.
            RuntimeError: If Docker fails while waiting.
        """
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

        raise TimeoutError(
            f"Timed out waiting for Fish container {self._container_name} to become healthy"
        )
