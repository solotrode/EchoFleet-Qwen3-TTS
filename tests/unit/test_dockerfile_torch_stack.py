from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"


def _extract_package_version(contents: str, package_name: str) -> str:
    match = re.search(rf"{re.escape(package_name)}==([^\s\\]+)", contents)
    assert match is not None, f"{package_name} must be pinned in Dockerfile"
    return match.group(1)


def test_dockerfile_pins_compatible_torch_stack_versions() -> None:
    contents = DOCKERFILE_PATH.read_text(encoding="utf-8")

    torch_version = _extract_package_version(contents, "torch")
    torchvision_version = _extract_package_version(contents, "torchvision")
    torchaudio_version = _extract_package_version(contents, "torchaudio")

    expected_pairs = {
        "2.10.0+cu128": ("0.25.0+cu128", "2.10.0+cu128"),
    }

    assert torch_version in expected_pairs, f"Add compatibility mapping for torch {torch_version}"
    assert (torchvision_version, torchaudio_version) == expected_pairs[torch_version]
