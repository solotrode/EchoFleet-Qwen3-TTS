#!/usr/bin/env python3
"""Download a Hugging Face repo into the local `models/` folder.

Usage examples:
  # with env token
  HUGGINGFACE_HUB_TOKEN=xxx python scripts/download_model.py --repo Qwen/Qwen3-TTS-12Hz-1.7B-Base

  # with explicit token
  python scripts/download_model.py --repo Qwen/Qwen3-TTS-12Hz-1.7B-Base --token xxx

This script uses `huggingface_hub.snapshot_download` to produce a local folder
that matches the repository layout, suitable for use as a model cache directory
inside the container (`/models/<repo-id>`).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from utils.logging import get_logger

logger = get_logger(__name__)

try:
    from huggingface_hub import snapshot_download
except Exception as e:  # pragma: no cover - runtime import error
    logger.error(
        "Missing dependency: huggingface_hub. Install with 'pip install huggingface-hub'",
        extra={"error": str(e)},
    )
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download HF repo to ./models/<repo>")
    p.add_argument(
        "--repo", required=False, help="Hugging Face repo id (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base)"
    )
    p.add_argument("--out", default="models", help="Output base folder (default: models)")
    p.add_argument(
        "--file", default=None, help="Path to file with one repo id per line (batch mode)"
    )
    p.add_argument("--revision", default=None, help="Repo revision/commit/tag to download")
    p.add_argument(
        "--token",
        default=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token (or set HUGGINGFACE_HUB_TOKEN)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_base: str = args.out
    revision: Optional[str] = args.revision
    token: Optional[str] = args.token

    repos = []
    if args.file:
        if not os.path.exists(args.file):
            logger.error("File not found", extra={"file": args.file})
            sys.exit(2)
        with open(args.file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    repos.append(line)
    elif args.repo:
        repos.append(args.repo)
    else:
        logger.error("No repo specified. Use --repo or --file.")
        sys.exit(2)

    for repo_id in repos:
        target_dir = os.path.join(out_base, repo_id)
        os.makedirs(target_dir, exist_ok=True)

        logger.info("Downloading repo", extra={"repo": repo_id, "target_dir": target_dir})
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                token=token,
            )
        except Exception as exc:
            logger.error("Download failed", extra={"repo": repo_id, "error": str(exc)})
            # continue with other repos instead of aborting all
            continue

        logger.info("Download completed", extra={"repo": repo_id})


if __name__ == "__main__":
    main()
