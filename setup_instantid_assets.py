#!/usr/bin/env python3
"""Download required InstantID assets from Hugging Face."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "InstantX/InstantID"
ASSETS = {
    "ip_adapter": ("ip-adapter.bin", Path("checkpoints/ip-adapter.bin")),
    "controlnet_config": (
        "ControlNetModel/config.json",
        Path("checkpoints/ControlNetModel/config.json"),
    ),
    "controlnet_weights": (
        "ControlNetModel/diffusion_pytorch_model.safetensors",
        Path("checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the InstantID ControlNet + IP-Adapter weights."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help="Hugging Face repo that hosts the InstantID checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("."),
        help="Root directory where checkpoints will be stored.",
    )
    return parser.parse_args()


def download_asset(repo_id: str, filename: str, target: Path, force: bool) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(f"[SKIP] {target} already exists.")
        return target

    print(f"[GET] {filename} -> {target}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target.parent),
        local_dir_use_symlinks=False,
    )
    if not target.exists():
        raise RuntimeError(f"Download reported success but {target} was not found.")
    return target


def main() -> None:
    args = parse_args()
    root = args.local_root.resolve()
    success = True

    for _, (filename, rel_target) in ASSETS.items():
        target = (root / rel_target).resolve()
        try:
            download_asset(args.repo_id, filename, target, args.force)
        except Exception as exc:  # pragma: no cover - network errors
            success = False
            print(f"[ERR] Failed to download {filename}: {exc}")

    if not success:
        raise SystemExit(1)

    print("\nInstantID assets are ready!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
