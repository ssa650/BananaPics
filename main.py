#!/usr/bin/env python3
"""Orchestrate the full BananaPics pipeline end-to-end."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Set
import shutil

BASE_DIR = Path(__file__).resolve().parent
SEGMENTED_DIR = BASE_DIR / "segmented_photos"
INSTANTID_DIR = BASE_DIR / "instantid_results"
PIPELINE = (
    ("Webcam capture & segmentation", "capture_selfie_segmentation.py"),
    ("Embedding extraction", "extract_embeddings.py"),
    ("InstantID generation", "run_instantid.py"),
)


def run_step(label: str, script: str) -> None:
    script_path = BASE_DIR / script
    if not script_path.exists():
        raise SystemExit(f"Required script missing: {script_path}")

    print(f"\n=== {label} ===")
    print(f"Running: {script_path}")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{label} failed with exit code {exc.returncode}.") from exc


def list_top_level_files(directory: Path) -> Set[Path]:
    if not directory.exists():
        return set()
    return {path.resolve() for path in directory.iterdir() if path.is_file()}


def archive_instantid_outputs(new_files: Iterable[Path]) -> Path | None:
    files = sorted(new_files)
    if not files:
        return None

    INSTANTID_DIR.mkdir(parents=True, exist_ok=True)
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = INSTANTID_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        target = session_dir / Path(path).name
        shutil.move(str(path), target)
    return session_dir


def clean_segmented_photos() -> None:
    if SEGMENTED_DIR.exists():
        shutil.rmtree(SEGMENTED_DIR)
    SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cleared segmented photos in {SEGMENTED_DIR}")


def main() -> None:
    print("Starting BananaPics pipeline...")
    instantid_before: Set[Path] = set()
    archived_dir: Path | None = None

    for label, script in PIPELINE:
        if script == "run_instantid.py":
            instantid_before = list_top_level_files(INSTANTID_DIR)
        run_step(label, script)
        if script == "run_instantid.py":
            instantid_after = list_top_level_files(INSTANTID_DIR)
            new_files = instantid_after - instantid_before
            archived_dir = archive_instantid_outputs(new_files)
            if archived_dir:
                print(f"InstantID outputs moved to {archived_dir}")
                clean_segmented_photos()
            else:
                print("InstantID produced no new files; skipping cleanup.")

    print("\nAll steps completed successfully!")


if __name__ == "__main__":
    main()
