#!/usr/bin/env python3
"""Automated multi-pose capture with MediaPipe background removal."""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

WINDOW_NAME = "Pose Capture (press Q/ESC to abort)"
POSE_SEQUENCE: Sequence[Tuple[str, str]] = (
    ("Face forward", "face_forward"),
    ("Turn head to the left", "head_left"),
    ("Turn head to the right", "head_right"),
    ("Tilt head down", "head_down"),
    ("Tilt head up", "head_up"),
    ("Look forward and smile", "smile"),
    ("Show an angry expression", "angry"),
    ("Open your mouth", "mouth_open"),
)


class CaptureAborted(Exception):
    """Raised when the user aborts the capture loop."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate a multi-pose capture session, remove the background, "
            "and save each segmented frame."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmented_photos"),
        help="Directory where the segmented images are saved.",
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=(0, 1),
        default=1,
        help="MediaPipe Selfie Segmentation model selection.",
    )
    parser.add_argument(
        "--bg-color",
        nargs=3,
        type=int,
        default=(255, 255, 255),
        metavar=("B", "G", "R"),
        help="Background color (BGR) to use when removing the background.",
    )
    parser.add_argument(
        "--photos-per-pose",
        type=int,
        default=3,
        help="How many photos to capture for each instruction.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=4.0,
        help="Seconds to give the user before the first photo of each pose.",
    )
    parser.add_argument(
        "--between-shots-seconds",
        type=float,
        default=1.5,
        help="Seconds between consecutive photos within the same pose.",
    )
    parser.add_argument(
        "--feedback-ms",
        type=int,
        default=400,
        help="Milliseconds to display the 'photo captured' feedback.",
    )
    return parser.parse_args()


def check_abort(key: int) -> None:
    if key in (27, ord("q")):
        raise CaptureAborted("Capture aborted by user.")


def overlay_text(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    overlay = frame.copy()
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, min(1.2, w / 800))
    thickness = 2
    line_height = int(40 * font_scale)
    start_y = h // 2 - ((len(lines) - 1) * line_height) // 2

    for idx, text in enumerate(lines):
        if not text:
            continue
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = start_y + idx * line_height
        cv2.putText(
            overlay, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
        )
    return overlay


def read_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from the webcam.")
    return frame


def countdown(
    cap: cv2.VideoCapture, instruction: str, seconds: float
) -> None:
    if seconds <= 0:
        return

    end_time = time.time() + seconds
    while time.time() < end_time:
        remaining = math.ceil(end_time - time.time())
        frame = read_frame(cap)
        display = overlay_text(
            frame, (instruction, f"Capturing starts in {remaining}s")
        )
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        check_abort(key)


def wait_for_capture(
    cap: cv2.VideoCapture,
    instruction: str,
    shot_idx: int,
    total_shots: int,
    delay_seconds: float,
) -> np.ndarray:
    if delay_seconds > 0:
        end_time = time.time() + delay_seconds
        while time.time() < end_time:
            remaining = max(0, math.ceil(end_time - time.time()))
            frame = read_frame(cap)
            display = overlay_text(
                frame,
                (
                    instruction,
                    f"Photo {shot_idx}/{total_shots} in {remaining}s",
                ),
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            check_abort(key)

    frame = read_frame(cap)
    display = overlay_text(
        frame,
        (
            instruction,
            f"Capturing photo {shot_idx}/{total_shots}",
        ),
    )
    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(1) & 0xFF
    check_abort(key)
    return frame


def display_feedback(
    frame: np.ndarray,
    instruction: str,
    shot_idx: int,
    total_shots: int,
    feedback_ms: int,
) -> None:
    display = overlay_text(
        frame,
        (
            instruction,
            f"Captured photo {shot_idx}/{total_shots}",
        ),
    )
    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(max(1, feedback_ms)) & 0xFF
    check_abort(key)


def segment_foreground(
    frame: np.ndarray,
    selfie: mp.solutions.selfie_segmentation.SelfieSegmentation,
    bg_color: Iterable[int],
) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = selfie.process(rgb_frame)

    if result.segmentation_mask is None:
        raise RuntimeError("Failed to compute segmentation mask.")

    mask = result.segmentation_mask
    condition = mask > 0.1
    background = np.zeros_like(frame, dtype=np.uint8)
    background[:] = np.array(bg_color, dtype=np.uint8)
    segmented = np.where(condition[..., None], frame, background)
    return segmented


def save_image(
    image: np.ndarray,
    output_dir: Path,
    pose_slug: str,
    shot_idx: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{pose_slug}_{shot_idx}.png"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), image)
    return output_path


def capture_pose_sequence(
    cap: cv2.VideoCapture,
    selfie: mp.solutions.selfie_segmentation.SelfieSegmentation,
    instruction: str,
    pose_slug: str,
    photos_per_pose: int,
    warmup_seconds: float,
    between_shots_seconds: float,
    feedback_ms: int,
    bg_color: Tuple[int, int, int],
    output_dir: Path,
) -> int:
    print(f"\nInstruction: {instruction}")
    countdown(cap, instruction, warmup_seconds)

    saved = 0
    for shot_idx in range(1, photos_per_pose + 1):
        delay = 0.0 if shot_idx == 1 else max(0.0, between_shots_seconds)
        frame = wait_for_capture(
            cap, instruction, shot_idx, photos_per_pose, delay
        )
        segmented = segment_foreground(frame, selfie, bg_color)
        output_path = save_image(segmented, output_dir, pose_slug, shot_idx)
        saved += 1
        display_feedback(frame, instruction, shot_idx, photos_per_pose, feedback_ms)
        print(f"  Saved: {output_path}")

    return saved


def run_capture_session(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access the webcam. Is it connected and free?")

    total_saved = 0
    bg_color: Tuple[int, int, int] = tuple(args.bg_color)  # type: ignore[assignment]

    try:
        with mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=args.model
        ) as selfie:
            for instruction, pose_slug in POSE_SEQUENCE:
                total_saved += capture_pose_sequence(
                    cap=cap,
                    selfie=selfie,
                    instruction=instruction,
                    pose_slug=pose_slug,
                    photos_per_pose=args.photos_per_pose,
                    warmup_seconds=max(0.0, args.warmup_seconds),
                    between_shots_seconds=max(0.0, args.between_shots_seconds),
                    feedback_ms=max(1, args.feedback_ms),
                    bg_color=bg_color,
                    output_dir=args.output_dir,
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nCapture complete. Saved {total_saved} segmented photos to {args.output_dir}")


def main() -> None:
    args = parse_args()
    try:
        run_capture_session(args)
    except CaptureAborted as exc:
        print(exc)
    except RuntimeError as exc:
        print(exc)


if __name__ == "__main__":
    main()
