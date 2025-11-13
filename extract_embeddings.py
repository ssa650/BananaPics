"""Extract InsightFace identity embeddings for captured photos."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class EmbeddingResult:
    image_path: Path
    embedding: np.ndarray
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk through segmented photos, run InsightFace, and export "
            "identity embeddings."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("segmented_photos"),
        help="Directory that contains the segmented images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("embeddings"),
        help="Directory to write embedding artifacts.",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="embeddings.json",
        help="Filename for the JSON summary (written inside --output-dir).",
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="InsightFace context id (-1 for CPU, >=0 for specific GPU).",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Square detection size to pass into FaceAnalysis (pixels).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Minimum detection score required to keep a face.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also write one .npy file per embedding.",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> Sequence[Path]:
    images = [
        path
        for path in sorted(input_dir.iterdir())
        if path.suffix.lower() in VALID_IMAGE_EXTS and path.is_file()
    ]
    return images


def init_face_analysis(ctx_id: int, det_size: int) -> FaceAnalysis:
    providers = None
    if ctx_id < 0:
        providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def choose_face(
    faces: Iterable, min_score: float
) -> Optional[Tuple[np.ndarray, float]]:
    best_face = None
    best_score = float("-inf")
    for face in faces:
        score = float(getattr(face, "det_score", 0.0) or 0.0)
        if score > best_score:
            best_face = face
            best_score = score

    if best_face is None or best_score < min_score:
        return None

    embedding = getattr(best_face, "normed_embedding", None)
    if embedding is None:
        return None

    return np.asarray(embedding, dtype=np.float32), best_score


def extract_embeddings(
    app: FaceAnalysis,
    images: Sequence[Path],
    min_score: float,
) -> List[EmbeddingResult]:
    results: List[EmbeddingResult] = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] Could not read image: {path}")
            continue
        faces = app.get(image)
        selection = choose_face(faces, min_score)
        if selection is None:
            print(f"[WARN] No face meeting score threshold in {path.name}")
            continue
        embedding, score = selection
        results.append(EmbeddingResult(path, embedding, score))
        print(f"[OK] {path.name}: score={score:.3f}, dim={embedding.shape[0]}")
    return results


def save_results(
    results: Sequence[EmbeddingResult],
    output_dir: Path,
    json_name: str,
    save_npy: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "count": len(results),
        "embedding_dim": int(results[0].embedding.shape[0]) if results else None,
        "files": [
            {
                "file": result.image_path.name,
                "embedding": result.embedding.tolist(),
                "det_score": float(result.score),
            }
            for result in results
        ],
    }
    json_path = output_dir / json_name
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if save_npy:
        for result in results:
            npy_path = output_dir / f"{result.image_path.stem}.npy"
            np.save(npy_path, result.embedding)

    return json_path


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory {args.input_dir} does not exist.")

    images = list_images(args.input_dir)
    if not images:
        raise SystemExit(f"No supported images found in {args.input_dir}.")

    print(
        f"Processing {len(images)} images from {args.input_dir} "
        f"(ctx_id={args.ctx_id}, det_size={args.det_size})"
    )
    app = init_face_analysis(args.ctx_id, args.det_size)
    results = extract_embeddings(app, images, args.min_score)
    if not results:
        raise SystemExit("No embeddings were generated.")

    json_path = save_results(results, args.output_dir, args.json_name, args.save_npy)
    print(f"Saved {len(results)} embeddings to {json_path}")


if __name__ == "__main__":
    main()
