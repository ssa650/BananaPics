"""Generate InstantID renders from stored InsightFace embeddings."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis

try:
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "StableDiffusionXLInstantIDPipeline is unavailable. "
        "Download pipeline_stable_diffusion_xl_instantid.py from the official InstantID repo "
        "and place it in this project directory (or ensure your diffusers install exposes the class)."
    ) from exc

PROMPT_TEMPLATE = "a portrait photo of me with a {expression}, realistic lighting, 4k"
DEFAULT_EXPRESSIONS: Sequence[str] = (
    "soft smile",
    "smile",
    "serious",
    "angry",
    "confident",
    "relaxed",
    "thoughtful",
)


@dataclass
class EmbeddingRecord:
    filename: str
    embedding: np.ndarray
    det_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick an InsightFace embedding, craft an InstantID prompt based on a user-selected "
            "expression, and run the InstantID pipeline."
        )
    )
    parser.add_argument(
        "--embeddings-file",
        type=Path,
        default=Path("embeddings/embeddings.json"),
        help="JSON output produced by extract_embeddings.py.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("segmented_photos"),
        help="Directory containing the original segmented photos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("instantid_results"),
        help="Directory that will receive generated renders.",
    )
    parser.add_argument(
        "--expression",
        type=str,
        default=None,
        help="Bypass the interactive expression picker with a custom value.",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=None,
        help="Skip the interactive photo selection and use this 0-based index.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Stable Diffusion XL base model or local path.",
    )
    parser.add_argument(
        "--controlnet",
        type=str,
        default="checkpoints/ControlNetModel",
        help="Path or repo id for the InstantID ControlNet weights.",
    )
    parser.add_argument(
        "--ip-adapter",
        type=Path,
        default=Path("checkpoints/ip-adapter.bin"),
        help="Path to the InstantID IP-Adapter checkpoint (ip-adapter.bin).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the diffusion pipeline (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32", "bf16"),
        default="fp16",
        help="Torch dtype for the pipeline.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion inference steps.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=0.8,
        help="ControlNet conditioning scale.",
    )
    parser.add_argument(
        "--ip-adapter-scale",
        type=float,
        default=0.8,
        help="InstantID IP-Adapter scale.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="ugly, deformed, blurry, low quality, text artifacts",
        help="Negative prompt passed to the pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 to randomize every run).",
    )
    parser.add_argument(
        "--insightface-ctx",
        type=int,
        default=0,
        help="InsightFace context id (-1 for CPU).",
    )
    parser.add_argument(
        "--insightface-det-size",
        type=int,
        default=640,
        help="Detector input size for InsightFace (applies to both dimensions).",
    )
    parser.add_argument(
        "--insightface-root",
        type=Path,
        default=None,
        help="Optional root directory containing downloaded InsightFace models.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime providers for InsightFace.",
    )
    parser.add_argument(
        "--enable-offload",
        action="store_true",
        help="Enable Diffusers CPU offload helpers to reduce VRAM usage.",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable VAE tiling (useful for limited VRAM).",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=PROMPT_TEMPLATE,
        help="Template used to build the final text prompt (must contain {expression}).",
    )
    return parser.parse_args()


def load_embedding_records(path: Path) -> List[EmbeddingRecord]:
    if not path.exists():
        raise SystemExit(f"Embeddings file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records: List[EmbeddingRecord] = []
    for entry in payload.get("files", []):
        vec = np.asarray(entry.get("embedding", []), dtype=np.float32)
        if vec.size == 0:
            continue
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        records.append(
            EmbeddingRecord(
                filename=entry.get("file"),
                embedding=vec,
                det_score=float(entry.get("det_score", 0.0)),
            )
        )
    if not records:
        raise SystemExit(f"No embeddings found in {path}")
    return records


def prompt_for_record(records: Sequence[EmbeddingRecord], image_index: int | None) -> EmbeddingRecord:
    if image_index is not None:
        if 0 <= image_index < len(records):
            return records[image_index]
        raise SystemExit(f"--image-index {image_index} is out of range (0-{len(records)-1})")

    print("\nAvailable embeddings:")
    for idx, record in enumerate(records, start=1):
        print(f"  [{idx:02d}] {record.filename}  (score={record.det_score:.3f})")
    selection = input("Select an image by number [1]: ").strip()
    idx = 0
    if selection:
        if not selection.isdigit():
            raise SystemExit("Selection must be a number.")
        idx = int(selection) - 1
    if not 0 <= idx < len(records):
        raise SystemExit("Selection is out of range.")
    return records[idx]


def prompt_for_expression(expression: str | None) -> str:
    if expression:
        return expression.strip()

    print("\nChoose an expression for the prompt:")
    for idx, option in enumerate(DEFAULT_EXPRESSIONS, start=1):
        print(f"  [{idx}] {option}")
    choice = input("Expression [1]: ").strip().lower()
    if not choice:
        return DEFAULT_EXPRESSIONS[0]
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(DEFAULT_EXPRESSIONS):
            return DEFAULT_EXPRESSIONS[idx]
        raise SystemExit("Expression choice is out of range.")
    return choice


def draw_kps(image: Image.Image, kps: np.ndarray) -> Image.Image:
    """Render InsightFace 5-keypoint landmarks into a control image."""
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.asarray(kps, dtype=np.float32)

    w, h = image.size
    canvas = np.zeros([h, w, 3], dtype=np.uint8)
    stickwidth = 4
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    for pair in limb_seq:
        color = colors[pair[0] % len(colors)]
        joint = kps[pair]
        x = joint[:, 0]
        y = joint[:, 1]
        length = math.hypot(x[0] - x[1], y[0] - y[1])
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1,
        )
        canvas = cv2.fillConvexPoly(canvas, polygon, color)
    canvas = (canvas * 0.6).astype(np.uint8)
    for idx, kp in enumerate(kps):
        color = colors[idx % len(colors)]
        canvas = cv2.circle(canvas, (int(kp[0]), int(kp[1])), 8, color, -1)
    return Image.fromarray(canvas)


def init_face_analyzer(args: argparse.Namespace) -> FaceAnalysis:
    kwargs = {
        "name": "antelopev2",
        "providers": args.providers,
    }
    if args.insightface_root is not None:
        kwargs["root"] = str(args.insightface_root)
    app = FaceAnalysis(**kwargs)
    app.prepare(ctx_id=args.insightface_ctx, det_size=(args.insightface_det_size, args.insightface_det_size))
    return app


def build_control_image(app: FaceAnalysis, image_path: Path) -> Image.Image:
    pil_image = Image.open(image_path).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        raise SystemExit(f"No face detected in {image_path}")

    def face_area(face) -> float:
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            bbox = face["bbox"]
        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    best_face = max(faces, key=face_area)
    kps = getattr(best_face, "kps", None)
    if kps is None:
        kps = getattr(best_face, "landmark_2d_106", None)
        if kps is not None:
            kps = kps[:5]
    if kps is None:
        raise SystemExit("Unable to extract landmarks for InstantID control image.")
    return draw_kps(pil_image, kps)


def load_pipeline(args: argparse.Namespace) -> StableDiffusionXLInstantIDPipeline:
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch_dtype)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe.to(args.device)
    pipe.load_ip_adapter_instantid(str(args.ip_adapter))
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    if args.enable_offload:
        pipe.enable_model_cpu_offload()
    if args.vae_tiling:
        pipe.enable_vae_tiling()
    return pipe


def main() -> None:
    args = parse_args()
    if not args.reference_dir.exists():
        raise SystemExit(f"Reference directory not found: {args.reference_dir}")
    if not args.ip_adapter.exists():
        raise SystemExit(f"IP-Adapter checkpoint not found: {args.ip_adapter}")

    records = load_embedding_records(args.embeddings_file)
    record = prompt_for_record(records, args.image_index)
    expression = prompt_for_expression(args.expression)
    prompt = args.prompt_template.format(expression=expression)

    reference_path = args.reference_dir / record.filename
    if not reference_path.exists():
        raise SystemExit(f"Reference image not found: {reference_path}")

    print(f"\nUsing {record.filename} (score={record.det_score:.3f})")
    print(f"Prompt: {prompt}")

    app = init_face_analyzer(args)
    control_image = build_control_image(app, reference_path)

    print("\nLoading InstantID pipeline (this can take a while the first time)...")
    pipe = load_pipeline(args)

    generator = None
    if args.seed >= 0:
        generator = torch.Generator(device=args.device)
        generator = generator.manual_seed(args.seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        image_embeds=record.embedding,
        image=control_image,
        controlnet_conditioning_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).images[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_expression = expression.replace(" ", "_")
    stem = Path(record.filename).stem
    output_path = args.output_dir / f"instantid_{stem}_{safe_expression}_{timestamp}.png"
    result.save(output_path)
    print(f"\nSaved InstantID render to {output_path}")


if __name__ == "__main__":
    main()
