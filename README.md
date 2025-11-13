# BananaPics

## Webcam Background Remover

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the capture script:
   ```bash
   python capture_selfie_segmentation.py
   ```

The OpenCV window guides you through a fully automated sequence—no button presses required. It shows messages such as:

- Face forward (3 photos)
- Turn head left (3 photos)
- Turn head right (3 photos)
- Tilt head down (3 photos)
- Tilt head up (3 photos)
- Look forward and smile (3 photos)
- Show an angry expression (3 photos)
- Open your mouth (3 photos)

Each capture is background-removed with MediaPipe Selfie Segmentation and saved to `segmented_photos` (created automatically). Use optional flags to customize behavior:

- `--output-dir PATH` – destination folder.
- `--bg-color B G R` – replacement background color (default white).
- `--warmup-seconds N` – prep time before each pose (default 4).
- `--between-shots-seconds N` – delay between photos inside a pose (default 1.5).
- `--photos-per-pose N` – number of photos per instruction (default 3).
- `--model {0,1}` – MediaPipe model to use.
- Press `Q`/`ESC` at any time to abort the session.

## Extract Identity Embeddings

After recording poses, run the embedding extractor (requires InsightFace + ONNXRuntime from `requirements.txt`):

```bash
python extract_embeddings.py
```

By default it scans `segmented_photos`, runs InsightFace on each image, and writes `embeddings/embeddings.json` (set `--output-dir` or `--json-name` to change this). Use `--save-npy` to dump one `.npy` per image, `--ctx-id` to pick GPU/CPU (`-1` is CPU), and `--min-score` to control the face-confidence threshold.

## InstantID Image Generation

To turn the stored embeddings into new renders, first download the official InstantID assets:

1. Run the helper to pull the official weights from Hugging Face (needs network access):
   ```bash
   python setup_instantid_assets.py
   ```
   This downloads `ip-adapter.bin` plus `ControlNetModel/config.json` and `ControlNetModel/diffusion_pytorch_model.safetensors` into the `checkpoints/` folder. Re-run with `--force` to refresh or use `--local-root` to target a different directory.
2. The repository already includes `pipeline_stable_diffusion_xl_instantid.py` (sourced from the [InstantID project](https://github.com/InstantID/InstantID)). Replace it with an updated upstream copy whenever you upgrade InstantID.

Then run:

```bash
python run_instantid.py --controlnet checkpoints/ControlNetModel --ip-adapter checkpoints/ip-adapter.bin
```

The script will:

- Ask you to choose which segmented photo/embedding to use.
- Prompt you to select one of the built-in expressions (`soft smile`, `smile`, `angry`, `serious`, etc.) and builds `a portrait photo of me with a {expression}, realistic lighting, 4k`.
- Load the InstantID pipeline with your checkpoints and generate an image saved to `instantid_results/`.

Flags such as `--expression`, `--image-index`, `--prompt-template`, `--steps`, or `--device` let you automate the run or fine-tune InstantID settings. Running InstantID is GPU-intensive; enable `--enable-offload`/`--vae-tiling` if you must fall back to CPU.
