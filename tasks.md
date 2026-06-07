# Handwriting Recognition App — Task List

Tasks are grouped by phase and layer. Complete Phase 0 before starting Phase 1. Phases 1 and 2 can overlap where noted.

---

## Phase 0 — Project Setup

- [x] **P0-1** Initialize monorepo structure: `/frontend`, `/backend`, `/ml`, `/data`, `/notebooks`
- [x] **P0-2** Create root `docker-compose.yml` wiring frontend dev server, backend, and a shared volume for model weights
- [x] **P0-3** Write `backend/Dockerfile` and `frontend/Dockerfile`
- [x] **P0-4** Set up Python virtual environment and `pyproject.toml` (or `requirements.txt`) with: `torch`, `torchvision`, `fastapi`, `uvicorn`, `opencv-python`, `pillow`, `numpy`, `lmdb` (for dataset storage), `wandb` (experiment tracking)
- [x] **P0-5** Scaffold React app in `/frontend` with Vite; install `react-canvas-draw` or implement a raw Canvas component
- [x] **P0-6** Set up a basic FastAPI server at `backend/main.py` with a health-check endpoint `GET /api/health`
- [x] **P0-7** Verify the full Docker stack starts and the frontend can reach the backend health endpoint

---

## Phase 1 — English Recognition + Tamil Basic

### 1A: Data — English

- [x] **P1-A1** Download the **IAM Handwriting Database** (lines split) via Hugging Face (`Teklia/IAM-line`); store in `/data/iam/`
- [x] **P1-A2** Write a dataset loader (`ml/datasets/iam.py`) that reads IAM line images and ground-truth transcriptions into PyTorch `Dataset`
- [x] **P1-A3** Implement data augmentation transforms: rotation, elastic distortion, noise, brightness jitter (`ml/transforms.py`)
- [x] **P1-A4** Build LMDB cache of preprocessed images for fast training I/O
- [x] **P1-A5** Validate the loader: sample 10 items, render them, confirm transcription alignment

### 1B: Data — Tamil Phase 1

- [ ] **P1-B1** Download the **IIIT-Indic Tamil** dataset (or HP Labs Tamil); store in `/data/tamil_basic/`
- [ ] **P1-B2** Write a Tamil character inventory file (`ml/vocab/tamil_phase1.py`) listing all Phase 1 Unicode codepoints
- [ ] **P1-B3** Write a Tamil dataset loader (`ml/datasets/tamil_basic.py`) analogous to the IAM loader
- [ ] **P1-B4** Build a synthetic data generator (`ml/synthetic/tamil_gen.py`) that:
  - Renders random Tamil Phase 1 text strings in handwriting-style fonts (Latha, Noto Sans Tamil, etc.)
  - Applies augmentation (elastic distortion, rotation, noise)
  - Outputs (image, transcription) pairs
- [ ] **P1-B5** Generate ≥ 20,000 synthetic Tamil Phase 1 line images; store in `/data/tamil_basic_synth/`

### 1C: Preprocessing Pipeline

- [ ] **P1-C1** Implement `ml/preprocess.py` with functions: `to_grayscale`, `denoise`, `binarize` (Otsu), `deskew`, `normalize_height(target_h=32)`
- [ ] **P1-C2** Implement line segmentation (`ml/segment.py`) using horizontal projection profiles
- [ ] **P1-C3** Write unit tests for each preprocessing step using sample images
- [ ] **P1-C4** Benchmark pipeline throughput; target ≥ 50 images/second on CPU

### 1D: Model — CRNN

- [ ] **P1-D1** Implement the CNN backbone (`ml/models/cnn_backbone.py`): ResNet-18 or custom VGG-style, output shape `(B, 512, H/8, W/4)`
- [ ] **P1-D2** Implement the full CRNN (`ml/models/crnn.py`): CNN → column-wise feature slices → BiLSTM (2 layers, hidden 256) → linear projection → logits
- [ ] **P1-D3** Implement CTC loss wrapper and beam-search decoder (`ml/decode.py`)
- [ ] **P1-D4** Write vocabulary classes for English and Tamil Phase 1 (`ml/vocab/english.py`, `ml/vocab/tamil_phase1.py`) with `encode`/`decode` methods
- [ ] **P1-D5** Write a unit test: feed a random tensor through the CRNN, confirm output shape matches vocab size

### 1E: Training — English Model

- [ ] **P1-E1** Write training script `ml/train_english.py` with: AdamW optimizer, OneCycleLR scheduler, CTC loss, gradient clipping
- [ ] **P1-E2** Integrate Weights & Biases (wandb) logging: loss, CER, WER per epoch
- [ ] **P1-E3** Train for ≥ 20 epochs; checkpoint best model by validation CER
- [ ] **P1-E4** Evaluate on IAM test set; confirm CER ≤ 8% and WER ≤ 20%
- [ ] **P1-E5** Export model weights to `ml/checkpoints/english_crnn.pt`

### 1F: Training — Tamil Phase 1 Model

- [ ] **P1-F1** Write training script `ml/train_tamil_p1.py` (mirror of English script with Tamil vocab and dataset)
- [ ] **P1-F2** Train on combined real + synthetic Tamil Phase 1 data
- [ ] **P1-F3** Evaluate on held-out Tamil test set; confirm CER ≤ 12%
- [ ] **P1-F4** Export weights to `ml/checkpoints/tamil_p1_crnn.pt`

### 1G: Backend API

- [ ] **P1-G1** Write `backend/inference.py`: loads both models at startup, exposes `recognize(image_bytes, language) -> RecognitionResult`
- [ ] **P1-G2** Implement `POST /api/recognize` endpoint: accept base64 image + language, run preprocessing + segmentation + model inference, return JSON response per API contract in design.md
- [ ] **P1-G3** Add input validation: max image size 10 MB, supported formats (PNG/JPG/WebP), valid language values
- [ ] **P1-G4** Write integration test: send a test handwritten image to the endpoint, assert the response schema is correct
- [ ] **P1-G5** Add CORS headers to allow the frontend dev server to call the API

### 1H: Frontend

- [ ] **P1-H1** Implement `Canvas` component: variable stroke width, color picker (black/blue/red), eraser tool, clear button; renders strokes to a PNG blob on demand
- [ ] **P1-H2** Implement `ImageUpload` component: file input accepting PNG/JPG/PDF; display preview thumbnail
- [ ] **P1-H3** Implement `CameraCapture` component: access `getUserMedia`, display live feed, capture frame as PNG blob
- [ ] **P1-H4** Implement `LanguageToggle` component: switch between English and Tamil; store selection in React state
- [ ] **P1-H5** Implement `RecognitionResult` component: display recognized text, per-line confidence, copy-to-clipboard button; highlight lines with confidence < 0.7
- [ ] **P1-H6** Wire all components into `App.tsx`: user picks input method, draws/uploads/captures, clicks "Recognize", result displays below
- [ ] **P1-H7** Make layout responsive (CSS Grid/Flexbox); test on mobile viewport in browser dev tools
- [ ] **P1-H8** Add loading spinner during API call; add error state for failed requests

### 1I: Integration & Testing

- [ ] **P1-I1** End-to-end test: draw English text on canvas → recognize → confirm sensible output
- [ ] **P1-I2** End-to-end test: upload a photo of handwritten English text → confirm recognition
- [ ] **P1-I3** End-to-end test: Tamil basic characters via canvas and upload
- [ ] **P1-I4** Test on mobile (Chrome Android or Safari iOS) — camera capture and canvas drawing
- [ ] **P1-I5** Measure end-to-end latency from button click to result display; confirm ≤ 3 seconds

---

## Phase 2 — Full Tamil Unicode

### 2A: Data — Tamil Compound Characters

- [ ] **P2-A1** Extend `ml/vocab/tamil_phase2.py` to include all uyirmei (216 compounds = 12 vowels × 18 consonants) + special characters in U+0B80–U+0BFF
- [ ] **P2-A2** Update synthetic generator to produce compound-character word and line images
- [ ] **P2-A3** Generate ≥ 50,000 synthetic Tamil Phase 2 line images with varied fonts and augmentations
- [ ] **P2-A4** Collect or curate at least 2,000 real handwritten Tamil compound-character samples if additional datasets are available
- [ ] **P2-A5** Apply Unicode normalization (NFC) to all Tamil ground-truth strings to ensure consistency

### 2B: Model — Tamil Phase 2

- [ ] **P2-B1** Evaluate whether CRNN is sufficient or a Transformer encoder-decoder is needed (compare accuracy on a small trial training run)
- [ ] **P2-B2** If upgrading to Transformer: implement a ViT-based encoder + autoregressive decoder (`ml/models/trocr_tamil.py`); otherwise extend CRNN vocab
- [ ] **P2-B3** Train Tamil Phase 2 model; checkpoint best by CER on compound-character test set
- [ ] **P2-B4** Evaluate; confirm CER ≤ 18% on compound-character test set
- [ ] **P2-B5** Export weights to `ml/checkpoints/tamil_p2_crnn.pt` (or `tamil_p2_transformer.pt`)

### 2C: Backend & Frontend Updates

- [ ] **P2-C1** Update `backend/inference.py` to load the Phase 2 Tamil model and route Tamil requests to it
- [ ] **P2-C2** Add Unicode normalization (NFC) post-processing step to Tamil output
- [ ] **P2-C3** Verify Tamil compound characters render correctly in the `RecognitionResult` component (test with system fonts on macOS, Windows, Linux, Android, iOS)
- [ ] **P2-C4** Update the language toggle UI to indicate Phase 2 Tamil is available (no visible change to user, just confirm no regression)

---

## Phase 3 — Optional Improvements

- [ ] **P3-1** Export models to ONNX; run inference via ONNX Runtime in backend for faster CPU inference
- [ ] **P3-2** Integrate a lightweight n-gram language model for post-processing CTC output (improves WER significantly)
- [ ] **P3-3** Package frontend as a PWA (add `manifest.json`, service worker) for mobile home-screen install
- [ ] **P3-4** Add per-word bounding box visualization: draw colored overlays on the original image next to each recognized word
- [ ] **P3-5** Add a feedback/correction UI: user can edit wrong recognized text, corrections are logged for future fine-tuning

---

## Ongoing / Cross-Cutting

- [ ] Keep model checkpoints versioned in a `/ml/checkpoints/` directory with a `model_registry.json` listing name, language, phase, CER, WER, training date
- [ ] Write a `CONTRIBUTING.md` explaining how to add a new language model
- [ ] Set up a GitHub Actions CI pipeline: lint, type-check frontend, run backend unit tests, confirm Docker build succeeds
