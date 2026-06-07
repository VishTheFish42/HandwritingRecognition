# Handwriting Recognition App — Design Document

## Overview

A web-based handwriting recognition system supporting **English** and **Tamil** scripts. Users can draw on screen, upload images, or use a camera to capture handwritten text. The system recognizes full sentences and paragraphs using deep learning models trained per language/script.

Tamil support is delivered in two phases:
- **Phase 1**: Basic Tamil characters (vowels, consonants, Aytham — ~247 base shapes)
- **Phase 2**: Full Unicode Tamil including all uyirmei (vowel-consonant compounds) and ligatures

---

## Platform

**Web application** (recommended). Rationale:
- HTML5 Canvas handles real-time stroke drawing natively
- `MediaDevices.getUserMedia` provides camera access without native APIs
- `<input type="file">` covers image upload
- No app store deployment; iterates fastest
- Can be wrapped as a Progressive Web App (PWA) for mobile home-screen install

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (Frontend)                   │
│                                                          │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────────┐  │
│  │  Canvas  │   │ File Upload │   │ Camera (WebRTC)  │  │
│  │ (draw)   │   │             │   │                  │  │
│  └────┬─────┘   └──────┬──────┘   └────────┬─────────┘  │
│       └────────────────┴──────────────────-┘             │
│                          │                               │
│                    Image / PNG blob                      │
│                          │                               │
│               ┌──────────▼──────────┐                   │
│               │   React UI Layer    │                    │
│               │  (preview, result,  │                    │
│               │   language toggle)  │                    │
│               └──────────┬──────────┘                   │
└──────────────────────────┼──────────────────────────────┘
                           │ HTTP POST /api/recognize
                           │ { image: base64, language: "en"|"ta" }
┌──────────────────────────▼──────────────────────────────┐
│                   Backend (FastAPI / Python)              │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │                  Preprocessing Pipeline            │  │
│  │  Grayscale → Denoise → Binarize → Deskew →        │  │
│  │  Line Segmentation → Word/Region Crop              │  │
│  └────────────────────────┬───────────────────────────┘  │
│                           │                              │
│          ┌────────────────┴────────────────┐             │
│          │                                 │             │
│  ┌───────▼────────┐               ┌────────▼──────────┐  │
│  │  English Model │               │   Tamil Model(s)  │  │
│  │  CRNN + CTC    │               │  CRNN + CTC       │  │
│  │  (IAM-trained) │               │  (IIIT-trained)   │  │
│  └───────┬────────┘               └────────┬──────────┘  │
│          └────────────────┬────────────────┘             │
│                           │                              │
│                  ┌────────▼────────┐                     │
│                  │  Post-processor │                     │
│                  │  (CTC decode,   │                     │
│                  │  Unicode norm)  │                     │
│                  └────────┬────────┘                     │
└───────────────────────────┼──────────────────────────────┘
                            │ { text: "...", confidence: 0.92 }
                    Back to browser
```

---

## ML Model Architecture

### Core Architecture: CRNN (Convolutional Recurrent Neural Network)

CRNN with CTC (Connectionist Temporal Classification) loss is the standard for sequence-level handwriting recognition. It handles variable-length input and output without explicit character segmentation.

```
Input image (H × W × 1, grayscale, height-normalized to 32px)
        │
        ▼
┌──────────────────────┐
│   CNN Feature        │  ResNet-style backbone
│   Extractor          │  Outputs feature map: (H/8) × (W/4) × 512
└──────────┬───────────┘
           │  Column-wise feature slices (sequence of vectors)
           ▼
┌──────────────────────┐
│  Bidirectional LSTM  │  2 layers, hidden size 256
│  (sequence model)    │  Input: sequence of 512-dim vectors
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Linear projection   │  → vocab_size logits per timestep
└──────────┬───────────┘
           │
           ▼
    CTC Decoder → character sequence
```

**Training loss**: CTC loss (no alignment labels needed, only the target string)
**Decoder**: Greedy or beam search with optional language model n-gram rescoring

### Vocabulary

| Language | Charset |
|----------|---------|
| English  | a–z, A–Z, digits, punctuation (~95 chars) |
| Tamil Phase 1 | 12 vowels + 18 consonants + Aytham + digits (~50 glyphs) |
| Tamil Phase 2 | Full Unicode Tamil block including all uyirmei compounds (~600+ codepoints) |

For Phase 2 Tamil, the model outputs Unicode codepoint sequences; compound characters are formed by the standard Unicode composition rules (base consonant + vowel sign).

### Alternative / Upgrade Path (Phase 2+)

For improved accuracy on long sequences, replace or augment CRNN with a **Transformer encoder-decoder** (TrOCR-style):
- Encoder: CNN + Vision Transformer (ViT) patches
- Decoder: Autoregressive token generation
- Pre-train on synthetic data, fine-tune on handwritten data
- Higher accuracy at the cost of more compute and longer inference time

---

## Preprocessing Pipeline

1. **Grayscale conversion** — strip color channels
2. **Denoising** — Gaussian blur or median filter
3. **Binarization** — Otsu's adaptive thresholding
4. **Deskewing** — Hough transform or projection profile to detect and correct slant
5. **Line segmentation** — horizontal projection profile valleys to separate text lines
6. **Height normalization** — resize each line strip to 32px height, preserve aspect ratio
7. **Padding** — pad width to fixed multiple for batching

For canvas-drawn strokes, preprocessing is lighter (strokes are already clean); focus is on rendering strokes to a raster image before the pipeline.

---

## Data Sources

### English
| Dataset | Description | Size |
|---------|-------------|------|
| IAM Handwriting Database (`Teklia/IAM-line` on Hugging Face) | Lines split — multiple writers; no registration required | 10,373 lines across train/val/test |
| EMNIST | Isolated characters | 800k+ samples |
| CVL Database | Additional writer variation | ~310 writers |

### Tamil
| Dataset | Description |
|---------|-------------|
| IIIT-Indic Handwriting Dataset | Line-level Tamil handwriting |
| HP Labs India Tamil Dataset | Character and word level |
| Synthetic generation | Use a Tamil handwriting font + augmentation to bootstrap training data |

Synthetic generation is critical for Tamil Phase 2 because labeled compound-character handwriting data is scarce. Strategy: render Tamil Unicode text in varied handwriting-style fonts (e.g., Latha, Bamini), apply elastic distortion, rotation jitter, and noise.

---

## Phased Delivery Plan

### Phase 1 — Foundation (English + Tamil basics)
- English full-sentence recognition
- Tamil basic character recognition (vowels, consonants, Aytham)
- All three input methods (canvas, upload, camera)
- Web UI with language toggle

### Phase 2 — Full Tamil Unicode
- Expand Tamil model to uyirmei compounds and ligatures
- Synthetic data pipeline for compound characters
- Upgrade decoder to handle Unicode normalization and composition
- Optional: Transformer-based model for improved accuracy

### Phase 3 — Quality & Polish (optional)
- Language model rescoring (n-gram or small LM)
- Confidence scores and per-region highlighting in UI
- PWA packaging for mobile install
- Offline inference via ONNX + WebAssembly (run model in browser)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, HTML5 Canvas, WebRTC |
| Backend | Python 3.11+, FastAPI |
| ML framework | PyTorch |
| Image processing | OpenCV, Pillow |
| Model serving | FastAPI (dev), TorchServe or ONNX Runtime (production) |
| Training | PyTorch Lightning (optional), Weights & Biases for experiment tracking |
| Containerization | Docker |

---

## API Contract

### `POST /api/recognize`

**Request**
```json
{
  "image": "<base64-encoded PNG>",
  "language": "en" | "ta"
}
```

**Response**
```json
{
  "text": "recognized text here",
  "confidence": 0.94,
  "lines": [
    { "text": "line one", "confidence": 0.96, "bbox": [x, y, w, h] },
    { "text": "line two", "confidence": 0.91, "bbox": [x, y, w, h] }
  ]
}
```

**Error response**
```json
{
  "error": "no_text_detected" | "unsupported_language" | "image_too_small",
  "message": "Human-readable description"
}
```
