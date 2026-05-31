# Handwriting Recognition App — Requirements

## Functional Requirements

### FR-1: Input Capture
- **FR-1.1** The app must allow users to draw handwriting directly on an HTML5 canvas with a pointer (mouse, touch, or stylus).
- **FR-1.2** The app must allow users to upload an image file (PNG, JPG, or PDF) containing handwritten text.
- **FR-1.3** The app must allow users to capture handwriting via the device camera using the browser MediaDevices API.
- **FR-1.4** The canvas must support variable stroke width, an eraser, and a clear button.

### FR-2: Language Support
- **FR-2.1** The app must support English handwriting recognition.
- **FR-2.2 (Phase 1)** The app must support recognition of basic Tamil characters: 12 vowels (உயிரெழுத்துக்கள்), 18 consonants (மெய்யெழுத்துக்கள்), and Aytham (ஃ).
- **FR-2.3 (Phase 2)** The app must support recognition of all Tamil uyirmei compound characters (vowel-consonant combinations) and common ligatures as defined in the Unicode Tamil block (U+0B80–U+0BFF).
- **FR-2.4** The user must be able to select the input language (English / Tamil) via a UI control before submitting.

### FR-3: Recognition
- **FR-3.1** The system must recognize full lines of handwritten text, not just isolated characters.
- **FR-3.2** The system must handle multi-line input (paragraphs) and return results per line.
- **FR-3.3** The output must be plain Unicode text.
- **FR-3.4** The system must return a confidence score per recognized line (0.0–1.0).

### FR-4: Output Display
- **FR-4.1** Recognized text must be displayed immediately after recognition completes.
- **FR-4.2** The user must be able to copy the recognized text to the clipboard with one click.
- **FR-4.3** The UI must visually indicate low-confidence lines (e.g., highlighted or marked).

### FR-5: Data Pipeline (internal)
- **FR-5.1** The system must include a reproducible training pipeline for both the English and Tamil models.
- **FR-5.2** The system must include a synthetic data generation pipeline for Tamil compound characters (Phase 2).

---

## Non-Functional Requirements

### NFR-1: Accuracy
| Metric | Target |
|--------|--------|
| English Character Error Rate (CER) | ≤ 8% on IAM test set |
| English Word Error Rate (WER) | ≤ 20% on IAM test set |
| Tamil Phase 1 CER | ≤ 12% on held-out test set |
| Tamil Phase 2 CER | ≤ 18% on held-out test set |

### NFR-2: Performance
- **NFR-2.1** API response time (image upload → text returned) must be ≤ 3 seconds for a single-page image on the production server.
- **NFR-2.2** Canvas stroke rendering must be real-time with no perceptible lag (< 16ms per frame).
- **NFR-2.3** The backend must handle at least 10 concurrent recognition requests without degradation.

### NFR-3: Usability
- **NFR-3.1** The UI must be functional on modern browsers: Chrome 110+, Firefox 110+, Safari 16+.
- **NFR-3.2** The UI must be responsive and usable on both desktop and mobile screen sizes.
- **NFR-3.3** Tamil output must render correctly using system Tamil fonts; the app must not require users to install custom fonts.

### NFR-4: Maintainability
- **NFR-4.1** Model training, evaluation, and inference must be decoupled from the web server so each can be updated independently.
- **NFR-4.2** All model training runs must be logged with hyperparameters, dataset versions, and evaluation metrics.
- **NFR-4.3** The backend and frontend must each have a Dockerfile for reproducible deployment.

### NFR-5: Security
- **NFR-5.1** Uploaded images must not be persisted to disk after inference completes.
- **NFR-5.2** The API must validate and cap incoming image size (max 10 MB) to prevent abuse.
- **NFR-5.3** Camera access must be requested with explicit browser permission prompts; no silent capture.

---

## Data Requirements

### DR-1: English Training Data
- IAM Handwriting Database (must register and download from official source)
- Minimum 10,000 labeled line images for training

### DR-2: Tamil Training Data
- At least one of: IIIT-Indic Tamil dataset or HP Labs India Tamil dataset
- Phase 2: Synthetic dataset of ≥ 50,000 compound-character line images generated from handwriting-style fonts with augmentation

### DR-3: Data Augmentation
- All training data must be augmented with: random rotation (±5°), elastic distortion, Gaussian noise, brightness/contrast jitter, and random stroke width variation (for synthetic data)

### DR-4: Train/Val/Test Split
- Minimum 80/10/10 split; test set must be writer-disjoint from training set

---

## Constraints

- The project is implemented using **PyTorch** as the ML framework.
- The backend language is **Python 3.11+**.
- The frontend framework is **React**.
- No proprietary third-party OCR APIs (e.g., Google Vision, AWS Textract) may be used as the recognition engine; models must be trained and served in-house.
- All datasets used must be publicly available or synthetically generated; no unlicensed data.
