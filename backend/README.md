---
title: Keyflow Ai
emoji: 💻
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false 
---

# KeyFlow AI Audio-to-MIDI Backend (`backend/`)

This Python backend powers **Studio AI Dedicated Piano Stem Separation (`BS-RoFormer` / `HT-Demucs 6S`)** and **State-of-the-Art Neural Polyphonic Piano Transcription (`Transkun` / `ByteDance Piano AI` / `Basic Pitch` / `Librosa CQT`)** with **Minimal Stem Preparation & Adaptive MIDI Bleed Filtering** for the KeyFlow Virtual Piano iPad & Web application.

> **Offline Fallback:** Even when no Python backend is running, the KeyFlow frontend includes an **In-Browser Web Audio API Polyphonic Piano Transcriber** (`js/audio-transcriber.js`), allowing `.mp3`, `.wav`, `.m4a`, and `.ogg` files to be transcribed 100% offline directly on iPad via GitHub Pages.

---

## Architecture & Audio Pipeline

1. **Multi-Tier Dedicated Piano Stem Isolation:**
   - **Tier 1 (Primary):** `BS-RoFormer` 6-stem separation via `audio-separator` (`BS-Roformer-SW.ckpt`), isolating piano with industry-leading SDR from vocals, guitars, bass, drums, and other accompaniment.
   - **Tier 2:** `HT-Demucs 6S` dedicated piano stem (`htdemucs_6s --two-stems piano`).
   - **Tier 3:** `HT-Demucs 4S` accompaniment stem (`htdemucs --two-stems other`).
   - **Tier 4 (Fallback):** Librosa HPSS harmonic separation with softened margin `(1.0, 1.5)`.

2. **Minimal Stem Preparation (`prepare_stem_for_transcription`):**
   - Preserves native sample rate and full audio frequency spectrum (no aggressive HPSS or 4.4 kHz low-pass filters that destroy piano hammer attack transients and harmonic overtones).
   - Applies only a gentle ~20 Hz 2nd-order Butterworth high-pass filter (`scipy.signal.butter` + `sosfiltfilt`) to strip sub-audible DC/sub-bass rumble.
   - Peak normalizes audio to 0.95 prior to neural transcription.
   - Configurable via `KEYFLOW_STEM_PREP` environment variable:
     - `minimal` (default): gentle 20 Hz high-pass + 0.95 peak norm.
     - `legacy`: previous HPSS percussive strip + 27.5–4400 Hz bandpass filtering.

3. **Multi-Engine Polyphonic Transcription & Bleed Filtering:**
   - **Tier 1 (Primary):** `Transkun` (Semi-CRF + Transformer), state-of-the-art benchmark-leading piano transcriber.
   - **Tier 2:** `ByteDance Piano AI` (CRNN High-Resolution Onsets & Frames + Pedal model).
   - **Tier 3:** Spotify `Basic Pitch` (neural polyphonic audio-to-MIDI, local server).
   - **Tier 4 (Fallback):** Librosa polyphonic CQT peak tracker (guarantees a valid MIDI even in zero-dependency environments).
   - Post-filters out-of-range notes (outside MIDI 21..108), micro-glitches (`< 55ms`), and background bleed under the sensitivity-adaptive velocity threshold (default 55% sensitivity -> velocity < 28 removed).
   - Caps simultaneous note onsets within any 30ms window to 8 notes (preserving highest-velocity notes).

---

## Environment Variables

- `KEYFLOW_STEM_PREP`: Stem preparation mode (`minimal` [default] or `legacy`).
- `KEYFLOW_SEPARATOR_MODEL`: Checkpoint filename for `audio-separator` BS-RoFormer (default: `BS-Roformer-SW.ckpt`).
- `KEYFLOW_SEPARATOR_OVERLAP`: Override the segment overlap count for MDXC/BS-RoFormer separation in `audio-separator`. Lower values are faster with slightly more seam artifacts; `2` is a reasonable speed setting (library default is typically `8`).
- `KEYFLOW_TRANSKUN_MIN_VEL`: Optional integer override for the Transkun velocity gate threshold in `filter_midi_bleed_notes` (defaults to sensitivity-mapped range 24..10).
- `KEYFLOW_MAX_CONCURRENT`: Maximum simultaneous `/api/transcribe` jobs (default: `2`). Excess requests wait briefly and then receive HTTP `503`.
- `KEYFLOW_RATE_LIMIT_PER_MIN`: Per-IP sliding-window request cap for `/api/transcribe` (default: `10` requests/minute). Excess requests receive HTTP `429`.
- `KEYFLOW_ALLOWED_ORIGINS`: Comma-separated CORS `allow_origins` list (default: `*`).
- `KEYFLOW_DEBUG_HEALTH`: Set to `true` or `1` to include `import_errors` and `installed_versions` in `/health` responses (disabled by default).
- `KEYFLOW_BYTEDANCE_SHA256`: Optional expected SHA-256 hex digest to verify the downloaded ByteDance CRNN checkpoint before use.
- `KEYFLOW_CACHE_DIR`: Private directory (mode `0700`) for model and checkpoint caches (default: `<tempdir>/keyflow_cache`).

---

## Deployment Options

You can run the backend locally on your computer or deploy it as a free **Hugging Face ZeroGPU Gradio Space**.

### Option A: Local FastAPI Server (`server.py`)

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   *(Optional)* For full neural stem isolation and Transkun/ByteDance CRNN piano transcription, install:
   ```bash
   pip install audio-separator[gpu] transkun demucs basic-pitch piano-transcription-inference
   ```

2. **Start the server:**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

3. **Connect from KeyFlow:**
   - From the KeyFlow **Home Screen**, click **`+ Add Song`** → **`Audio Recording`**.
   - If running on the same PC: keep the server URL as `http://localhost:8000` (when running the frontend locally over `http://`).
   - **Important (HTTPS Mixed Content):** When using the hosted GitHub Pages frontend (`https://...`), browsers (including iPad Safari and Chrome) block `http://` LAN URLs such as `http://192.168.1.50:8000` as Mixed Content. Either expose your local server via an HTTPS tunnel/reverse proxy, use a Hugging Face Space `https://...hf.space` URL, or serve the frontend locally over `http://`.

---

### Option B: Free Hugging Face ZeroGPU Space (`app.py`)

Hugging Face provides free ZeroGPU hardware (NVIDIA RTX Pro 6000 Blackwell) using the Gradio SDK:

1. **Create a Gradio Space on Hugging Face:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Space SDK: **Gradio**.
   - Hardware: **ZeroGPU (Free)**.

2. **Upload the backend files:**
   - Upload `app.py`, `pipeline.py`, `packages.txt`, and `requirements.txt` from `backend/` to the root of your Hugging Face Space repository.
   - Hugging Face automatically installs system Debian packages (`ffmpeg`, `libsndfile1`) and starts `app.py`.

3. **Connect from KeyFlow:**
   - Copy your public Space URL (e.g. `https://<username>-<space-name>.hf.space`).
   - Open KeyFlow, tap **`+ Add Song`** → **`Audio Recording`**, expand **Advanced: Local AI Server**, paste the `.hf.space` URL into the **AI Server URL** field, and tap **Check**.
   - The server will report active engines (`BS-RoFormer`, `HT-Demucs 6S`, `Transkun`, `ByteDance Piano AI`) on ZeroGPU.

---

## API Endpoints

- **`GET /health`** / **`GET /api/health`** — Returns server status, version, and available AI engines (`BS-RoFormer`, `HT-Demucs-6S`, `Transkun`, `ByteDance-Piano`, `Basic-Pitch`, `Librosa-CQT`). Set `KEYFLOW_DEBUG_HEALTH=true` to also expose package versions and import diagnostics.
- **`POST /api/transcribe`** — Accepts `multipart/form-data`:
  - `file`: Audio file (`.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`, `.webm`, max 50 MB).
  - `isolate_piano`: `"true"` (default) to run BS-RoFormer / HT-Demucs 6S dedicated piano isolation or `"false"`.
  - `sensitivity`: integer/float `15..85` (default `55`).
  - `engine`: optional string `"auto"` (default), `"transkun"`, or `"bytedance"`.
  - Returns JSON: `{"midi_base64": "...", "title": "...", "separator_used": "...", "transcriber_used": "...", "bpm": 120.0, "grid_offset": 0.8, "timings": {...}}`.
