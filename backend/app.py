import os

# Configure writable cache directories and Gradio SSR mode for Hugging Face ZeroGPU Spaces
os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba"
os.environ["GRADIO_SSR_MODE"] = "False"

for _cache_dir in ("/tmp/torch", "/tmp/hf", "/tmp/numba"):
    os.makedirs(_cache_dir, exist_ok=True)

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(*args, **kwargs):
            def _decorator(fn):
                return fn
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return _decorator

    spaces = _SpacesFallback()

import base64
import importlib.metadata
import importlib.util
import inspect
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB limit
ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".webm", ".mp4"}
ALLOWED_OMR_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".musicxml", ".mxl", ".xml"}

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import scipy.ndimage as ndi
    HAS_NDI = True
except Exception:
    HAS_NDI = False

try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except Exception:
    HAS_PDFIUM = False

try:
    import music21
    HAS_MUSIC21 = True
except Exception:
    HAS_MUSIC21 = False

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    from scipy.signal import butter, sosfiltfilt
    HAS_SCIPY_SIGNAL = True
except Exception:
    HAS_SCIPY_SIGNAL = False

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except Exception:
    HAS_PRETTY_MIDI = False

# Records import failures so /health can report the real reason an engine is unavailable.
IMPORT_ERRORS: Dict[str, str] = {}

try:
    import torch
    HAS_TORCH = True
except Exception as _exc:
    HAS_TORCH = False
    IMPORT_ERRORS["torch"] = f"{type(_exc).__name__}: {_exc}"[:500]

try:
    from piano_transcription_inference import PianoTranscription, sample_rate as BD_SAMPLE_RATE
    HAS_BYTEDANCE = True
except Exception as _exc:
    HAS_BYTEDANCE = False
    BD_SAMPLE_RATE = 16000
    IMPORT_ERRORS["piano_transcription_inference"] = f"{type(_exc).__name__}: {_exc}"[:500]

try:
    from audio_separator.separator import Separator
    HAS_AUDIO_SEPARATOR = True
except Exception as _exc:
    HAS_AUDIO_SEPARATOR = False
    IMPORT_ERRORS["audio_separator"] = f"{type(_exc).__name__}: {_exc}"[:500]

for _name, _msg in IMPORT_ERRORS.items():
    print(f"[KeyFlow Space] Import failed for {_name}: {_msg}")


def _installed_versions() -> Dict[str, Any]:
    """Returns installed distribution versions (None if not installed), independent of import success."""
    out: Dict[str, Any] = {"python": sys.version.split()[0]}
    for dist in ("numpy", "torch", "torchaudio", "audio-separator", "onnxruntime-gpu",
                 "onnxruntime", "transkun", "demucs", "piano_transcription_inference", "librosa", "audioread", "setuptools", "homr"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except Exception:
            out[dist] = None
    return out

HAS_DEMUCS = shutil.which("demucs") is not None

# Runner snippet that shims pkg_resources.resource_filename if setuptools>=82 is installed
# (which removed pkg_resources, breaking transkun/transcribe.py line 22).
_TRANSKUN_RUNNER_CODE = (
    "import sys, types, importlib.util, pathlib\n"
    "try:\n"
    "    import pkg_resources\n"
    "except ImportError:\n"
    "    m = types.ModuleType('pkg_resources')\n"
    "    m.resource_filename = lambda pkg, res: str(pathlib.Path(importlib.util.find_spec(pkg).origin).parent / res)\n"
    "    sys.modules['pkg_resources'] = m\n"
    "from transkun.transcribe import main\n"
    "sys.exit(main())\n"
)

HAS_TRANSKUN = False
TRANSKUN_CMD: List[str] = []

try:
    if importlib.util.find_spec("transkun") is not None:
        HAS_TRANSKUN = True
        TRANSKUN_CMD = [sys.executable, "-c", _TRANSKUN_RUNNER_CODE]
except Exception:
    pass

if not HAS_TRANSKUN and shutil.which("transkun") is not None:
    HAS_TRANSKUN = True
    TRANSKUN_CMD = ["transkun"]

HAS_HOMR = False
HOMR_CMD: List[str] = []

if shutil.which("homr") is not None:
    HAS_HOMR = True
    HOMR_CMD = ["homr"]
else:
    try:
        if importlib.util.find_spec("homr") is not None:
            HAS_HOMR = True
            HOMR_CMD = [sys.executable, "-m", "homr.main"]
    except Exception as _exc:
        IMPORT_ERRORS["homr"] = f"{type(_exc).__name__}: {_exc}"[:500]

HAS_AUDIVERIS = shutil.which("audiveris") is not None
HAS_OEMER = shutil.which("oemer") is not None

# BS-RoFormer SW 6-stem model (BS-Roformer-SW.ckpt with BS-Roformer-SW.yaml), extracting ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano']
PIANO_SEPARATOR_MODEL = os.environ.get("KEYFLOW_SEPARATOR_MODEL", "BS-Roformer-SW.ckpt")
STEM_PREP_MODE = os.environ.get("KEYFLOW_STEM_PREP", "minimal")
SEPARATOR_OVERLAP = os.environ.get("KEYFLOW_SEPARATOR_OVERLAP", "")

MODEL_CACHE_DIR = Path("/tmp/audio-separator-models")
_separator_lock = threading.Lock()


def _get_separator_mdxc_params() -> Optional[Dict[str, Any]]:
    """
    Inspects Separator.__init__ signature and returns merged mdxc_params if KEYFLOW_SEPARATOR_OVERLAP
    is a valid integer >= 1. Otherwise returns None so library default applies.
    """
    if not SEPARATOR_OVERLAP:
        return None
    try:
        overlap_val = int(SEPARATOR_OVERLAP)
        if overlap_val < 1:
            return None
    except ValueError:
        return None

    try:
        if not HAS_AUDIO_SEPARATOR:
            return None
        sig = inspect.signature(Separator.__init__)
        if "mdxc_params" not in sig.parameters:
            print("[KeyFlow Space] WARNING: 'mdxc_params' parameter not found in Separator.__init__ signature, skipping overlap override.")
            return None
        default_val = sig.parameters["mdxc_params"].default
        if isinstance(default_val, dict):
            params = dict(default_val)
        else:
            params = {}
        params["overlap"] = overlap_val
        return params
    except Exception as exc:
        print(f"[KeyFlow Space] WARNING: Failed to inspect Separator.__init__ signature ({exc}), skipping overlap override.")
        return None


def _prefetch_separator_model_worker() -> None:
    """
    Background worker to download BS-RoFormer model weights outside @spaces.GPU duration.
    ZeroGPU charges from the moment the GPU function starts; pre-downloading weights saves GPU quota.
    Uses Separator.download_model_files (pure HTTP download without CUDA initialization).
    """
    if not HAS_AUDIO_SEPARATOR:
        return
    try:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        target_ckpt = MODEL_CACHE_DIR / PIANO_SEPARATOR_MODEL
        if target_ckpt.exists():
            return

        sep = Separator(model_file_dir=str(MODEL_CACHE_DIR))
        if hasattr(sep, "download_model_files"):
            print(f"[KeyFlow Space] Prefetching separator model '{PIANO_SEPARATOR_MODEL}' via download_model_files...")
            sep.download_model_files(PIANO_SEPARATOR_MODEL)
            print(f"[KeyFlow Space] Separator model '{PIANO_SEPARATOR_MODEL}' prefetched successfully.")
        else:
            # Note: We verified against audio-separator 0.48.0 source:
            # Separator.download_model_files handles HTTP fetching and SHA-256 verification.
            # If download_model_files is absent, calling load_model() directly on CPU risks
            # initializing CUDA providers in PyTorch/ONNX, which raises RuntimeError on ZeroGPU
            # when outside an active @spaces.GPU context. Therefore, we do not call load_model.
            print(f"[KeyFlow Space] Separator.download_model_files not found; skipping CPU prefetch to avoid CUDA initialization outside @spaces.GPU.")
    except Exception as exc:
        print(f"[KeyFlow Space] Separator model prefetch warning (non-fatal): {exc}")


if HAS_AUDIO_SEPARATOR:
    try:
        threading.Thread(target=_prefetch_separator_model_worker, daemon=True, name="prefetch_separator").start()
    except Exception as t_exc:
        print(f"[KeyFlow Space] Failed to start separator prefetch thread: {t_exc}")

BYTEDANCE_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
BYTEDANCE_CHECKPOINT_PATH = Path("/tmp/piano_model.pth")
_checkpoint_lock = threading.Lock()


def ensure_bytedance_checkpoint() -> Path:
    """Thread-safely downloads the ByteDance Piano Transcription model weights to /tmp/piano_model.pth on first use."""
    if BYTEDANCE_CHECKPOINT_PATH.exists() and BYTEDANCE_CHECKPOINT_PATH.stat().st_size > 10_000_000:
        return BYTEDANCE_CHECKPOINT_PATH

    with _checkpoint_lock:
        if BYTEDANCE_CHECKPOINT_PATH.exists() and BYTEDANCE_CHECKPOINT_PATH.stat().st_size > 10_000_000:
            return BYTEDANCE_CHECKPOINT_PATH

        tmp_download = Path(f"/tmp/piano_model_{os.getpid()}_{threading.get_ident()}.pth.part")
        try:
            urllib.request.urlretrieve(BYTEDANCE_CHECKPOINT_URL, str(tmp_download))
            tmp_download.replace(BYTEDANCE_CHECKPOINT_PATH)
        finally:
            if tmp_download.exists():
                try:
                    tmp_download.unlink()
                except Exception:
                    pass
        return BYTEDANCE_CHECKPOINT_PATH


def _legacy_clean_isolated_piano_audio(stem_wav_path: Path, work_dir: Path) -> Path:
    """
    Legacy post-Demucs harmonic and piano-range filtering:
    1. Loads audio at ByteDance sample rate (16 kHz mono).
    2. Strips residual drum/percussive transients via HPSS (margin 1.0 harm vs 2.2 perc).
    3. Bandpasses to piano fundamental range (27.5 Hz A0 to 4400 Hz C8).
    4. Normalizes peak amplitude and saves to piano_cleaned.wav.
    """
    if not HAS_LIBROSA:
        return stem_wav_path

    try:
        y, sr = librosa.load(str(stem_wav_path), sr=BD_SAMPLE_RATE, mono=True)
        if y.size == 0 or np.max(np.abs(y)) < 1e-6:
            return stem_wav_path

        # 1. HPSS to strip percussive bleed from drums/cymbals
        y_harm, _ = librosa.effects.hpss(y, margin=(1.0, 2.2))

        # 2. Bandpass filter for piano fundamental range (27.5 Hz - 4400 Hz)
        y_filtered = y_harm
        if HAS_SCIPY_SIGNAL and sr > 8800:
            try:
                nyq = 0.5 * sr
                low = 27.5 / nyq
                high = min(4400.0, nyq - 50.0) / nyq
                sos = butter(2, [low, high], btype="bandpass", output="sos")
                y_filtered = sosfiltfilt(sos, y_harm)
            except Exception as f_err:
                print(f"[KeyFlow Space] Bandpass filter skipped: {f_err}")
                y_filtered = y_harm

        # 3. Peak amplitude normalization
        peak = float(np.max(np.abs(y_filtered)))
        if peak > 1e-6:
            y_norm = y_filtered * (0.95 / peak)
        else:
            y_norm = y_filtered

        cleaned_path = work_dir / "piano_cleaned.wav"
        sf.write(str(cleaned_path), y_norm.astype(np.float32), sr)
        if cleaned_path.exists() and cleaned_path.stat().st_size > 1000:
            return cleaned_path
    except Exception as exc:
        print(f"[KeyFlow Space] Post-Demucs cleaning error: {exc}")

    return stem_wav_path


def prepare_stem_for_transcription(stem_path: Path, work_dir: Path, mode: str | None = None) -> Path:
    """
    Prepares an isolated piano stem for neural transcription:
    - Minimal mode (default):
      1. Loads audio preserving native sample rate (sr=None, mono=True).
      2. Applies a gentle 20 Hz high-pass filter (2nd order Butterworth) to remove DC/rumble.
      3. Peak-normalizes to 0.95.
      4. Saves as piano_prepared.wav (float32).
      No HPSS, no low-pass, no bandpass so transients and high harmonics remain intact.
    - Legacy mode (mode="legacy" or KEYFLOW_STEM_PREP="legacy"):
      Delegates to _legacy_clean_isolated_piano_audio.
    """
    current_mode = mode or STEM_PREP_MODE
    if current_mode == "legacy":
        return _legacy_clean_isolated_piano_audio(stem_path, work_dir)

    if not HAS_LIBROSA:
        return stem_path

    try:
        y, sr = librosa.load(str(stem_path), sr=None, mono=True)
        if y.size == 0 or np.max(np.abs(y)) < 1e-6:
            return stem_path

        # Gentle high-pass at ~20 Hz to remove DC/sub-bass rumble
        y_filtered = y
        if HAS_SCIPY_SIGNAL and sr is not None and sr > 40:
            try:
                nyq = 0.5 * sr
                low = 20.0 / nyq
                if 0 < low < 1:
                    sos = butter(2, low, btype="highpass", output="sos")
                    y_filtered = sosfiltfilt(sos, y)
            except Exception as f_err:
                print(f"[KeyFlow Space] Highpass filter skipped: {f_err}")
                y_filtered = y

        # Peak normalization to 0.95
        peak = float(np.max(np.abs(y_filtered)))
        if peak > 1e-6:
            y_norm = y_filtered * (0.95 / peak)
        else:
            y_norm = y_filtered

        prepared_path = work_dir / "piano_prepared.wav"
        sf.write(str(prepared_path), y_norm.astype(np.float32), sr)
        if prepared_path.exists() and prepared_path.stat().st_size > 1000:
            return prepared_path
    except Exception as exc:
        print(f"[KeyFlow Space] Stem preparation error: {exc}")

    return stem_path


def isolate_piano_with_roformer(input_audio_path: Path, work_dir: Path) -> Path | None:
    """Isolates dedicated piano stem using BS-RoFormer 6-stem model via audio-separator."""
    if not HAS_AUDIO_SEPARATOR:
        return None
    try:
        sep_out = work_dir / "sep_out"
        sep_out.mkdir(parents=True, exist_ok=True)
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with _separator_lock:
            sep_kwargs: Dict[str, Any] = {
                "output_dir": str(sep_out),
                "model_file_dir": str(MODEL_CACHE_DIR),
                "output_format": "WAV",
            }
            mdxc_ov = _get_separator_mdxc_params()
            if mdxc_ov is not None:
                sep_kwargs["mdxc_params"] = mdxc_ov

            separator = Separator(**sep_kwargs)
            # On ZeroGPU, load_model must execute inside the active GPU context:
            try:
                separator.load_model(model_filename=PIANO_SEPARATOR_MODEL)
            except Exception as lm_exc:
                if "not found in supported model files" in str(lm_exc).lower():
                    try:
                        as_ver = importlib.metadata.version("audio-separator")
                    except Exception:
                        as_ver = "unknown"
                    print(f"[KeyFlow Space] ERROR: separator model '{PIANO_SEPARATOR_MODEL}' not available in audio-separator {as_ver} — upgrade to >=0.48.0. Falling back to htdemucs_6s.")
                    return None
                raise

            separator.output_dir = str(sep_out)
            mi = getattr(separator, "model_instance", None)
            if mi is not None and hasattr(mi, "output_dir"):
                mi.output_dir = str(sep_out)

            output_files = separator.separate(str(input_audio_path))

        if not output_files:
            return None

        for fname in output_files:
            candidate = Path(fname)
            resolved_path = None
            if candidate.is_absolute() and candidate.exists():
                resolved_path = candidate
            elif (sep_out / fname).exists():
                resolved_path = sep_out / fname
            elif (Path(separator.output_dir) / fname).exists():
                resolved_path = Path(separator.output_dir) / fname

            if resolved_path and "piano" in resolved_path.name.lower() and resolved_path.stat().st_size > 1000:
                return resolved_path
    except Exception as exc:
        print(f"[KeyFlow Space] BS-RoFormer isolation error: {exc}")
        return None
    finally:
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return None


def isolate_piano_with_htdemucs_6s(input_audio_path: Path, work_dir: Path) -> Path | None:
    """Runs 6-stem HT-Demucs dedicated piano stem isolation."""
    if not HAS_DEMUCS:
        return None
    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    out_dir = work_dir / "demucs_out"
    try:
        cmd_6s = [
            "demucs",
            "-n", "htdemucs_6s",
            "--two-stems", "piano",
            "-d", device,
            "-o", str(out_dir),
            str(input_audio_path),
        ]
        subprocess.run(
            cmd_6s,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        piano_wav = out_dir / "htdemucs_6s" / input_audio_path.stem / "piano.wav"
        if piano_wav.exists() and piano_wav.stat().st_size > 1000:
            return piano_wav
    except Exception as exc:
        print(f"[KeyFlow Space] htdemucs_6s dedicated piano isolation fallback: {exc}")
    finally:
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return None


def isolate_piano_stem(input_audio_path: Path, work_dir: Path) -> Tuple[Path, str]:
    """
    Separates piano stem using hierarchical fallback chain:
    1. BS-RoFormer SW 6-stem via audio-separator.
    2. HT-Demucs 6S dedicated piano stem (demucs -n htdemucs_6s --two-stems piano).
    3. HT-Demucs 4-stem other stem (demucs -n htdemucs --two-stems other).
    4. Librosa HPSS harmonic separation with softer margin=(1.0, 1.5).
    5. Raw input audio fallback.

    Returns (prepared_stem_path, separator_name).
    """
    # 1. Primary: BS-RoFormer via audio-separator
    if HAS_AUDIO_SEPARATOR:
        try:
            roformer_stem = isolate_piano_with_roformer(input_audio_path, work_dir)
            if roformer_stem is not None and roformer_stem.exists():
                return prepare_stem_for_transcription(roformer_stem, work_dir), "bs-roformer-sw"
        except Exception as exc:
            print(f"[KeyFlow Space] BS-RoFormer fallback: {exc}")

    # 2. Secondary: 6-stem dedicated piano model (htdemucs_6s --two-stems piano)
    if HAS_DEMUCS:
        piano_6s = isolate_piano_with_htdemucs_6s(input_audio_path, work_dir)
        if piano_6s is not None and piano_6s.exists():
            return prepare_stem_for_transcription(piano_6s, work_dir), "htdemucs_6s"

        # 3. Tertiary: 4-stem model (htdemucs --two-stems other)
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        out_dir = work_dir / "demucs_out"
        try:
            cmd_4s = [
                "demucs",
                "-n", "htdemucs",
                "--two-stems", "other",
                "-d", device,
                "-o", str(out_dir),
                str(input_audio_path),
            ]
            subprocess.run(
                cmd_4s,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=240,
            )
            other_wav = out_dir / "htdemucs" / input_audio_path.stem / "other.wav"
            if other_wav.exists() and other_wav.stat().st_size > 1000:
                return prepare_stem_for_transcription(other_wav, work_dir), "htdemucs"
        except Exception as exc:
            print(f"[KeyFlow Space] Demucs 4-stem fallback failed: {exc}")
        finally:
            if HAS_TORCH and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # 4. Quaternary: Librosa HPSS crude harmonic separation (softer margin)
    if HAS_LIBROSA:
        try:
            y, sr = librosa.load(str(input_audio_path), sr=22050, mono=True)
            y_harm, _ = librosa.effects.hpss(y, margin=(1.0, 1.5))
            harm_path = work_dir / "harmonic_isolated.wav"
            sf.write(str(harm_path), y_harm, sr)
            if harm_path.exists() and harm_path.stat().st_size > 1000:
                return prepare_stem_for_transcription(harm_path, work_dir), "librosa-hpss"
        except Exception as exc:
            print(f"[KeyFlow Space] Librosa HPSS fallback skipped: {exc}")

    return prepare_stem_for_transcription(input_audio_path, work_dir), "none"


def transcribe_with_transkun(audio_path: Path, output_midi_path: Path) -> bool:
    """Transcribes audio into a Standard MIDI file using Transkun."""
    if not HAS_TRANSKUN or not TRANSKUN_CMD:
        return False
    try:
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        cmd = list(TRANSKUN_CMD) + [str(audio_path), str(output_midi_path), "--device", device]
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=300,
        )
        return output_midi_path.exists() and output_midi_path.stat().st_size > 80
    except subprocess.CalledProcessError as exc:
        err_msg = (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc))[-1000:]
        print(f"[KeyFlow Space] Transkun transcription failed: {err_msg}")
        return False
    except Exception as exc:
        print(f"[KeyFlow Space] Transkun transcription error: {str(exc)[:500]}")
        return False
    finally:
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def transcribe_with_bytedance(audio_path: Path, output_midi_path: Path) -> bool:
    """Transcribes audio into a Standard MIDI file using ByteDance PianoTranscription."""
    if not HAS_BYTEDANCE or not HAS_LIBROSA:
        return False
    try:
        ckpt_path = ensure_bytedance_checkpoint()
        audio, _ = librosa.load(str(audio_path), sr=BD_SAMPLE_RATE, mono=True)
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        transcriber = PianoTranscription(device=device, checkpoint_path=str(ckpt_path))
        transcriber.transcribe(audio, str(output_midi_path))
        return output_midi_path.exists() and output_midi_path.stat().st_size > 80
    except Exception as exc:
        print(f"[KeyFlow Space] ByteDance transcription error (falling back to Librosa CQT): {exc}")
        return False
    finally:
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def transcribe_with_librosa_cqt_to_midi(audio_path: Path, output_midi_path: Path, sensitivity: float) -> bool:
    """
    Guaranteed polyphonic CQT + Harmonic Overtone Suppression fallback that writes a
    Standard MIDI file via pretty_midi so transcription never fails.
    """
    if not HAS_LIBROSA or not HAS_PRETTY_MIDI:
        return False

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    y_harm, _ = librosa.effects.hpss(y, margin=(1.2, 2.0))

    hop_length = 512
    fmin = librosa.midi_to_hz(28)  # E1
    n_bins = 69                    # MIDI 28..96 (E1..C7)

    cqt_mag = np.abs(
        librosa.cqt(
            y_harm,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=12,
        )
    )

    cleaned = cqt_mag.copy()
    for b in range(n_bins):
        e0 = cqt_mag[b]
        if b + 12 < n_bins:
            cleaned[b + 12] = np.maximum(0.0, cleaned[b + 12] - e0 * 0.45)
        if b + 19 < n_bins:
            cleaned[b + 19] = np.maximum(0.0, cleaned[b + 19] - e0 * 0.26)
        if b + 24 < n_bins:
            cleaned[b + 24] = np.maximum(0.0, cleaned[b + 24] - e0 * 0.16)

    max_val = float(np.max(cleaned)) if cleaned.size > 0 else 1.0
    if max_val <= 1e-8:
        return False
    norm = cleaned / max_val

    sens_norm = float(np.clip(sensitivity / 100.0 if sensitivity > 1.0 else sensitivity, 0.15, 0.90))
    act_thresh = 0.24 - (sens_norm - 0.5) * 0.18
    max_poly = int(round(3 + sens_norm * 4))
    frame_sec = hop_length / float(sr)

    active: Dict[int, Dict[str, float]] = {}
    notes: List[Dict[str, Any]] = []

    num_frames = norm.shape[1]
    for f_idx in range(num_frames):
        t = f_idx * frame_sec
        col = norm[:, f_idx]
        peaks = []
        for b in range(1, n_bins - 1):
            val = float(col[b])
            if val >= act_thresh and val >= float(col[b - 1]) * 0.94 and val >= float(col[b + 1]) * 0.94:
                peaks.append((b + 28, val))
        peaks.sort(key=lambda x: x[1], reverse=True)
        chosen = dict(peaks[:max_poly])

        for midi_num in list(active.keys()):
            st = active[midi_num]
            cur_val = chosen.get(midi_num, 0.0)
            if cur_val < act_thresh * 0.55:
                dur = t - st["start"]
                if dur >= 0.07:
                    notes.append({
                        "midi": int(midi_num),
                        "start": float(st["start"]),
                        "end": float(st["start"] + min(4.5, dur)),
                        "velocity": int(np.clip(round((0.38 + np.sqrt(st["peak"]) * 0.56) * 127), 32, 122)),
                    })
                del active[midi_num]
            else:
                if cur_val > st["peak"]:
                    st["peak"] = cur_val

        for midi_num, val in chosen.items():
            if midi_num not in active:
                active[midi_num] = {"start": t, "peak": val}

    end_t = num_frames * frame_sec
    for midi_num, st in active.items():
        dur = end_t - st["start"]
        if dur >= 0.07:
            notes.append({
                "midi": int(midi_num),
                "start": float(st["start"]),
                "end": float(st["start"] + min(4.5, dur)),
                "velocity": int(np.clip(round((0.38 + np.sqrt(st["peak"]) * 0.56) * 127), 32, 122)),
            })

    if not notes:
        return False

    try:
        tempo_arr = librosa.beat.tempo(y=y_harm, sr=sr)
        raw_bpm = float(tempo_arr[0]) if hasattr(tempo_arr, "__getitem__") and len(tempo_arr) > 0 else float(tempo_arr)
        est_bpm = float(np.clip(raw_bpm, 60.0, 180.0))
    except Exception:
        est_bpm = 112.0

    pm = pretty_midi.PrettyMIDI(initial_tempo=est_bpm)
    piano = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for n in sorted(notes, key=lambda item: (item["start"], item["midi"])):
        piano.notes.append(
            pretty_midi.Note(
                velocity=n["velocity"],
                pitch=n["midi"],
                start=n["start"],
                end=n["end"],
            )
        )
    pm.instruments.append(piano)
    pm.write(str(output_midi_path))
    return output_midi_path.exists()


def filter_midi_bleed_notes(
    midi_path: Path,
    sensitivity: float = 55.0,
    transcriber: str = "bytedance",
) -> bool:
    """
    Post-filters low-velocity bleed and micro-glitches from neural transcription:
    1. Removes notes outside 88-key piano range (MIDI 21..108).
    2. Velocity gate mapped from sensitivity:
       - ByteDance: 15..85 -> min_vel 38..20 (default 55 -> 28).
       - Transkun: 15..85 -> min_vel 24..10 (default 55 -> 16), overridable via KEYFLOW_TRANSKUN_MIN_VEL.
    3. Removes sub-55ms micro-glitches (duration < 0.055s).
    4. Caps simultaneous note onsets within any 30ms window to 8 notes (keeping highest velocity).
    """
    if not HAS_PRETTY_MIDI or not midi_path.exists():
        return False
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        sens_clamped = float(np.clip(sensitivity, 15.0, 85.0))
        t_name = (transcriber or "bytedance").strip().lower()

        if t_name == "transkun":
            min_vel = int(round(24.0 - ((sens_clamped - 15.0) / 70.0) * 14.0))
            min_vel = max(1, min(45, min_vel))
            env_override = os.environ.get("KEYFLOW_TRANSKUN_MIN_VEL", "").strip()
            if env_override:
                try:
                    min_vel = int(env_override)
                except ValueError:
                    pass
        else:
            min_vel = int(round(38.0 - ((sens_clamped - 15.0) / 70.0) * 18.0))
            min_vel = max(15, min(45, min_vel))

        total_kept = 0
        for inst in pm.instruments:
            if inst.is_drum:
                continue

            # Step 1: Filter pitch range, minimum velocity, and minimum duration
            valid_notes = []
            for n in inst.notes:
                dur = n.end - n.start
                if 21 <= n.pitch <= 108 and n.velocity >= min_vel and dur >= 0.055:
                    valid_notes.append(n)

            valid_notes.sort(key=lambda n: n.start)

            # Step 2: Cap simultaneous note onsets within 30ms window to 8 notes
            filtered_notes = []
            i = 0
            while i < len(valid_notes):
                w_start = valid_notes[i].start
                w_notes = []
                while i < len(valid_notes) and (valid_notes[i].start - w_start) <= 0.030:
                    w_notes.append(valid_notes[i])
                    i += 1
                if len(w_notes) > 8:
                    w_notes.sort(key=lambda n: n.velocity, reverse=True)
                    w_notes = w_notes[:8]
                filtered_notes.extend(w_notes)

            filtered_notes.sort(key=lambda n: (n.start, n.pitch))

            # Step 3: Bake CC 64 sustain pedal holds into note end times
            cc64_events = sorted(
                [cc for cc in inst.control_changes if cc.number == 64],
                key=lambda cc: cc.time,
            )
            pedal_intervals: List[Tuple[float, float]] = []
            p_start: Optional[float] = None
            for cc in cc64_events:
                if cc.value >= 64 and p_start is None:
                    p_start = float(cc.time)
                elif cc.value < 64 and p_start is not None:
                    if float(cc.time) > p_start:
                        pedal_intervals.append((p_start, float(cc.time)))
                    p_start = None

            # Map next onset of the same pitch so sustained notes never overlap a re-strike
            next_same_pitch: List[float] = [1e9] * len(filtered_notes)
            last_seen_pitch: Dict[int, float] = {}
            for idx_n in range(len(filtered_notes) - 1, -1, -1):
                p_val = filtered_notes[idx_n].pitch
                if p_val in last_seen_pitch:
                    next_same_pitch[idx_n] = last_seen_pitch[p_val]
                last_seen_pitch[p_val] = float(filtered_notes[idx_n].start)

            if pedal_intervals:
                for idx_n, n in enumerate(filtered_notes):
                    for p_down, p_up in pedal_intervals:
                        if p_down <= n.end + 0.04 and p_up > n.end:
                            capped_end = min(p_up - 0.02, next_same_pitch[idx_n] - 0.03)
                            if capped_end > n.end:
                                n.end = capped_end
                            break

            # Step 4: Extend clipped chord clusters (2+ notes within 45ms) up to next chord change
            clusters_idx: List[List[int]] = []
            if filtered_notes:
                cur_c = [0]
                for idx_n in range(1, len(filtered_notes)):
                    if (filtered_notes[idx_n].start - filtered_notes[cur_c[0]].start) <= 0.045:
                        cur_c.append(idx_n)
                    else:
                        clusters_idx.append(cur_c)
                        cur_c = [idx_n]
                clusters_idx.append(cur_c)

            for c_pos in range(len(clusters_idx) - 1):
                c_list = clusters_idx[c_pos]
                c_start = float(filtered_notes[c_list[0]].start)
                if len(c_list) >= 2:
                    target_next_t: Optional[float] = None
                    for next_c_pos in range(c_pos + 1, len(clusters_idx)):
                        cand_t = float(filtered_notes[clusters_idx[next_c_pos][0]].start)
                        dt = cand_t - c_start
                        if dt < 0.22:
                            continue
                        if len(clusters_idx[next_c_pos]) >= 2 or dt >= 0.28:
                            target_next_t = cand_t
                            break
                    if target_next_t is not None and 0.25 <= (target_next_t - c_start) <= 2.6:
                        for idx_n in c_list:
                            n = filtered_notes[idx_n]
                            dur = n.end - n.start
                            max_end = min(target_next_t - 0.04, next_same_pitch[idx_n] - 0.03)
                            if dur <= 1.25 and (max_end - n.start) > dur + 0.05:
                                n.end = min(n.start + 3.5, max_end)

            inst.notes = filtered_notes
            total_kept += len(filtered_notes)

        if total_kept > 0:
            pm.write(str(midi_path))
            return True
        return False
    except Exception as exc:
        print(f"[KeyFlow Space] Error filtering MIDI bleed notes: {exc}")
        return False


def estimate_midi_and_audio_rhythm(
    audio_path: Path,
    midi_path: Path,
) -> Dict[str, float]:
    """
    Estimates musical tempo (BPM) and Bar 1 beat grid offset (including the frontend's +0.8s lead-in)
    by combining librosa beat tracking on the audio with phase-alignment against transcribed MIDI note onsets.
    """
    default_res = {"bpm": 120.0, "grid_offset": 0.8}
    if not HAS_PRETTY_MIDI or not midi_path.exists():
        return default_res

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        raw_notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                raw_notes.extend(inst.notes)
        if not raw_notes:
            return default_res

        raw_notes.sort(key=lambda n: (n.start, n.pitch))

        # Cluster simultaneous note onsets within 35ms
        clusters: List[Tuple[float, float]] = []
        idx = 0
        n_total = len(raw_notes)
        while idx < n_total:
            t0 = float(raw_notes[idx].start)
            w_sum = 0.0
            while idx < n_total and (float(raw_notes[idx].start) - t0) <= 0.035:
                dur = max(0.05, min(2.0, float(raw_notes[idx].end - raw_notes[idx].start)))
                w_sum += float(raw_notes[idx].velocity) * (0.5 + 0.5 * dur)
                idx += 1
            clusters.append((t0, w_sum))

        # 1. Estimate BPM via librosa beat_track on first 90s of audio
        est_bpm = 0.0
        if HAS_LIBROSA and audio_path.exists():
            try:
                y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=90.0)
                if y.size > sr * 2:
                    tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr)
                    arr = np.asarray(tempo_val).flat
                    if arr.size > 0:
                        est_bpm = float(arr[0])
            except Exception:
                est_bpm = 0.0

        # 2. Fallback BPM from MIDI inter-onset intervals if librosa beat_track failed
        if est_bpm <= 30.0 and len(clusters) >= 6:
            iois = []
            for i in range(1, len(clusters)):
                dt = clusters[i][0] - clusters[i - 1][0]
                if 0.18 <= dt <= 1.5:
                    iois.append(dt)
            if iois:
                med_ioi = float(np.median(iois))
                if med_ioi > 0:
                    est_bpm = 60.0 / med_ioi

        if est_bpm <= 30.0 or not np.isfinite(est_bpm):
            est_bpm = 120.0

        # Fold half/double tempo into natural piano practice range [64, 164]
        while est_bpm < 64.0:
            est_bpm *= 2.0
        while est_bpm > 164.0:
            est_bpm *= 0.5
        est_bpm = float(round(np.clip(est_bpm, 48.0, 208.0)))

        # 3. Find beat phase in [0, beat_sec) that best aligns with weighted MIDI note onsets
        beat_sec = 60.0 / est_bpm
        times = np.array([c[0] for c in clusters], dtype=np.float64)
        weights = np.array([c[1] for c in clusters], dtype=np.float64)

        angles = 2.0 * np.pi * (times / beat_sec)
        sin_sum = float(np.sum(weights * np.sin(angles)))
        cos_sum = float(np.sum(weights * np.cos(angles)))
        best_phase = (np.arctan2(sin_sum, cos_sum) / (2.0 * np.pi)) * beat_sec
        best_phase = float(best_phase % beat_sec)

        # Find the first strong musical onset to anchor Bar 1 near the start of the piece
        med_w = float(np.median(weights)) if weights.size > 0 else 0.0
        first_strong_t = float(times[0])
        for t_val, w_val in clusters[:12]:
            if w_val >= med_w * 0.45:
                first_strong_t = float(t_val)
                break

        k_steps = round((first_strong_t - best_phase) / beat_sec)
        bar1_midi_sec = best_phase + k_steps * beat_sec
        if bar1_midi_sec < 0.0:
            bar1_midi_sec += beat_sec

        # Frontend adds +0.8s lead-in offset to all parsed MIDI note times
        frontend_grid_offset = round(0.8 + bar1_midi_sec, 3)
        return {
            "bpm": est_bpm,
            "grid_offset": float(np.clip(frontend_grid_offset, 0.05, 10.0)),
        }
    except Exception as exc:
        print(f"[KeyFlow Rhythm] Rhythm estimation fallback: {exc}")
        return default_res


@spaces.GPU(duration=180)
def run_gpu_transcription_pipeline(
    input_path_str: str,
    should_isolate: bool = True,
    sens_val: float = 55.0,
    work_dir_str: str = "",
    engine: str = "auto",
) -> Dict[str, Any]:
    """
    ZeroGPU-decorated transcription pipeline.
    Runs BS-RoFormer / HT-Demucs 6S dedicated piano stem isolation and Transkun / ByteDance Piano Transcription on GPU,
    followed by harmonic and MIDI bleed filtering (with Librosa CQT fallback),
    returning a dict with {"midi_base64", "separator_used", "transcriber_used"}.
    """
    if not input_path_str:
        raise ValueError("Missing input audio path.")

    input_path = Path(input_path_str)
    safe_work_dir = None
    if work_dir_str:
        try:
            cand = Path(work_dir_str).resolve()
            if cand.is_relative_to(Path("/tmp").resolve()) and cand.is_dir():
                safe_work_dir = cand
        except Exception:
            safe_work_dir = None

    def _execute(work_dir: Path) -> Dict[str, Any]:
        t_sep_start = time.perf_counter()
        if should_isolate:
            target_audio, separator_used = isolate_piano_stem(input_path, work_dir)
        else:
            target_audio, separator_used = input_path, "none"
        t_sep_end = time.perf_counter()
        sep_s = round(t_sep_end - t_sep_start, 2)

        output_midi = work_dir / "transcribed.mid"
        transcriber_used = None
        req_engine = (engine or "auto").strip().lower()

        t_trans_start = time.perf_counter()
        if req_engine == "bytedance":
            if transcribe_with_bytedance(target_audio, output_midi):
                transcriber_used = "bytedance"
            elif transcribe_with_transkun(target_audio, output_midi):
                transcriber_used = "transkun"
        elif req_engine == "transkun":
            if transcribe_with_transkun(target_audio, output_midi):
                transcriber_used = "transkun"
            elif transcribe_with_bytedance(target_audio, output_midi):
                transcriber_used = "bytedance"
        else:  # "auto" or any fallback
            if transcribe_with_transkun(target_audio, output_midi):
                transcriber_used = "transkun"
            elif transcribe_with_bytedance(target_audio, output_midi):
                transcriber_used = "bytedance"

        if transcriber_used is not None:
            filter_midi_bleed_notes(output_midi, float(sens_val), transcriber=transcriber_used)
        else:
            if transcribe_with_librosa_cqt_to_midi(target_audio, output_midi, float(sens_val)):
                transcriber_used = "librosa-cqt"
        t_trans_end = time.perf_counter()
        trans_s = round(t_trans_end - t_trans_start, 2)

        print(f"[KeyFlow Space] separation={separator_used} {sep_s:.1f}s | transcription={transcriber_used} {trans_s:.1f}s")

        if not transcriber_used or not output_midi.exists() or output_midi.stat().st_size <= 80:
            raise RuntimeError("Could not transcribe piano notes from audio file.")

        rhythm = estimate_midi_and_audio_rhythm(target_audio, output_midi)

        return {
            "midi_base64": base64.b64encode(output_midi.read_bytes()).decode("ascii"),
            "separator_used": separator_used,
            "transcriber_used": transcriber_used,
            "bpm": rhythm["bpm"],
            "grid_offset": rhythm["grid_offset"],
            "timings": {
                "separation_s": sep_s,
                "transcription_s": trans_s,
            },
        }

    if safe_work_dir is not None:
        return _execute(safe_work_dir)

    os.makedirs("/tmp", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="keyflow_zerogpu_") as tmp_dir:
        return _execute(Path(tmp_dir))


async def health_check_handler() -> Dict[str, Any]:
    engines = []
    as_ver = None
    if HAS_AUDIO_SEPARATOR:
        engines.append("BS-RoFormer")
        try:
            as_ver = importlib.metadata.version("audio-separator")
        except Exception:
            try:
                as_ver = importlib.metadata.version("audio_separator")
            except Exception:
                as_ver = "installed"
    if HAS_DEMUCS:
        engines.append("HT-Demucs 6S")
    if HAS_TRANSKUN:
        engines.append("Transkun")
    if HAS_BYTEDANCE:
        engines.append("ByteDance Piano AI")
    if HAS_LIBROSA:
        engines.append("Librosa-CQT")
    if HAS_HOMR:
        engines.append("HOMR")
    if HAS_AUDIVERIS:
        engines.append("Audiveris")
    engines.append("Sheet-OMR")
    return {
        "status": "ok",
        "engines": engines,
        "audio_separator_version": as_ver,
        "import_errors": IMPORT_ERRORS,
        "installed_versions": _installed_versions(),
    }


async def transcribe_endpoint_handler(
    file: UploadFile = File(...),
    isolate_piano: str = Form("true"),
    sensitivity: str = Form("55"),
    engine: str = Form("auto"),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio filename.")

    should_isolate = str(isolate_piano).strip().lower() in ("true", "1", "yes")
    try:
        sens_val = float(sensitivity)
    except ValueError:
        sens_val = 55.0

    clean_title = Path(file.filename).stem or "Transcribed Song"
    raw_suffix = Path(file.filename).suffix.lower()
    suffix = raw_suffix if raw_suffix in ALLOWED_SUFFIXES else ".mp3"

    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 50 MB limit.")

    def _sync_transcribe() -> Dict[str, Any]:
        os.makedirs("/tmp", exist_ok=True)
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="keyflow_space_") as tmp_dir:
            work_dir = Path(tmp_dir)
            input_path = work_dir / f"upload{suffix}"
            input_path.write_bytes(raw_bytes)
            return run_gpu_transcription_pipeline(
                str(input_path),
                should_isolate=should_isolate,
                sens_val=sens_val,
                work_dir_str=str(work_dir),
                engine=engine,
            )

    try:
        pipeline_res = await run_in_threadpool(_sync_transcribe)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "midi_base64": pipeline_res["midi_base64"],
        "title": clean_title,
        "separator_used": pipeline_res.get("separator_used", "none"),
        "transcriber_used": pipeline_res.get("transcriber_used", "none"),
        "bpm": pipeline_res.get("bpm", 120.0),
        "grid_offset": pipeline_res.get("grid_offset", 0.8),
        "timings": pipeline_res.get("timings", {"separation_s": 0.0, "transcription_s": 0.0}),
    }


def render_pdf_to_images(pdf_path: Path, work_dir: Path, max_pages: int = 8) -> List[Path]:
    """Renders PDF pages to 200 DPI PNG images using pypdfium2."""
    if not HAS_PDFIUM:
        raise RuntimeError("pypdfium2 is required for PDF sheet music rendering.")
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_paths = []
    num_pages = min(len(pdf), max_pages)
    for i in range(num_pages):
        page = pdf[i]
        image = page.render(scale=2.0).to_pil()
        out_path = work_dir / f"page_{i + 1}.png"
        image.save(str(out_path))
        page_paths.append(out_path)
    return page_paths


DIATONIC_PATTERN = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B


def diatonic_step_to_midi(base_step_in_c: int, base_octave: int, step_offset: int) -> int:
    """Maps diatonic line/space offsets on standard staves to MIDI pitch numbers."""
    target_step = base_step_in_c + step_offset
    octave_shift = target_step // 7
    note_in_oct = target_step % 7
    midi = (base_octave * 12) + 12 + (octave_shift * 12) + DIATONIC_PATTERN[note_in_oct]
    return int(np.clip(midi, 21, 108))


def transcribe_sheet_image_cv_to_midi(image_paths: List[Path], output_midi_path: Path) -> bool:
    """
    Deterministic Computer Vision Grand-Staff Visual OMR:
    1. Grayscale + adaptive binarization to detect sheet ink.
    2. Horizontal projection profile to detect 5-line staves & line spacing (d).
    3. Group staves into Grand Staff pairs (Top = Treble, Bottom = Bass).
    4. Remove thin staff lines via vertical morphology and locate notehead blobs.
    5. Map vertical line/space position to MIDI pitch and horizontal position to timeline.
    6. Writes Right Hand (Track 1) and Left Hand (Track 2) to Standard MIDI via pretty_midi.
    """
    if not HAS_PIL or not HAS_PRETTY_MIDI or not HAS_NDI:
        return False

    all_notes: List[Dict[str, Any]] = []
    current_time_offset = 0.0
    system_duration = 7.0  # seconds per grand staff system

    for img_path in image_paths:
        try:
            with Image.open(str(img_path)) as pil_img:
                gray = pil_img.convert("L")
                w, h = gray.size
                if w < 100 or h < 100:
                    continue
                if w > 2400:
                    scale = 2400.0 / w
                    gray = gray.resize((2400, int(round(h * scale))), Image.Resampling.BILINEAR)
                    w, h = gray.size
                img_arr = np.array(gray)
        except Exception as e:
            print(f"[KeyFlow OMR] Could not load image {img_path}: {e}")
            continue

        # 1. Binarize: black ink = 1, white paper = 0
        thresh = float(np.mean(img_arr)) * 0.85
        binary = (img_arr < thresh).astype(np.uint8)

        # 2. Horizontal projection profile
        h_proj = np.sum(binary, axis=1)
        line_thresh = max(int(w * 0.22), int(np.percentile(h_proj, 88)))

        line_ys = []
        in_run = False
        run_start = 0
        for y in range(h):
            if h_proj[y] >= line_thresh:
                if not in_run:
                    in_run = True
                    run_start = y
            else:
                if in_run:
                    in_run = False
                    line_ys.append(int(round((run_start + y - 1) / 2.0)))
        if in_run:
            line_ys.append(int(round((run_start + h - 1) / 2.0)))

        if len(line_ys) < 5:
            continue

        # 3. Group lines into 5-line staves
        detected_staves = []
        i = 0
        while i <= len(line_ys) - 5:
            spacings = [line_ys[i + j + 1] - line_ys[i + j] for j in range(4)]
            avg_d = np.mean(spacings)
            std_d = np.std(spacings)
            if 6.0 <= avg_d <= 45.0 and std_d <= 3.5:
                staff_lines = line_ys[i:i + 5]
                detected_staves.append({
                    "lines": staff_lines,
                    "top": staff_lines[0],
                    "bottom": staff_lines[4],
                    "d": float(avg_d),
                    "center": float(np.mean(staff_lines))
                })
                i += 5
            else:
                i += 1

        if not detected_staves:
            continue

        # 4. Group into grand staff pairs (Top = Treble, Bottom = Bass)
        grand_staves = []
        s_idx = 0
        while s_idx < len(detected_staves):
            st1 = detected_staves[s_idx]
            if s_idx + 1 < len(detected_staves):
                st2 = detected_staves[s_idx + 1]
                gap = st2["top"] - st1["bottom"]
                if 1.5 * st1["d"] <= gap <= 10.0 * st1["d"]:
                    grand_staves.append({"treble": st1, "bass": st2, "d": (st1["d"] + st2["d"]) / 2.0})
                    s_idx += 2
                    continue
            grand_staves.append({"treble": st1, "bass": None, "d": st1["d"]})
            s_idx += 1

        # 5. Notehead segmentation: vertical morphological opening
        typical_d = int(round(np.median([gs["d"] for gs in grand_staves])))
        v_kernel_h = max(3, int(round(typical_d * 0.55)))
        v_kernel = np.ones((v_kernel_h, 1), dtype=np.uint8)
        note_mask = ndi.binary_opening(binary, structure=v_kernel)

        labeled, num_features = ndi.label(note_mask)
        objects = ndi.find_objects(labeled)

        # 6. Map notehead blobs to pitch and time
        for gs_idx, gs in enumerate(grand_staves):
            d = gs["d"]
            treble = gs["treble"]
            bass = gs["bass"]
            sys_time_start = current_time_offset + (gs_idx * system_duration)

            min_y = treble["top"] - int(d * 4.0)
            max_y = (bass["bottom"] if bass else treble["bottom"]) + int(d * 4.0)
            x_start_margin = int(w * 0.08)
            x_end_margin = int(w * 0.96)
            sys_width = max(100, x_end_margin - x_start_margin)

            for sl_y, sl_x in objects:
                if sl_y is None or sl_x is None:
                    continue
                y_min, y_max = sl_y.start, sl_y.stop
                x_min, x_max = sl_x.start, sl_x.stop

                yc = (y_min + y_max) / 2.0
                xc = (x_min + x_max) / 2.0

                if not (min_y <= yc <= max_y and x_start_margin <= xc <= x_end_margin):
                    continue

                blob_h = y_max - y_min
                blob_w = x_max - x_min

                if not (0.55 * d <= blob_h <= 1.9 * d and 0.55 * d <= blob_w <= 2.6 * d):
                    continue
                if not (0.35 <= (blob_w / max(1.0, blob_h)) <= 3.2):
                    continue

                hand = 'R'
                if bass is not None:
                    dist_to_treble = abs(yc - treble["center"])
                    dist_to_bass = abs(yc - bass["center"])
                    if dist_to_bass < dist_to_treble:
                        hand = 'L'

                if hand == 'R':
                    step_offset = int(round((treble["bottom"] - yc) / (d / 2.0)))
                    midi = diatonic_step_to_midi(base_step_in_c=2, base_octave=4, step_offset=step_offset)
                else:
                    ref_bottom = bass["bottom"] if bass else treble["bottom"]
                    step_offset = int(round((ref_bottom - yc) / (d / 2.0)))
                    midi = diatonic_step_to_midi(base_step_in_c=4, base_octave=2, step_offset=step_offset)

                t = sys_time_start + ((xc - x_start_margin) / float(sys_width)) * (system_duration * 0.92)
                all_notes.append({
                    "midi": midi,
                    "time": round(float(t), 3),
                    "duration": 0.48,
                    "hand": hand,
                    "velocity": 85 if hand == 'R' else 78
                })

        current_time_offset += len(grand_staves) * system_duration

    if not all_notes:
        return False

    all_notes.sort(key=lambda n: (n["time"], n["midi"]))

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    rh_inst = pretty_midi.Instrument(program=0, name="Right Hand")
    lh_inst = pretty_midi.Instrument(program=0, name="Left Hand")

    for n in all_notes:
        note_obj = pretty_midi.Note(
            velocity=n["velocity"],
            pitch=n["midi"],
            start=n["time"],
            end=n["time"] + n["duration"]
        )
        if n["hand"] == 'L':
            lh_inst.notes.append(note_obj)
        else:
            rh_inst.notes.append(note_obj)

    pm.instruments.append(rh_inst)
    pm.instruments.append(lh_inst)
    pm.write(str(output_midi_path))
    return output_midi_path.exists() and output_midi_path.stat().st_size > 80


def extract_musicxml_text_from_mxl(mxl_bytes: bytes) -> str:
    """Extracts score XML text from a compressed .mxl ZIP archive."""
    buf = io.BytesIO(mxl_bytes)
    with zipfile.ZipFile(buf, "r") as z:
        target_name = None
        for name in z.namelist():
            if name.lower() == "meta-inf/container.xml":
                try:
                    c_xml = z.read(name).decode("utf-8", errors="ignore")
                    m = re.search(r'full-path\s*=\s*["\']([^"\']+)["\']', c_xml, re.IGNORECASE)
                    if m:
                        target_name = m.group(1).strip()
                except Exception:
                    pass

        if target_name and target_name in z.namelist():
            return z.read(target_name).decode("utf-8", errors="ignore")

        candidates = [n for n in z.namelist() if (n.lower().endswith(".musicxml") or n.lower().endswith(".xml")) and not n.lower().startswith("meta-inf/")]
        if not candidates:
            raise ValueError("No score XML file found in .mxl archive.")
        candidates.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
        return z.read(candidates[0]).decode("utf-8", errors="ignore")


def merge_musicxml_pages(xml_pages: List[str]) -> str:
    """Merges multiple single-page MusicXML strings into a single continuous multi-measure score."""
    valid_pages = [x.strip() for x in xml_pages if x and "<score-partwise" in x]
    if not valid_pages:
        return xml_pages[0] if xml_pages else ""
    if len(valid_pages) == 1:
        return valid_pages[0]

    try:
        base_root = ET.fromstring(valid_pages[0])
        base_parts = base_root.findall("part")
        if not base_parts:
            return valid_pages[0]

        measure_counters = [len(p.findall("measure")) for p in base_parts]

        for page_xml in valid_pages[1:]:
            try:
                page_root = ET.fromstring(page_xml)
                page_parts = page_root.findall("part")
                for p_idx, base_part in enumerate(base_parts):
                    if p_idx >= len(page_parts):
                        break
                    for m_el in page_parts[p_idx].findall("measure"):
                        measure_counters[p_idx] += 1
                        m_el.set("number", str(measure_counters[p_idx]))
                        base_part.append(m_el)
            except Exception:
                continue

        return ET.tostring(base_root, encoding="unicode", xml_declaration=True)
    except Exception:
        return valid_pages[0]


def transcribe_sheet_with_homr(image_paths: List[Path], work_dir: Path) -> Optional[str]:
    """
    Runs deep-learning Optical Music Recognition via `homr` on one or more page images
    and returns the resulting MusicXML string (or None on failure).
    """
    if not HAS_HOMR or not HOMR_CMD or not image_paths:
        return None

    try:
        cmd = list(HOMR_CMD) + [str(p) for p in image_paths]
        subprocess.run(
            cmd,
            cwd=str(work_dir),
            check=True,
            capture_output=True,
            timeout=240,
        )

        # 1. Check for homr's native multi-page merged output first
        first_stem = image_paths[0].stem
        merged_cand = work_dir / f"merged_{first_stem}.musicxml"
        if merged_cand.exists() and merged_cand.stat().st_size > 100:
            return merged_cand.read_text(encoding="utf-8", errors="ignore")

        # 2. Otherwise collect per-page .musicxml files in page order
        page_xmls: List[str] = []
        for img_p in image_paths:
            for ext in (".musicxml", ".xml"):
                cand = img_p.with_suffix(ext)
                if cand.exists() and cand.stat().st_size > 100:
                    page_xmls.append(cand.read_text(encoding="utf-8", errors="ignore"))
                    break

        if page_xmls:
            return merge_musicxml_pages(page_xmls)

        # 3. Fallback scan for any .musicxml or .mxl generated in work_dir
        for f_path in sorted(work_dir.glob("*.musicxml")):
            if f_path.stat().st_size > 100:
                page_xmls.append(f_path.read_text(encoding="utf-8", errors="ignore"))
        if page_xmls:
            return merge_musicxml_pages(page_xmls)

        for mxl_path in sorted(work_dir.glob("*.mxl")):
            if mxl_path.stat().st_size > 100:
                return extract_musicxml_text_from_mxl(mxl_path.read_bytes())

    except subprocess.CalledProcessError as exc:
        err_msg = (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc))[-1000:]
        print(f"[KeyFlow OMR] homr failed: {err_msg}")
    except Exception as exc:
        print(f"[KeyFlow OMR] homr error: {str(exc)[:500]}")
    return None


def transcribe_sheet_with_cli_omr(
    input_path: Path,
    image_paths: List[Path],
    work_dir: Path,
) -> Tuple[Optional[str], str]:
    """Tries Audiveris or oemer CLI if installed on PATH. Returns (musicxml_or_none, engine_name)."""
    if HAS_AUDIVERIS:
        try:
            out_dir = work_dir / "audiveris_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = ["audiveris", "-batch", "-export", "-output", str(out_dir), str(input_path)]
            subprocess.run(cmd, check=True, capture_output=True, timeout=240)
            mxl_files = sorted(out_dir.rglob("*.mxl"))
            if mxl_files:
                xml_pages = [extract_musicxml_text_from_mxl(p.read_bytes()) for p in mxl_files]
                return merge_musicxml_pages(xml_pages), "audiveris"
            xml_files = sorted(out_dir.rglob("*.musicxml")) + sorted(out_dir.rglob("*.xml"))
            if xml_files:
                xml_pages = [p.read_text(encoding="utf-8", errors="ignore") for p in xml_files]
                return merge_musicxml_pages(xml_pages), "audiveris"
        except Exception as exc:
            print(f"[KeyFlow OMR] Audiveris fallback error: {exc}")

    if HAS_OEMER and image_paths:
        try:
            out_dir = work_dir / "oemer_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            xml_pages = []
            for img_p in image_paths:
                cmd = ["oemer", "-o", str(out_dir), str(img_p)]
                subprocess.run(cmd, check=True, capture_output=True, timeout=240)
                for cand in sorted(out_dir.glob(f"{img_p.stem}*.musicxml")):
                    if cand.stat().st_size > 100:
                        xml_pages.append(cand.read_text(encoding="utf-8", errors="ignore"))
            if xml_pages:
                return merge_musicxml_pages(xml_pages), "oemer"
        except Exception as exc:
            print(f"[KeyFlow OMR] oemer fallback error: {exc}")

    return None, "none"


def transcribe_sheet_image_cv_to_musicxml(
    image_paths: List[Path],
    title: str = "Sheet Music",
) -> Optional[str]:
    """
    Rhythm- and Grand-Staff-aware CV fallback OMR that detects 5-line staves,
    hollow vs. filled noteheads, and chord clusters, emitting a valid 2-staff
    Grand Staff MusicXML document (<score-partwise version="3.1">).
    """
    if not HAS_PIL or not HAS_NDI or not image_paths:
        return None

    NOTE_STEPS = [
        ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
        ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
    ]
    DIVISIONS = 4
    MEASURE_TICKS = 16  # 4/4 time

    def _midi_to_xml_pitch(midi: int) -> str:
        m = int(np.clip(midi, 21, 108))
        step, alter = NOTE_STEPS[m % 12]
        octave = (m // 12) - 1
        alter_xml = f"<alter>{alter}</alter>" if alter else ""
        return f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"

    def _ticks_to_type(ticks: int) -> str:
        if ticks >= 8:
            return "half"
        if ticks >= 4:
            return "quarter"
        return "eighth"

    measures: List[Tuple[List[Tuple[int, int, bool]], List[Tuple[int, int, bool]]]] = []

    for img_path in image_paths:
        try:
            with Image.open(str(img_path)) as pil_img:
                gray = pil_img.convert("L")
                w, h = gray.size
                if w < 100 or h < 100:
                    continue
                if w > 2400:
                    scale = 2400.0 / w
                    gray = gray.resize((2400, int(round(h * scale))), Image.Resampling.BILINEAR)
                    w, h = gray.size
                img_arr = np.array(gray)
        except Exception:
            continue

        thresh = float(np.mean(img_arr)) * 0.85
        binary = (img_arr < thresh).astype(np.uint8)

        h_proj = np.sum(binary, axis=1)
        line_thresh = max(int(w * 0.22), int(np.percentile(h_proj, 88)))

        line_ys: List[int] = []
        in_run = False
        run_start = 0
        for y in range(h):
            if h_proj[y] >= line_thresh:
                if not in_run:
                    in_run = True
                    run_start = y
            elif in_run:
                in_run = False
                line_ys.append(int(round((run_start + y - 1) / 2.0)))
        if in_run:
            line_ys.append(int(round((run_start + h - 1) / 2.0)))

        if len(line_ys) < 5:
            continue

        detected_staves = []
        i = 0
        while i <= len(line_ys) - 5:
            spacings = [line_ys[i + j + 1] - line_ys[i + j] for j in range(4)]
            avg_d = float(np.mean(spacings))
            std_d = float(np.std(spacings))
            if 6.0 <= avg_d <= 45.0 and std_d <= 3.5:
                staff_lines = line_ys[i:i + 5]
                detected_staves.append({
                    "lines": staff_lines,
                    "top": staff_lines[0],
                    "bottom": staff_lines[4],
                    "d": avg_d,
                    "center": float(np.mean(staff_lines)),
                })
                i += 5
            else:
                i += 1

        if not detected_staves:
            continue

        grand_staves = []
        s_idx = 0
        while s_idx < len(detected_staves):
            st1 = detected_staves[s_idx]
            if s_idx + 1 < len(detected_staves):
                st2 = detected_staves[s_idx + 1]
                gap = st2["top"] - st1["bottom"]
                if 1.5 * st1["d"] <= gap <= 10.0 * st1["d"]:
                    grand_staves.append({"treble": st1, "bass": st2, "d": (st1["d"] + st2["d"]) / 2.0})
                    s_idx += 2
                    continue
            grand_staves.append({"treble": st1, "bass": None, "d": st1["d"]})
            s_idx += 1

        typical_d = int(round(np.median([gs["d"] for gs in grand_staves])))
        v_kernel_h = max(3, int(round(typical_d * 0.55)))
        v_kernel = np.ones((v_kernel_h, 1), dtype=np.uint8)
        note_mask = ndi.binary_opening(binary, structure=v_kernel)

        labeled, _ = ndi.label(note_mask)
        objects = ndi.find_objects(labeled)

        for gs in grand_staves:
            d = gs["d"]
            treble = gs["treble"]
            bass = gs["bass"]
            min_y = treble["top"] - int(d * 4.0)
            max_y = (bass["bottom"] if bass else treble["bottom"]) + int(d * 4.0)
            x_start_margin = int(w * 0.08)
            x_end_margin = int(w * 0.96)

            sys_events: List[Dict[str, Any]] = []
            for sl_y, sl_x in objects:
                if sl_y is None or sl_x is None:
                    continue
                y_min, y_max = sl_y.start, sl_y.stop
                x_min, x_max = sl_x.start, sl_x.stop
                yc = (y_min + y_max) / 2.0
                xc = (x_min + x_max) / 2.0
                if not (min_y <= yc <= max_y and x_start_margin <= xc <= x_end_margin):
                    continue

                blob_h = y_max - y_min
                blob_w = x_max - x_min
                if not (0.55 * d <= blob_h <= 1.9 * d and 0.55 * d <= blob_w <= 2.6 * d):
                    continue
                if not (0.35 <= (blob_w / max(1.0, blob_h)) <= 3.2):
                    continue

                # Inspect inner core (35%..65%) to classify hollow vs. filled notehead
                iy0 = y_min + int(round(blob_h * 0.35))
                iy1 = max(iy0 + 1, y_min + int(round(blob_h * 0.65)))
                ix0 = x_min + int(round(blob_w * 0.35))
                ix1 = max(ix0 + 1, x_min + int(round(blob_w * 0.65)))
                inner_core = binary[iy0:iy1, ix0:ix1]
                is_hollow = bool(inner_core.size > 0 and float(np.mean(inner_core)) < 0.45)

                staff_num = 1
                if bass is not None and abs(yc - bass["center"]) < abs(yc - treble["center"]):
                    staff_num = 2

                if staff_num == 1:
                    step_offset = int(round((treble["bottom"] - yc) / (d / 2.0)))
                    midi = diatonic_step_to_midi(base_step_in_c=2, base_octave=4, step_offset=step_offset)
                else:
                    ref_bottom = bass["bottom"] if bass else treble["bottom"]
                    step_offset = int(round((ref_bottom - yc) / (d / 2.0)))
                    midi = diatonic_step_to_midi(base_step_in_c=4, base_octave=2, step_offset=step_offset)

                sys_events.append({
                    "xc": float(xc),
                    "midi": midi,
                    "staff": staff_num,
                    "hollow": is_hollow,
                })

            if not sys_events:
                continue

            sys_events.sort(key=lambda e: (e["xc"], e["staff"], e["midi"]))

            # Group into horizontal onset clusters (within 0.6 * d)
            clusters: List[List[Dict[str, Any]]] = []
            cur_c = [sys_events[0]]
            for ev in sys_events[1:]:
                if abs(ev["xc"] - cur_c[0]["xc"]) <= max(4.0, d * 0.6):
                    cur_c.append(ev)
                else:
                    clusters.append(cur_c)
                    cur_c = [ev]
            clusters.append(cur_c)

            cur_t_notes: List[Tuple[int, int, bool]] = []
            cur_b_notes: List[Tuple[int, int, bool]] = []
            cur_ticks = 0

            for c_idx, cluster in enumerate(clusters):
                has_hollow = any(ev["hollow"] for ev in cluster)
                if has_hollow:
                    dur_ticks = 8
                elif c_idx + 1 < len(clusters) and (clusters[c_idx + 1][0]["xc"] - cluster[0]["xc"]) < (d * 2.3):
                    dur_ticks = 2
                else:
                    dur_ticks = 4

                if cur_ticks + dur_ticks > MEASURE_TICKS and (cur_t_notes or cur_b_notes):
                    measures.append((cur_t_notes, cur_b_notes))
                    cur_t_notes = []
                    cur_b_notes = []
                    cur_ticks = 0

                t_first = True
                b_first = True
                for ev in cluster:
                    if ev["staff"] == 1:
                        cur_t_notes.append((ev["midi"], dur_ticks, not t_first))
                        t_first = False
                    else:
                        cur_b_notes.append((ev["midi"], dur_ticks, not b_first))
                        b_first = False
                if t_first:
                    cur_t_notes.append((-1, dur_ticks, False))
                if b_first:
                    cur_b_notes.append((-1, dur_ticks, False))
                cur_ticks += dur_ticks

            if cur_t_notes or cur_b_notes:
                measures.append((cur_t_notes, cur_b_notes))

    if not measures:
        return None

    safe_title = xml_escape(title or "Sheet Music")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<score-partwise version="3.1">',
        f'  <work><work-title>{safe_title}</work-title></work>',
        '  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>',
        '  <part id="P1">',
    ]

    for m_num, (t_notes, b_notes) in enumerate(measures, start=1):
        lines.append(f'    <measure number="{m_num}">')
        if m_num == 1:
            lines.append('      <attributes>')
            lines.append(f'        <divisions>{DIVISIONS}</divisions>')
            lines.append('        <key><fifths>0</fifths></key>')
            lines.append('        <time><beats>4</beats><beat-type>4</beat-type></time>')
            lines.append('        <staves>2</staves>')
            lines.append('        <clef number="1"><sign>G</sign><line>2</line></clef>')
            lines.append('        <clef number="2"><sign>F</sign><line>4</line></clef>')
            lines.append('      </attributes>')
            lines.append('      <direction placement="above"><sound tempo="110"/></direction>')

        t_used = 0
        for midi, dur, is_chord in t_notes:
            if not is_chord:
                t_used += dur
            if midi < 0:
                lines.append(f'      <note><rest/><duration>{dur}</duration><voice>1</voice><type>{_ticks_to_type(dur)}</type><staff>1</staff></note>')
            else:
                chord_tag = "<chord/>" if is_chord else ""
                lines.append(f'      <note>{chord_tag}{_midi_to_xml_pitch(midi)}<duration>{dur}</duration><voice>1</voice><type>{_ticks_to_type(dur)}</type><staff>1</staff></note>')
        if t_used < MEASURE_TICKS:
            rem = MEASURE_TICKS - t_used
            lines.append(f'      <note><rest/><duration>{rem}</duration><voice>1</voice><staff>1</staff></note>')

        lines.append(f'      <backup><duration>{MEASURE_TICKS}</duration></backup>')

        b_used = 0
        for midi, dur, is_chord in b_notes:
            if not is_chord:
                b_used += dur
            if midi < 0:
                lines.append(f'      <note><rest/><duration>{dur}</duration><voice>2</voice><type>{_ticks_to_type(dur)}</type><staff>2</staff></note>')
            else:
                chord_tag = "<chord/>" if is_chord else ""
                lines.append(f'      <note>{chord_tag}{_midi_to_xml_pitch(midi)}<duration>{dur}</duration><voice>2</voice><type>{_ticks_to_type(dur)}</type><staff>2</staff></note>')
        if b_used < MEASURE_TICKS:
            rem = MEASURE_TICKS - b_used
            lines.append(f'      <note><rest/><duration>{rem}</duration><voice>2</voice><staff>2</staff></note>')

        lines.append('    </measure>')

    lines.append('  </part>')
    lines.append('</score-partwise>')
    return "\n".join(lines)


async def omr_endpoint_handler(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Phase 5: Optical Music Recognition (OMR) & Sheet Music Endpoint.
    Accepts .pdf, .png, .jpg, .jpeg, .webp, .musicxml, or .mxl and returns
    either parsed MusicXML or base64 Standard MIDI.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing sheet music filename.")

    clean_title = Path(file.filename).stem or "Sheet Music Song"
    raw_suffix = Path(file.filename).suffix.lower()
    if raw_suffix not in ALLOWED_OMR_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported sheet music file extension: {raw_suffix}")

    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded sheet music file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Sheet music file exceeds 50 MB limit.")

    # 1. Digital MusicXML or MXL uploads: extract and return MusicXML directly
    if raw_suffix in (".musicxml", ".xml"):
        return {
            "musicxml": raw_bytes.decode("utf-8", errors="ignore"),
            "title": clean_title,
        }

    if raw_suffix == ".mxl":
        try:
            xml_text = extract_musicxml_text_from_mxl(raw_bytes)
            return {
                "musicxml": xml_text,
                "title": clean_title,
            }
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not unpack .mxl archive: {exc}")

    # 2. Visual scans (.pdf, .png, .jpg, .jpeg, .webp): run 3-tier OMR pipeline
    def _sync_omr() -> Dict[str, Any]:
        os.makedirs("/tmp", exist_ok=True)
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="keyflow_omr_") as tmp_dir:
            work_dir = Path(tmp_dir)
            input_path = work_dir / f"upload{raw_suffix}"
            input_path.write_bytes(raw_bytes)

            image_paths = []
            if raw_suffix == ".pdf":
                image_paths = render_pdf_to_images(input_path, work_dir, max_pages=8)
            else:
                image_paths = [input_path]

            if not image_paths:
                raise RuntimeError("No rendered pages available for OMR processing.")

            # Tier 1: homr (deep-learning MusicXML OMR)
            xml_text = transcribe_sheet_with_homr(image_paths, work_dir)
            omr_engine = "homr"

            # Tier 2: Audiveris / oemer CLI
            if not xml_text:
                xml_text, omr_engine = transcribe_sheet_with_cli_omr(input_path, image_paths, work_dir)

            # Tier 3: Rhythm- & Grand-Staff-aware CV-to-MusicXML fallback
            if not xml_text:
                xml_text = transcribe_sheet_image_cv_to_musicxml(image_paths, title=clean_title)
                omr_engine = "cv-musicxml"

            if xml_text and xml_text.strip():
                return {
                    "musicxml": xml_text,
                    "title": clean_title,
                    "omr_engine": omr_engine,
                }

            # Final safety net: legacy CV-to-MIDI
            output_midi = work_dir / "transcribed_sheet.mid"
            ok = transcribe_sheet_image_cv_to_midi(image_paths, output_midi)
            if not ok or not output_midi.exists():
                raise RuntimeError("No readable piano staves or noteheads detected in sheet music image.")

            return {
                "midi_base64": base64.b64encode(output_midi.read_bytes()).decode("ascii"),
                "title": clean_title,
                "omr_engine": "cv-midi",
            }

    try:
        return await run_in_threadpool(_sync_omr)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def register_keyflow_routes(fastapi_app: FastAPI) -> FastAPI:
    """Idempotently attaches CORS, /health, /api/transcribe, and /api/omr endpoints to Gradio's FastAPI app."""
    if fastapi_app is None or getattr(fastapi_app, "_keyflow_routes_registered", False):
        return fastapi_app

    fastapi_app._keyflow_routes_registered = True
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    fastapi_app.get("/health")(health_check_handler)
    fastapi_app.get("/api/health")(health_check_handler)
    fastapi_app.post("/api/transcribe")(transcribe_endpoint_handler)
    fastapi_app.post("/api/omr")(omr_endpoint_handler)
    return fastapi_app


# Hook Gradio's App.create_app so routes are guaranteed to be attached even when demo.app is initialized lazily
if hasattr(gr, "routes") and hasattr(gr.routes, "App") and hasattr(gr.routes.App, "create_app"):
    _orig_create_app = gr.routes.App.create_app

    def _patched_create_app(*args, **kwargs):
        created_app = _orig_create_app(*args, **kwargs)
        register_keyflow_routes(created_app)
        return created_app

    gr.routes.App.create_app = _patched_create_app


# 1. Create the Gradio Blocks UI and wire the @spaces.GPU function so ZeroGPU detects it during startup
with gr.Blocks(title="KeyFlow AI Audio & Sheet Music Server") as demo:
    gr.Markdown(
        """
        # 🎹 KeyFlow AI Audio & Sheet Music Server (ZeroGPU Online)
        This Hugging Face Gradio Space powers **Phase 3 AI Audio-to-MIDI Transcription** (`BS-RoFormer` + `Transkun` / `ByteDance Piano AI`) and **Phase 5 Sheet Music Optical Music Recognition (OMR)** on **ZeroGPU** for the **KeyFlow Virtual Piano** iPad web app.

        ### Active API Endpoints
        - **`GET /health`** — Returns server health & active engines (`BS-RoFormer`, `HT-Demucs 6S`, `Transkun`, `ByteDance Piano AI`, `Sheet-OMR`).
        - **`POST /api/transcribe`** — Accepts `file`, `isolate_piano`, `sensitivity`, and optional `engine` (`multipart/form-data`) and returns `{"midi_base64": "...", "title": "...", "separator_used": "...", "transcriber_used": "..."}`.
        - **`POST /api/omr`** — Accepts sheet music (`.pdf`, `.png`, `.jpg`, `.musicxml`, `.mxl`) and returns `{"musicxml": "..."}` or `{"midi_base64": "..."}`.

        **How to connect from KeyFlow:** Copy this Space's public `.hf.space` URL (e.g. `https://<username>-<space-name>.hf.space`) into the **AI Server URL** field inside KeyFlow's **AI Audio** or **Sheet Music** modal and tap **Check**.
        """
    )
    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Audio File (.mp3, .wav, .m4a)")
        isolate_in = gr.Checkbox(value=True, label="Isolate Piano Stem (BS-RoFormer)")
        sens_in = gr.Slider(minimum=15, maximum=85, value=55, step=1, label="Sensitivity (%)")
    transcribe_btn = gr.Button("Transcribe Audio to MIDI (Base64)")
    midi_b64_out = gr.Textbox(label="Base64 Standard MIDI Output")

    def _gradio_transcribe(audio_file, isolate_piano, sensitivity):
        pipeline_res = run_gpu_transcription_pipeline(
            audio_file,
            should_isolate=isolate_piano,
            sens_val=sensitivity,
            engine="auto",
        )
        return pipeline_res["midi_base64"]

    transcribe_btn.click(
        fn=_gradio_transcribe,
        inputs=[audio_in, isolate_in, sens_in],
        outputs=midi_b64_out,
        api_name="transcribe",
    )

# 2. Reference Gradio's built-in FastAPI instance (if already initialized) and attach routes
app = getattr(demo, "app", None)
if app is not None:
    register_keyflow_routes(app)


if __name__ == "__main__":
    try:
        demo.launch(ssr_mode=False)
    except TypeError:
        demo.launch()
