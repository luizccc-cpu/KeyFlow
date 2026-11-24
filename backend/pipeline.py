"""
KeyFlow (Piano-Flow) — Shared AI Audio-to-MIDI Pipeline & Security Guards
=========================================================================
Shared between `backend/server.py` (local FastAPI server) and `backend/app.py`
(Hugging Face ZeroGPU Gradio Space).
"""

import asyncio
import base64
from contextlib import asynccontextmanager
import hashlib
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB limit
MAX_CHECKPOINT_BYTES = 300 * 1024 * 1024  # 300 MB safety ceiling for checkpoint download
ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".webm", ".mp4"}

# Records import failures so debug health checks can report why an engine is unavailable.
IMPORT_ERRORS: Dict[str, str] = {}

try:
    from audio_separator.separator import Separator
    HAS_AUDIO_SEPARATOR = True
except Exception as _exc:
    Separator = None  # type: ignore[assignment]
    HAS_AUDIO_SEPARATOR = False
    IMPORT_ERRORS["audio_separator"] = f"{type(_exc).__name__}: {_exc}"[:500]

try:
    import torch
    HAS_TORCH = True
except Exception as _exc:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False
    IMPORT_ERRORS["torch"] = f"{type(_exc).__name__}: {_exc}"[:500]

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    sf = None  # type: ignore[assignment]
    HAS_SOUNDFILE = False

try:
    from scipy.signal import butter, sosfiltfilt
    HAS_SCIPY_SIGNAL = True
except Exception:
    butter = None  # type: ignore[assignment]
    sosfiltfilt = None  # type: ignore[assignment]
    HAS_SCIPY_SIGNAL = False

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    librosa = None  # type: ignore[assignment]
    HAS_LIBROSA = False

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except Exception:
    pretty_midi = None  # type: ignore[assignment]
    HAS_PRETTY_MIDI = False

try:
    from piano_transcription_inference import PianoTranscription, sample_rate as BD_SAMPLE_RATE
    HAS_BYTEDANCE = True
except Exception as _exc:
    PianoTranscription = None  # type: ignore[assignment]
    HAS_BYTEDANCE = False
    BD_SAMPLE_RATE = 16000
    IMPORT_ERRORS["piano_transcription_inference"] = f"{type(_exc).__name__}: {_exc}"[:500]

try:
    from basic_pitch.inference import predict as basic_pitch_predict
    HAS_BASIC_PITCH = True
except Exception:
    basic_pitch_predict = None  # type: ignore[assignment]
    HAS_BASIC_PITCH = False

for _name, _msg in IMPORT_ERRORS.items():
    print(f"[KeyFlow Backend] Import failed for {_name}: {_msg}")


def _installed_versions() -> Dict[str, Any]:
    """Returns installed distribution versions (None if not installed), independent of import success."""
    out: Dict[str, Any] = {"python": sys.version.split()[0]}
    for dist in (
        "numpy",
        "torch",
        "torchaudio",
        "audio-separator",
        "onnxruntime-gpu",
        "onnxruntime",
        "transkun",
        "demucs",
        "piano_transcription_inference",
        "librosa",
        "audioread",
        "setuptools",
    ):
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

# BS-RoFormer SW 6-stem model (BS-Roformer-SW.ckpt with BS-Roformer-SW.yaml),
# extracting ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano']
PIANO_SEPARATOR_MODEL = os.environ.get("KEYFLOW_SEPARATOR_MODEL", "BS-Roformer-SW.ckpt")
STEM_PREP_MODE = os.environ.get("KEYFLOW_STEM_PREP", "minimal")
SEPARATOR_OVERLAP = os.environ.get("KEYFLOW_SEPARATOR_OVERLAP", "")


def get_secure_cache_dir(subdir: str = "") -> Path:
    """
    Returns a private, symlink-checked cache directory (mode 0700 where supported).
    Respects `KEYFLOW_CACHE_DIR` or defaults to `<tempdir>/keyflow_cache`.
    """
    env_dir = os.environ.get("KEYFLOW_CACHE_DIR", "").strip()
    base = Path(env_dir) if env_dir else (Path(tempfile.gettempdir()) / "keyflow_cache")
    if base.is_symlink():
        raise RuntimeError(f"Refusing symlink cache directory: {base}")
    base.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(base, 0o700)
    except Exception:
        pass

    if not subdir:
        return base

    target = base / subdir
    if target.is_symlink():
        raise RuntimeError(f"Refusing symlink cache subdirectory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target, 0o700)
    except Exception:
        pass
    return target


MODEL_CACHE_DIR = get_secure_cache_dir("audio-separator-models")
BYTEDANCE_CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
BYTEDANCE_CHECKPOINT_PATH = get_secure_cache_dir("checkpoints") / "piano_model.pth"

_separator_lock = threading.Lock()
_cached_separator = None

_bytedance_lock = threading.Lock()
_cached_bytedance = None
_cached_bytedance_device: Optional[str] = None

_checkpoint_lock = threading.Lock()


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
        if not HAS_AUDIO_SEPARATOR or Separator is None:
            return None
        sig = inspect.signature(Separator.__init__)
        if "mdxc_params" not in sig.parameters:
            print("[KeyFlow Backend] WARNING: 'mdxc_params' parameter not found in Separator.__init__ signature, skipping overlap override.")
            return None
        default_val = sig.parameters["mdxc_params"].default
        params = dict(default_val) if isinstance(default_val, dict) else {}
        params["overlap"] = overlap_val
        return params
    except Exception as exc:
        print(f"[KeyFlow Backend] WARNING: Failed to inspect Separator.__init__ signature ({exc}), skipping overlap override.")
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower()


def ensure_bytedance_checkpoint() -> Path:
    """
    Thread-safely streams the ByteDance Piano Transcription model weights into a private
    cache directory, enforcing HTTPS, size bounds, and optional KEYFLOW_BYTEDANCE_SHA256 verification.
    """
    expected_sha = os.environ.get("KEYFLOW_BYTEDANCE_SHA256", "").strip().lower()
    if BYTEDANCE_CHECKPOINT_PATH.is_symlink():
        raise RuntimeError("Refusing symlink checkpoint path.")

    if BYTEDANCE_CHECKPOINT_PATH.exists() and BYTEDANCE_CHECKPOINT_PATH.stat().st_size > 10_000_000:
        if not expected_sha or _sha256_file(BYTEDANCE_CHECKPOINT_PATH) == expected_sha:
            return BYTEDANCE_CHECKPOINT_PATH

    with _checkpoint_lock:
        if BYTEDANCE_CHECKPOINT_PATH.is_symlink():
            raise RuntimeError("Refusing symlink checkpoint path.")
        if BYTEDANCE_CHECKPOINT_PATH.exists() and BYTEDANCE_CHECKPOINT_PATH.stat().st_size > 10_000_000:
            if not expected_sha or _sha256_file(BYTEDANCE_CHECKPOINT_PATH) == expected_sha:
                return BYTEDANCE_CHECKPOINT_PATH

        if not BYTEDANCE_CHECKPOINT_URL.lower().startswith("https://"):
            raise RuntimeError("Refusing non-HTTPS checkpoint download URL.")

        tmp_download = BYTEDANCE_CHECKPOINT_PATH.with_name(
            f"piano_model_{os.getpid()}_{threading.get_ident()}.pth.part"
        )
        hasher = hashlib.sha256()
        total_bytes = 0
        try:
            req = urllib.request.Request(
                BYTEDANCE_CHECKPOINT_URL,
                headers={"User-Agent": "KeyFlow-Backend/3.2"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                with tmp_download.open("wb") as out_fh:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > MAX_CHECKPOINT_BYTES:
                            raise RuntimeError("Checkpoint download exceeded 300 MB safety limit.")
                        hasher.update(chunk)
                        out_fh.write(chunk)

            if total_bytes <= 10_000_000:
                raise RuntimeError("Downloaded checkpoint is unexpectedly small or incomplete.")

            if expected_sha and hasher.hexdigest().lower() != expected_sha:
                raise RuntimeError("ByteDance checkpoint SHA-256 verification failed.")

            os.replace(tmp_download, BYTEDANCE_CHECKPOINT_PATH)
            try:
                os.chmod(BYTEDANCE_CHECKPOINT_PATH, 0o600)
            except Exception:
                pass
        finally:
            if tmp_download.exists():
                try:
                    tmp_download.unlink()
                except Exception:
                    pass
        return BYTEDANCE_CHECKPOINT_PATH


def prefetch_separator_model() -> None:
    """
    Thread-safely pre-downloads BS-RoFormer model weights outside @spaces.GPU duration.
    Acquires `_separator_lock` so startup prefetch never races with an early request.
    """
    if not HAS_AUDIO_SEPARATOR or Separator is None:
        return
    try:
        with _separator_lock:
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            target_ckpt = MODEL_CACHE_DIR / PIANO_SEPARATOR_MODEL
            if target_ckpt.exists():
                return

            sep = Separator(model_file_dir=str(MODEL_CACHE_DIR))
            if hasattr(sep, "download_model_files"):
                print(f"[KeyFlow Backend] Prefetching separator model '{PIANO_SEPARATOR_MODEL}' via download_model_files...")
                sep.download_model_files(PIANO_SEPARATOR_MODEL)
                print(f"[KeyFlow Backend] Separator model '{PIANO_SEPARATOR_MODEL}' prefetched successfully.")
            else:
                print("[KeyFlow Backend] Separator.download_model_files not found; skipping CPU prefetch to avoid CUDA initialization outside @spaces.GPU.")
    except Exception as exc:
        print(f"[KeyFlow Backend] Separator model prefetch warning (non-fatal): {exc}")


_NON_PIANO_STEM_TAGS = (
    "(vocals)",
    "(drums)",
    "(bass)",
    "(guitar)",
    "(other)",
    "(instrumental)",
    "_vocals.",
    "_drums.",
    "_bass.",
    "_guitar.",
    "_other.",
)


def _select_piano_stem_path(candidates: List[Path]) -> Optional[Path]:
    """
    Selects the piano stem from `audio-separator` outputs without matching an input
    filename that happens to contain 'piano' (e.g. `my_piano_song_(Vocals).wav`).
    """
    existing = [
        p for p in candidates
        if p is not None and p.exists() and p.is_file() and p.stat().st_size > 1000
    ]
    if not existing:
        return None

    for p in existing:
        if "(piano)" in p.name.lower():
            return p

    for p in existing:
        lower = p.name.lower()
        if any(tag in lower for tag in _NON_PIANO_STEM_TAGS):
            continue
        if re.search(r"(?:^|[_\-\s(\[])piano(?:$|[_\-\s)\].])", lower):
            return p

    for p in existing:
        lower = p.name.lower()
        if any(tag in lower for tag in _NON_PIANO_STEM_TAGS):
            continue
        if "piano" in lower:
            return p

    return None


def _legacy_clean_isolated_piano_audio(stem_wav_path: Path, work_dir: Path) -> Path:
    """
    Legacy post-Demucs harmonic and piano-range filtering:
    1. Loads audio at ByteDance sample rate (16 kHz mono).
    2. Strips residual drum/percussive transients via HPSS (margin 1.0 harm vs 2.2 perc).
    3. Bandpasses to piano fundamental range (27.5 Hz A0 to 4400 Hz C8).
    4. Normalizes peak amplitude and saves to piano_cleaned.wav.
    """
    if not HAS_LIBROSA or librosa is None:
        return stem_wav_path

    try:
        y, sr = librosa.load(str(stem_wav_path), sr=BD_SAMPLE_RATE, mono=True)
        if y.size == 0 or np.max(np.abs(y)) < 1e-6:
            return stem_wav_path

        y_harm, _ = librosa.effects.hpss(y, margin=(1.0, 2.2))

        y_filtered = y_harm
        if HAS_SCIPY_SIGNAL and butter is not None and sosfiltfilt is not None and sr > 8800:
            try:
                nyq = 0.5 * sr
                low = 27.5 / nyq
                high = min(4400.0, nyq - 50.0) / nyq
                sos = butter(2, [low, high], btype="bandpass", output="sos")
                y_filtered = sosfiltfilt(sos, y_harm)
            except Exception as f_err:
                print(f"[KeyFlow Backend] Bandpass filter skipped: {f_err}")
                y_filtered = y_harm

        peak = float(np.max(np.abs(y_filtered)))
        y_norm = y_filtered * (0.95 / peak) if peak > 1e-6 else y_filtered

        cleaned_path = work_dir / "piano_cleaned.wav"
        if HAS_SOUNDFILE and sf is not None:
            sf.write(str(cleaned_path), y_norm.astype(np.float32), sr)
            if cleaned_path.exists() and cleaned_path.stat().st_size > 1000:
                return cleaned_path
    except Exception as exc:
        print(f"[KeyFlow Backend] Post-Demucs cleaning error: {exc}")

    return stem_wav_path


def prepare_stem_for_transcription(stem_path: Path, work_dir: Path, mode: Optional[str] = None) -> Path:
    """
    Prepares an isolated piano stem for neural transcription:
    - Minimal mode (default): native sample rate, ~20 Hz 2nd-order Butterworth high-pass,
      0.95 peak normalization, preserving transients and high harmonics.
    - Legacy mode (`mode="legacy"` or `KEYFLOW_STEM_PREP="legacy"`):
      delegates to `_legacy_clean_isolated_piano_audio`.
    """
    current_mode = mode or STEM_PREP_MODE
    if current_mode == "legacy":
        return _legacy_clean_isolated_piano_audio(stem_path, work_dir)

    if not HAS_LIBROSA or librosa is None:
        return stem_path

    try:
        y, sr = librosa.load(str(stem_path), sr=None, mono=True)
        if y.size == 0 or np.max(np.abs(y)) < 1e-6:
            return stem_path

        y_filtered = y
        if HAS_SCIPY_SIGNAL and butter is not None and sosfiltfilt is not None and sr is not None and sr > 40:
            try:
                nyq = 0.5 * sr
                low = 20.0 / nyq
                if 0 < low < 1:
                    sos = butter(2, low, btype="highpass", output="sos")
                    y_filtered = sosfiltfilt(sos, y)
            except Exception as f_err:
                print(f"[KeyFlow Backend] Highpass filter skipped: {f_err}")
                y_filtered = y

        peak = float(np.max(np.abs(y_filtered)))
        y_norm = y_filtered * (0.95 / peak) if peak > 1e-6 else y_filtered

        prepared_path = work_dir / "piano_prepared.wav"
        if HAS_SOUNDFILE and sf is not None:
            sf.write(str(prepared_path), y_norm.astype(np.float32), sr)
            if prepared_path.exists() and prepared_path.stat().st_size > 1000:
                return prepared_path
    except Exception as exc:
        print(f"[KeyFlow Backend] Stem preparation error: {exc}")

    return stem_path


def isolate_piano_with_roformer(
    input_audio_path: Path,
    work_dir: Path,
    cache_separator: bool = True,
) -> Optional[Path]:
    """Isolates dedicated piano stem using BS-RoFormer 6-stem model via audio-separator."""
    global _cached_separator
    if not HAS_AUDIO_SEPARATOR or Separator is None:
        return None

    sep_out = work_dir / "sep_out"
    sep_out.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with _separator_lock:
        try:
            separator = _cached_separator if cache_separator else None
            if separator is None:
                sep_kwargs: Dict[str, Any] = {
                    "output_dir": str(sep_out),
                    "model_file_dir": str(MODEL_CACHE_DIR),
                    "output_format": "WAV",
                }
                mdxc_ov = _get_separator_mdxc_params()
                if mdxc_ov is not None:
                    sep_kwargs["mdxc_params"] = mdxc_ov

                sep = Separator(**sep_kwargs)
                try:
                    sep.load_model(model_filename=PIANO_SEPARATOR_MODEL)
                except Exception as lm_exc:
                    if "not found in supported model files" in str(lm_exc).lower():
                        try:
                            as_ver = importlib.metadata.version("audio-separator")
                        except Exception:
                            as_ver = "unknown"
                        print(
                            f"[KeyFlow Backend] ERROR: separator model '{PIANO_SEPARATOR_MODEL}' not available "
                            f"in audio-separator {as_ver} — upgrade to >=0.48.0. Falling back to htdemucs_6s."
                        )
                        if cache_separator:
                            _cached_separator = None
                        return None
                    raise
                separator = sep
                if cache_separator:
                    _cached_separator = sep

            separator.output_dir = str(sep_out)
            mi = getattr(separator, "model_instance", None)
            if mi is not None and hasattr(mi, "output_dir"):
                mi.output_dir = str(sep_out)

            output_files = separator.separate(str(input_audio_path))
            if not output_files:
                return None

            resolved_candidates: List[Path] = []
            for fname in output_files:
                candidate = Path(fname)
                if candidate.is_absolute() and candidate.exists():
                    resolved_candidates.append(candidate)
                elif (sep_out / fname).exists():
                    resolved_candidates.append(sep_out / fname)
                elif (Path(separator.output_dir) / fname).exists():
                    resolved_candidates.append(Path(separator.output_dir) / fname)

            return _select_piano_stem_path(resolved_candidates)
        except Exception as exc:
            print(f"[KeyFlow Backend] BS-RoFormer isolation error: {exc}")
            if cache_separator:
                _cached_separator = None
            return None
        finally:
            if HAS_TORCH and torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass


def isolate_piano_with_htdemucs_6s(input_audio_path: Path, work_dir: Path) -> Optional[Path]:
    """Runs 6-stem HT-Demucs dedicated piano stem isolation."""
    if not HAS_DEMUCS:
        return None
    device = "cuda" if (HAS_TORCH and torch is not None and torch.cuda.is_available()) else "cpu"
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
        print(f"[KeyFlow Backend] htdemucs_6s dedicated piano isolation fallback: {exc}")
    finally:
        if HAS_TORCH and torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return None


def isolate_piano_stem(
    input_audio_path: Path,
    work_dir: Path,
    cache_separator: bool = True,
) -> Tuple[Path, str]:
    """
    Separates piano stem using hierarchical fallback chain:
    1. BS-RoFormer SW 6-stem via audio-separator.
    2. HT-Demucs 6S dedicated piano stem (`demucs -n htdemucs_6s --two-stems piano`).
    3. HT-Demucs 4-stem other stem (`demucs -n htdemucs --two-stems other`).
    4. Librosa HPSS harmonic separation with softer margin=(1.0, 1.5).
    5. Raw input audio fallback.

    Returns (prepared_stem_path, separator_name).
    """
    if HAS_AUDIO_SEPARATOR:
        try:
            roformer_stem = isolate_piano_with_roformer(
                input_audio_path, work_dir, cache_separator=cache_separator
            )
            if roformer_stem is not None and roformer_stem.exists():
                return prepare_stem_for_transcription(roformer_stem, work_dir), "bs-roformer-sw"
        except Exception as exc:
            print(f"[KeyFlow Backend] BS-RoFormer fallback: {exc}")

    if HAS_DEMUCS:
        piano_6s = isolate_piano_with_htdemucs_6s(input_audio_path, work_dir)
        if piano_6s is not None and piano_6s.exists():
            return prepare_stem_for_transcription(piano_6s, work_dir), "htdemucs_6s"

        device = "cuda" if (HAS_TORCH and torch is not None and torch.cuda.is_available()) else "cpu"
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
            print(f"[KeyFlow Backend] Demucs 4-stem fallback failed: {exc}")
        finally:
            if HAS_TORCH and torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    if HAS_LIBROSA and librosa is not None:
        try:
            y, sr = librosa.load(str(input_audio_path), sr=22050, mono=True)
            y_harm, _ = librosa.effects.hpss(y, margin=(1.0, 1.5))
            harm_path = work_dir / "harmonic_isolated.wav"
            if HAS_SOUNDFILE and sf is not None:
                sf.write(str(harm_path), y_harm, sr)
            if harm_path.exists() and harm_path.stat().st_size > 1000:
                return prepare_stem_for_transcription(harm_path, work_dir), "librosa-hpss"
        except Exception as exc:
            print(f"[KeyFlow Backend] Librosa HPSS fallback skipped: {exc}")

    return prepare_stem_for_transcription(input_audio_path, work_dir), "none"


# Backward compatibility alias
isolate_piano_with_demucs = isolate_piano_stem


def transcribe_with_transkun(audio_path: Path, output_midi_path: Path) -> bool:
    """Transcribes audio into a Standard MIDI file using Transkun."""
    if not HAS_TRANSKUN or not TRANSKUN_CMD:
        return False
    try:
        device = "cuda" if (HAS_TORCH and torch is not None and torch.cuda.is_available()) else "cpu"
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
        print(f"[KeyFlow Backend] Transkun transcription failed: {err_msg}")
        return False
    except Exception as exc:
        print(f"[KeyFlow Backend] Transkun transcription error: {str(exc)[:500]}")
        return False
    finally:
        if HAS_TORCH and torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def transcribe_with_bytedance(
    audio_path: Path,
    output_midi_path: Path,
    cache_model: bool = True,
) -> bool:
    """Transcribes audio into a Standard MIDI file using ByteDance PianoTranscription."""
    global _cached_bytedance, _cached_bytedance_device
    if not HAS_BYTEDANCE or PianoTranscription is None or not HAS_LIBROSA or librosa is None:
        return False
    try:
        ckpt_path = ensure_bytedance_checkpoint()
        audio, _ = librosa.load(str(audio_path), sr=BD_SAMPLE_RATE, mono=True)
        device = "cuda" if (HAS_TORCH and torch is not None and torch.cuda.is_available()) else "cpu"

        if cache_model:
            with _bytedance_lock:
                if _cached_bytedance is None or _cached_bytedance_device != device:
                    _cached_bytedance = PianoTranscription(device=device, checkpoint_path=str(ckpt_path))
                    _cached_bytedance_device = device
                _cached_bytedance.transcribe(audio, str(output_midi_path))
        else:
            transcriber = PianoTranscription(device=device, checkpoint_path=str(ckpt_path))
            transcriber.transcribe(audio, str(output_midi_path))

        return output_midi_path.exists() and output_midi_path.stat().st_size > 80
    except Exception as exc:
        print(f"[KeyFlow Backend] ByteDance transcription failed: {exc}")
        if cache_model:
            with _bytedance_lock:
                _cached_bytedance = None
                _cached_bytedance_device = None
        return False
    finally:
        if HAS_TORCH and torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def transcribe_with_basic_pitch(audio_path: Path, output_midi_path: Path, sensitivity: float) -> bool:
    if not HAS_BASIC_PITCH or basic_pitch_predict is None:
        return False
    try:
        sens_norm = float(np.clip(sensitivity / 100.0 if sensitivity > 1.0 else sensitivity, 0.15, 0.90))
        onset_thresh = float(np.clip(0.65 - sens_norm * 0.35, 0.25, 0.70))
        frame_thresh = float(np.clip(0.45 - sens_norm * 0.22, 0.18, 0.55))
        _, midi_data, _ = basic_pitch_predict(
            str(audio_path),
            onset_threshold=onset_thresh,
            frame_threshold=frame_thresh,
            minimum_note_length=55.0,
            minimum_frequency=librosa.midi_to_hz(28) if (HAS_LIBROSA and librosa is not None) else 41.2,
            maximum_frequency=librosa.midi_to_hz(96) if (HAS_LIBROSA and librosa is not None) else 2093.0,
        )
        midi_data.write(str(output_midi_path))
        return output_midi_path.exists()
    except Exception as exc:
        print(f"[KeyFlow Backend] Basic Pitch transcription failed: {exc}")
        return False


def filter_midi_bleed_notes(
    midi_path: Path,
    sensitivity: float = 55.0,
    transcriber: str = "bytedance",
) -> bool:
    """
    Post-filters low-velocity bleed and micro-glitches from neural transcription:
    1. Removes notes outside 88-key piano range (MIDI 21..108).
    2. Velocity gate mapped from sensitivity (accepts 0..1 or 15..85):
       - ByteDance: 15..85 -> min_vel 38..20 (default 55 -> 28).
       - Transkun: 15..85 -> min_vel 24..10 (default 55 -> 16), overridable via KEYFLOW_TRANSKUN_MIN_VEL.
    3. Removes sub-55ms micro-glitches (duration < 0.055s).
    4. Caps simultaneous note onsets within any 30ms window to 8 notes (keeping highest velocity).
    5. Bakes CC 64 sustain pedal intervals into note end times and strips CC 64 events to prevent
       double-application on the frontend.
    6. Extends clipped chord clusters up to the next chord change.
    """
    if not HAS_PRETTY_MIDI or pretty_midi is None or not midi_path.exists():
        return False
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        raw_sens = float(sensitivity)
        sens_pct = raw_sens * 100.0 if (0.0 < raw_sens <= 1.0) else raw_sens
        sens_clamped = float(np.clip(sens_pct, 15.0, 85.0))
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

            valid_notes = []
            for n in inst.notes:
                dur = n.end - n.start
                if 21 <= n.pitch <= 108 and n.velocity >= min_vel and dur >= 0.055:
                    valid_notes.append(n)

            valid_notes.sort(key=lambda n: n.start)

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
                inst.control_changes = [cc for cc in inst.control_changes if cc.number != 64]

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
        print(f"[KeyFlow Filter] Error filtering MIDI bleed notes: {exc}")
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
    if not HAS_PRETTY_MIDI or pretty_midi is None or not midi_path.exists():
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

        est_bpm = 0.0
        if HAS_LIBROSA and librosa is not None and audio_path.exists():
            try:
                y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=90.0)
                if y.size > sr * 2:
                    tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr)
                    arr = np.asarray(tempo_val).flat
                    if arr.size > 0:
                        est_bpm = float(arr[0])
            except Exception:
                est_bpm = 0.0

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

        while est_bpm < 64.0:
            est_bpm *= 2.0
        while est_bpm > 164.0:
            est_bpm *= 0.5
        est_bpm = float(round(np.clip(est_bpm, 48.0, 208.0)))

        beat_sec = 60.0 / est_bpm
        times = np.array([c[0] for c in clusters], dtype=np.float64)
        weights = np.array([c[1] for c in clusters], dtype=np.float64)

        angles = 2.0 * np.pi * (times / beat_sec)
        sin_sum = float(np.sum(weights * np.sin(angles)))
        cos_sum = float(np.sum(weights * np.cos(angles)))
        best_phase = (np.arctan2(sin_sum, cos_sum) / (2.0 * np.pi)) * beat_sec
        best_phase = float(best_phase % beat_sec)

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

        frontend_grid_offset = round(0.8 + bar1_midi_sec, 3)
        return {
            "bpm": est_bpm,
            "grid_offset": float(np.clip(frontend_grid_offset, 0.05, 10.0)),
        }
    except Exception as exc:
        print(f"[KeyFlow Rhythm] Rhythm estimation fallback: {exc}")
        return default_res


def _estimate_tempo_bpm(y_harm: np.ndarray, sr: int) -> float:
    if not HAS_LIBROSA or librosa is None:
        return 112.0
    try:
        if hasattr(librosa, "feature") and hasattr(librosa.feature, "tempo"):
            tempo_arr = librosa.feature.tempo(y=y_harm, sr=sr)
        else:
            tempo_arr = librosa.beat.tempo(y=y_harm, sr=sr)
        raw_bpm = (
            float(tempo_arr[0])
            if hasattr(tempo_arr, "__getitem__") and len(tempo_arr) > 0
            else float(tempo_arr)
        )
        return float(np.clip(raw_bpm, 60.0, 180.0))
    except Exception:
        return 112.0


def transcribe_with_librosa_cqt(audio_path: Path, sensitivity: float) -> Tuple[List[Dict[str, Any]], float]:
    """
    Polyphonic Constant-Q Transform (CQT) + Harmonic Suppression Piano Transcriber.
    Works out-of-the-box with standard `librosa` + `numpy`.
    """
    if not HAS_LIBROSA or librosa is None:
        raise RuntimeError("librosa is required for audio analysis.")

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    y_harm, _ = librosa.effects.hpss(y, margin=(1.2, 2.0))
    est_bpm = _estimate_tempo_bpm(y_harm, sr)

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
        return [], est_bpm
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
                if dur >= 0.065:
                    notes.append({
                        "midi": int(midi_num),
                        "time": round(float(st["start"]), 4),
                        "duration": round(float(min(4.5, dur)), 4),
                        "velocity": round(float(np.clip(0.38 + np.sqrt(st["peak"]) * 0.56, 0.35, 0.96)), 3),
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
        if dur >= 0.065:
            notes.append({
                "midi": int(midi_num),
                "time": round(float(st["start"]), 4),
                "duration": round(float(min(4.5, dur)), 4),
                "velocity": round(float(np.clip(0.38 + np.sqrt(st["peak"]) * 0.56, 0.35, 0.96)), 3),
            })

    notes.sort(key=lambda n: (n["time"], n["midi"]))
    return notes, est_bpm


def notes_to_midi_bytes(notes: List[Dict[str, Any]], bpm: float = 110.0) -> bytes:
    if not HAS_PRETTY_MIDI or pretty_midi is None:
        raise RuntimeError("pretty_midi is not installed")
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for n in notes:
        vel = int(np.clip(round(float(n.get("velocity", 0.78)) * 127), 28, 124))
        start = float(n["time"])
        end = start + max(0.06, float(n["duration"]))
        piano.notes.append(pretty_midi.Note(velocity=vel, pitch=int(n["midi"]), start=start, end=end))
    pm.instruments.append(piano)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def transcribe_with_librosa_cqt_to_midi(audio_path: Path, output_midi_path: Path, sensitivity: float) -> bool:
    """Writes a Standard MIDI file via Librosa CQT + pretty_midi."""
    if not HAS_LIBROSA or not HAS_PRETTY_MIDI:
        return False
    notes, est_bpm = transcribe_with_librosa_cqt(audio_path, sensitivity)
    if not notes:
        return False
    midi_bytes = notes_to_midi_bytes(notes, bpm=est_bpm)
    output_midi_path.write_bytes(midi_bytes)
    return output_midi_path.exists() and output_midi_path.stat().st_size > 80


def run_transcription_pipeline(
    input_path: Path,
    work_dir: Path,
    should_isolate: bool = True,
    sens_val: float = 55.0,
    engine: str = "auto",
    cache_models: bool = True,
    enable_basic_pitch: bool = False,
) -> Dict[str, Any]:
    """
    Unified stem separation -> polyphonic transcription -> MIDI bleed filtering -> rhythm estimation.
    """
    sens_norm = float(np.clip(sens_val / 100.0 if sens_val > 1.0 else sens_val, 0.15, 0.90))

    t_sep_start = time.perf_counter()
    if should_isolate:
        target_audio, separator_used = isolate_piano_stem(
            input_path, work_dir, cache_separator=cache_models
        )
    else:
        target_audio, separator_used = input_path, "none"
    sep_s = round(time.perf_counter() - t_sep_start, 2)

    output_midi = work_dir / "transcribed.mid"
    transcriber_used = None
    req_engine = (engine or "auto").strip().lower()

    t_trans_start = time.perf_counter()
    if req_engine == "bytedance":
        if transcribe_with_bytedance(target_audio, output_midi, cache_model=cache_models):
            transcriber_used = "bytedance"
        elif transcribe_with_transkun(target_audio, output_midi):
            transcriber_used = "transkun"
        elif enable_basic_pitch and transcribe_with_basic_pitch(target_audio, output_midi, sens_norm):
            transcriber_used = "basic-pitch"
    elif req_engine == "transkun":
        if transcribe_with_transkun(target_audio, output_midi):
            transcriber_used = "transkun"
        elif transcribe_with_bytedance(target_audio, output_midi, cache_model=cache_models):
            transcriber_used = "bytedance"
        elif enable_basic_pitch and transcribe_with_basic_pitch(target_audio, output_midi, sens_norm):
            transcriber_used = "basic-pitch"
    else:
        if transcribe_with_transkun(target_audio, output_midi):
            transcriber_used = "transkun"
        elif transcribe_with_bytedance(target_audio, output_midi, cache_model=cache_models):
            transcriber_used = "bytedance"
        elif enable_basic_pitch and transcribe_with_basic_pitch(target_audio, output_midi, sens_norm):
            transcriber_used = "basic-pitch"

    if transcriber_used is not None:
        filter_midi_bleed_notes(output_midi, float(sens_val), transcriber=transcriber_used)
        if not output_midi.exists() or output_midi.stat().st_size <= 80:
            raise ValueError("No pitched piano notes detected in audio file.")
        midi_bytes = output_midi.read_bytes()
    else:
        notes, est_bpm = transcribe_with_librosa_cqt(target_audio, sens_norm)
        if not notes:
            raise ValueError("No pitched piano notes detected in audio file.")
        if HAS_PRETTY_MIDI:
            midi_bytes = notes_to_midi_bytes(notes, bpm=est_bpm)
            output_midi.write_bytes(midi_bytes)
            transcriber_used = "librosa-cqt"
        else:
            trans_s = round(time.perf_counter() - t_trans_start, 2)
            print(f"[KeyFlow Backend] separation={separator_used} {sep_s:.1f}s | transcription=librosa-cqt {trans_s:.1f}s")
            return {
                "notes": notes,
                "bpm": round(est_bpm, 1),
                "separator_used": separator_used,
                "transcriber_used": "librosa-cqt",
                "timings": {
                    "separation_s": sep_s,
                    "transcription_s": trans_s,
                },
            }

    rhythm = estimate_midi_and_audio_rhythm(target_audio, output_midi)
    trans_s = round(time.perf_counter() - t_trans_start, 2)
    print(f"[KeyFlow Backend] separation={separator_used} {sep_s:.1f}s | transcription={transcriber_used} {trans_s:.1f}s")

    return {
        "midi_base64": base64.b64encode(midi_bytes).decode("ascii"),
        "separator_used": separator_used,
        "transcriber_used": transcriber_used,
        "bpm": rhythm["bpm"],
        "grid_offset": rhythm["grid_offset"],
        "timings": {
            "separation_s": sep_s,
            "transcription_s": trans_s,
        },
    }


def get_available_engines(include_basic_pitch: bool = True) -> List[str]:
    engines = []
    if HAS_AUDIO_SEPARATOR:
        engines.append("BS-RoFormer")
    if HAS_DEMUCS:
        engines.append("HT-Demucs-6S")
    if HAS_TRANSKUN:
        engines.append("Transkun")
    if HAS_BYTEDANCE:
        engines.append("ByteDance-Piano")
    if include_basic_pitch and HAS_BASIC_PITCH:
        engines.append("Basic-Pitch")
    if HAS_LIBROSA:
        engines.append("Librosa-CQT")
    return engines


def build_health_payload(version: str = "3.2.0", include_basic_pitch: bool = True) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "ok",
        "version": version,
        "engines": get_available_engines(include_basic_pitch=include_basic_pitch),
        "audio_separator_available": HAS_AUDIO_SEPARATOR,
        "demucs_available": HAS_DEMUCS,
        "transkun_available": HAS_TRANSKUN,
        "bytedance_available": HAS_BYTEDANCE,
        "librosa_available": HAS_LIBROSA,
    }
    if include_basic_pitch:
        payload["basic_pitch_available"] = HAS_BASIC_PITCH

    debug_health = os.environ.get("KEYFLOW_DEBUG_HEALTH", "").strip().lower() in ("1", "true", "yes")
    if debug_health:
        as_ver = None
        if HAS_AUDIO_SEPARATOR:
            try:
                as_ver = importlib.metadata.version("audio-separator")
            except Exception:
                try:
                    as_ver = importlib.metadata.version("audio_separator")
                except Exception:
                    as_ver = "installed"
        payload["audio_separator_version"] = as_ver
        payload["import_errors"] = IMPORT_ERRORS
        payload["installed_versions"] = _installed_versions()

    return payload


def is_path_within_allowed_temp(candidate: Path, extra_roots: Optional[List[Path]] = None) -> bool:
    """Validates that `candidate` resolves inside allowed temp directories."""
    try:
        resolved = candidate.resolve()
    except Exception:
        return False

    roots: List[Path] = [Path(tempfile.gettempdir())]
    if os.name != "nt":
        roots.append(Path("/tmp"))
    gradio_tmp = os.environ.get("GRADIO_TEMP_DIR", "").strip()
    if gradio_tmp:
        roots.append(Path(gradio_tmp))
    if extra_roots:
        roots.extend(extra_roots)

    for root in roots:
        try:
            if resolved.is_relative_to(root.resolve()):
                return True
        except Exception:
            continue
    return False


async def save_upload_to_path(
    upload_file: UploadFile,
    dest_path: Path,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> int:
    """Streams an uploaded file in 1 MB chunks to disk, enforcing max_bytes without full RAM buffering."""
    total_bytes = 0
    chunk_size = 1024 * 1024
    try:
        with dest_path.open("wb") as out_fh:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio file exceeds {max_bytes // (1024 * 1024)} MB limit.",
                    )
                out_fh.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        return total_bytes
    except HTTPException:
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        raise
    except Exception:
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        raise


class TranscriptionGuard:
    """Per-IP sliding-window rate limiter and async concurrency cap for transcription endpoints."""

    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        rate_limit_per_min: Optional[int] = None,
        acquire_timeout_s: float = 10.0,
    ) -> None:
        if max_concurrent is None:
            try:
                max_concurrent = int(os.environ.get("KEYFLOW_MAX_CONCURRENT", "2"))
            except ValueError:
                max_concurrent = 2
        if rate_limit_per_min is None:
            try:
                rate_limit_per_min = int(os.environ.get("KEYFLOW_RATE_LIMIT_PER_MIN", "10"))
            except ValueError:
                rate_limit_per_min = 10

        self.max_concurrent = max(1, max_concurrent)
        self.rate_limit_per_min = max(1, rate_limit_per_min)
        self.acquire_timeout_s = max(0.1, float(acquire_timeout_s))

        self._rate_lock = threading.Lock()
        self._requests: Dict[str, List[float]] = {}
        self._sem_lock = threading.Lock()
        self._sem_loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._sem_lock:
            if self._semaphore is None or self._sem_loop is not loop:
                self._semaphore = asyncio.Semaphore(self.max_concurrent)
                self._sem_loop = loop
            return self._semaphore

    def check_rate_limit(self, client_ip: str) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        key = client_ip or "unknown"

        with self._rate_lock:
            if len(self._requests) > 1000:
                stale = [
                    ip for ip, ts_list in self._requests.items()
                    if not ts_list or ts_list[-1] < cutoff
                ]
                for ip in stale:
                    del self._requests[ip]

            timestamps = [t for t in self._requests.get(key, []) if t >= cutoff]
            if len(timestamps) >= self.rate_limit_per_min:
                self._requests[key] = timestamps
                raise HTTPException(
                    status_code=429,
                    detail="Too many transcription requests. Please wait a minute and try again.",
                )
            timestamps.append(now)
            self._requests[key] = timestamps

    @asynccontextmanager
    async def slot(self):
        sem = self._get_semaphore()
        try:
            await asyncio.wait_for(sem.acquire(), timeout=self.acquire_timeout_s)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Server is busy processing another audio file. Please retry shortly.",
            )
        try:
            yield
        finally:
            sem.release()


def add_cors_and_pna_middleware(fastapi_app: FastAPI) -> None:
    """Attaches configurable CORS and Chrome Private Network Access header middleware."""
    raw_origins = os.environ.get("KEYFLOW_ALLOWED_ORIGINS", "*").strip()
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()] or ["*"]

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.middleware("http")
    async def _private_network_access_middleware(request: Request, call_next):
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network", "").lower() == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response
