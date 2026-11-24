import os
import tempfile
from pathlib import Path
from typing import Any, Dict

# Configure writable cache directories and Gradio SSR mode before importing pipeline/ML libs
_CACHE_BASE = Path("/tmp") if os.name != "nt" else Path(tempfile.gettempdir())
os.environ.setdefault("TORCH_HOME", str(_CACHE_BASE / "torch"))
os.environ.setdefault("HF_HOME", str(_CACHE_BASE / "hf"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_BASE / "numba"))
os.environ.setdefault("KEYFLOW_CACHE_DIR", str(_CACHE_BASE / "keyflow_cache"))
os.environ["GRADIO_SSR_MODE"] = "False"

for _cache_key in ("TORCH_HOME", "HF_HOME", "NUMBA_CACHE_DIR", "KEYFLOW_CACHE_DIR"):
    os.makedirs(os.environ[_cache_key], exist_ok=True)

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

import threading
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
import gradio as gr

try:
    from pipeline import (
        ALLOWED_SUFFIXES,
        HAS_AUDIO_SEPARATOR,
        MAX_UPLOAD_BYTES,
        TranscriptionGuard,
        add_cors_and_pna_middleware,
        build_health_payload,
        is_path_within_allowed_temp,
        prefetch_separator_model,
        run_transcription_pipeline,
        save_upload_to_path,
    )
except ImportError:
    from backend.pipeline import (
        ALLOWED_SUFFIXES,
        HAS_AUDIO_SEPARATOR,
        MAX_UPLOAD_BYTES,
        TranscriptionGuard,
        add_cors_and_pna_middleware,
        build_health_payload,
        is_path_within_allowed_temp,
        prefetch_separator_model,
        run_transcription_pipeline,
        save_upload_to_path,
    )

if HAS_AUDIO_SEPARATOR:
    try:
        threading.Thread(
            target=prefetch_separator_model,
            daemon=True,
            name="prefetch_separator",
        ).start()
    except Exception as t_exc:
        print(f"[KeyFlow Space] Failed to start separator prefetch thread: {t_exc}")

transcription_guard = TranscriptionGuard()


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
    Validates temp paths and runs BS-RoFormer / HT-Demucs 6S isolation + Transkun / ByteDance
    transcription without cross-request mutable GPU model caching.
    """
    if not input_path_str:
        raise ValueError("Invalid input audio path.")

    input_path = Path(input_path_str)
    if not is_path_within_allowed_temp(input_path) or not input_path.is_file():
        raise ValueError("Invalid input audio path.")

    safe_work_dir = None
    if work_dir_str:
        cand = Path(work_dir_str)
        if is_path_within_allowed_temp(cand) and cand.is_dir():
            safe_work_dir = cand.resolve()

    def _execute(work_dir: Path) -> Dict[str, Any]:
        res = run_transcription_pipeline(
            input_path,
            work_dir,
            should_isolate=should_isolate,
            sens_val=float(sens_val),
            engine=engine,
            cache_models=False,
            enable_basic_pitch=False,
        )
        if "midi_base64" not in res:
            raise RuntimeError("Could not transcribe piano notes from audio file.")
        return res

    if safe_work_dir is not None:
        return _execute(safe_work_dir)

    tmp_root = Path(os.environ.get("KEYFLOW_CACHE_DIR", str(_CACHE_BASE))).parent
    with tempfile.TemporaryDirectory(dir=str(tmp_root), prefix="keyflow_zerogpu_") as tmp_dir:
        return _execute(Path(tmp_dir))


async def health_check_handler() -> Dict[str, Any]:
    return build_health_payload(version="3.2.0", include_basic_pitch=False)


async def transcribe_endpoint_handler(
    request: Request,
    file: UploadFile = File(...),
    isolate_piano: str = Form("true"),
    sensitivity: str = Form("55"),
    engine: str = Form("auto"),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio filename.")

    client_ip = request.client.host if request.client else "unknown"
    transcription_guard.check_rate_limit(client_ip)

    should_isolate = str(isolate_piano).strip().lower() in ("true", "1", "yes")
    try:
        sens_val = float(sensitivity)
    except ValueError:
        sens_val = 55.0

    clean_title = Path(file.filename).stem or "Transcribed Song"
    raw_suffix = Path(file.filename).suffix.lower()
    suffix = raw_suffix if raw_suffix in ALLOWED_SUFFIXES else ".mp3"

    async with transcription_guard.slot():
        with tempfile.TemporaryDirectory(prefix="keyflow_space_") as tmp_dir:
            work_dir = Path(tmp_dir)
            input_path = work_dir / f"upload{suffix}"
            await save_upload_to_path(file, input_path, max_bytes=MAX_UPLOAD_BYTES)

            from fastapi.concurrency import run_in_threadpool

            try:
                pipeline_res = await run_in_threadpool(
                    run_gpu_transcription_pipeline,
                    str(input_path),
                    should_isolate=should_isolate,
                    sens_val=sens_val,
                    work_dir_str=str(work_dir),
                    engine=engine,
                )
            except HTTPException:
                raise
            except ValueError as val_exc:
                msg = str(val_exc)
                if "No pitched piano notes detected" in msg:
                    raise HTTPException(status_code=422, detail=msg)
                print(f"[KeyFlow Space] Validation error: {type(val_exc).__name__}: {val_exc}")
                raise HTTPException(
                    status_code=422,
                    detail="Could not transcribe piano notes from this audio file.",
                )
            except Exception as exc:
                print(f"[KeyFlow Space] Transcription error: {type(exc).__name__}: {exc}")
                raise HTTPException(
                    status_code=422,
                    detail="Could not transcribe piano notes from this audio file.",
                )

    return {
        "midi_base64": pipeline_res["midi_base64"],
        "title": clean_title,
        "separator_used": pipeline_res.get("separator_used", "none"),
        "transcriber_used": pipeline_res.get("transcriber_used", "none"),
        "bpm": pipeline_res.get("bpm", 120.0),
        "grid_offset": pipeline_res.get("grid_offset", 0.8),
        "timings": pipeline_res.get("timings", {"separation_s": 0.0, "transcription_s": 0.0}),
    }


def register_keyflow_routes(fastapi_app: FastAPI) -> FastAPI:
    """Idempotently attaches CORS/PNA middleware, /health, and /api/transcribe endpoints to Gradio's FastAPI app."""
    if fastapi_app is None or getattr(fastapi_app, "_keyflow_routes_registered", False):
        return fastapi_app

    fastapi_app._keyflow_routes_registered = True
    add_cors_and_pna_middleware(fastapi_app)
    fastapi_app.get("/health")(health_check_handler)
    fastapi_app.get("/api/health")(health_check_handler)
    fastapi_app.post("/api/transcribe")(transcribe_endpoint_handler)
    return fastapi_app


# Hook Gradio's App.create_app so routes are guaranteed to be attached even when demo.app is initialized lazily
if hasattr(gr, "routes") and hasattr(gr.routes, "App") and hasattr(gr.routes.App, "create_app"):
    _orig_create_app = gr.routes.App.create_app

    def _patched_create_app(*args, **kwargs):
        created_app = _orig_create_app(*args, **kwargs)
        register_keyflow_routes(created_app)
        return created_app

    gr.routes.App.create_app = _patched_create_app


with gr.Blocks(title="KeyFlow AI Audio Server") as demo:
    gr.Markdown(
        """
        # 🎹 KeyFlow AI Audio-to-MIDI Server (ZeroGPU Online)
        This Hugging Face Gradio Space powers **Phase 3 AI Audio-to-MIDI Transcription** (`BS-RoFormer` + `Transkun` / `ByteDance Piano AI`) on **ZeroGPU** for the **KeyFlow Virtual Piano** iPad web app.

        ### Active API Endpoints
        - **`GET /health`** — Returns server health & active engines (`BS-RoFormer`, `HT-Demucs 6S`, `Transkun`, `ByteDance Piano AI`).
        - **`POST /api/transcribe`** — Accepts `file`, `isolate_piano`, `sensitivity`, and optional `engine` (`multipart/form-data`) and returns `{"midi_base64": "...", "title": "...", "separator_used": "...", "transcriber_used": "...", "bpm": 120.0, "grid_offset": 0.8}`.

        **How to connect from KeyFlow:** Copy this Space's public `.hf.space` URL (e.g. `https://<username>-<space-name>.hf.space`) into the **AI Server URL** field inside KeyFlow's **Audio Song → Falling Piano Notes** modal and tap **Check**.
        """
    )
    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Audio File (.mp3, .wav, .m4a)")
        isolate_in = gr.Checkbox(value=True, label="Isolate Piano Stem (BS-RoFormer)")
        sens_in = gr.Slider(minimum=15, maximum=85, value=55, step=1, label="Sensitivity (%)")
    transcribe_btn = gr.Button("Transcribe Audio to MIDI (Base64)")
    midi_b64_out = gr.Textbox(label="Base64 Standard MIDI Output")

    def _gradio_transcribe(audio_file, isolate_piano, sensitivity):
        if not audio_file:
            raise gr.Error("Please upload an audio file first.")
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

app = getattr(demo, "app", None)
if app is not None:
    register_keyflow_routes(app)


if __name__ == "__main__":
    try:
        demo.launch(ssr_mode=False)
    except TypeError:
        demo.launch()
