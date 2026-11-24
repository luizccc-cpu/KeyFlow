"""
KeyFlow (Piano-Flow) — Phase 3 Hybrid AI Audio-to-MIDI Local Backend Server
===========================================================================
FastAPI service delegating separation, transcription, post-filtering, and
security guards to `backend/pipeline.py`.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

try:
    from pipeline import (
        ALLOWED_SUFFIXES,
        HAS_AUDIO_SEPARATOR,
        HAS_BASIC_PITCH,
        HAS_BYTEDANCE,
        HAS_DEMUCS,
        HAS_LIBROSA,
        HAS_PRETTY_MIDI,
        HAS_TRANSKUN,
        MAX_UPLOAD_BYTES,
        TranscriptionGuard,
        _legacy_clean_isolated_piano_audio,
        add_cors_and_pna_middleware,
        build_health_payload,
        estimate_midi_and_audio_rhythm,
        filter_midi_bleed_notes,
        get_available_engines,
        isolate_piano_stem,
        isolate_piano_with_demucs,
        isolate_piano_with_htdemucs_6s,
        isolate_piano_with_roformer,
        notes_to_midi_bytes,
        prepare_stem_for_transcription,
        run_transcription_pipeline,
        save_upload_to_path,
        transcribe_with_basic_pitch,
        transcribe_with_bytedance,
        transcribe_with_librosa_cqt,
        transcribe_with_transkun,
    )
except ImportError:
    from backend.pipeline import (
        ALLOWED_SUFFIXES,
        HAS_AUDIO_SEPARATOR,
        HAS_BASIC_PITCH,
        HAS_BYTEDANCE,
        HAS_DEMUCS,
        HAS_LIBROSA,
        HAS_PRETTY_MIDI,
        HAS_TRANSKUN,
        MAX_UPLOAD_BYTES,
        TranscriptionGuard,
        _legacy_clean_isolated_piano_audio,
        add_cors_and_pna_middleware,
        build_health_payload,
        estimate_midi_and_audio_rhythm,
        filter_midi_bleed_notes,
        get_available_engines,
        isolate_piano_stem,
        isolate_piano_with_demucs,
        isolate_piano_with_htdemucs_6s,
        isolate_piano_with_roformer,
        notes_to_midi_bytes,
        prepare_stem_for_transcription,
        run_transcription_pipeline,
        save_upload_to_path,
        transcribe_with_basic_pitch,
        transcribe_with_bytedance,
        transcribe_with_librosa_cqt,
        transcribe_with_transkun,
    )

app = FastAPI(
    title="KeyFlow AI Audio-to-MIDI Transcription Server",
    version="3.2.0",
    description="BS-RoFormer / HT-Demucs stem isolation + Multi-engine Piano Transcription for KeyFlow Virtual Piano",
)

add_cors_and_pna_middleware(app)
transcription_guard = TranscriptionGuard()


@app.get("/health")
@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return build_health_payload(version="3.2.0", include_basic_pitch=True)


@app.post("/api/transcribe")
async def transcribe_endpoint(
    request: Request,
    file: UploadFile = File(...),
    isolate_piano: str = Form("true"),
    sensitivity: str = Form("55"),
    engine: str = Form("auto"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio filename.")

    client_ip = request.client.host if request.client else "unknown"
    transcription_guard.check_rate_limit(client_ip)

    should_isolate = str(isolate_piano).strip().lower() in ("true", "1", "yes")
    raw_suffix = Path(file.filename).suffix.lower()
    suffix = raw_suffix if raw_suffix in ALLOWED_SUFFIXES else ".mp3"
    clean_title = Path(file.filename).stem or "Transcribed Song"

    try:
        sens_val = float(sensitivity)
    except ValueError:
        sens_val = 55.0

    async with transcription_guard.slot():
        with tempfile.TemporaryDirectory(prefix="keyflow_ai_") as tmp_dir:
            work_dir = Path(tmp_dir)
            input_path = work_dir / f"upload{suffix}"
            await save_upload_to_path(file, input_path, max_bytes=MAX_UPLOAD_BYTES)

            try:
                result = await run_in_threadpool(
                    run_transcription_pipeline,
                    input_path,
                    work_dir,
                    should_isolate=should_isolate,
                    sens_val=sens_val,
                    engine=engine,
                    cache_models=True,
                    enable_basic_pitch=True,
                )
            except HTTPException:
                raise
            except ValueError as val_exc:
                msg = str(val_exc)
                if "No pitched piano notes detected" in msg:
                    raise HTTPException(status_code=422, detail=msg)
                print(f"[KeyFlow Backend] Validation error: {type(val_exc).__name__}: {val_exc}")
                raise HTTPException(
                    status_code=422,
                    detail="Could not transcribe piano notes from this audio file.",
                )
            except Exception as exc:
                print(f"[KeyFlow Backend] Transcription error: {type(exc).__name__}: {exc}")
                raise HTTPException(
                    status_code=422,
                    detail="Could not transcribe piano notes from this audio file.",
                )

    result["title"] = clean_title
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
