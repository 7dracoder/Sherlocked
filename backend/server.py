"""FastAPI server for Sherlocked keeping Gemini credentials server-side."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import FIXTURES_DIR, RESEARCH_DIR, LearnedWeights, get_settings
from .gemini_pipeline import SherlockedPipeline
from .schemas import (
    CaptureSourceKind,
    ManualCorrectionRequest,
    ModalityMode,
    SessionAnalyzeRequest,
    SessionAnalyzeResponse,
)
from .vision_tracker import render_face_masked_video

app = FastAPI(
    title="Sherlocked Multimodal Inference Server",
    version="1.0.0",
    description="Server-side Gemini 4 Neon, Gemini 4 Carbon, and Gemini Decisions API pipeline for Ray-Ban Meta & iPhone.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PIPELINE: SherlockedPipeline | None = None


def get_pipeline() -> SherlockedPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SherlockedPipeline()
    else:
        _PIPELINE.reload_weights()
    return _PIPELINE


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    weights = LearnedWeights.load()
    return {
        "status": "ok",
        "live_gemini_configured": settings.has_live_gemini,
        "execution_mode": "live_gemini" if settings.has_live_gemini else "local_multimodal_engine",
        "models": {
            "neon": settings.neon_model,
            "carbon": settings.carbon_model,
            "decisions": settings.decision_model,
        },
        "learned_weights": {
            "facial_evidence_weight": weights.facial_evidence_weight,
            "vocal_prosody_weight": weights.vocal_prosody_weight,
            "lexical_consistency_weight": weights.lexical_consistency_weight,
            "present_guess_as_exploratory": weights.present_guess_as_exploratory,
            "scientific_status": weights.scientific_status,
        },
    }


@app.post("/api/v1/session/analyze", response_model=SessionAnalyzeResponse)
def analyze_session(req: SessionAnalyzeRequest) -> SessionAnalyzeResponse:
    pipeline = get_pipeline()
    try:
        return pipeline.analyze_session_turn(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/session/manual-correction")
def manual_correction(req: ManualCorrectionRequest) -> dict[str, Any]:
    pipeline = get_pipeline()
    record = pipeline.apply_manual_correction(
        session_id=req.session_id,
        corrected_stories=req.corrected_stories,
        revealed_lie_story_number=req.revealed_lie_story_number,
    )
    return {
        "status": "updated",
        "session_id": record.session_id,
        "game_revealed": record.game_revealed,
        "revealed_lie_story_number": record.revealed_lie_story_number,
        "initial_guess_locked": record.initial_guess.chosen_story_number if record.initial_guess else None,
        "updated_guess": record.updated_guess.chosen_story_number if record.updated_guess else None,
        "stories_count": len(record.stories),
    }


@app.post("/api/v1/session/{session_id}/reset")
def reset_session(session_id: str) -> dict[str, str]:
    pipeline = get_pipeline()
    pipeline.store.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/api/v1/fixtures")
def list_fixtures() -> dict[str, Any]:
    manifest_path = FIXTURES_DIR / "manifest.json"
    if not manifest_path.exists():
        return {"fixtures": []}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Strip any ground truth fields from public fixture listing so UI/guesser remains blind until reveal
    sanitized = []
    for item in data.get("fixtures", []):
        clean_item = {
            k: v
            for k, v in item.items()
            if k not in ("actual_lie_story_number", "secret_constraint", "ground_truth_intervals")
        }
        sanitized.append(clean_item)
    return {"fixtures": sanitized, "is_synthetic_consented_fixture_set": True}


@app.get("/api/v1/fixtures/{fixture_id}/media/{variant}")
def get_fixture_media(fixture_id: str, variant: str = "original") -> FileResponse:
    manifest_path = FIXTURES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Fixtures manifest not found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    match = next((f for f in manifest.get("fixtures", []) if f["fixture_id"] == fixture_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found.")

    orig_path = FIXTURES_DIR / match["video_relpath"]
    if not orig_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing.")

    if variant == "masked":
        masked_path = orig_path.with_name(f"{orig_path.stem}_masked.mp4")
        if not masked_path.exists():
            render_face_masked_video(orig_path, masked_path)
        return FileResponse(masked_path, media_type="video/mp4")

    return FileResponse(orig_path, media_type="video/mp4")


@app.post("/api/v1/fixtures/{fixture_id}/replay", response_model=SessionAnalyzeResponse)
def replay_fixture(
    fixture_id: str,
    face_masked: bool = Query(default=False),
    include_followup: bool = Query(default=False),
) -> SessionAnalyzeResponse:
    manifest_path = FIXTURES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Fixtures manifest not found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    match = next((f for f in manifest.get("fixtures", []) if f["fixture_id"] == fixture_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found.")

    video_path = FIXTURES_DIR / match["video_relpath"]
    if face_masked:
        masked_path = video_path.with_name(f"{video_path.stem}_masked.mp4")
        if not masked_path.exists():
            render_face_masked_video(video_path, masked_path)
        video_path = masked_path

    pipeline = get_pipeline()
    session_id = f"replay_{fixture_id}_{'masked' if face_masked else 'visible'}"
    pipeline.store.reset_session(session_id)

    req = SessionAnalyzeRequest(
        session_id=session_id,
        turn_id="replay_run_1",
        capture_source=CaptureSourceKind.REPLAY,
        modality_mode=(
            ModalityMode.AUDIO_PLUS_FACE_MASKED_VIDEO
            if face_masked
            else ModalityMode.AUDIO_PLUS_VISIBLE_FACE
        ),
        is_replay=True,
        request_followup_questions=include_followup,
        is_followup_round=include_followup,
        audio_video_drift_ms=float(match.get("audio_video_drift_ms", 0.0)),
        video_mp4_base64=base64.b64encode(video_path.read_bytes()).decode("ascii"),
    )
    return pipeline.analyze_session_turn(req)


@app.get("/api/v1/research/summary")
def research_summary() -> dict[str, Any]:
    results_path = RESEARCH_DIR / "results" / "benchmark_report.json"
    if not results_path.exists():
        return {"status": "pending", "weights": LearnedWeights.load().__dict__}
    return json.loads(results_path.read_text(encoding="utf-8"))
