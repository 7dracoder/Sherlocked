"""Verifies google-genai >= 3.2.0 SDK method signatures & schema contracts for Neon, Carbon & Decisions API."""

from __future__ import annotations

import inspect
from google import genai


def test_google_genai_sdk_supports_interactions_and_decisions_v1beta() -> None:
    client_v1 = genai.Client(api_key="test_schema_inspection_only")
    client_beta = genai.Client(
        api_key="test_schema_inspection_only",
        http_options={"api_version": "v1beta"},
    )

    assert hasattr(client_v1, "interactions"), "google-genai >= 3.2 must expose client.interactions"
    assert callable(client_v1.interactions.create)

    assert hasattr(client_beta, "decisions"), "google-genai >= 3.2 must expose client.decisions on v1beta"
    assert callable(client_beta.decisions.create)

    # Verify underlying GAOS resource create signature accepts model, input, and questions
    base_cls = type(client_beta.decisions).__mro__[1]
    sig_dec = inspect.signature(base_cls.create)
    params = set(sig_dec.parameters.keys())
    assert {"model", "input", "questions"}.issubset(params)

    # Verify Managed Agents (`client.agents` + `interactions.create(agent=..., environment="remote")`) contract
    assert hasattr(client_beta, "agents"), "google-genai >= 3.2 must expose client.agents for Managed Agents"
    assert callable(client_beta.agents.create)


def test_post_session_managed_agent_debrief_endpoint() -> None:
    from fastapi.testclient import TestClient
    from backend.server import app

    client = TestClient(app)
    replay = client.post("/api/v1/fixtures/example_live_lie_demo/replay?face_masked=false&include_followup=false")
    assert replay.status_code == 200
    sid = replay.json()["session_id"]

    # Record post-round reveal (#2) and compute calibration debrief
    corr = client.post("/api/v1/session/manual-correction", json={"session_id": sid, "revealed_lie_story_number": 2})
    assert corr.status_code == 200

    debrief = client.post(f"/api/v1/session/{sid}/agent-debrief")
    assert debrief.status_code == 200
    body = debrief.json()
    assert body["agent"] == "antigravity-preview-09-2026"
    assert body["model"] == "gemini-4-argon"
    assert body["brier_score"] is not None and body["brier_score"] < 0.20
