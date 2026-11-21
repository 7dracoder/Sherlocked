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
