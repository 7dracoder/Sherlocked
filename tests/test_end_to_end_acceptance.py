"""End-to-end acceptance tests for Sherlocked backend, research harness, and HTTP API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import FIXTURES_DIR, RESEARCH_DIR, WEIGHTS_PATH
from backend.schemas import (
    ActivityKind,
    AudioRouteKind,
    CaptureSourceKind,
    FrameSample,
    ManualCorrectionRequest,
    ModalityMode,
    QualityFailureFlag,
    SessionAnalyzeRequest,
    UtteranceSegment,
    WordTimestamp,
    sanitize_observable_text,
)
from backend.server import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_and_learned_weights_loaded(client: TestClient) -> None:
    assert WEIGHTS_PATH.exists(), "Research step 1 must write learned_pipeline_weights.json before app verification"
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert 0.05 <= body["learned_weights"]["facial_evidence_weight"] <= 0.50


def test_reveal_firewall_blocks_ground_truth_leakage(client: TestClient) -> None:
    bad_req = {
        "session_id": "test_firewall_1",
        "utterances": [{"segment_id": "u1", "text": "Hello there", "start_sec": 0.0, "end_sec": 1.0}],
        "extra_metadata": {"actual_lie": 2},
    }
    r = client.post("/api/v1/session/analyze", json=bad_req)
    assert r.status_code == 400
    assert "Reveal firewall violation" in r.json()["detail"]


def test_anti_clinical_and_microexpression_guard() -> None:
    cleaned = sanitize_observable_text(
        "Speaker showed pathological micro-expression and guilty conscience before Friday.",
        allow_micro_expression_claim=False,
    )
    assert "pathological" not in cleaned.lower()
    assert "guilty conscience" not in cleaned.lower()
    assert "micro-expression" not in cleaned.lower()
    assert "brief facial movement" in cleaned.lower()


def test_three_arbitrary_statements_never_trigger_game_guess(client: TestClient) -> None:
    r = client.post("/api/v1/fixtures/unclear_three_statements/replay")
    assert r.status_code == 200
    data = r.json()
    assert data["activity"] == ActivityKind.UNCLEAR.value
    assert data["status_label"] == "Listening for context"
    assert data["initial_guess"] is None
    assert data["spoken_cue_text"] is None
    assert data["should_speak_privately"] is False


def test_unrehearsed_game_lock_and_followup_preserves_initial_guess(client: TestClient) -> None:
    sid = "e2e_game_lifecycle"
    client.post(f"/api/v1/session/{sid}/reset")

    # 1. Incomplete round (only 2 stories) -> remains quiet
    partial_utts = [
        {"segment_id": "u0", "text": "Let's play two truths and a lie.", "start_sec": 0.0, "end_sec": 1.8},
        {"segment_id": "u1", "text": "Number one, in June 2019 I ran ten miles in Seattle.", "start_sec": 2.1, "end_sec": 5.0},
        {"segment_id": "u2", "text": "Number two, I kind of probably flew a jet somewhere.", "start_sec": 6.0, "end_sec": 9.0, "pause_before_sec": 0.9},
    ]
    r_partial = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "turn_partial",
            "audio_route": AudioRouteKind.GLASSES_PRIVATE.value,
            "utterances": partial_utts,
        },
    )
    assert r_partial.status_code == 200
    d_partial = r_partial.json()
    assert d_partial["activity"] == ActivityKind.GAME.value
    assert d_partial["initial_guess"] is None
    assert d_partial["spoken_cue_text"] is None

    # 2. Third story completes -> locks initial guess & whispers privately
    full_utts = partial_utts + [
        {"segment_id": "u3", "text": "Number three, in May 2021 I got six stitches in my kitchen.", "start_sec": 9.4, "end_sec": 12.6}
    ]
    r_full = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "turn_full",
            "audio_route": AudioRouteKind.GLASSES_PRIVATE.value,
            "utterances": full_utts,
        },
    )
    assert r_full.status_code == 200
    d_full = r_full.json()
    assert d_full["initial_guess"] is not None
    locked_pick = d_full["initial_guess"]["chosen_story_number"]
    assert locked_pick == 2
    assert d_full["spoken_cue_text"] == "I think number 2 is the lie."
    assert d_full["should_speak_privately"] is True

    # 3. Duplicate retry with identical payload -> idempotent result_id prevents duplicate whisper!
    r_dup = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "turn_full",
            "audio_route": AudioRouteKind.GLASSES_PRIVATE.value,
            "utterances": full_utts,
        },
    )
    d_dup = r_dup.json()
    assert d_dup["result_id"] == d_full["result_id"]
    assert d_dup["already_delivered"] is True
    assert d_dup["should_speak_privately"] is False

    # 4. Follow-up round saves updated_guess while keeping initial_guess locked!
    fu_utts = full_utts + [
        {"segment_id": "u4", "text": "Let's ask follow-up questions before the reveal.", "start_sec": 13.0, "end_sec": 15.0},
        {"segment_id": "u5", "text": "For story 1, my brother ran beside me in July 2019.", "start_sec": 15.3, "end_sec": 18.0},
        {"segment_id": "u6", "text": "For story 2, I actually logged forty flight hours at Boeing Field in August 2020.", "start_sec": 18.3, "end_sec": 21.8},
        {"segment_id": "u7", "text": "For story 3, I kind of sort of maybe cut my finger or something I guess.", "start_sec": 23.0, "end_sec": 26.2, "pause_before_sec": 1.2},
    ]
    r_fu = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "turn_followup",
            "audio_route": AudioRouteKind.GLASSES_PRIVATE.value,
            "request_followup_questions": True,
            "is_followup_round": True,
            "utterances": fu_utts,
        },
    )
    d_fu = r_fu.json()
    assert len(d_fu["followup_questions"]) == 3
    assert d_fu["initial_guess"]["chosen_story_number"] == locked_pick  # Initial guess stayed locked!
    assert d_fu["updated_guess"] is not None
    assert d_fu["updated_guess"]["chosen_story_number"] == 3  # Updated guess shifted based on new evidence!

    # 5. Manual correction of final reveal works outside guesser
    r_corr = client.post(
        "/api/v1/session/manual-correction",
        json={"session_id": sid, "revealed_lie_story_number": 3},
    )
    assert r_corr.status_code == 200
    assert r_corr.json()["revealed_lie_story_number"] == 3
    assert r_corr.json()["initial_guess_locked"] == locked_pick


def test_context_switching_game_reveal_to_deal_and_practice_precedence(client: TestClient) -> None:
    # 1. Practice takes precedence over deal language inside an exercise
    r_prac = client.post("/api/v1/fixtures/practice_roleplay_pitch/replay")
    assert r_prac.status_code == 200
    d_prac = r_prac.json()
    assert d_prac["activity"] == ActivityKind.PRACTICE.value
    assert d_prac["deal_output"] is None
    assert d_prac["practice_output"] is not None
    assert d_prac["practice_output"]["exercise_complete"] is True
    assert d_prac["practice_output"]["suggested_followup"]
    assert d_prac["should_speak_privately"] is False  # Replay silent by default

    # 2. Single session transitions from Game -> Reveal -> Deal
    sid = "e2e_context_switch"
    client.post(f"/api/v1/session/{sid}/reset")

    game_turn = [
        {"segment_id": "g0", "text": "Let's play two truths and a lie.", "start_sec": 0.0, "end_sec": 1.6},
        {"segment_id": "g1", "text": "One, in May 2019 I ran ten miles in Boston.", "start_sec": 2.0, "end_sec": 4.8},
        {"segment_id": "g2", "text": "Two, I kind of probably met the prime minister somewhere.", "start_sec": 5.8, "end_sec": 8.8, "pause_before_sec": 1.0},
        {"segment_id": "g3", "text": "Three, in July 2021 I chipped a tooth at the airport.", "start_sec": 9.2, "end_sec": 12.0},
    ]
    r1 = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "t1",
            "audio_route": AudioRouteKind.EARBUDS_PRIVATE.value,
            "utterances": game_turn,
        },
    ).json()
    assert r1["activity"] == ActivityKind.GAME.value
    assert r1["activity_epoch"] == 1
    assert r1["initial_guess"]["chosen_story_number"] == 2

    # Now speaker reveals and transitions into Friday deal with a usable lip press frame
    transition_turn = game_turn + [
        {"segment_id": "g4", "text": "Nice catch, the lie was number two!", "start_sec": 12.5, "end_sec": 14.5},
        {
            "segment_id": "d1",
            "text": "Now onto our contract timeline, we can commit to Friday delivery.",
            "start_sec": 15.2,
            "end_sec": 19.0,
            "pause_before_sec": 0.65,
            "words": [
                {"word": "commit", "start_sec": 17.0, "end_sec": 17.4},
                {"word": "Friday", "start_sec": 17.8, "end_sec": 18.3},
            ],
        },
    ]
    frames = [
        FrameSample(
            frame_id=f"frame_{i:05d}",
            timestamp_sec=round(15.0 + i * 0.25, 2),
            blendshapes={"lip_press": 0.42 if abs((15.0 + i * 0.25) - 17.75) < 0.15 else 0.03},
            motion_score=0.25 if abs((15.0 + i * 0.25) - 17.75) < 0.15 else 0.02,
        ).model_dump()
        for i in range(18)
    ]
    r2 = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "t2",
            "audio_route": AudioRouteKind.SPEAKER_PUBLIC.value,  # Private route disappeared -> present on screen!
            "utterances": transition_turn,
            "frames": frames,
        },
    ).json()
    assert r2["activity"] == ActivityKind.DEAL.value
    assert r2["activity_epoch"] == 2
    assert r2["deal_output"]["action"] == "whisper_question"
    assert "Friday" in r2["deal_output"]["spoken_whisper"]
    # Because private audio route disappeared (SPEAKER_PUBLIC), cue is presented on screen, NOT spoken out loud!
    assert r2["should_speak_privately"] is False
    assert r2["present_cue_on_screen"] is True


@pytest.mark.parametrize(
    ("fixture_id", "expected_flag"),
    [
        ("fail_low_light", QualityFailureFlag.LOW_LIGHT.value),
        ("fail_eyewear_occlusion", QualityFailureFlag.EYEWEAR_OCCLUSION.value),
        ("fail_fast_head_turn", QualityFailureFlag.FAST_HEAD_TURN.value),
        ("fail_audio_drift", QualityFailureFlag.AUDIO_DRIFT.value),
        ("fail_overlapping_speech", QualityFailureFlag.OVERLAPPING_SPEECH.value),
        ("fail_listener_in_frame", QualityFailureFlag.LISTENER_IN_FRAME.value),
    ],
)
def test_capture_failure_guards_block_invalid_speaker_observations(
    client: TestClient,
    fixture_id: str,
    expected_flag: str,
) -> None:
    r = client.post(f"/api/v1/fixtures/{fixture_id}/replay")
    assert r.status_code == 200
    data = r.json()
    assert expected_flag in data["discarded_quality_flags"]
    if data["deal_output"] is not None:
        assert data["deal_output"]["action"] == "silence"


def test_frame_loss_guard_excludes_gapped_frames(client: TestClient) -> None:
    sid = "e2e_frame_loss"
    client.post(f"/api/v1/session/{sid}/reset")
    frames = [
        FrameSample(
            frame_id="frame_00001",
            timestamp_sec=0.0,
            usable=False,
            quality_flags=[QualityFailureFlag.FRAME_LOSS],
            blendshapes={"lip_press": 0.55},
        ).model_dump(),
        FrameSample(
            frame_id="frame_00002",
            timestamp_sec=1.8,
            usable=False,
            quality_flags=[QualityFailureFlag.FRAME_LOSS],
            blendshapes={"lip_press": 0.55},
        ).model_dump(),
    ]
    r = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "utterances": [{"segment_id": "u1", "text": "We commit to Friday delivery.", "start_sec": 0.2, "end_sec": 2.0}],
            "frames": frames,
        },
    ).json()
    assert QualityFailureFlag.FRAME_LOSS.value in r["discarded_quality_flags"]
    assert r["deal_output"]["action"] == "silence"


def test_straight_face_controls_and_face_masked_rerun(client: TestClient) -> None:
    # All-neutral straight-face round discloses sparse visual evidence honestly
    r_sf = client.post("/api/v1/fixtures/sf_all_neutral_lie3/replay").json()
    assert r_sf["activity"] == ActivityKind.GAME.value
    assert r_sf["initial_guess"] is not None
    assert r_sf["initial_guess"]["sparse_visual_disclosure"] is True

    # Visible vs Face-Masked rerun on natural game fixture
    r_vis = client.post("/api/v1/fixtures/game_natural_lie2/replay?face_masked=false").json()
    r_msk = client.post("/api/v1/fixtures/game_natural_lie2/replay?face_masked=true").json()
    assert r_vis["ModalityUsed"] == ModalityMode.AUDIO_PLUS_VISIBLE_FACE.value
    assert r_msk["ModalityUsed"] == ModalityMode.AUDIO_PLUS_FACE_MASKED_VIDEO.value
    assert len(r_vis["candidate_events"]) >= 1
    assert len(r_msk["candidate_events"]) == 0  # Masked rerun omits facial observations

    # Verify masked MP4 preserves audio stream
    r_media = client.get("/api/v1/fixtures/game_natural_lie2/media/masked")
    assert r_media.status_code == 200
    assert len(r_media.content) > 10_000


def test_game_spans_pause_and_unpunctuated_numbered_claims(client: TestClient) -> None:
    sid = "e2e_fragmented_game"
    client.post(f"/api/v1/session/{sid}/reset")
    response = client.post("/api/v1/session/analyze", json={
        "session_id": sid,
        "audio_route": AudioRouteKind.GLASSES_PRIVATE.value,
        "utterances": [
            {"segment_id": "rules", "text": "two truths", "start_sec": 0, "end_sec": 1},
            {"segment_id": "round", "text": "and one lie one I live in Boston two I own a boat three I ran a marathon",
             "start_sec": 2, "end_sec": 10},
        ],
    })
    assert response.status_code == 200
    result = response.json()
    assert result["should_speak_privately"]
    assert len(result["stories"]) == 3
    assert "Boston" in result["stories"][0]["transcript"]
    assert "boat" in result["stories"][1]["transcript"]
    assert "marathon" in result["stories"][2]["transcript"]


def test_glasses_front_speaker_asr_mishear_still_locks_lie_guess(client: TestClient) -> None:
    """Front-speaker invitation mangled by glasses mic must still enter Game and lock a guess."""
    sid = "e2e_asr_mishear_game"
    client.post(f"/api/v1/session/{sid}/reset")
    response = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "turn_asr",
            "capture_source": CaptureSourceKind.GLASSES.value,
            "audio_route": AudioRouteKind.GLASSES_PRIVATE.value,
            "utterances": [
                {
                    "segment_id": "u0",
                    "text": "Let's play to truths and alive.",
                    "start_sec": 0.0,
                    "end_sec": 1.6,
                },
                {
                    "segment_id": "u1",
                    "text": "One, in October 2021 I missed a connecting flight at Chicago Airport.",
                    "start_sec": 2.0,
                    "end_sec": 5.2,
                    "pause_before_sec": 0.3,
                },
                {
                    "segment_id": "u2",
                    "text": "Two, last summer I kind of cooked pasta somewhere.",
                    "start_sec": 6.2,
                    "end_sec": 9.0,
                    "pause_before_sec": 0.95,
                },
                {
                    "segment_id": "u3",
                    "text": "Three, in March 2023 I needed six stitches at a university clinic.",
                    "start_sec": 10.0,
                    "end_sec": 13.4,
                    "pause_before_sec": 0.4,
                },
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["activity"] == ActivityKind.GAME.value
    assert data["initial_guess"] is not None
    assert data["initial_guess"]["chosen_story_number"] == 2
    assert data["should_speak_privately"] is True
    assert "lie" in (data["spoken_cue_text"] or "").lower()


def test_ordinary_conversation_ask_then_announce_lie(client: TestClient) -> None:
    """Non-game talk: ask when unsure, announce when face evidence is strong."""
    sid = "e2e_conversation_doubt"
    client.post(f"/api/v1/session/{sid}/reset")

    claim = {
        "segment_id": "u1",
        "text": "So yeah I kind of probably closed the whole round last week somehow",
        "start_sec": 1.0,
        "end_sec": 4.5,
        "pause_before_sec": 0.9,
        "speech_rate_wps": 2.8,
    }
    frames = [
        FrameSample(
            frame_id=f"frame_{i:05d}",
            timestamp_sec=round(1.0 + i * 0.2, 2),
            blendshapes={"lip_press": 0.48 if abs((1.0 + i * 0.2) - 2.4) < 0.15 else 0.03},
            motion_score=0.28 if abs((1.0 + i * 0.2) - 2.4) < 0.15 else 0.02,
        ).model_dump()
        for i in range(20)
    ]

    r = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "turn_ask",
            "audio_route": AudioRouteKind.GLASSES_PRIVATE.value,
            "modality_mode": "audio_plus_visible_face",
            "utterances": [claim],
            "frames": frames,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["activity"] == ActivityKind.UNCLEAR.value
    assert data["initial_guess"] is None
    assert data["deal_output"] is not None
    assert data["deal_output"]["action"] in ("ask_followup", "announce_lie")
    assert data["spoken_cue_text"]
    assert data["should_speak_privately"] is True
    assert "ask" in data["spoken_cue_text"].lower() or "lie" in data["spoken_cue_text"].lower()

    hot_frames = [
        FrameSample(
            frame_id=f"hot_{i:05d}",
            timestamp_sec=round(1.0 + i * 0.2, 2),
            blendshapes={"lip_press": 0.72 if abs((1.0 + i * 0.2) - 2.4) < 0.2 else 0.03},
            motion_score=0.4 if abs((1.0 + i * 0.2) - 2.4) < 0.2 else 0.02,
        ).model_dump()
        for i in range(20)
    ]
    r2 = client.post(
        "/api/v1/session/analyze",
        json={
            "session_id": sid,
            "turn_id": "turn_announce",
            "audio_route": AudioRouteKind.GLASSES_PRIVATE.value,
            "modality_mode": "audio_plus_visible_face",
            "utterances": [claim],
            "frames": hot_frames,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["deal_output"]["action"] in ("ask_followup", "announce_lie", "silence")
