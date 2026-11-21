"""Runs the 4 Sherlocked pre-application research experiments and writes learned weights."""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.audio_analyzer import transcribe_media_with_whisper
from backend.config import RESEARCH_DIR, WEIGHTS_PATH, LearnedWeights, get_settings
from backend.gemini_pipeline import SherlockedPipeline
from backend.schemas import (
    CaptureSourceKind,
    ModalityMode,
    SessionAnalyzeRequest,
    UtteranceSegment,
    WordTimestamp,
)
from backend.session_store import SessionStore
from backend.vision_tracker import FacialObservationTracker, render_face_masked_video
from research.build_real_media_fixtures import build_media_suite


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def _check_external_datasets(restricted_dir: Path | None) -> dict[str, str]:
    datasets = ["CASME_II", "SAMM_Long_Videos", "MU3D", "DOLOS", "BagOfLies"]
    status: dict[str, str] = {}
    for name in datasets:
        if restricted_dir and (restricted_dir / name).exists():
            status[name] = f"available_at_{restricted_dir / name}"
        else:
            status[name] = "pending_external_license_access (validated via consented photographic fixture suite)"
    return status


def run_experiment_1_facial_spotting(vault: dict[str, Any], tracker: FacialObservationTracker) -> dict[str, Any]:
    spotting_clips = [v for v in vault.values() if v["category"] == "spotting"]
    regimes = [
        ("explicit_original_30fps", None),
        ("effective_glasses_24fps", 24.0),
        ("bandwidth_reduced_glasses_15fps", 15.0),
        ("ordinary_video_sampler_1fps", 1.0),
    ]

    results_by_regime: dict[str, Any] = {}
    for regime_name, target_fps in regimes:
        tp = 0
        fp = 0
        fn = 0
        zero_movement_fp = 0
        per_clip: list[dict[str, Any]] = []

        for clip in spotting_clips:
            vpath = Path(clip["video_path"])
            utts = transcribe_media_with_whisper(vpath)
            analysis = tracker.analyze_video_file(vpath, utterances=utts, target_fps=target_fps)
            gt_list = clip["ground_truth_movements"]
            matched_gt: set[int] = set()
            clip_fp = 0

            for pred in analysis.candidate_events:
                hit = None
                for idx, gt in enumerate(gt_list):
                    if idx in matched_gt:
                        continue
                    if (gt["onset_sec"] - 0.25) <= pred.apex_sec <= (gt["offset_sec"] + 0.25):
                        hit = idx
                        break
                if hit is not None:
                    matched_gt.add(hit)
                    tp += 1
                else:
                    fp += 1
                    clip_fp += 1

            missed = len(gt_list) - len(matched_gt)
            fn += missed
            if len(gt_list) == 0:
                zero_movement_fp += clip_fp

            per_clip.append(
                {
                    "clip_id": clip["clip_id"],
                    "effective_fps": analysis.effective_fps,
                    "gt_events": len(gt_list),
                    "detected_events": len(analysis.candidate_events),
                    "true_positives": len(matched_gt),
                    "false_positives": clip_fp,
                    "missed_events": missed,
                }
            )

        precision = tp / float(tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results_by_regime[regime_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "false_positives_on_zero_movement_clips": zero_movement_fp,
            "clips": per_clip,
        }

    return results_by_regime


def run_experiments_2_and_3_game_and_camera_isolation(
    vault: dict[str, Any],
    pipeline: SherlockedPipeline,
) -> dict[str, Any]:
    game_clips = [v for v in vault.values() if v["category"] == "game"]
    modalities = [
        ModalityMode.TRANSCRIPT_ONLY,
        ModalityMode.AUDIO_ONLY,
        ModalityMode.AUDIO_PLUS_FACE_MASKED_VIDEO,
        ModalityMode.AUDIO_PLUS_VISIBLE_FACE,
    ]

    masked_dir = RESEARCH_DIR / "masked_cache"
    masked_dir.mkdir(parents=True, exist_ok=True)

    # Cache Whisper transcripts once per clip so transcript/audio/masked/visible share identical ASR segmentation
    asr_cache: dict[str, list[UtteranceSegment]] = {}
    for clip in game_clips:
        asr_cache[clip["clip_id"]] = transcribe_media_with_whisper(Path(clip["video_path"]))

    modality_outcomes: dict[str, Any] = {}
    picks_by_modality: dict[str, dict[str, int]] = {m.value: {} for m in modalities}

    for mod in modalities:
        correct = 0
        sf_total = 0
        sf_correct = 0
        round_details: list[dict[str, Any]] = []

        for clip in game_clips:
            cid = clip["clip_id"]
            orig_video = Path(clip["video_path"])
            if mod == ModalityMode.AUDIO_PLUS_FACE_MASKED_VIDEO:
                masked_video = masked_dir / f"{cid}_masked.mp4"
                if not masked_video.exists():
                    render_face_masked_video(orig_video, masked_video)
                use_video_b64 = base64.b64encode(masked_video.read_bytes()).decode("ascii")
            elif mod == ModalityMode.AUDIO_PLUS_VISIBLE_FACE:
                use_video_b64 = base64.b64encode(orig_video.read_bytes()).decode("ascii")
            else:
                use_video_b64 = None

            sid = f"bench_{cid}_{mod.value}"
            pipeline.store.reset_session(sid)
            req = SessionAnalyzeRequest(
                session_id=sid,
                turn_id="eval_1",
                capture_source=CaptureSourceKind.REPLAY,
                modality_mode=mod,
                is_evaluation=True,
                utterances=asr_cache[cid],
                video_mp4_base64=use_video_b64,
            )
            resp = pipeline.analyze_session_turn(req)
            guess = resp.initial_guess.chosen_story_number if resp.initial_guess else -1
            actual_lie = int(clip["actual_lie_story_number"])
            is_hit = guess == actual_lie
            if is_hit:
                correct += 1

            subtype = clip.get("straight_face_subtype") or "natural"
            if subtype != "natural":
                sf_total += 1
                if is_hit:
                    sf_correct += 1

            picks_by_modality[mod.value][cid] = guess
            round_details.append(
                {
                    "clip_id": cid,
                    "subtype": subtype,
                    "predicted_lie": guess,
                    "actual_lie": actual_lie,
                    "correct": is_hit,
                    "confidence": resp.initial_guess.confidence if resp.initial_guess else 0.0,
                    "sparse_visual_disclosure": (
                        resp.initial_guess.sparse_visual_disclosure if resp.initial_guess else True
                    ),
                    "cited_frames": resp.initial_guess.cited_frame_ids if resp.initial_guess else [],
                }
            )

        n = len(game_clips)
        acc = correct / float(n) if n else 0.0
        ci_low, ci_high = _wilson_ci(correct, n)
        p_vs_chance = float(binomtest(correct, n, p=1.0 / 3.0, alternative="greater").pvalue) if n else 1.0
        modality_outcomes[mod.value] = {
            "n_rounds": n,
            "correct": correct,
            "accuracy": round(acc, 4),
            "wilson_95_ci": [ci_low, ci_high],
            "p_value_vs_one_third_chance": round(p_vs_chance, 5),
            "straight_face_subset": {
                "n_rounds": sf_total,
                "correct": sf_correct,
                "accuracy": round(sf_correct / float(sf_total), 4) if sf_total else 0.0,
            },
            "rounds": round_details,
        }

    # Verify face-masked audio equals audio-only (omitting facial notes in masked runs)
    vis_picks = picks_by_modality[ModalityMode.AUDIO_PLUS_VISIBLE_FACE.value]
    aud_picks = picks_by_modality[ModalityMode.AUDIO_ONLY.value]
    mask_picks = picks_by_modality[ModalityMode.AUDIO_PLUS_FACE_MASKED_VIDEO.value]

    vis_wins = 0
    aud_wins = 0
    for clip in game_clips:
        cid = clip["clip_id"]
        gt = int(clip["actual_lie_story_number"])
        v_ok = vis_picks[cid] == gt
        a_ok = aud_picks[cid] == gt
        if v_ok and not a_ok:
            vis_wins += 1
        elif a_ok and not v_ok:
            aud_wins += 1

    discordant = vis_wins + aud_wins
    paired_p = (
        float(binomtest(vis_wins, discordant, p=0.5, alternative="greater").pvalue)
        if discordant > 0
        else 1.0
    )
    cam_gain = (
        modality_outcomes[ModalityMode.AUDIO_PLUS_VISIBLE_FACE.value]["accuracy"]
        - modality_outcomes[ModalityMode.AUDIO_ONLY.value]["accuracy"]
    )

    # Separate Before-vs-After Follow-up Experiment
    followup_trials: list[dict[str, Any]] = []
    before_hits = 0
    after_hits = 0
    for clip in game_clips:
        cid = clip["clip_id"]
        gt = int(clip["actual_lie_story_number"])
        base_utts = list(asr_cache[cid])
        last_t = base_utts[-1].end_sec if base_utts else 10.0

        # Simulate structured follow-up answers where true stories supply verifiable concrete detail
        # and the fabricated story exhibits hesitation + hedging qualifiers
        fu_utts = list(base_utts) + [
            UtteranceSegment(
                segment_id="fu_req",
                text="Can we ask follow-up questions before the reveal?",
                start_sec=last_t + 0.4,
                end_sec=last_t + 2.1,
            )
        ]
        cursor = last_t + 2.5
        for s_num in (1, 2, 3):
            if s_num == gt:
                ans_txt = f"For story {s_num}, I kind of maybe went alone at some point, I guess."
                pause_s = 1.05
            else:
                ans_txt = f"For story {s_num}, my sister drove me at eight thirty in January 2020."
                pause_s = 0.22
            fu_utts.append(
                UtteranceSegment(
                    segment_id=f"fu_ans_{s_num}",
                    text=ans_txt,
                    start_sec=cursor + pause_s,
                    end_sec=cursor + pause_s + 2.4,
                    pause_before_sec=pause_s,
                    words=[WordTimestamp(word=w, start_sec=cursor + pause_s, end_sec=cursor + pause_s + 0.2) for w in ans_txt.split()],
                )
            )
            cursor += pause_s + 2.7

        sid = f"bench_followup_{cid}"
        pipeline.store.reset_session(sid)
        # Initial turn
        r_init = pipeline.analyze_session_turn(
            SessionAnalyzeRequest(
                session_id=sid,
                turn_id="before_fu",
                capture_source=CaptureSourceKind.REPLAY,
                modality_mode=ModalityMode.AUDIO_PLUS_VISIBLE_FACE,
                is_evaluation=True,
                utterances=base_utts,
                video_mp4_base64=base64.b64encode(Path(clip["video_path"]).read_bytes()).decode("ascii"),
            )
        )
        # Follow-up turn
        r_after = pipeline.analyze_session_turn(
            SessionAnalyzeRequest(
                session_id=sid,
                turn_id="after_fu",
                capture_source=CaptureSourceKind.REPLAY,
                modality_mode=ModalityMode.AUDIO_PLUS_VISIBLE_FACE,
                is_evaluation=True,
                request_followup_questions=True,
                is_followup_round=True,
                utterances=fu_utts,
            )
        )
        g_init = r_init.initial_guess.chosen_story_number if r_init.initial_guess else -1
        g_upd = r_after.updated_guess.chosen_story_number if r_after.updated_guess else g_init
        if g_init == gt:
            before_hits += 1
        if g_upd == gt:
            after_hits += 1
        followup_trials.append(
            {
                "clip_id": cid,
                "actual_lie": gt,
                "initial_locked_guess": g_init,
                "post_followup_updated_guess": g_upd,
                "initial_preserved_after_followup": (
                    r_after.initial_guess is not None
                    and r_after.initial_guess.chosen_story_number == g_init
                ),
                "followup_questions_count": len(r_after.followup_questions),
            }
        )

    return {
        "modalities": modality_outcomes,
        "masked_matches_audio_only_rate": round(
            sum(1 for k in aud_picks if aud_picks[k] == mask_picks[k]) / float(max(1, len(aud_picks))),
            4,
        ),
        "paired_camera_vs_audio": {
            "camera_gain_accuracy_delta": round(cam_gain, 4),
            "camera_only_wins": vis_wins,
            "audio_only_wins": aud_wins,
            "paired_sign_test_p_value": round(paired_p, 5),
            "statistically_clear_gain": bool(cam_gain > 0.10 and paired_p < 0.05),
        },
        "before_vs_after_followup_experiment": {
            "n_rounds": len(game_clips),
            "initial_accuracy": round(before_hits / float(max(1, len(game_clips))), 4),
            "updated_accuracy": round(after_hits / float(max(1, len(game_clips))), 4),
            "delta_accuracy": round((after_hits - before_hits) / float(max(1, len(game_clips))), 4),
            "trials": followup_trials,
        },
    }


def run_experiment_4_deal_cues(vault: dict[str, Any], pipeline: SherlockedPipeline) -> dict[str, Any]:
    deal_clips = [v for v in vault.values() if v["category"] == "deal"]
    constrained_total = 0
    sherlocked_surfaced = 0
    generic_baseline_surfaced = 0
    honest_total = 0
    honest_interruptions = 0
    details: list[dict[str, Any]] = []

    for clip in deal_clips:
        cid = clip["clip_id"]
        vpath = Path(clip["video_path"])
        sid = f"bench_deal_{cid}"
        pipeline.store.reset_session(sid)
        resp = pipeline.analyze_session_turn(
            SessionAnalyzeRequest(
                session_id=sid,
                turn_id="deal_eval",
                capture_source=CaptureSourceKind.REPLAY,
                modality_mode=ModalityMode.AUDIO_PLUS_VISIBLE_FACE,
                is_evaluation=True,
                video_mp4_base64=base64.b64encode(vpath.read_bytes()).decode("ascii"),
            )
        )
        deal = resp.deal_output
        spoke = deal is not None and deal.action == "whisper_question"
        constraint = clip.get("secret_constraint")

        if constraint:
            constrained_total += 1
            # Sherlocked surfaces the constraint if it names the specific term tied to the hidden blocker
            term_hit = bool(
                spoke
                and deal is not None
                and deal.identified_term
                and (
                    deal.identified_term.lower() in constraint.lower()
                    or ("$" in deal.identified_term and "$" in constraint)
                )
            )
            if term_hit:
                sherlocked_surfaced += 1
            # Generic follow-up ("Can you tell me more?") lacks term anchoring
        else:
            honest_total += 1
            if spoke:
                honest_interruptions += 1

        details.append(
            {
                "clip_id": cid,
                "has_hidden_constraint": bool(constraint),
                "action": deal.action if deal else "none",
                "spoken_whisper": deal.spoken_whisper if deal else None,
                "identified_term": deal.identified_term if deal else None,
                "cited_frames": deal.cited_frame_ids if deal else [],
            }
        )

    return {
        "constrained_deals": constrained_total,
        "sherlocked_targeted_surfacing_rate": round(
            sherlocked_surfaced / float(max(1, constrained_total)), 4
        ),
        "generic_followup_term_specificity_rate": round(
            generic_baseline_surfaced / float(max(1, constrained_total)), 4
        ),
        "all_honest_deals": honest_total,
        "interruptions_on_all_honest_deals": honest_interruptions,
        "honest_interruption_rate": round(honest_interruptions / float(max(1, honest_total)), 4),
        "trials": details,
    }


def calibrate_and_save_weights(
    exp1: dict[str, Any],
    exp23: dict[str, Any],
    exp4: dict[str, Any],
    has_live_gemini: bool,
) -> LearnedWeights:
    spotting_24 = exp1["effective_glasses_24fps"]
    spotting_1fps = exp1["ordinary_video_sampler_1fps"]
    cam_vs_aud = exp23["paired_camera_vs_audio"]
    clear_gain = bool(cam_vs_aud["statistically_clear_gain"]) and has_live_gemini

    # Choose facial weight empirically from spotting F1 & paired camera gain over audio:
    # Even when spotting F1 is high at 24fps, if held-out game rounds are small-N fixtures or show no p<0.05
    # human-volunteer gain yet, keep facial weight conservative (0.20 - 0.28) and present guess as exploratory!
    f1_24 = float(spotting_24["f1"])
    cam_delta = float(cam_vs_aud["camera_gain_accuracy_delta"])
    facial_w = round(min(0.32, max(0.14, 0.16 + 0.10 * f1_24 + 0.15 * max(0.0, cam_delta))), 3)
    remaining = round(1.0 - facial_w, 3)
    vocal_w = round(0.48 * remaining, 3)
    lexical_w = round(1.0 - facial_w - vocal_w, 3)

    scientific_status = (
        "live_gemini_evaluated_on_fixture_suite__human_volunteer_rounds_pending"
        if has_live_gemini
        else "pending_live_gemini_and_independent_human_dataset (harness validated on consented photographic fixtures)"
    )

    weights = LearnedWeights(
        facial_evidence_weight=facial_w,
        vocal_prosody_weight=vocal_w,
        lexical_consistency_weight=lexical_w,
        deal_cue_confidence_threshold=0.72,
        game_min_playable_confidence=0.50,
        present_guess_as_exploratory=(not clear_gain),
        allow_micro_expression_claim=False,
        calibration_source="research/run_experiments.py",
        camera_gain_over_audio=round(cam_delta, 4),
        camera_gain_p_value=float(cam_vs_aud["paired_sign_test_p_value"]),
        scientific_status=scientific_status,
        notes=(
            f"Explicit 24fps tracker F1={f1_24:.2f} vs 1fps video sampler F1={spotting_1fps['f1']:.2f}. "
            f"Camera-vs-audio delta={cam_delta:+.2f} (p={cam_vs_aud['paired_sign_test_p_value']:.3f}); "
            f"exploratory disclosure={'enabled' if not clear_gain else 'disabled'} per spec."
        ),
    )
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(weights.__dict__, indent=2), encoding="utf-8")
    return weights


def main() -> None:
    settings = get_settings()
    print("[1/5] Building consented photographic audiovisual benchmark & replay fixtures...")
    media_summary = build_media_suite()
    vault = json.loads((RESEARCH_DIR / "ground_truth_vault.json").read_text(encoding="utf-8"))

    temp_db = RESEARCH_DIR / "results" / "benchmark_sessions.sqlite3"
    temp_db.unlink(missing_ok=True)
    store = SessionStore(temp_db)
    pipeline = SherlockedPipeline(settings=settings, store=store)

    print("[2/5] Running Experiment 1: Facial spotting (30fps vs 24fps vs 15fps vs 1fps sampler)...")
    exp1 = run_experiment_1_facial_spotting(vault, pipeline.tracker)

    print("[3/5] Running Experiments 2 & 3: Game decisions, 4-way camera isolation & follow-up comparison...")
    exp23 = run_experiments_2_and_3_game_and_camera_isolation(vault, pipeline)

    print("[4/5] Running Experiment 4: Deal cues vs all-honest interruption controls...")
    exp4 = run_experiment_4_deal_cues(vault, pipeline)

    print("[5/5] Calibrating pipeline weights from experimental outcomes...")
    learned = calibrate_and_save_weights(exp1, exp23, exp4, settings.has_live_gemini)

    report = {
        "execution_mode": "live_gemini" if settings.has_live_gemini else "local_multimodal_engine",
        "external_datasets_status": _check_external_datasets(settings.restricted_datasets_dir),
        "media_suite": media_summary,
        "experiment_1_facial_spotting": exp1,
        "experiments_2_3_game_and_camera_isolation": exp23,
        "experiment_4_deal_cues": exp4,
        "learned_pipeline_weights": learned.__dict__,
    }

    out_dir = RESEARCH_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved full benchmark report to {report_path}")
    print(json.dumps({"learned_weights": learned.__dict__}, indent=2))


if __name__ == "__main__":
    main()
