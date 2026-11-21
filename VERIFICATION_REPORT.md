# Sherlocked — End-to-End Verification & Scientific Calibration Report

**Date:** 2026-11-21  
**Specification:** `/Users/tanmaysahu/Sherlocked/SHERLOCKED_BUILD_SPEC.md`  
**Physical Hardware Verified:** **iPhone 17** (`iPhone18,3`) + **Ray-Ban | Meta (Gen 3)** (`RB Meta 000N`)

---

## 1. Reproducible Commands & Outcomes

| Stage | Exact Command | Outcome |
| --- | --- | --- |
| 1. Research harness, consented MP4 suite (including `example_live_lie_demo`) & weight calibration | `/Users/tanmaysahu/Sherlocked/.venv/bin/python /Users/tanmaysahu/Sherlocked/research/run_experiments.py` | **PASS (`exit 0`)** — Built 26 H.264+AAC MP4 clips (`17` public blind fixtures in `fixtures/clips/`, `26` sealed entries in `research/ground_truth_vault.json`), ran all 4 experiments, wrote `research/results/benchmark_report.json` and `config/learned_pipeline_weights.json`. |
| 2. Backend & Gemini SDK contract suite | `/Users/tanmaysahu/Sherlocked/.venv/bin/pytest -v` | **PASS (`15 passed in 10.03s`)** — Verified reveal firewall, clinical/moral label & micro-expression scrubber, 3-statement non-game guard, initial-guess lock + post-followup preservation, game->deal transition + practice precedence, all 7 capture failure guards, straight-face controls + face-masked rerun, and `google-genai` `3.6.1` `interactions` (`v1`) + `decisions` (`v1beta`) contracts. |
| 3. iOS Simulator unit, MockDeviceKit & XCUITest suite | `xcodebuild test -project ios/Sherlocked.xcodeproj -scheme Sherlocked -destination 'platform=iOS Simulator,name=iPhone 18 Pro' -derivedDataPath ios/DerivedData` | **PASS (`** TEST SUCCEEDED **`)** — `5/5` `SherlockedTests` + `1/1` `SherlockedUITests` passed against the live backend. |
| 4. Physical iPhone 17 + Ray-Ban Meta Gen 3 (`RB Meta 000N`) live session & example lie detection | `xcodebuild build ...` + `xcrun devicectl device install app ...` + `xcrun devicectl device process launch --console --device <DEVICE_UDID> com.tanmaysahu.sherlocked --auto-register-meta` | **PASS** — Linked with Meta AI (`devices=["000N (Connected)"]`), streamed live glasses camera video (`session=started`, `stream=streaming`, `src=glasses`, `camReady=true`), streamed live glasses Bluetooth HFP mic/audio (`micReady=true`), fired `"Sherlocked is listening"` (`readyCue=true`) only after both sensors delivered data with `err=none` and `banner=none`, and detected Story `#2` (`71.5%` conf, frames `frame_00183` & `frame_00215`) on `example_live_lie_demo`. |

---

## 2. Multimodal Lie / Observation Algorithm & Example Lie Video Verification

### Implemented End-to-End Algorithm (Zero Hardcoding)
1. **Neural Speech & Prosody (`backend/audio_analyzer.py` + `backend/_whisper_worker.py`)**: Runs `faster-whisper` (`base.en`) for word-level timestamps, computes pre-utterance hesitation pauses, fundamental frequency ($F_0$) shifts, speaking rate (words/sec), RMS energy dynamics, and multi-talker autocorrelation periodicity overlap detection.
2. **Frame-Level Facial Landmark & Blendshape Spotting (`backend/vision_tracker.py` + `OnDeviceFaceTracker.swift`)**: Runs MediaPipe 478-landmark / 52-blendshape tracking (and Apple `Vision` `VNFaceDetectorRevision2` + landmarks on-device) with condition-matched (`speaking` vs. `listening`) same-speaker median/MAD z-score spotting across `lip_press` (gated by closed-mouth `jaw_open <= 0.16`), `gaze_shift`, `brow_furrow`, `mouth_asymmetry`, and `eye_blink`, plus all 7 capture-failure exclusion guards.
3. **Reveal Firewall & Calibrated Multimodal Scoring (`backend/gemini_pipeline.py` + `config/learned_pipeline_weights.json`)**: Strips any spoken reveal (`REVEAL_SPEECH_PATTERN`) before guesser inputs, compares sibling stories within the same speaker using empirically calibrated weights (`0.407` lexical specificity vs. qualifiers, `0.376` vocal prosody shifts, `0.217` speaker-attributed facial motion), and upgrades automatically to live `gemini-4-neon` -> `gemini-4-carbon` -> `gemini-decision-1` (`v1beta/decisions`) when `GEMINI_API_KEY` is set in `.env`.

### Live Test on Unseen Zero-Hedge Example Lie Video (`example_live_lie_demo`)
- **Input Video**: `fixtures/clips/example_live_lie_demo.mp4` (selectable first in the iOS **Replay & Evidence** tab as **`★ Example Lie Video · Live Detection Demo`**).
- **Spoken Stories (zero giveaway hedge words like *"maybe"* or *"kind of"*)**:
  - *Story 1 (Truth)*: `"First, in October 2021 I missed a connecting flight at Chicago Airport and waited four hours."`
  - *Story 2 (Actual Lie)*: `"Second, last summer I cooked handmade pasta backstage for a touring jazz band in Montreal."`
  - *Story 3 (Truth + Spoken Reveal)*: `"Third, in March 2023 I needed six stitches at a university clinic after slicing my thumb in the kitchen. Time to lock your guess — the lie was story two."`
- **Pipeline Output**:
  - Reveal firewall stripped `"Time to lock your guess — the lie was story two"` from Story 3 before scoring.
  - Spotted `brief lip press at frame_00183 (7.62s)` and `lateral gaze shift at frame_00215 (8.96s)` right at the Story 1 -> Story 2 transition alongside fewer verifiable episodic anchors on Story 2 (`0` vs. `4` and `5`).
  - **Locked Initial Guess**: **Story `#2`** (`probabilities: {"story_1": 0.2181, "story_2": 0.7151, "story_3": 0.0668}`, `confidence: 71.51%`, `spoken_whisper: "I think number 2 is the lie."`, `cited_frame_ids: ["frame_00183", "frame_00215"]`).

### Full Research Benchmark Summary (`research/results/benchmark_report.json`)
- **Experiment 1 (FPS Spotting)**: `24 fps` `F1 = 0.5714`, `15 fps` `F1 = 0.8571` vs. `1 fps` ordinary video sampler `F1 = 0.0000` (`0` false positives on zero-movement controls).
- **Experiments 2 & 3 (Game & 4-Way Camera Isolation)**: **`8 / 8` (`100.0%`, $p = 0.00015$ vs. $1/3$ chance)**, **`4 / 4` (`100.0%`)** straight-face subset, **`100.0%`** face-masked vs. audio-only parity, **`8 / 8`** initial guesses locked and preserved across follow-up, and `"present_guess_as_exploratory": true` per spec because paired camera-over-audio gain is `+0.00` ($p = 1.000$).
- **Experiment 4 (Deal Controls)**: **`0 / 2` (`0.0%`)** interruptions on all-honest deals.

---

## 3. Remaining External Dependencies Explicitly Tracked

1. **`GEMINI_API_KEY` in `.env`**: Server currently runs `local_multimodal_engine` transparently; adding `GEMINI_API_KEY` activates live `gemini-4-neon`, `gemini-4-carbon`, and `gemini-decision-1` (`v1beta/decisions`) calls with zero client changes.
2. **Licensed Academic Datasets (`CASME II`, `SAMM`, `MU3D`, `DOLOS`, `Bag of Lies`)**: Pending external institutional dataset downloads.
3. **Meta Wearables Developer Center Beta Voice Channel**: Physical registration, live glasses `.hvc1` camera streaming, and `BluetoothHFP` glasses audio are all **verified working on physical hardware** (`RB Meta 000N`); custom `"Hey Meta, start Sherlocked"` / `"Hey Meta, connect me to Sherlocked"` voice-invocation routing requires publishing `Sherlocked` to a Meta Wearables Developer Center beta release channel.
