# Sherlocked iOS build specification

Sherlocked is a native iPhone app for Ray-Ban Meta glasses. The wearer starts a session by voice or with the app’s Start button. Sherlocked recognizes the activity from the conversation and uses timestamped facial movements and speech to decide what to say. Build the research harness first, then use its results to drive the same pipeline in the app.

## Instructions for the builder

Use this specification as the implementation brief for one continuous build. **There is no timebox.** Take the time needed to research the uncertain parts, build the complete flow, and run the verification. Process real sessions and research clips through their natural boundaries. The product has no fixed calibration period or required decision cadence. Ordinary network timeouts and cancellation still apply.

**Tweak the implementation to suit the project.** Inspect the workspace, installed Xcode and SDKs, Meta device access, and available Gemini endpoints first. Adapt the capture strategy, dependencies, project structure, backend interfaces, prompts, and UI when the evidence or tooling calls for it. Choose and implement routine adjustments directly. Record material deviations and the reason for each in the handoff.

Preserve the product requirements while making those adjustments. Sherlocked remains a native iOS app with Ray-Ban support, voice start, automatic activity detection without a mode picker, facial analysis tied to specific frames, activity-specific output, and replay. Keep the straight-face controls and the research-first order. The runtime uses Gemini 4 Neon, Gemini 4 Carbon, and the Decisions API. For the coding agent, prefer Gemini 4 Carbon when both options have equivalent repo, terminal, and Xcode access. Claude Opus 5.5 is suitable when it has the stronger local tool access. [Gemini 4 Carbon](https://deepmind.google/models/model-cards/gemini-4-carbon/) · [Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5)

Build and verify in this order:

1. Establish a repeatable replay benchmark and run the facial-spotting and straight-face experiments with the data and Gemini access available. Use those results to choose how much weight facial evidence gets. If independent data or model access is unavailable, build the harness and validate it with clearly identified fixtures, then mark the scientific result pending.
2. Build the full phone and replay flow against the shared pipeline, including automatic context changes, the locked game guess, deal and practice outputs, and frame-linked replay. Run the build and tests and fix failures within scope.
3. Wire the glasses to the same session code. Test hardware capture, voice launch, and audio routing on a paired device when available. If hardware, credentials, permissions, or Meta provisioning block live testing, complete the independent work and report the exact remaining dependency and the test it blocks.
4. Rerun the acceptance tests from a clean setup. Deliver the working code, reproducible commands, and a verification report that identifies which runs used fixtures, live Gemini, and a physical iPhone with glasses. Keep failed tests and pending checks visible in that report.

## User experience

After one-time pairing and permission setup, the primary voice command is **“Hey Meta, start Sherlocked.”** Test **“Hey Meta, connect me to Sherlocked”** as the preferred wording. Register the app under the spoken name Sherlocked and use Meta’s `LaunchApp` invocation. Use the Meta beta release channel for device validation. The exact preferred phrase and locked-phone behavior must be tested on the actual device and recorded as verified or unavailable. Meta documents app-name launch as experimental. [Meta Voice Invocations](https://github.com/meta/meta-wearables-dat-ios/blob/main/plugins/mwdat-ios/skills/voice-invocations/SKILL.md) · [Meta device testing guidance](https://github.com/meta/meta-wearables-dat-ios/discussions/326)

The voice command and Start button enter the same session flow. Play **“Sherlocked is listening”** only once the intended camera and microphone are delivering data. A repeated launch returns to the active session. Show the capture source, recording state, audio route, and a read-only activity indicator. Include a prominent Stop control. If the phone cannot begin capture from a locked or background state, show the required unlock or foreground action. Default to the paired glasses when available and offer iPhone camera and mic with earbuds as a fallback. If a private playback route disappears, present the cue on screen.

### Automatic activity detection

| Activity | How it starts | What Sherlocked produces |
| --- | --- | --- |
| Two Truths and a Lie | Speech establishes the game rules or clearly names the game. Count complete stories from the same player. | After three stories, lock one guess and whisper “I think number two is the lie,” substituting the chosen number. If the group asks for follow-ups before the reveal, ask one comparable question per story and keep both the initial guess and the updated guess. |
| Deal discussion | Speech establishes that people are discussing a specific term or commitment. | When a usable facial observation coincides with an identifiable term and supports an actionable follow-up, whisper a short question like “Ask what could delay Friday.” Otherwise stay silent. |
| Practice | The conversation explicitly establishes role-play, rehearsal, or coaching. This takes precedence over deal language inside the exercise. | After the exercise, show a replayable moment and a suggested follow-up. |
| Unclear | The conversation is ordinary, incomplete, or ambiguous. | Stay silent and keep the status at Listening for context. |

Keep a recognized game stable through the reveal. A clear transition can start a new activity in the same session. Three arbitrary statements never trigger a game guess. Missing or overlapping speech can leave a story pending. Offer a manual correction for story boundaries and the final reveal. Keep the reveal label outside every input to the guesser.

## Capture and Gemini pipeline

- **iOS capture.** Use SwiftUI, Meta’s Device Access Toolkit, and AVFoundation. Put glasses, phone, and replay behind the same capture interface. Use glasses video plus iPhone mic as the baseline route and use synchronized glasses audio when enabled and verified. Normalize timestamps to one session clock and check alignment against a reference recording. Handle permission denial, interruptions, disconnects, and repeat starts without leaving capture running after Stop.
- **Facial observations.** Use an on-device landmark or blendshape tracker to propose motion intervals. Preserve original frames through processing, measure the delivered frame rate, and send ordered, timestamped face crops before, during, and after candidate events. Compare with usable moments from the same speaker and similar speaking or listening conditions. Avoid fixed-duration calibration. Distinguish a visible, still face from low-quality or missing frames. Only attribute a movement to a speaker when audio and face association are usable. Benchmark explicit frames against the video path because the default video sampler can miss brief events. [Gemini video sampling](https://aistudio.google.com/docs/video-understanding) · [Meta capture sample](https://github.com/meta/meta-wearables-dat-ios/tree/main/samples/CameraAccess)
- **Context and observations with Gemini 4 Neon.** Infer the activity and story boundaries from the exchange. In the visual pass, describe observable changes and link them to frame identifiers and audio timestamps. For example, record a brief lip press before the word *Friday*.
- **Comparison with Gemini 4 Carbon.** Compare all three stories and their evidence in Game. In Deal or Practice, generate a short follow-up tied to the actual exchange when there is a useful candidate. Compare facial movement, voice, and wording. Keep observed actions distinct from interpretations of emotion or intent.
- **Final selection with the Gemini Decisions API.** Choose one story for a complete, usable game round. Select a proposed question or silence in Deal. Select a replay moment or none in Practice. Keep API credentials on the server. Validate structured results and referenced frames before presenting them. Use a stable result ID so retries, reconnects, or duplicate callbacks cannot replay a private cue. Keep replay and evaluation calls silent by default. [Gemini Decisions API](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/)

A visible, low-motion face is valid input. In a playable round with sparse facial evidence, Sherlocked can still make its best guess from the complete exchange and disclose the sparse visual evidence in replay. If capture prevents the app from distinguishing the stories, ask for a retry. In Deal, sparse or unusable facial evidence defaults to silence. A straight face, a gaze shift, a blink, a pause, and a change in expression are observations. Clinical or moral labels have no role in the product. [DePaulo et al.](https://smg.media.mit.edu/library/DePauloEtAl.Cues%20to%20Deception.pdf) · [Matsumoto and Hwang](https://pmc.ncbi.nlm.nih.gov/articles/PMC6305322/)

## Research before application implementation

1. **Measure facial spotting.** Start with annotated CASME II or SAMM Long Videos. Compare the original frames, ordinary video sampling, and versions at effective glasses frame rates. Score event precision and recall against annotated intervals and include clips with no target movement. This measures what the camera and model can see. [CAS database access](https://melabipcas.github.io/melab/en/databases.html) · [SAMM Long Videos](https://arxiv.org/abs/1911.01519)
2. **Measure game decisions.** Prototype with MU3D and use DOLOS or Bag of Lies as separate benchmarks where access and terms permit. Collect consenting volunteers’ unrehearsed same-speaker three-story rounds for the primary test. Have the player privately record the lie before speaking. Balance lie position and hold out whole speakers. Include natural and deliberately neutral delivery, with reviewers labeling visible movement blind to the answer. [MU3D](https://www.lloydsocialdetectionlab.com/research-materials) · [DOLOS](https://rose1.ntu.edu.sg/dataset/DOLOS/) · [Bag of Lies](https://iab-rubric.org/index.php/bag-of-lies)
3. **Isolate the camera’s contribution.** Run the same held-out rounds with transcript only, audio only, audio plus face-masked video, and audio plus visible face. Keep the task and model settings fixed and omit facial notes from masked runs. Record picks before reveal. Compare paired accuracy against audio and one-in-three chance, with sample sizes and uncertainty. Report straight-face results separately. Compare before and after follow-up in a separate experiment. If the camera shows no clear gain, present the guess as exploratory and report the result. Reserve a micro-expression claim for evidence tied to independently annotated brief movements.
4. **Measure deal cues separately.** Use consensual mock deals where the counterpart privately records a constraint ahead of time. Compare whether Sherlocked’s question surfaces it against a generic follow-up, and count interruptions on all-honest exchanges. The game benchmark establishes game performance. The deal test measures follow-up usefulness.

## End-to-end acceptance tests

| Test | Passing behavior |
| --- | --- |
| Fresh installation | One-time onboarding connects the glasses and handles permissions. The phone capture path completes the same game when glasses are unavailable. |
| Hands-free start | A supported command starts exactly one session on real glasses without touching the phone. Record results separately for the documented and preferred phrases and for locked and unlocked states. The ready cue corresponds to actual capture readiness. |
| Unrehearsed game | The app recognizes the spoken rules, tracks three stories, whispers exactly one initial guess, and locks it before the reveal. Optional follow-up saves a distinct guess. |
| Straight-face controls | Replay covers a neutral truth and neutral lie from the same speaker, all-neutral delivery, an uncomfortable truth, a rehearsed lie, and a mostly true story with one fabricated detail. Report misses and sparse visual evidence honestly. |
| Context switching | One session can move from the game reveal into a mock deal. A role-play deal produces practice feedback. Unclear activity and unfinished games remain quiet. |
| Capture failures | Low light, eyewear, fast head turns, frame loss, audio drift, overlapping speech, and a listener in frame cannot silently appear as valid observations from the speaker. |
| Recovery | Network timeout, cancelled analysis, repeated invocation, glasses disconnect, and audio route change produce clear state and no duplicate playback. Replay and Stop remain usable. |
| Evidence and evaluation | The app replays the cited frames with matching audio. A face-masked rerun preserves audio and records the separate result. A clean install can run the documented replay demo with explicitly identified fixtures. Live Gemini and real glasses tests are recorded separately. |

## Deliverables for the build

Deliver a buildable Xcode project, backend, setup and launch instructions, a consented replay fixture set, an evaluation runner and results, and a real-device test record. Keep credentials and restricted research footage outside the public repository. For Element 119, the live demo starts by voice, proceeds through a fresh game and reveal, and transitions into a mock deal without a mode selector. The one-minute submission video and public repo identify the work built for the event.

The handoff includes the exact commands run and their outcomes, any changes made to this specification during implementation, and any checks still pending external access. Treat success on the phone or in replay as verification of that path. The voice-start path is verified when it starts capture and delivers the ready cue on a paired physical device.
