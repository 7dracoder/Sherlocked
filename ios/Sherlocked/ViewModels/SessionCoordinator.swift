import AVFoundation
import Combine
import Foundation
import UIKit

enum SessionLaunchTrigger: String {
    case startButton = "start_button"
    case voiceCommand = "voice_command"
}

@MainActor
final class SessionCoordinator: ObservableObject {
    // MARK: - Core Session State (No Mode Picker!)
    @Published private(set) var sessionID: String = UUID().uuidString
    @Published private(set) var isSessionActive: Bool = false
    @Published private(set) var captureSource: CaptureSourceKind = .iphone
    @Published private(set) var hasCameraDeliveredFirstFrame: Bool = false
    @Published private(set) var hasMicDeliveredFirstSample: Bool = false
    @Published private(set) var hasPlayedReadyCue: Bool = false
    @Published private(set) var repeatedLaunchCount: Int = 0
    @Published var requiresForegroundUnlockPrompt: Bool = false

    // MARK: - Read-Only Activity & Analysis State
    @Published private(set) var detectedActivity: ActivityKind = .unclear
    @Published private(set) var statusLabel: String = "Idle · Ready to start"
    @Published private(set) var isAnalyzing: Bool = false
    @Published private(set) var latestPreviewImage: UIImage?
    @Published private(set) var effectiveFPS: Double = 0.0
    @Published private(set) var latestResponse: SessionAnalyzeResponseDTO?
    @Published private(set) var maskedComparisonResponse: SessionAnalyzeResponseDTO?
    @Published var onScreenCueBanner: String?
    @Published var recoveryBannerMessage: String?

    // MARK: - Replay & Onboarding State
    @Published var showOnboarding: Bool = false
    @Published var showManualCorrectionSheet: Bool = false
    @Published var showMockGlassesSheet: Bool = false
    @Published var availableFixtures: [ReplayFixtureItemDTO] = []
    @Published var selectedReplayFixture: ReplayFixtureItemDTO?
    @Published var activeReplayVideoURL: URL?
    /// DAT PCM is optional fallback only — Bluetooth HFP is the accurate listen + speak path.
    @Published var useSynchronizedGlassesAudio: Bool = false
    @Published var configuredBackendURL: String = ""

    let glassesCoordinator: GlassesWearablesCoordinator
    let audioEngine: AudioAndSpeechEngine
    let faceTracker = OnDeviceFaceTracker()
    let apiClient: BackendAPIClient

    private let phoneCamera = PhoneCameraCaptureSource()
    private var sessionStartMonotonic: TimeInterval = 0
    private var collectedFrames: [FrameSampleDTO] = []
    private var deliveredResultIDs: Set<String> = []
    private var analysisTask: Task<Void, Never>?
    private var cancellables = Set<AnyCancellable>()
    private var speechDebounce: Task<Void, Never>?
    private var pendingPayload: SessionAnalyzeRequestDTO?
    private var simulatedUtterances: [UtteranceSegmentDTO] = []
    private var frameProcessing = false
    private var turnSequence: Int = 0

    init(
        glassesCoordinator: GlassesWearablesCoordinator? = nil,
        audioEngine: AudioAndSpeechEngine? = nil,
        apiClient: BackendAPIClient = BackendAPIClient()
    ) {
        self.glassesCoordinator = glassesCoordinator ?? GlassesWearablesCoordinator()
        self.audioEngine = audioEngine ?? AudioAndSpeechEngine()
        self.apiClient = apiClient

        wireCallbacks()

        if !UserDefaults.standard.bool(forKey: "sherlocked_onboarding_completed") {
            self.showOnboarding = true
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            let url = await self.apiClient.baseURL
            self.configuredBackendURL = url
            self.audioEngine.configureBackendURL(url)
        }

        if ProcessInfo.processInfo.arguments.contains("--auto-register-meta") {
            Task { @MainActor [weak self] in
                await self?.runLiveDeviceMetaLinkAndBackendProbe()
            }
        }
        #if DEBUG
        if ProcessInfo.processInfo.arguments.contains("--live-audio-check") {
            Task { @MainActor in
                for _ in 0..<12 where !self.glassesCoordinator.hasEligibleGlasses {
                    try? await Task.sleep(for: .milliseconds(250))
                }
                await startOrReturnToSession()
                let route = AVAudioSession.sharedInstance().currentRoute
                print("[Sherlocked] live audio check \(sessionID), input \(route.inputs.map(\.portType.rawValue)), output \(route.outputs.map(\.portType.rawValue))")
            }
        }
        #endif
    }

    func runLiveDeviceMetaLinkAndBackendProbe() async {
        let outputs = AVAudioSession.sharedInstance().currentRoute.outputs.map { "\($0.portName) (\($0.portType.rawValue))" }
        print("[SHERLOCKED_DEVICE_PROBE] Initial audio outputs: \(outputs)")
        print("[SHERLOCKED_DEVICE_PROBE] Initial DAT registrationState: \(glassesCoordinator.registrationStateDescription)")
        print("[SHERLOCKED_DEVICE_PROBE] Initial DAT devices: \(glassesCoordinator.connectedDeviceNames)")

        await loadFixtures()
        print("[SHERLOCKED_DEVICE_PROBE] Backend \(await apiClient.baseURL) fixtures loaded: \(availableFixtures.count)")

        if let gameFix = availableFixtures.first(where: { $0.fixture_id == "example_live_lie_demo" }) ?? availableFixtures.first(where: { $0.fixture_id == "game_natural_lie2" }) {
            await runFixtureReplay(gameFix)
            print("[SHERLOCKED_DEVICE_PROBE] Live iPhone->Backend Example Lie Video (\(gameFix.fixture_id)) Guess: #\(latestResponse?.initial_guess?.chosen_story_number ?? -1) conf=\(latestResponse?.initial_guess?.confidence ?? 0) frames=\(latestResponse?.initial_guess?.cited_frame_ids ?? []) status=\(statusLabel)")
            await runFaceMaskedRerunForSelectedFixture()
            print("[SHERLOCKED_DEVICE_PROBE] Live iPhone->Backend Masked Rerun Modality: \(maskedComparisonResponse?.ModalityUsed.rawValue ?? "none") events=\(maskedComparisonResponse?.candidate_events.count ?? -1)")
        }

        if glassesCoordinator.registrationState != .registered {
            print("[SHERLOCKED_DEVICE_PROBE] Calling Wearables.shared.startRegistration() to link with Meta AI app...")
            await glassesCoordinator.startRegistration()
        }

        for tick in 1...14 {
            try? await Task.sleep(for: .seconds(1))
            let curOut = AVAudioSession.sharedInstance().currentRoute.outputs.map { "\($0.portName) (\($0.portType.rawValue))" }
            let curIn = AVAudioSession.sharedInstance().currentRoute.inputs.map { "\($0.portName) (\($0.portType.rawValue))" }
            print("[SHERLOCKED_DEVICE_PROBE] t=\(tick)s reg=\(glassesCoordinator.registrationStateDescription) eligible=\(glassesCoordinator.hasEligibleGlasses) link=\(glassesCoordinator.isDeviceLinkConnected) devices=\(glassesCoordinator.connectedDeviceNames) session=\(glassesCoordinator.sessionState.description) stream=\(glassesCoordinator.streamState) src=\(captureSource.rawValue) camReady=\(hasCameraDeliveredFirstFrame) micReady=\(hasMicDeliveredFirstSample) readyCue=\(hasPlayedReadyCue) audioOut=\(curOut) audioIn=\(curIn) err=\(glassesCoordinator.lastWearablesError ?? "none") banner=\(recoveryBannerMessage ?? "none")")
            if glassesCoordinator.hasEligibleGlasses && !isSessionActive && tick == 2 {
                print("[SHERLOCKED_DEVICE_PROBE] Eligible Ray-Ban Meta glasses active! Starting live session...")
                await startOrReturnToSession(trigger: .startButton)
            }
        }
    }

    private func wireCallbacks() {
        glassesCoordinator.onVoiceLaunchReceived = { [weak self] in
            Task { @MainActor [weak self] in
                await self?.startOrReturnToSession(trigger: .voiceCommand)
            }
        }

        glassesCoordinator.onVideoFrameDelivered = { [weak self] img, relSec in
            Task { @MainActor [weak self] in
                self?.ingestLiveVideoFrame(img, timestampSec: relSec)
            }
        }

        glassesCoordinator.onGlassesAudioDelivered = { [weak self] pcm in
            Task { @MainActor [weak self] in
                guard let self else { return }
                // Never mix DAT glasses PCM into recognition when the user forced iPhone mic.
                if UserDefaults.standard.bool(forKey: "sherlocked_phone_mic") { return }
                self.audioEngine.consumeExternalGlassesPCM(pcm)
            }
        }

        glassesCoordinator.onGlassesDisconnectedMidSession = { [weak self] in
            Task { @MainActor [weak self] in
                guard let self, self.isSessionActive, self.captureSource == .glasses else { return }
                self.recoveryBannerMessage = "Ray-Ban Meta disconnected. Stream paused safely — tap Start to continue on iPhone or Stop to review replay."
                await self.glassesCoordinator.stopGlassesCapture()
                self.hasCameraDeliveredFirstFrame = false
            }
        }

        audioEngine.$livePartialTranscript.sink { [weak self] _ in
            self?.objectWillChange.send()
        }.store(in: &cancellables)
        audioEngine.$micStatusLine.sink { [weak self] _ in
            self?.objectWillChange.send()
        }.store(in: &cancellables)
        audioEngine.$micLevel.sink { [weak self] _ in
            self?.objectWillChange.send()
        }.store(in: &cancellables)
        audioEngine.$recognitionIssue.sink { [weak self] message in
            // Soft ASR reconnect noise — don't block the live UI with it.
            if let message, message.contains("ElevenLabs reconnecting") {
                return
            }
            if let message {
                self?.recoveryBannerMessage = message
            } else if self?.recoveryBannerMessage?.hasPrefix("Speech recognition is reconnecting.") == true {
                self?.recoveryBannerMessage = nil
            }
        }.store(in: &cancellables)

        audioEngine.$currentRoute
            .receive(on: RunLoop.main)
            .sink { [weak self] newRoute in
                guard let self else { return }
                // If private playback route disappears while a cue exists, surface it on screen!
                if !newRoute.isPrivate, self.onScreenCueBanner == nil, let cue = self.latestResponse?.spoken_cue_text, self.isSessionActive {
                    self.onScreenCueBanner = cue
                }
            }
            .store(in: &cancellables)
    }

    func completeOnboarding() {
        UserDefaults.standard.set(true, forKey: "sherlocked_onboarding_completed")
        showOnboarding = false
    }

    // MARK: - Unified Voice & Button Session Start

    func startOrReturnToSession(
        trigger: SessionLaunchTrigger = .startButton,
        forcePhoneFallback: Bool = false
    ) async {
        // Repeated launch contract: return to the active session without restarting or duplicating cues!
        if isSessionActive {
            repeatedLaunchCount += 1
            recoveryBannerMessage = "Returned to active Sherlocked session."
            return
        }

        // Check locked / background capture restriction on iOS
        let appState = UIApplication.shared.applicationState
        let protectedUnlocked = UIApplication.shared.isProtectedDataAvailable
        if !protectedUnlocked || appState == .background {
            requiresForegroundUnlockPrompt = true
            statusLabel = "Unlock & foreground iPhone to start capture"
            return
        }
        requiresForegroundUnlockPrompt = false
        recoveryBannerMessage = nil
        onScreenCueBanner = nil
        maskedComparisonResponse = nil
        latestResponse = nil

        sessionID = "live_\(UUID().uuidString.prefix(8))"
        sessionStartMonotonic = CACurrentMediaTime()
        simulatedUtterances.removeAll()
        turnSequence = 0
        collectedFrames.removeAll()
        faceTracker.reset()

        hasCameraDeliveredFirstFrame = false
        hasMicDeliveredFirstSample = false
        hasPlayedReadyCue = false
        detectedActivity = .unclear
        statusLabel = "Starting sensors..."
        isSessionActive = true

        let useGlasses = glassesCoordinator.hasEligibleGlasses && !forcePhoneFallback
        captureSource = useGlasses ? .glasses : .iphone

        // 1. Request Permissions (if undetermined) & Start Microphone + Live Speech Recognition
        let perms = await audioEngine.requestPermissions()
        guard isSessionActive else { return }
        if !perms.speech || !perms.mic {
            recoveryBannerMessage = "Enable Microphone and Speech Recognition in Settings to start live transcription."
            stopSession()
            return
        } else {
            do {
                try audioEngine.startLiveCapture(
                    sessionID: sessionID,
                    sessionStartMonotonic: sessionStartMonotonic,
                    hasPairedGlasses: useGlasses,
                    onFirstAudioBuffer: { [weak self] in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            self.hasMicDeliveredFirstSample = true
                            self.checkAndAnnounceReadyCue()
                        }
                    },
                    onUtteranceCommitted: { [weak self] in
                        Task { @MainActor [weak self] in
                            self?.scheduleSpeechUpdate()
                        }
                    }
                )
            } catch {
                recoveryBannerMessage = error.localizedDescription
                stopSession()
                return
            }
        }

        // 2. Start Intended Camera Source (Glasses primary -> seamless iPhone camera fallback while keeping Glasses HFP audio!)
        if useGlasses {
            do {
                let forcePhoneMic = UserDefaults.standard.bool(forKey: "sherlocked_phone_mic")
                try await glassesCoordinator.startGlassesCapture(
                    sessionStartClock: sessionStartMonotonic,
                    enableSynchronizedGlassesAudio: useSynchronizedGlassesAudio && !forcePhoneMic
                )
            } catch {
                captureSource = .iphone
                statusLabel = audioEngine.currentRoute == .glassesPrivate
                    ? "Glasses Audio + iPhone Camera Active"
                    : "iPhone Camera + Mic Active"
                do {
                    try await startPhoneCameraFallback()
                } catch {
                    recoveryBannerMessage = error.localizedDescription
                    stopSession()
                }
            }
        } else {
            do {
                try await startPhoneCameraFallback()
            } catch {
                recoveryBannerMessage = error.localizedDescription
                stopSession()
            }
        }
        // Glasses DAT / camera init can steal the audio route — put phone mic back.
        audioEngine.reassertPreferredMicIfNeeded()
        statusLabel = "Listening · \(audioEngine.micStatusLine)"
    }

    private func startPhoneCameraFallback() async throws {
        try await phoneCamera.start(sessionStartClock: sessionStartMonotonic) { [weak self] img, relSec in
            Task { @MainActor [weak self] in
                self?.ingestLiveVideoFrame(img, timestampSec: relSec)
            }
        }
    }

    private func ingestLiveVideoFrame(_ image: UIImage, timestampSec: Double) {
        guard isSessionActive else { return }
        latestPreviewImage = image
        guard !frameProcessing else { return }
        frameProcessing = true
        let sid = sessionID
        let tracker = faceTracker
        Task {
            let sample = await Task.detached(priority: .userInitiated) {
                tracker.processFrame(image: image, sessionTimestampSec: timestampSec)
            }.value
            frameProcessing = false
            guard isSessionActive, sessionID == sid else { return }
            acceptFrame(sample)
        }
    }

    private func acceptFrame(_ sample: FrameSampleDTO) {
        effectiveFPS = (faceTracker.effectiveFPS * 10).rounded() / 10
        collectedFrames.append(sample)
        if collectedFrames.count > 240 {
            collectedFrames.removeFirst(collectedFrames.count - 240)
        }
        if !hasCameraDeliveredFirstFrame {
            hasCameraDeliveredFirstFrame = true
            checkAndAnnounceReadyCue()
        }
    }

    /// Plays "Sherlocked is listening" ONLY once BOTH intended camera and microphone are delivering data.
    func checkAndAnnounceReadyCue() {
        guard isSessionActive, !hasPlayedReadyCue else { return }
        guard hasCameraDeliveredFirstFrame && hasMicDeliveredFirstSample else { return }

        hasPlayedReadyCue = true
        statusLabel = "Listening for context"
        onScreenCueBanner = "Sherlocked is listening"
        // Always attempt speak-back into glasses (falls back to speaker if needed).
        _ = audioEngine.speakCuePrivatelyIfPermitted("Sherlocked is listening", forceSpeak: true)
    }

    /// Used by automated tests and Simulator harness to verify sensor-readiness gating & live speech pipeline.
    func injectSimulatedMicrophoneSampleAndUtterance(_ text: String, startSec: Double, endSec: Double) {
        if !hasMicDeliveredFirstSample {
            hasMicDeliveredFirstSample = true
            checkAndAnnounceReadyCue()
        }
        let words = text.split(separator: " ").enumerated().map { idx, tok in
            let dur = max(0.1, (endSec - startSec) / Double(max(1, text.split(separator: " ").count)))
            let w0 = startSec + Double(idx) * dur
            return WordTimestampDTO(word: String(tok), start_sec: w0, end_sec: w0 + dur)
        }
        var current = simulatedUtterances
        let pause = max(0.0, startSec - (current.last?.end_sec ?? 0.0))
        current.append(
            UtteranceSegmentDTO(
                segment_id: String(format: "utt_%03d", current.count + 1),
                text: text,
                start_sec: startSec,
                end_sec: endSec,
                pause_before_sec: pause,
                words: words
            )
        )
        simulatedUtterances = current
        scheduleTurnAnalysis(overrideUtterances: current)
    }

    // MARK: - Stop & Cancellation

    func cancelInFlightAnalysis() {
        speechDebounce?.cancel()
        pendingPayload = nil
        analysisTask?.cancel()
        analysisTask = nil
        isAnalyzing = false
        recoveryBannerMessage = "Cancelled in-flight analysis. Session state preserved."
    }

    func stopSession() {
        speechDebounce?.cancel()
        pendingPayload = nil
        analysisTask?.cancel()
        analysisTask = nil
        isAnalyzing = false

        phoneCamera.stop()
        audioEngine.stopCapture()
        Task {
            await glassesCoordinator.stopGlassesCapture()
        }

        isSessionActive = false
        hasCameraDeliveredFirstFrame = false
        hasMicDeliveredFirstSample = false
        statusLabel = "Stopped · Session ready for replay"
    }

    // MARK: - Shared Pipeline Turn Execution

    private func scheduleSpeechUpdate() {
        speechDebounce?.cancel()
        speechDebounce = Task {
            // Short pause so front-speaker ASR can settle — keep snappy for live cue.
            try? await Task.sleep(for: .milliseconds(380))
            guard !Task.isCancelled, isSessionActive else { return }
            scheduleTurnAnalysis()
        }
    }

    func scheduleTurnAnalysis(
        requestFollowups: Bool = false,
        isFollowupRound: Bool = false,
        overrideUtterances: [UtteranceSegmentDTO]? = nil
    ) {
        let utts = overrideUtterances ?? audioEngine.utterances
        guard !utts.isEmpty else { return }

        turnSequence += 1
        let turnID = "turn_\(turnSequence)"
        // Prefer speaker-attributed usable frames; strip bulky JPEG if any slipped in.
        let framesSnapshot = Array(
            collectedFrames
                .suffix(96)
                .filter { $0.face_visible }
                .suffix(48)
                .map { frame in
                    var f = frame
                    f.jpeg_base64 = nil
                    return f
                }
        )
        let fpsSnapshot = effectiveFPS > 0 ? effectiveFPS : 24.0
        let routeSnapshot = audioEngine.currentRoute
        let sourceSnapshot = captureSource
        let sidSnapshot = sessionID

        let payload = SessionAnalyzeRequestDTO(
            session_id: sidSnapshot,
            turn_id: turnID,
            capture_source: sourceSnapshot,
            audio_route: routeSnapshot,
            modality_mode: .audioPlusVisibleFace,
            is_replay: false,
            is_evaluation: false,
            request_followup_questions: requestFollowups,
            is_followup_round: isFollowupRound,
            effective_fps: fpsSnapshot,
            utterances: utts,
            frames: framesSnapshot
        )

        pendingPayload = payload
        guard analysisTask == nil else { return }
        isAnalyzing = true
        analysisTask = Task { [weak self] in
            guard let self else { return }
            while let next = pendingPayload {
                pendingPayload = nil
                do {
                    let response = try await apiClient.analyzeTurn(next)
                    guard !Task.isCancelled, sessionID == next.session_id else { return }
                    applyAnalysisResponse(response)
                } catch {
                    guard !Task.isCancelled else { return }
                    recoveryBannerMessage = error.localizedDescription
                }
            }
            analysisTask = nil
            self.isAnalyzing = false
        }
    }

    private func applyAnalysisResponse(_ response: SessionAnalyzeResponseDTO) {
        latestResponse = response
        detectedActivity = response.activity
        statusLabel = response.status_label

        if response.retry_requested, let reason = response.retry_reason {
            recoveryBannerMessage = reason
        }

        guard isSessionActive, response.session_id == sessionID else { return }

        var cue = response.spoken_cue_text
        var key = response.result_id
        var banner: String? = nil
        if let guess = response.initial_guess, response.activity == .game {
            let conf = Int((guess.confidence * 100).rounded())
            banner = "Answer: Story #\(guess.chosen_story_number) is the lie (\(conf)%)."
            statusLabel = "Guess locked · Story #\(guess.chosen_story_number)"
            if response.updated_guess == nil, response.revealed_lie_story_number == nil {
                cue = guess.spoken_whisper
                if let story = response.stories.first(where: { $0.story_number == guess.chosen_story_number }) {
                    let opening = story.transcript.split(separator: " ").prefix(10).joined(separator: " ")
                    if !opening.isEmpty {
                        cue = guess.spoken_whisper + " The one starting: " + opening
                        banner = (banner ?? "") + " " + opening
                    }
                }
                key = "\(sessionID)/\(response.activity_epoch)/initial"
            }
        } else if let deal = response.deal_output, deal.action == "whisper_question", let whisper = deal.spoken_whisper {
            banner = "Deal cue: \(whisper)"
            cue = cue ?? whisper
        }

        if let banner, !banner.isEmpty {
            onScreenCueBanner = banner
        }
        guard let cue, !cue.isEmpty, deliveredResultIDs.insert(key).inserted else { return }
        if onScreenCueBanner == nil { onScreenCueBanner = cue }
        _ = audioEngine.speakCuePrivatelyIfPermitted(cue, forceSpeak: true)
    }

    // MARK: - Replay & Face-Masked Comparison

    func loadFixtures() async {
        for attempt in 1...2 {
            do {
                availableFixtures = try await apiClient.fetchFixtures()
                if selectedReplayFixture == nil {
                    selectedReplayFixture = availableFixtures.first(where: { $0.fixture_id == "example_live_lie_demo" })
                        ?? availableFixtures.first(where: { $0.fixture_id == "game_natural_lie2" })
                        ?? availableFixtures.first
                }
                if recoveryBannerMessage?.contains("Could not fetch fixtures") == true {
                    recoveryBannerMessage = nil
                }
                return
            } catch {
                if attempt == 1 {
                    try? await Task.sleep(for: .milliseconds(900))
                } else {
                    recoveryBannerMessage = "Backend unreachable — check Setup > Server Backend Endpoint."
                }
            }
        }
    }

    func runFixtureReplay(_ fixture: ReplayFixtureItemDTO, includeFollowup: Bool = false) async {
        if isSessionActive {
            stopSession()
        }
        selectedReplayFixture = fixture
        captureSource = .replay
        isAnalyzing = true
        maskedComparisonResponse = nil
        onScreenCueBanner = nil
        recoveryBannerMessage = nil
        activeReplayVideoURL = await apiClient.mediaURL(for: fixture.fixture_id, masked: false)

        do {
            let resp = try await apiClient.replayFixture(
                fixtureID: fixture.fixture_id,
                faceMasked: false,
                includeFollowup: includeFollowup
            )
            latestResponse = resp
            detectedActivity = resp.activity
            statusLabel = "Replay · \(resp.status_label)"
        } catch {
            recoveryBannerMessage = "Replay failed: \(error.localizedDescription)"
        }
        isAnalyzing = false
    }

    func runFaceMaskedRerunForSelectedFixture() async {
        guard let fixture = selectedReplayFixture else { return }
        isAnalyzing = true
        activeReplayVideoURL = await apiClient.mediaURL(for: fixture.fixture_id, masked: true)
        do {
            let maskedResp = try await apiClient.replayFixture(
                fixtureID: fixture.fixture_id,
                faceMasked: true,
                includeFollowup: false
            )
            maskedComparisonResponse = maskedResp
            statusLabel = "Face-Masked Rerun Complete (Audio Preserved)"
        } catch {
            recoveryBannerMessage = "Face-masked rerun failed: \(error.localizedDescription)"
        }
        isAnalyzing = false
    }

    func submitManualReveal(_ lieStoryNumber: Int, correctedStories: [StorySpanDTO]? = nil) async {
        let targetSessionID = latestResponse?.session_id ?? sessionID
        do {
            try await apiClient.sendManualCorrection(
                ManualCorrectionRequestDTO(
                    session_id: targetSessionID,
                    corrected_stories: correctedStories,
                    revealed_lie_story_number: lieStoryNumber
                )
            )
            if var current = latestResponse {
                current.revealed_lie_story_number = lieStoryNumber
                if let correctedStories {
                    current.stories = correctedStories
                }
                latestResponse = current
            }
        } catch {
            recoveryBannerMessage = error.localizedDescription
        }
    }

    func updateBackendEndpoint(_ urlString: String) async {
        await apiClient.updateBaseURL(urlString)
        configuredBackendURL = await apiClient.baseURL
        audioEngine.configureBackendURL(configuredBackendURL)
        await loadFixtures()
    }
}
