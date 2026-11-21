import AVFoundation
import Foundation
import Speech

// Apple's input tap runs on an audio thread. Prefer one mic source at a time
// (glasses HFP) so DAT PCM and the phone mic never fight for recognition.
private final class SpeechInput: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?

    func replace(_ next: SFSpeechAudioBufferRecognitionRequest?) {
        lock.lock()
        defer { lock.unlock() }
        request = next
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        request?.append(buffer)
    }
}

@MainActor
final class AudioAndSpeechEngine: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var currentRoute: AudioRouteKind = .speakerPublic
    @Published private(set) var isMicrophoneFlowing: Bool = false
    @Published private(set) var livePartialTranscript: String = ""
    @Published private(set) var utterances: [UtteranceSegmentDTO] = []
    @Published private(set) var recordedAudioURL: URL?

    private var audioEngine = AVAudioEngine()
    private let speechSynthesizer = AVSpeechSynthesizer()
    private var speechRecognizer: SFSpeechRecognizer? = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var routeObserver: NSObjectProtocol?
    private var interruptionObserver: NSObjectProtocol?

    private var sessionStartClock: TimeInterval = 0
    private let speechInput = SpeechInput()
    private var completedUtterances: [UtteranceSegmentDTO] = []
    private var offset: Double = 0
    private var generation = UUID()
    private var tapInstalled = false
    private var retryTask: Task<Void, Never>?
    private var converter: AVAudioConverter?
    private var targetASRFormat: AVAudioFormat?
    @Published private(set) var inputName = ""
    @Published private(set) var recognitionIssue: String?
    @Published private(set) var asrProviderLabel: String = "Apple + ElevenLabs"
    @Published private(set) var micSourceLabel: String = "iPhone mic"
    /// Human-readable live mic diagnostics for the UI (intended vs actual route).
    @Published private(set) var micStatusLine: String = "Mic: idle"
    @Published private(set) var micLevel: Double = 0

    private var pairedGlassesHint: Bool = false
    private var usingGlassesMic: Bool = false
    private var onFirstAudioDelivered: (() -> Void)?
    private var onUtteranceUpdated: (() -> Void)?
    private let elevenASR = ElevenLabsASRClient()
    private var backendBaseURL: String = "https://sherlocked-backend.onrender.com"
    private let pcmLock = NSLock()
    private var pcmScratch = Data()
    private var pcmChunkStartRel: Double = 0
    private var pcmHasSpeechEnergy = false
    private var elevenFlushLoopTask: Task<Void, Never>?
    private var elevenTranscribeTask: Task<Void, Never>?
    private var committedUtteranceCount = 0
    private var datPCMFlowing = false
    /// Soft gain applied before ElevenLabs (HFP/glasses are quiet & narrowband).
    private var elevenLabsGain: Float = 1.6
    /// Audio-tap thread reads these without hopping to MainActor.
    private let tapFlags = TapFlags()
    private var lastSpeechEnergyAt: TimeInterval = 0
    private var silenceWatchTask: Task<Void, Never>?
    private var silenceFinalized = false
    private var latestPartialFormatted: String = ""
    /// Fired when the speaker goes quiet so the coordinator can analyze immediately.
    var onSpeechPaused: (() -> Void)?

    private final class TapFlags: @unchecked Sendable {
        private let lock = NSLock()
        private var _skipHFP = false
        var skipHFPTap: Bool {
            get { lock.lock(); defer { lock.unlock() }; return _skipHFP }
            set { lock.lock(); _skipHFP = newValue; lock.unlock() }
        }
    }

    override init() {
        super.init()
        speechSynthesizer.delegate = self
        speechSynthesizer.usesApplicationAudioSession = true
        // Front person holds the phone → default to iPhone mic. Glasses mic is optional.
        if UserDefaults.standard.object(forKey: "sherlocked_mic_source_v3") == nil {
            MicCapturePreference.preferPhoneMic = true
            UserDefaults.standard.set(true, forKey: "sherlocked_mic_source_v3")
        }
        usingGlassesMic = false
        publishMicStatus()
        refreshAudioRoute(hasPairedGlasses: false)
        observeRouteAndInterruptions()
    }

    func configureBackendURL(_ url: String) {
        let trimmed = url.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if !trimmed.isEmpty {
            backendBaseURL = trimmed
            elevenASR.configure(backendBaseURL: trimmed)
        }
    }

    private func appendElevenCommitted(text: String, words: [ElevenLabsASRClient.Word]) {
        let fixed = Self.normalizeFrontSpeakerASR(text).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !fixed.isEmpty else { return }
        let nowRel = max(0, CACurrentMediaTime() - sessionStartClock)
        let mappedWords: [WordTimestampDTO] = words.map { w in
            WordTimestampDTO(
                word: w.word,
                start_sec: (w.start * 100).rounded() / 100,
                end_sec: (w.end * 100).rounded() / 100,
                confidence: w.confidence
            )
        }
        let start = mappedWords.first?.start_sec ?? max(0, (utterances.last?.end_sec ?? 0) + 0.05)
        let end = mappedWords.last?.end_sec ?? max(start + 0.3, nowRel)
        let pause = max(0, start - (utterances.last?.end_sec ?? 0))
        committedUtteranceCount += 1
        var next = utterances
        // Prefer ElevenLabs text for the latest overlapping clause when Apple already drafted one.
        if let last = next.last, abs(last.start_sec - start) < 1.2 {
            next[next.count - 1] = UtteranceSegmentDTO(
                segment_id: last.segment_id,
                text: fixed,
                start_sec: min(last.start_sec, start),
                end_sec: max(last.end_sec, end),
                pause_before_sec: last.pause_before_sec,
                speech_rate_wps: Double(max(1, mappedWords.count)) / max(0.2, end - start),
                words: mappedWords.isEmpty ? last.words : mappedWords
            )
        } else {
            next.append(
                UtteranceSegmentDTO(
                    segment_id: String(format: "utt_%03d", committedUtteranceCount),
                    text: fixed,
                    start_sec: (start * 100).rounded() / 100,
                    end_sec: (end * 100).rounded() / 100,
                    pause_before_sec: (pause * 100).rounded() / 100,
                    speech_rate_wps: Double(max(1, mappedWords.count)) / max(0.2, end - start),
                    words: mappedWords
                )
            )
        }
        utterances = next
        livePartialTranscript = next.map(\.text).joined(separator: " ")
        asrProviderLabel = "ElevenLabs + Apple"
        onUtteranceUpdated?()
    }

    private func enqueuePCM16ForElevenLabs(_ pcm: Data) {
        let boosted = Self.amplifyPCM16(pcm, gain: elevenLabsGain)
        let rel = max(0, CACurrentMediaTime() - sessionStartClock)
        // Glasses/HFP is quieter — accept a lower peak so chunks actually flush.
        let peakThreshold: Int16 = usingGlassesMic ? 500 : 900
        // Longer chunks for glasses (better Scribe accuracy on narrowband).
        let flushBytes = usingGlassesMic ? 64_000 : 44_800 // ~2.0s / ~1.4s @16k
        var shouldFlush = false
        pcmLock.lock()
        if pcmScratch.isEmpty {
            pcmChunkStartRel = rel
        }
        if !pcmHasSpeechEnergy {
            boosted.withUnsafeBytes { raw in
                guard let samples = raw.bindMemory(to: Int16.self).baseAddress else { return }
                let n = boosted.count / 2
                var peak: Int16 = 0
                for i in stride(from: 0, to: n, by: 8) {
                    let v = samples[i]
                    let a = v == Int16.min ? Int16.max : abs(v)
                    if a > peak { peak = a }
                }
                if peak > peakThreshold { pcmHasSpeechEnergy = true }
            }
        }
        pcmScratch.append(boosted)
        if pcmScratch.count >= flushBytes {
            shouldFlush = true
        }
        pcmLock.unlock()
        if shouldFlush {
            flushElevenLabsChunk()
        }
    }

    nonisolated private static func amplifyPCM16(_ pcm: Data, gain: Float) -> Data {
        guard gain > 1.01, !pcm.isEmpty else { return pcm }
        var out = Data(count: pcm.count)
        out.withUnsafeMutableBytes { dstRaw in
            pcm.withUnsafeBytes { srcRaw in
                guard let src = srcRaw.bindMemory(to: Int16.self).baseAddress,
                      let dst = dstRaw.bindMemory(to: Int16.self).baseAddress else { return }
                let n = pcm.count / 2
                for i in 0..<n {
                    let v = Float(src[i]) * gain
                    let clipped = max(-32768.0, min(32767.0, v))
                    dst[i] = Int16(clipped)
                }
            }
        }
        return out
    }

    private func flushElevenLabsChunk() {
        pcmLock.lock()
        let chunk = pcmScratch
        let startRel = pcmChunkStartRel
        let hadSpeech = pcmHasSpeechEnergy
        pcmScratch.removeAll(keepingCapacity: true)
        pcmHasSpeechEnergy = false
        pcmLock.unlock()
        guard hadSpeech, chunk.count >= 8_000 else { return } // >=0.25s
        // Don't cancel the flush *loop* — only replace the in-flight HTTP job.
        if elevenTranscribeTask != nil { return }
        let wav = ElevenLabsASRClient.wavDataFromPCM16(chunk)
        let startIndex = max(1, utterances.count + 1)
        elevenTranscribeTask = Task { [weak self] in
            defer {
                Task { @MainActor [weak self] in
                    self?.elevenTranscribeTask = nil
                }
            }
            guard let self else { return }
            do {
                let result = try await self.elevenASR.transcribeWAV(
                    wav,
                    timeOffsetSec: startRel,
                    utteranceStartIndex: startIndex
                )
                guard !Task.isCancelled else { return }
                if !result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.appendElevenCommitted(text: result.text, words: result.words)
                    self.recognitionIssue = nil
                }
            } catch {
                // Keep Apple live path running; soft-warn only.
                self.recognitionIssue = nil
            }
        }
    }

    func refreshAudioRoute(hasPairedGlasses: Bool? = nil) {
        if let hasPairedGlasses {
            pairedGlassesHint = hasPairedGlasses
        }
        let session = AVAudioSession.sharedInstance()
        inputName = session.currentRoute.inputs.first?.portName ?? ""
        let outputs = session.currentRoute.outputs
        if outputs.isEmpty {
            currentRoute = .none
            return
        }
        for port in outputs {
            switch port.portType {
            case .bluetoothHFP, .bluetoothA2DP, .bluetoothLE:
                let name = port.portName
                if pairedGlassesHint
                    || name.localizedCaseInsensitiveContains("Ray-Ban")
                    || name.localizedCaseInsensitiveContains("RB Meta")
                    || name.localizedCaseInsensitiveContains("Meta") {
                    currentRoute = .glassesPrivate
                } else {
                    currentRoute = .earbudsPrivate
                }
                return
            case .headphones, .usbAudio:
                currentRoute = .earbudsPrivate
                return
            default:
                continue
            }
        }
        // Glasses may still be linked even if iOS briefly reports speaker output.
        if pairedGlassesHint {
            currentRoute = .glassesPrivate
            return
        }
        currentRoute = .speakerPublic
    }

    private func observeRouteAndInterruptions() {
        routeObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshAudioRoute()
            }
        }
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            let typeVal = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            Task { @MainActor [weak self] in
                guard let self, let typeVal, let type = AVAudioSession.InterruptionType(rawValue: typeVal) else { return }
                if type == .began {
                    self.isMicrophoneFlowing = false
                }
            }
        }
    }

    func requestPermissions() async -> (mic: Bool, speech: Bool) {
        #if targetEnvironment(simulator)
        return (true, true)
        #else
        let micGranted: Bool
        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            micGranted = true
        case .undetermined:
            micGranted = await AVAudioApplication.requestRecordPermission()
        case .denied:
            micGranted = false
        @unknown default:
            micGranted = false
        }

        let currentSpeech = SFSpeechRecognizer.authorizationStatus()
        let speechAuth: SFSpeechRecognizerAuthorizationStatus
        if currentSpeech == .notDetermined {
            speechAuth = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status)
                }
            }
        } else {
            speechAuth = currentSpeech
        }
        return (micGranted, speechAuth == .authorized)
        #endif
    }

    func startLiveCapture(
        sessionID: String,
        sessionStartMonotonic: TimeInterval,
        hasPairedGlasses: Bool,
        onFirstAudioBuffer: @escaping () -> Void,
        onUtteranceCommitted: @escaping () -> Void
    ) throws {
        stopCapture()

        sessionStartClock = sessionStartMonotonic
        completedUtterances = []
        recognitionIssue = nil
        utterances.removeAll()
        livePartialTranscript = ""
        onFirstAudioDelivered = onFirstAudioBuffer
        onUtteranceUpdated = onUtteranceCommitted
        silenceFinalized = false
        lastSpeechEnergyAt = CACurrentMediaTime()
        converter = nil
        targetASRFormat = nil

        let audioSession = AVAudioSession.sharedInstance()
        #if targetEnvironment(simulator)
        try? audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        refreshAudioRoute(hasPairedGlasses: hasPairedGlasses)
        return
        #else
        // Default: iPhone mic (front person holds the phone). Glasses mic is opt-in.
        applyMicPreferenceLocked(hasPairedGlasses: hasPairedGlasses)
        elevenASR.configure(backendBaseURL: backendBaseURL)
        datPCMFlowing = false
        elevenLabsGain = usingGlassesMic ? 2.4 : 1.8
        try configureCaptureSession(hasPairedGlasses: hasPairedGlasses)

        let freshEngine = AVAudioEngine()
        self.audioEngine = freshEngine
        let inputNode = freshEngine.inputNode
        // Prefer native node format; if route just switched, inputFormat can briefly be 0 —
        // installTap(format: nil) still works and we adapt from the first buffer.
        var hwFormat = inputNode.inputFormat(forBus: 0)
        if hwFormat.sampleRate < 1 || hwFormat.channelCount < 1 {
            // One more activate+prefer cycle, then fall back to a sane stereo/mono float format.
            try? configureCaptureSession(hasPairedGlasses: hasPairedGlasses)
            hwFormat = inputNode.inputFormat(forBus: 0)
        }
        if hwFormat.sampleRate < 1 || hwFormat.channelCount < 1 {
            hwFormat = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)
                ?? AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48_000, channels: 1, interleaved: false)!
        }
        guard speechRecognizer?.isAvailable == true else {
            throw NSError(domain: "SherlockedSpeech", code: 1, userInfo: [NSLocalizedDescriptionKey: "English speech recognition is unavailable."])
        }

        let elevenFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)
            ?? hwFormat
        targetASRFormat = elevenFormat
        converter = nil // built lazily from the first live buffer (handles route flips)

        asrProviderLabel = usingGlassesMic ? "Glasses · Apple + ElevenLabs" : "Phone · Apple + ElevenLabs"
        committedUtteranceCount = 0
        isMicrophoneFlowing = false
        pcmLock.lock()
        pcmScratch.removeAll(keepingCapacity: true)
        pcmHasSpeechEnergy = false
        pcmLock.unlock()

        beginRecognition()

        let speechInput = self.speechInput
        tapFlags.skipHFPTap = false
        // Always nil format so the tap tracks live hardware after Bluetooth route flips.
        Self.installDualASRAudioTap(
            on: inputNode,
            preferredFormat: nil,
            elevenFormat: elevenFormat,
            converterBox: ConverterBox(),
            input: speechInput,
            tapFlags: tapFlags
        ) { [weak self] pcm16, level in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.micLevel = max(self.micLevel * 0.6, level)
                if level > 0.045 {
                    self.lastSpeechEnergyAt = CACurrentMediaTime()
                    self.silenceFinalized = false
                }
                if !self.isMicrophoneFlowing {
                    self.isMicrophoneFlowing = true
                    self.onFirstAudioDelivered?()
                    self.publishMicStatus()
                }
                self.enqueuePCM16ForElevenLabs(pcm16)
                self.armSilenceWatch()
            }
        }

        tapInstalled = true
        freshEngine.prepare()
        try freshEngine.start()
        applyPreferredMicInput(session: AVAudioSession.sharedInstance())
        publishMicStatus()

        elevenFlushLoopTask?.cancel()
        elevenFlushLoopTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(1200))
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self?.reassertPreferredMicIfNeeded()
                    self?.flushElevenLabsChunk()
                }
            }
        }
        #endif
    }

    private func configureCaptureSession(hasPairedGlasses: Bool) throws {
        let audioSession = AVAudioSession.sharedInstance()
        // Phone mic: NEVER enable HFP — it steals input to glasses and kills front-speaker ASR.
        // allowBluetoothA2DP keeps glasses/buds as output-only when needed.
        if usingGlassesMic {
            try audioSession.setCategory(
                .playAndRecord,
                mode: .voiceChat,
                options: [.allowBluetoothHFP, .mixWithOthers]
            )
        } else {
            try audioSession.setCategory(
                .playAndRecord,
                mode: .default,
                options: [.defaultToSpeaker, .allowBluetoothA2DP, .mixWithOthers]
            )
        }
        try? audioSession.setPreferredSampleRate(48_000)
        try? audioSession.setPreferredIOBufferDuration(0.02)
        try audioSession.setActive(true, options: [.notifyOthersOnDeactivation])
        applyPreferredMicInput(session: audioSession)
        if !usingGlassesMic {
            try? audioSession.overrideOutputAudioPort(.speaker)
        }
        refreshAudioRoute(hasPairedGlasses: hasPairedGlasses)
        publishMicStatus()
    }

    /// Mutable converter holder so the audio tap can create it from the first real buffer.
    private final class ConverterBox: @unchecked Sendable {
        let lock = NSLock()
        var converter: AVAudioConverter?
        var sourceRate: Double = 0
    }

    /// Call after glasses camera starts — Meta session can steal the audio route.
    func reassertPreferredMicIfNeeded() {
        let session = AVAudioSession.sharedInstance()
        let inputType = session.currentRoute.inputs.first?.portType
        let stuckOnHFP = !usingGlassesMic && (inputType == .bluetoothHFP)
        if stuckOnHFP {
            // Meta/DAT stole the route — full category reset without HFP.
            try? configureCaptureSession(hasPairedGlasses: pairedGlassesHint)
        } else {
            applyPreferredMicInput(session: session)
            if !usingGlassesMic {
                try? session.overrideOutputAudioPort(.speaker)
            }
            publishMicStatus()
        }
    }

    /// Live toggle: switch phone ↔ glasses mic and update the status line immediately.
    func applyMicPreference(preferPhoneMic: Bool, hasPairedGlasses: Bool) {
        MicCapturePreference.preferPhoneMic = preferPhoneMic
        applyMicPreferenceLocked(hasPairedGlasses: hasPairedGlasses)
        elevenLabsGain = usingGlassesMic ? 2.4 : 1.8
        do {
            try configureCaptureSession(hasPairedGlasses: hasPairedGlasses)
        } catch {
            print("[Sherlocked] mic preference apply failed: \(error.localizedDescription)")
        }
    }

    private func applyMicPreferenceLocked(hasPairedGlasses: Bool) {
        let forcePhoneMic = MicCapturePreference.preferPhoneMic
        usingGlassesMic = hasPairedGlasses && !forcePhoneMic
        micSourceLabel = usingGlassesMic ? "Glasses mic" : "iPhone mic"
    }

    var prefersGlassesMic: Bool { usingGlassesMic }

    private func applyPreferredMicInput(session: AVAudioSession) {
        if usingGlassesMic {
            if let hfp = session.availableInputs?.first(where: {
                $0.portType == .bluetoothHFP
                    || $0.portName.localizedCaseInsensitiveContains("Ray-Ban")
                    || $0.portName.localizedCaseInsensitiveContains("RB Meta")
                    || $0.portName.localizedCaseInsensitiveContains("Meta")
            }) {
                try? session.setPreferredInput(hfp)
            }
        } else if let builtIn = session.availableInputs?.first(where: { $0.portType == .builtInMic }) {
            try? session.setPreferredInput(builtIn)
            if let bottom = builtIn.dataSources?.first(where: {
                $0.orientation == .bottom || $0.orientation == .front
            }) ?? builtIn.dataSources?.first {
                try? builtIn.setPreferredDataSource(bottom)
            }
        }
        inputName = session.currentRoute.inputs.first.map { "\($0.portName) (\($0.portType.rawValue))" } ?? "none"
    }

    private func publishMicStatus() {
        let intended = usingGlassesMic ? "Glasses mic" : "iPhone mic"
        micSourceLabel = intended
        let actual: String
        if usingGlassesMic && datPCMFlowing {
            actual = "DAT 16k"
        } else {
            actual = inputName.isEmpty ? "unknown" : inputName
        }
        let flowing = isMicrophoneFlowing ? "live" : "waiting"
        micStatusLine = "Mic: \(intended) → \(actual) · \(flowing) · lvl \(String(format: "%.0f", micLevel * 100))%"
    }

    private func beginRecognition() {
        guard let recognizer = speechRecognizer else { return }
        let id = UUID()
        generation = id
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        // Cloud ASR is much better for front-speaker + glasses HFP; on-device struggles with both.
        if #available(iOS 16.0, *) {
            request.requiresOnDeviceRecognition = false
        }
        request.taskHint = .dictation
        request.contextualStrings = [
            "two truths and a lie",
            "two truths and one lie",
            "let's play two truths and a lie",
            "story one", "story two", "story three",
            "number one", "number two", "number three",
            "first", "second", "third", "firstly", "secondly", "thirdly",
            "the lie was story", "the lie is number",
            "follow-up questions", "ask me questions",
            "I once", "last summer", "in March", "in October",
        ]
        completedUtterances = utterances
        offset = max(0, CACurrentMediaTime() - sessionStartClock)
        recognitionRequest = request
        speechInput.replace(request)
        recognitionTask = Self.startNonisolatedRecognitionTask(recognizer: recognizer, request: request) { [weak self] snapshot, error in
            Task { @MainActor [weak self] in
                guard let self, self.generation == id else { return }
                if let snapshot {
                    self.recognitionIssue = nil
                    self.consumeSpeechSnapshot(snapshot)
                }
                if snapshot?.isFinal == true || error != nil {
                    self.recognitionIssue = error.map { "Speech recognition is reconnecting. " + $0 }
                    self.generation = UUID()
                    self.retryTask = Task { @MainActor in
                        if error != nil { try? await Task.sleep(for: .milliseconds(400)) }
                        guard !Task.isCancelled else { return }
                        self.beginRecognition()
                    }
                }
            }
        }
    }

    struct SpeechSegmentItem: Sendable {
        let substring: String
        let timestamp: Double
        let duration: Double
        let confidence: Float
    }

    struct SpeechSnapshot: Sendable {
        let formattedString: String
        let isFinal: Bool
        let segments: [SpeechSegmentItem]
    }

    nonisolated private static func startNonisolatedRecognitionTask(
        recognizer: SFSpeechRecognizer,
        request: SFSpeechAudioBufferRecognitionRequest,
        onResult: @escaping @Sendable (SpeechSnapshot?, String?) -> Void
    ) -> SFSpeechRecognitionTask {
        recognizer.recognitionTask(with: request) { result, error in
            guard let result else { onResult(nil, error?.localizedDescription); return }
            let items = result.bestTranscription.segments.map {
                SpeechSegmentItem(
                    substring: $0.substring,
                    timestamp: $0.timestamp,
                    duration: $0.duration,
                    confidence: $0.confidence
                )
            }
            onResult(
                SpeechSnapshot(
                    formattedString: result.bestTranscription.formattedString,
                    isFinal: result.isFinal,
                    segments: items
                ), error?.localizedDescription
            )
        }
    }

    nonisolated private static func installDualASRAudioTap(
        on inputNode: AVAudioInputNode,
        preferredFormat: AVAudioFormat?,
        elevenFormat: AVAudioFormat,
        converterBox: ConverterBox,
        input: SpeechInput,
        tapFlags: TapFlags,
        onPCM16: @escaping @Sendable (Data, Double) -> Void
    ) {
        // nil format = node's native format (survives route changes better than a stale hwFormat).
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: preferredFormat) { buffer, _ in
            guard buffer.frameLength > 0 else { return }
            if tapFlags.skipHFPTap { return }

            // Apple Speech MUST get the native hardware buffers (no resample).
            input.append(buffer)

            var level = 0.0
            if let channels = buffer.floatChannelData {
                let frames = Int(buffer.frameLength)
                var sum = 0.0
                let step = max(1, frames / 64)
                for i in stride(from: 0, to: frames, by: step) {
                    sum += abs(Double(channels[0][i]))
                }
                level = min(1.0, (sum / Double(max(1, frames / step))) * 5.0)
            }

            let srcFormat = buffer.format
            let elevenBuf: AVAudioPCMBuffer
            if srcFormat.sampleRate != elevenFormat.sampleRate || srcFormat.channelCount != elevenFormat.channelCount {
                converterBox.lock.lock()
                if converterBox.converter == nil || abs(converterBox.sourceRate - srcFormat.sampleRate) > 0.5 {
                    converterBox.converter = AVAudioConverter(from: srcFormat, to: elevenFormat)
                    converterBox.sourceRate = srcFormat.sampleRate
                }
                let converter = converterBox.converter
                converterBox.lock.unlock()
                guard let converter else { return }
                let ratio = elevenFormat.sampleRate / max(1.0, srcFormat.sampleRate)
                let outFrames = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
                guard let out = AVAudioPCMBuffer(pcmFormat: elevenFormat, frameCapacity: outFrames) else { return }
                var error: NSError?
                var consumed = false
                converter.convert(to: out, error: &error) { _, status in
                    if consumed {
                        status.pointee = .noDataNow
                        return nil
                    }
                    consumed = true
                    status.pointee = .haveData
                    return buffer
                }
                guard error == nil, out.frameLength > 0 else { return }
                elevenBuf = out
            } else {
                elevenBuf = buffer
            }
            if let pcm = Self.floatBufferToPCM16Data(elevenBuf) {
                onPCM16(pcm, level)
            }
        }
    }

    nonisolated private static func floatBufferToPCM16Data(_ buffer: AVAudioPCMBuffer) -> Data? {
        guard let channels = buffer.floatChannelData else { return nil }
        let frames = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        var data = Data(count: frames * 2)
        data.withUnsafeMutableBytes { raw in
            guard let dst = raw.bindMemory(to: Int16.self).baseAddress else { return }
            for i in 0..<frames {
                var sample: Float = 0
                for ch in 0..<channelCount {
                    sample += channels[ch][i]
                }
                if channelCount > 1 { sample /= Float(channelCount) }
                let clipped = max(-1.0, min(1.0, sample))
                dst[i] = Int16(clipped * 32767.0)
            }
        }
        return data
    }

    nonisolated private static func pcm16DataToFloatBuffer(_ data: Data, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let frames = data.count / 2
        guard frames > 0,
              let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)),
              let channels = buf.floatChannelData else { return nil }
        buf.frameLength = AVAudioFrameCount(frames)
        data.withUnsafeBytes { raw in
            guard let src = raw.bindMemory(to: Int16.self).baseAddress else { return }
            for i in 0..<frames {
                channels[0][i] = Float(src[i]) / 32768.0
            }
        }
        return buf
    }

    /// DAT glasses mic (16 kHz PCM) — much better than HFP for front-speaker ASR.
    func consumeExternalGlassesPCM(_ buffer: AVAudioPCMBuffer) {
        guard usingGlassesMic, recognitionRequest != nil else { return }
        datPCMFlowing = true
        tapFlags.skipHFPTap = true // stop double-feeding lossy HFP once DAT is alive

        // Feed native DAT buffers to Apple Speech (already float PCM).
        speechInput.append(buffer)
        let floatBuf: AVAudioPCMBuffer
        if let asrFormat = targetASRFormat,
           (buffer.format.sampleRate != asrFormat.sampleRate || buffer.format.channelCount != asrFormat.channelCount),
           let converted = Self.convertBuffer(buffer, to: asrFormat) {
            floatBuf = converted
        } else {
            floatBuf = buffer
        }
        if let pcm = Self.floatBufferToPCM16Data(floatBuf) {
            enqueuePCM16ForElevenLabs(pcm)
        }
        var level = 0.05
        if let channels = buffer.floatChannelData {
            let frames = Int(buffer.frameLength)
            var sum = 0.0
            let step = max(1, frames / 32)
            for i in stride(from: 0, to: frames, by: step) {
                sum += abs(Double(channels[0][i]))
            }
            level = min(1.0, (sum / Double(max(1, frames / step))) * 5.0)
        }
        micLevel = max(micLevel * 0.5, level)
        if level > 0.045 {
            lastSpeechEnergyAt = CACurrentMediaTime()
            silenceFinalized = false
        }
        armSilenceWatch()
        if !isMicrophoneFlowing {
            isMicrophoneFlowing = true
            onFirstAudioDelivered?()
            publishMicStatus()
        }
    }

    nonisolated private static func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) -> AVAudioPCMBuffer? {
        guard let converter = AVAudioConverter(from: buffer.format, to: format) else { return nil }
        let ratio = format.sampleRate / buffer.format.sampleRate
        let outFrames = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: outFrames) else { return nil }
        var error: NSError?
        var consumed = false
        converter.convert(to: out, error: &error) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        return (error == nil && out.frameLength > 0) ? out : nil
    }

    /// Soft-corrects common glasses-mic ASR misses for the two-truths invitation and ordinals.
    private static func normalizeFrontSpeakerASR(_ text: String) -> String {
        var out = text
        let swaps: [(String, String)] = [
            (#"(?i)\bto truths\b"#, "two truths"),
            (#"(?i)\btoo truths\b"#, "two truths"),
            (#"(?i)\btwo truth\b"#, "two truths"),
            (#"(?i)\btrue truths\b"#, "two truths"),
            (#"(?i)\bto truth\b"#, "two truths"),
            (#"(?i)\band alive\b"#, "and a lie"),
            (#"(?i)\ban alive\b"#, "and a lie"),
            (#"(?i)\band the lie\b"#, "and a lie"),
            (#"(?i)\bin a lie\b"#, "and a lie"),
            (#"(?i)\band rely\b"#, "and a lie"),
            (#"(?i)\band a light\b"#, "and a lie"),
            (#"(?i)\band ally\b"#, "and a lie"),
            (#"(?i)\btwo truths and lie\b"#, "two truths and a lie"),
            (#"(?i)\blet'?s play to\b"#, "let's play two"),
            (#"(?i)\bstory to\b"#, "story two"),
            (#"(?i)\bnumber to\b"#, "number two"),
            (#"(?i)\bstory tree\b"#, "story three"),
            (#"(?i)\bnumber tree\b"#, "number three"),
        ]
        for (pattern, replacement) in swaps {
            if let regex = try? NSRegularExpression(pattern: pattern) {
                let range = NSRange(out.startIndex..<out.endIndex, in: out)
                out = regex.stringByReplacingMatches(in: out, range: range, withTemplate: replacement)
            }
        }
        return out
    }

    private func commitPlainTranscript(_ text: String) {
        let normalized = Self.normalizeFrontSpeakerASR(text)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        // Avoid duplicating the same committed clause.
        if let last = utterances.last, last.text == normalized { return }
        let now = max(0, CACurrentMediaTime() - sessionStartClock)
        let start = utterances.last?.end_sec ?? max(0, now - 1.0)
        var rebuilt = completedUtterances.isEmpty ? utterances : completedUtterances
        // If the last utterance is a growing prefix of this text, replace it.
        if let last = rebuilt.last, normalized.hasPrefix(last.text), normalized.count > last.text.count {
            rebuilt[rebuilt.count - 1] = UtteranceSegmentDTO(
                segment_id: last.segment_id,
                text: normalized,
                start_sec: last.start_sec,
                end_sec: (now * 100).rounded() / 100,
                pause_before_sec: last.pause_before_sec,
                speech_rate_wps: Double(normalized.split(separator: " ").count) / max(0.2, now - last.start_sec)
            )
        } else {
            rebuilt.append(
                UtteranceSegmentDTO(
                    segment_id: String(format: "utt_%03d", rebuilt.count + 1),
                    text: normalized,
                    start_sec: (start * 100).rounded() / 100,
                    end_sec: (now * 100).rounded() / 100,
                    pause_before_sec: 0.2,
                    speech_rate_wps: Double(normalized.split(separator: " ").count) / max(0.2, now - start)
                )
            )
        }
        utterances = rebuilt
        completedUtterances = rebuilt
        livePartialTranscript = rebuilt.map(\.text).joined(separator: " ")
        onUtteranceUpdated?()
    }

    private func armSilenceWatch() {
        silenceWatchTask?.cancel()
        silenceWatchTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(700))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                self?.handlePossibleSpeechPause()
            }
        }
    }

    private func handlePossibleSpeechPause() {
        let quietFor = CACurrentMediaTime() - lastSpeechEnergyAt
        guard quietFor >= 0.65, !silenceFinalized else { return }
        let hasContent = !livePartialTranscript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !utterances.isEmpty
        guard hasContent else { return }
        silenceFinalized = true
        // Snapshot the newest partial clause only (not the whole joined transcript).
        let newest = latestPartialFormatted.trimmingCharacters(in: .whitespacesAndNewlines)
        if !newest.isEmpty {
            commitPlainTranscript(newest)
        } else if !utterances.isEmpty {
            onUtteranceUpdated?()
        }
        flushElevenLabsChunk()
        recognitionRequest?.endAudio()
        onSpeechPaused?()
        // isFinal handler in beginRecognition restarts the task; clear the gate shortly after.
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(900))
            self.silenceFinalized = false
        }
    }

    func consumeSpeechSnapshot(_ result: SpeechSnapshot) {
        let normalizedFormatted = Self.normalizeFrontSpeakerASR(result.formattedString)
        // Always update the live UI text — even when word timestamps are missing.
        if !normalizedFormatted.isEmpty {
            latestPartialFormatted = normalizedFormatted
            livePartialTranscript = (completedUtterances.map(\.text) + [normalizedFormatted])
                .filter { !$0.isEmpty }
                .joined(separator: " ")
        }

        let segments = result.segments
        guard let first = segments.first else {
            // No timestamps — commit plain text on final OR when we already have enough words.
            if !normalizedFormatted.isEmpty, result.isFinal || normalizedFormatted.split(separator: " ").count >= 6 {
                commitPlainTranscript(normalizedFormatted)
            }
            return
        }
        if let last = utterances.last, first.timestamp + offset >= last.end_sec + 0.15 {
            completedUtterances = utterances
        }

        var currentWords: [WordTimestampDTO] = []
        var clauseStart: Double?
        var prevEnd = offset
        var rebuilt = completedUtterances

        for seg in segments {
            let wStart = offset + max(0.0, seg.timestamp)
            let wEnd = wStart + max(0.08, seg.duration)
            let gap = wStart - prevEnd

            // Split on short pauses so stories don't collapse into one utterance.
            if !currentWords.isEmpty && gap > 0.45 {
                let cText = Self.normalizeFrontSpeakerASR(currentWords.map(\.word).joined(separator: " "))
                let s0 = clauseStart ?? 0.0
                let s1 = currentWords.last?.end_sec ?? wStart
                let pauseBefore = max(0.0, s0 - (rebuilt.last?.end_sec ?? 0.0))
                rebuilt.append(
                    UtteranceSegmentDTO(
                        segment_id: String(format: "utt_%03d", rebuilt.count + 1),
                        text: cText,
                        start_sec: (s0 * 100).rounded() / 100,
                        end_sec: (s1 * 100).rounded() / 100,
                        pause_before_sec: (pauseBefore * 100).rounded() / 100,
                        speech_rate_wps: Double(currentWords.count) / max(0.2, s1 - s0),
                        words: currentWords
                    )
                )
                currentWords.removeAll()
                clauseStart = nil
            }

            if clauseStart == nil {
                clauseStart = wStart
            }
            currentWords.append(
                WordTimestampDTO(
                    word: seg.substring,
                    start_sec: (wStart * 100).rounded() / 100,
                    end_sec: (wEnd * 100).rounded() / 100,
                    confidence: Double(seg.confidence)
                )
            )
            prevEnd = wEnd
        }

        if !currentWords.isEmpty {
            let cText = Self.normalizeFrontSpeakerASR(currentWords.map(\.word).joined(separator: " "))
            let s0 = clauseStart ?? 0.0
            let s1 = currentWords.last?.end_sec ?? s0 + 0.3
            let pauseBefore = max(0.0, s0 - (rebuilt.last?.end_sec ?? 0.0))
            rebuilt.append(
                UtteranceSegmentDTO(
                    segment_id: String(format: "utt_%03d", rebuilt.count + 1),
                    text: cText,
                    start_sec: (s0 * 100).rounded() / 100,
                    end_sec: (s1 * 100).rounded() / 100,
                    pause_before_sec: (pauseBefore * 100).rounded() / 100,
                    speech_rate_wps: Double(currentWords.count) / max(0.2, s1 - s0),
                    words: currentWords
                )
            )
        }

        for i in rebuilt.indices {
            let fixed = Self.normalizeFrontSpeakerASR(rebuilt[i].text)
            if fixed != rebuilt[i].text {
                rebuilt[i].text = fixed
            }
        }

        if rebuilt != utterances || result.isFinal {
            utterances = rebuilt
            onUtteranceUpdated?()
        }
    }

    /// Speaks private cues into glasses when possible; still speaks on speaker as a fallback
    /// so the wearer always hears the locked guess.
    @discardableResult
    func speakCuePrivatelyIfPermitted(_ text: String, forceSpeak: Bool = false) -> Bool {
        refreshAudioRoute()
        let wantPrivate = pairedGlassesHint || currentRoute.isPrivate || usingGlassesMic
        guard forceSpeak || wantPrivate || currentRoute.isPrivate else {
            return false
        }

        prepareSessionForPlayback(preferGlasses: wantPrivate)

        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.48
        utterance.volume = 1.0
        utterance.pitchMultiplier = 1.0
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            ?? AVSpeechSynthesisVoice(language: "en-GB")
            ?? AVSpeechSynthesisVoice.speechVoices().first
        speechSynthesizer.speak(utterance)
        return true
    }

    private func prepareSessionForPlayback(preferGlasses: Bool) {
        let session = AVAudioSession.sharedInstance()
        do {
            // Enable HFP only for speak-back so glasses ear gets the cue.
            try session.setCategory(
                .playAndRecord,
                mode: preferGlasses ? .voiceChat : .default,
                options: preferGlasses
                    ? [.allowBluetoothHFP, .mixWithOthers]
                    : [.defaultToSpeaker, .mixWithOthers]
            )
            try session.setActive(true)
        } catch {
            print("[Sherlocked] playback session prep failed: \(error.localizedDescription)")
        }
        refreshAudioRoute()
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        let ports = AVAudioSession.sharedInstance().currentRoute.outputs.map { "\($0.portName):\($0.portType.rawValue)" }
        print("[Sherlocked] speech playback started on \(ports)")
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            // Restore phone-mic capture route after private HFP speak-back.
            guard let self, !self.usingGlassesMic else { return }
            let session = AVAudioSession.sharedInstance()
            try? session.setCategory(
                .playAndRecord,
                mode: .measurement,
                options: [.defaultToSpeaker, .mixWithOthers]
            )
            try? session.setActive(true)
            self.applyPreferredMicInput(session: session)
            self.publishMicStatus()
        }
    }

    func stopCapture() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        generation = UUID()
        retryTask?.cancel()
        elevenFlushLoopTask?.cancel()
        elevenFlushLoopTask = nil
        elevenTranscribeTask?.cancel()
        elevenTranscribeTask = nil
        silenceWatchTask?.cancel()
        silenceWatchTask = nil
        flushElevenLabsChunk()
        speechInput.replace(nil)
        datPCMFlowing = false
        tapFlags.skipHFPTap = false
        silenceFinalized = false
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        pcmLock.lock()
        pcmScratch.removeAll()
        pcmHasSpeechEnergy = false
        pcmLock.unlock()
        isMicrophoneFlowing = false
        converter = nil
        targetASRFormat = nil
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
    }
}
