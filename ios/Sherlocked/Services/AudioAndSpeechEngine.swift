import AVFoundation
import Foundation
import Speech

// Apple's input tap runs on an audio thread. Swap recognition requests without
// reinstalling the tap or mixing glasses PCM with the local microphone.
private final class SpeechInput: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var external = false

    func replace(_ next: SFSpeechAudioBufferRecognitionRequest?, external: Bool) {
        lock.lock()
        defer { lock.unlock() }
        request = next
        self.external = external
    }

    func append(_ buffer: AVAudioPCMBuffer, external: Bool) {
        lock.lock()
        defer { lock.unlock() }
        if self.external == external { request?.append(buffer) }
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
    private var external = false
    private var retryTask: Task<Void, Never>?
    @Published private(set) var inputName = ""
    @Published private(set) var recognitionIssue: String?

    private var pairedGlassesHint: Bool = false
    private var onFirstAudioDelivered: (() -> Void)?
    private var onUtteranceUpdated: (() -> Void)?

    override init() {
        super.init()
        speechSynthesizer.delegate = self
        speechSynthesizer.usesApplicationAudioSession = true
        refreshAudioRoute(hasPairedGlasses: false)
        observeRouteAndInterruptions()
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
        external = false
        utterances.removeAll()
        livePartialTranscript = ""
        onFirstAudioDelivered = onFirstAudioBuffer
        onUtteranceUpdated = onUtteranceCommitted

        let audioSession = AVAudioSession.sharedInstance()
        #if targetEnvironment(simulator)
        try? audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        refreshAudioRoute(hasPairedGlasses: hasPairedGlasses)
        return
        #else
        // Meta glasses HFP is the listen source whenever glasses are paired.
        // Phone mic is only an explicit override (or fallback with no glasses).
        // One-time migration: older builds defaulted phone mic ON; force glasses-first.
        if UserDefaults.standard.object(forKey: "sherlocked_glasses_mic_primary_v1") == nil {
            UserDefaults.standard.set(false, forKey: "sherlocked_phone_mic")
            UserDefaults.standard.set(true, forKey: "sherlocked_glasses_mic_primary_v1")
        }
        let forcePhoneMic = UserDefaults.standard.bool(forKey: "sherlocked_phone_mic")
        let useGlassesMic = hasPairedGlasses && !forcePhoneMic

        if useGlassesMic {
            // Verified on Ray-Ban Meta Gen 3: videoRecording + HFP avoids OSStatus -50 / -10868.
            try audioSession.setCategory(
                .playAndRecord,
                mode: .videoRecording,
                options: [.allowBluetoothHFP, .mixWithOthers, .defaultToSpeaker]
            )
        } else {
            // Phone mic near the speaker; Bluetooth A2DP/HFP still available for private cues.
            try audioSession.setCategory(
                .playAndRecord,
                mode: .default,
                options: [.allowBluetoothA2DP, .allowBluetoothHFP, .defaultToSpeaker]
            )
        }
        try audioSession.setActive(true)

        let preferredPort: AVAudioSession.Port = useGlassesMic ? .bluetoothHFP : .builtInMic
        if let preferredInput = audioSession.availableInputs?.first(where: { $0.portType == preferredPort }) {
            try audioSession.setPreferredInput(preferredInput)
        } else if useGlassesMic,
                  let anyHFP = audioSession.availableInputs?.first(where: {
                      $0.portType == .bluetoothHFP
                      || $0.portName.localizedCaseInsensitiveContains("Ray-Ban")
                      || $0.portName.localizedCaseInsensitiveContains("RB Meta")
                      || $0.portName.localizedCaseInsensitiveContains("Meta")
                  }) {
            try audioSession.setPreferredInput(anyHFP)
        }
        refreshAudioRoute(hasPairedGlasses: hasPairedGlasses)

        // Obtain the hardware format after selecting the recording route.
        let freshEngine = AVAudioEngine()
        self.audioEngine = freshEngine
        let inputNode = freshEngine.inputNode
        let hwFormat = inputNode.inputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0 && hwFormat.channelCount > 0 else {
            throw NSError(domain: "SherlockedAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "No microphone input is available."])
        }
        guard speechRecognizer?.isAvailable == true else {
            throw NSError(domain: "SherlockedSpeech", code: 1, userInfo: [NSLocalizedDescriptionKey: "English speech recognition is unavailable."])
        }
        beginRecognition()

        Self.installNonisolatedAudioTap(
            on: inputNode,
            format: hwFormat,
            input: speechInput
        ) { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if !self.isMicrophoneFlowing {
                    self.isMicrophoneFlowing = true
                    self.onFirstAudioDelivered?()
                }
            }
        }

        tapInstalled = true
        freshEngine.prepare()
        try freshEngine.start()
        #endif
    }

    private func beginRecognition() {
        guard let recognizer = speechRecognizer else { return }
        let id = UUID()
        generation = id
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        // Dictation + bias phrases help the glasses mic catch a front speaker at conversational distance.
        request.taskHint = .dictation
        request.contextualStrings = [
            "two truths and a lie",
            "two truths and one lie",
            "let's play two truths and a lie",
            "story one", "story two", "story three",
            "number one", "number two", "number three",
            "first", "second", "third",
            "the lie was story", "the lie is number",
            "follow-up questions", "ask me questions",
        ]
        completedUtterances = utterances
        offset = max(0, CACurrentMediaTime() - sessionStartClock)
        recognitionRequest = request
        speechInput.replace(request, external: external)
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
                        if error != nil { try? await Task.sleep(for: .milliseconds(280)) }
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

    nonisolated private static func installNonisolatedAudioTap(
        on inputNode: AVAudioInputNode,
        format: AVAudioFormat,
        input: SpeechInput,
        onSampleSeen: @escaping @Sendable () -> Void
    ) {
        nonisolated(unsafe) var emittedFirst = false

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            guard buffer.frameLength > 0 else { return }
            input.append(buffer, external: false)
            if !emittedFirst {
                emittedFirst = true
                onSampleSeen()
            }
        }
    }

    /// Consumes synchronized PCM buffers coming directly from Ray-Ban Meta glasses when enabled.
    func consumeExternalGlassesPCM(_ buffer: AVAudioPCMBuffer) {
        guard recognitionRequest != nil else { return }
        if !external {
            external = true
            generation = UUID()
            retryTask?.cancel()
            recognitionTask?.cancel()
            beginRecognition()
        }
        speechInput.append(buffer, external: true)
        if !isMicrophoneFlowing {
            isMicrophoneFlowing = true
            onFirstAudioDelivered?()
        }
    }

    /// Soft-corrects common glasses-mic ASR misses for the two-truths invitation.
    private static func normalizeFrontSpeakerASR(_ text: String) -> String {
        var out = text
        let swaps: [(String, String)] = [
            (#"(?i)\bto truths\b"#, "two truths"),
            (#"(?i)\btoo truths\b"#, "two truths"),
            (#"(?i)\btwo truth\b"#, "two truths"),
            (#"(?i)\btrue truths\b"#, "two truths"),
            (#"(?i)\band alive\b"#, "and a lie"),
            (#"(?i)\ban alive\b"#, "and a lie"),
            (#"(?i)\band the lie\b"#, "and a lie"),
            (#"(?i)\bin a lie\b"#, "and a lie"),
            (#"(?i)\band rely\b"#, "and a lie"),
            (#"(?i)\btwo truths and lie\b"#, "two truths and a lie"),
        ]
        for (pattern, replacement) in swaps {
            if let regex = try? NSRegularExpression(pattern: pattern) {
                let range = NSRange(out.startIndex..<out.endIndex, in: out)
                out = regex.stringByReplacingMatches(in: out, range: range, withTemplate: replacement)
            }
        }
        return out
    }

    func consumeSpeechSnapshot(_ result: SpeechSnapshot) {
        let segments = result.segments
        guard let first = segments.first else { return }
        // Some recognizers publish a new time slice after a pause in the same task.
        // Keep earlier, disjoint audio while allowing revisions to overlapping words.
        if let last = utterances.last, first.timestamp + offset >= last.end_sec + 0.15 {
            completedUtterances = utterances
        }
        let normalizedFormatted = Self.normalizeFrontSpeakerASR(result.formattedString)
        livePartialTranscript = (completedUtterances.map(\.text) + [normalizedFormatted]).joined(separator: " ")

        var currentWords: [WordTimestampDTO] = []
        var clauseStart: Double?
        var prevEnd = offset
        var rebuilt = completedUtterances

        for seg in segments {
            let wStart = offset + max(0.0, seg.timestamp)
            let wEnd = wStart + max(0.08, seg.duration)
            let gap = wStart - prevEnd

            // Slightly tighter clause split so front-speaker pauses still form story boundaries quickly.
            if !currentWords.isEmpty && gap > 0.55 {
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

        // Also normalize already-committed clause text (invitation often lands mid-stream).
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

    func speakCuePrivatelyIfPermitted(_ text: String) -> Bool {
        refreshAudioRoute()
        guard currentRoute.isPrivate else {
            return false
        }
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.51
        utterance.volume = 1.0
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        speechSynthesizer.speak(utterance)
        return true
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        let ports = AVAudioSession.sharedInstance().currentRoute.outputs.map { $0.portType.rawValue }
        print("[Sherlocked] speech playback started on \(ports)")
    }

    func stopCapture() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        generation = UUID()
        retryTask?.cancel()
        speechInput.replace(nil, external: false)
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        isMicrophoneFlowing = false
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
    }
}
