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

    private var pairedGlassesHint: Bool = false
    private var usingGlassesMic: Bool = false
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
        converter = nil
        targetASRFormat = nil

        let audioSession = AVAudioSession.sharedInstance()
        #if targetEnvironment(simulator)
        try? audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        refreshAudioRoute(hasPairedGlasses: hasPairedGlasses)
        return
        #else
        if UserDefaults.standard.object(forKey: "sherlocked_glasses_mic_primary_v1") == nil {
            UserDefaults.standard.set(false, forKey: "sherlocked_phone_mic")
            UserDefaults.standard.set(true, forKey: "sherlocked_glasses_mic_primary_v1")
        }
        let forcePhoneMic = UserDefaults.standard.bool(forKey: "sherlocked_phone_mic")
        usingGlassesMic = hasPairedGlasses && !forcePhoneMic

        // Keep mic + private earpiece on HFP. Prefer videoRecording over voiceChat —
        // voiceChat has caused AVAudioSession / CoreAudio start failures on Meta HFP.
        // Do NOT use defaultToSpeaker — that steals speak-back from Ray-Ban Meta.
        if usingGlassesMic {
            try audioSession.setCategory(
                .playAndRecord,
                mode: .videoRecording,
                options: [.allowBluetoothHFP, .mixWithOthers]
            )
            try? audioSession.setPreferredSampleRate(16_000)
            try? audioSession.setPreferredIOBufferDuration(0.02)
        } else {
            try audioSession.setCategory(
                .playAndRecord,
                mode: .measurement,
                options: [.allowBluetoothA2DP, .allowBluetoothHFP, .defaultToSpeaker]
            )
        }
        try audioSession.setActive(true, options: [.notifyOthersOnDeactivation])

        let preferredPort: AVAudioSession.Port = usingGlassesMic ? .bluetoothHFP : .builtInMic
        if let preferredInput = audioSession.availableInputs?.first(where: { $0.portType == preferredPort }) {
            try audioSession.setPreferredInput(preferredInput)
        } else if usingGlassesMic,
                  let anyHFP = audioSession.availableInputs?.first(where: {
                      $0.portType == .bluetoothHFP
                      || $0.portName.localizedCaseInsensitiveContains("Ray-Ban")
                      || $0.portName.localizedCaseInsensitiveContains("RB Meta")
                      || $0.portName.localizedCaseInsensitiveContains("Meta")
                  }) {
            try audioSession.setPreferredInput(anyHFP)
        }
        refreshAudioRoute(hasPairedGlasses: hasPairedGlasses)

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

        // Normalize to 16 kHz mono for SFSpeech — better accuracy for distant front speakers.
        let asrFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)
            ?? hwFormat
        targetASRFormat = asrFormat
        if hwFormat.sampleRate != asrFormat.sampleRate || hwFormat.channelCount != asrFormat.channelCount {
            converter = AVAudioConverter(from: hwFormat, to: asrFormat)
        } else {
            converter = nil
        }

        beginRecognition()

        let converter = self.converter
        let speechInput = self.speechInput
        Self.installNonisolatedAudioTap(
            on: inputNode,
            format: hwFormat,
            asrFormat: asrFormat,
            converter: converter,
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
        // Cloud ASR + spoken-audio hint works better for glasses HFP / distant talkers.
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

    nonisolated private static func installNonisolatedAudioTap(
        on inputNode: AVAudioInputNode,
        format: AVAudioFormat,
        asrFormat: AVAudioFormat,
        converter: AVAudioConverter?,
        input: SpeechInput,
        onSampleSeen: @escaping @Sendable () -> Void
    ) {
        nonisolated(unsafe) var emittedFirst = false

        inputNode.installTap(onBus: 0, bufferSize: 2048, format: format) { buffer, _ in
            guard buffer.frameLength > 0 else { return }
            if let converter {
                let ratio = asrFormat.sampleRate / format.sampleRate
                let outFrames = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
                guard let out = AVAudioPCMBuffer(pcmFormat: asrFormat, frameCapacity: outFrames) else { return }
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
                if error == nil, out.frameLength > 0 {
                    input.append(out)
                }
            } else {
                input.append(buffer)
            }
            if !emittedFirst {
                emittedFirst = true
                onSampleSeen()
            }
        }
    }

    /// Optional DAT PCM path. Only used as a fallback if HFP has not started flowing yet.
    func consumeExternalGlassesPCM(_ buffer: AVAudioPCMBuffer) {
        guard recognitionRequest != nil else { return }
        // Prefer Bluetooth HFP mic once it is delivering samples — DAT PCM often
        // arrives in a format that hurts Apple Speech accuracy for front speakers.
        if isMicrophoneFlowing { return }
        guard let asrFormat = targetASRFormat else {
            speechInput.append(buffer)
            return
        }
        if buffer.format.sampleRate == asrFormat.sampleRate && buffer.format.channelCount == asrFormat.channelCount {
            speechInput.append(buffer)
        } else if let converted = Self.convertBuffer(buffer, to: asrFormat) {
            speechInput.append(converted)
        } else {
            speechInput.append(buffer)
        }
        if !isMicrophoneFlowing {
            isMicrophoneFlowing = true
            onFirstAudioDelivered?()
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

    func consumeSpeechSnapshot(_ result: SpeechSnapshot) {
        let segments = result.segments
        guard let first = segments.first else { return }
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

            // Longer gap keeps clauses intact for clearer story segmentation.
            if !currentWords.isEmpty && gap > 0.72 {
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
            if preferGlasses {
                try session.setCategory(
                    .playAndRecord,
                    mode: .videoRecording,
                    options: [.allowBluetoothHFP, .mixWithOthers]
                )
            } else {
                try session.setCategory(
                    .playAndRecord,
                    mode: .default,
                    options: [.allowBluetoothA2DP, .allowBluetoothHFP, .defaultToSpeaker, .mixWithOthers]
                )
            }
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

    func stopCapture() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        generation = UUID()
        retryTask?.cancel()
        speechInput.replace(nil)
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        isMicrophoneFlowing = false
        converter = nil
        targetASRFormat = nil
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
    }
}
