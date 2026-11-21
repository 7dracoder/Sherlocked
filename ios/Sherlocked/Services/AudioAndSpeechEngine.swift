import AVFoundation
import Foundation
import Speech

@MainActor
final class AudioAndSpeechEngine: NSObject, ObservableObject {
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
    private var lastCommittedWordCount: Int = 0
    private var lastSegmentEndSec: Double = 0.0
    private var pairedGlassesHint: Bool = false
    private var onFirstAudioDelivered: (() -> Void)?
    private var onUtteranceUpdated: (() -> Void)?

    override init() {
        super.init()
        refreshAudioRoute(hasPairedGlasses: false)
        observeRouteAndInterruptions()
    }

    func refreshAudioRoute(hasPairedGlasses: Bool? = nil) {
        if let hasPairedGlasses {
            pairedGlassesHint = hasPairedGlasses
        }
        let session = AVAudioSession.sharedInstance()
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
        lastCommittedWordCount = 0
        lastSegmentEndSec = 0.0
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
        // Avoid combining .voiceChat with .allowBluetoothA2DP (causes OSStatus -50 on iOS hardware).
        // Use .default with .allowBluetoothHFP + .defaultToSpeaker so both Ray-Ban Meta HFP mic/audio
        // and iPhone built-in mic/earbuds work cleanly without error!
        do {
            try audioSession.setCategory(
                .playAndRecord,
                mode: .videoRecording,
                options: [.allowBluetoothHFP, .mixWithOthers, .defaultToSpeaker]
            )
        } catch {
            try audioSession.setCategory(.playAndRecord, mode: .default, options: [.allowBluetoothHFP, .defaultToSpeaker])
        }
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        if let btInput = audioSession.availableInputs?.first(where: { $0.portType == .bluetoothHFP }) {
            try? audioSession.setPreferredInput(btInput)
        }
        refreshAudioRoute(hasPairedGlasses: hasPairedGlasses)

        // Instantiate a fresh AVAudioEngine AFTER setActive + setPreferredInput so its input AUHAL
        // matches Ray-Ban Meta BluetoothHFP (16 kHz mono) without throwing CoreAudio -10868!
        let freshEngine = AVAudioEngine()
        self.audioEngine = freshEngine
        let inputNode = freshEngine.inputNode
        let hwFormat = inputNode.inputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0 && hwFormat.channelCount > 0 else {
            return
        }

        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        recognitionRequest = req

        if let recognizer = speechRecognizer, recognizer.isAvailable {
            recognitionTask = Self.startNonisolatedRecognitionTask(
                recognizer: recognizer,
                request: req
            ) { [weak self] snapshot in
                Task { @MainActor [weak self] in
                    self?.consumeSpeechSnapshot(snapshot)
                }
            }
        }

        Self.installNonisolatedAudioTap(
            on: inputNode,
            format: hwFormat,
            request: req
        ) { [weak self] in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if !self.isMicrophoneFlowing {
                    self.isMicrophoneFlowing = true
                    self.onFirstAudioDelivered?()
                }
            }
        }

        freshEngine.prepare()
        try freshEngine.start()
        #endif
    }

    private struct SpeechSegmentItem: Sendable {
        let substring: String
        let timestamp: Double
        let duration: Double
        let confidence: Float
    }

    private struct SpeechSnapshot: Sendable {
        let formattedString: String
        let isFinal: Bool
        let segments: [SpeechSegmentItem]
    }

    nonisolated private static func startNonisolatedRecognitionTask(
        recognizer: SFSpeechRecognizer,
        request: SFSpeechAudioBufferRecognitionRequest,
        onResult: @escaping @Sendable (SpeechSnapshot) -> Void
    ) -> SFSpeechRecognitionTask {
        recognizer.recognitionTask(with: request) { result, _ in
            guard let result else { return }
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
                )
            )
        }
    }

    nonisolated private static func installNonisolatedAudioTap(
        on inputNode: AVAudioInputNode,
        format: AVAudioFormat,
        request: SFSpeechAudioBufferRecognitionRequest,
        onSampleSeen: @escaping @Sendable () -> Void
    ) {
        nonisolated(unsafe) let unsafeReq = request
        nonisolated(unsafe) var emittedFirst = false

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            guard buffer.frameLength > 0 else { return }
            unsafeReq.append(buffer)
            if !emittedFirst {
                emittedFirst = true
                onSampleSeen()
            }
        }
    }

    /// Consumes synchronized PCM buffers coming directly from Ray-Ban Meta glasses when enabled.
    func consumeExternalGlassesPCM(_ buffer: AVAudioPCMBuffer) {
        recognitionRequest?.append(buffer)
        if !isMicrophoneFlowing {
            isMicrophoneFlowing = true
            onFirstAudioDelivered?()
        }
    }

    private func consumeSpeechSnapshot(_ result: SpeechSnapshot) {
        let text = result.formattedString
        livePartialTranscript = text

        let segments = result.segments
        guard segments.count > lastCommittedWordCount else { return }

        var currentWords: [WordTimestampDTO] = []
        var clauseStart: Double?
        var prevEnd = lastSegmentEndSec
        var rebuilt: [UtteranceSegmentDTO] = []

        for seg in segments {
            let wStart = max(0.0, seg.timestamp)
            let wEnd = wStart + max(0.08, seg.duration)
            let gap = wStart - prevEnd

            if !currentWords.isEmpty && gap > 0.68 {
                let cText = currentWords.map(\.word).joined(separator: " ")
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
                    confidence: Double(seg.confidence > 0 ? seg.confidence : 0.9)
                )
            )
            prevEnd = wEnd
        }

        if !currentWords.isEmpty {
            let cText = currentWords.map(\.word).joined(separator: " ")
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

        if rebuilt.count != utterances.count || result.isFinal {
            utterances = rebuilt
            lastCommittedWordCount = segments.count
            onUtteranceUpdated?()
        }
    }

    func speakCuePrivatelyIfPermitted(_ text: String, forceReadyCue: Bool = false) -> Bool {
        guard currentRoute.isPrivate || forceReadyCue else {
            return false
        }
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.51
        utterance.volume = forceReadyCue ? 0.85 : 0.72
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        speechSynthesizer.speak(utterance)
        return true
    }

    func stopCapture() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
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
