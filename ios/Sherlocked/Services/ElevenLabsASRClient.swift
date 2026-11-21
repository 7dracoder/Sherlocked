import Foundation

/// Streams 16 kHz mono PCM to the Sherlocked backend, which proxies ElevenLabs Scribe realtime.
@MainActor
final class ElevenLabsASRClient: NSObject {
    struct Word: Sendable {
        let word: String
        let start: Double
        let end: Double
        let confidence: Double
    }

    var onPartial: ((String) -> Void)?
    var onCommitted: ((String, [Word]) -> Void)?
    var onError: ((String) -> Void)?
    var onConnected: (() -> Void)?

    private var task: URLSessionWebSocketTask?
    private var session: URLSession?
    private var receiveLoop: Task<Void, Never>?
    private(set) var isConnected = false

    func connect(backendBaseURL: String) {
        disconnect()
        let wsBase: String
        if backendBaseURL.hasPrefix("https://") {
            wsBase = "wss://" + backendBaseURL.dropFirst("https://".count)
        } else if backendBaseURL.hasPrefix("http://") {
            wsBase = "ws://" + backendBaseURL.dropFirst("http://".count)
        } else {
            wsBase = backendBaseURL
        }
        guard let url = URL(string: wsBase.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/api/v1/asr/live") else {
            onError?("Invalid ASR WebSocket URL")
            return
        }
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        let sess = URLSession(configuration: config)
        session = sess
        let ws = sess.webSocketTask(with: url)
        task = ws
        ws.resume()
        isConnected = true
        onConnected?()
        receiveLoop = Task { [weak self] in
            await self?.pumpIncoming()
        }
    }

    func sendPCM16(_ data: Data) {
        guard isConnected, let task else { return }
        task.send(.data(data)) { [weak self] error in
            if let error {
                Task { @MainActor in
                    self?.onError?(error.localizedDescription)
                }
            }
        }
    }

    func requestCommit() {
        guard isConnected, let task else { return }
        guard let payload = try? JSONSerialization.data(withJSONObject: ["type": "commit"]) else { return }
        guard let text = String(data: payload, encoding: .utf8) else { return }
        task.send(.string(text)) { _ in }
    }

    func disconnect() {
        receiveLoop?.cancel()
        receiveLoop = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        session?.invalidateAndCancel()
        session = nil
        isConnected = false
    }

    private func pumpIncoming() async {
        while !Task.isCancelled, let task {
            do {
                let message = try await task.receive()
                switch message {
                case .string(let text):
                    handleJSONText(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        handleJSONText(text)
                    }
                @unknown default:
                    break
                }
            } catch {
                if !Task.isCancelled {
                    onError?(error.localizedDescription)
                    isConnected = false
                }
                break
            }
        }
    }

    private func handleJSONText(_ text: String) {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String else { return }

        switch type {
        case "partial":
            onPartial?((obj["text"] as? String) ?? "")
        case "committed":
            let transcript = (obj["text"] as? String) ?? ""
            var words: [Word] = []
            if let rawWords = obj["words"] as? [[String: Any]] {
                for w in rawWords {
                    let token = (w["text"] as? String) ?? (w["word"] as? String) ?? ""
                    guard !token.isEmpty else { continue }
                    let start = (w["start"] as? Double) ?? (w["start_sec"] as? Double) ?? 0
                    let end = (w["end"] as? Double) ?? (w["end_sec"] as? Double) ?? (start + 0.12)
                    let conf = (w["logprob"] as? Double).map { max(0, min(1, exp($0))) } ?? 0.9
                    words.append(Word(word: token, start: start, end: end, confidence: conf))
                }
            }
            onCommitted?(transcript, words)
        case "error":
            let detail: String
            if let d = obj["detail"] as? String {
                detail = d
            } else if let d = obj["detail"] {
                detail = String(describing: d)
            } else {
                detail = "ElevenLabs ASR error"
            }
            onError?(detail)
        case "session":
            break
        default:
            break
        }
    }
}
