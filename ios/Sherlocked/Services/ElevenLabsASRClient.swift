import Foundation

/// Posts short WAV clips to the Sherlocked backend → ElevenLabs Scribe (reliable HTTP path).
@MainActor
final class ElevenLabsASRClient {
    struct Word: Sendable {
        let word: String
        let start: Double
        let end: Double
        let confidence: Double
    }

    struct TranscriptResult: Sendable {
        let text: String
        let words: [Word]
    }

    private var backendBaseURL: String = "https://sherlocked-backend.onrender.com"
    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 25
        config.timeoutIntervalForResource = 30
        return URLSession(configuration: config)
    }()

    private var inFlight = false

    func configure(backendBaseURL: String) {
        let trimmed = backendBaseURL
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if !trimmed.isEmpty {
            self.backendBaseURL = trimmed
        }
    }

    /// Transcribe a PCM16 mono 16 kHz clip wrapped as WAV.
    func transcribeWAV(
        _ wavData: Data,
        timeOffsetSec: Double,
        utteranceStartIndex: Int
    ) async throws -> TranscriptResult {
        guard !inFlight else {
            throw NSError(domain: "ElevenASR", code: 1, userInfo: [NSLocalizedDescriptionKey: "busy"])
        }
        inFlight = true
        defer { inFlight = false }

        guard let url = URL(string: "\(backendBaseURL)/api/v1/asr/transcribe") else {
            throw NSError(domain: "ElevenASR", code: 2, userInfo: [NSLocalizedDescriptionKey: "bad url"])
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "audio_wav_base64": wavData.base64EncodedString(),
            "time_offset_sec": timeOffsetSec,
            "utterance_start_index": utteranceStartIndex,
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let msg = String(data: data, encoding: .utf8) ?? "ElevenLabs transcribe failed"
            throw NSError(domain: "ElevenASR", code: 3, userInfo: [NSLocalizedDescriptionKey: msg])
        }
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "ElevenASR", code: 4, userInfo: [NSLocalizedDescriptionKey: "bad json"])
        }
        let text = (obj["text"] as? String) ?? ""
        var words: [Word] = []
        if let utts = obj["utterances"] as? [[String: Any]] {
            for u in utts {
                if let ws = u["words"] as? [[String: Any]] {
                    for w in ws {
                        let token = (w["word"] as? String) ?? ""
                        guard !token.isEmpty else { continue }
                        words.append(
                            Word(
                                word: token,
                                start: (w["start_sec"] as? Double) ?? 0,
                                end: (w["end_sec"] as? Double) ?? 0,
                                confidence: (w["confidence"] as? Double) ?? 0.9
                            )
                        )
                    }
                }
            }
        }
        return TranscriptResult(text: text, words: words)
    }

    static func wavDataFromPCM16(_ pcm: Data, sampleRate: Int = 16_000) -> Data {
        var data = Data()
        let dataSize = UInt32(pcm.count)
        let fileSize = UInt32(36 + pcm.count)
        let sr = UInt32(sampleRate)
        let byteRate = UInt32(sampleRate * 2)
        func appendU16(_ v: UInt16) { var le = v.littleEndian; data.append(Data(bytes: &le, count: 2)) }
        func appendU32(_ v: UInt32) { var le = v.littleEndian; data.append(Data(bytes: &le, count: 4)) }
        data.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // RIFF
        appendU32(fileSize)
        data.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // WAVE
        data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20]) // fmt
        appendU32(16)
        appendU16(1) // PCM
        appendU16(1) // mono
        appendU32(sr)
        appendU32(byteRate)
        appendU16(2) // block align
        appendU16(16) // bits
        data.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // data
        appendU32(dataSize)
        data.append(pcm)
        return data
    }
}
