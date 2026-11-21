import Foundation

enum BackendError: LocalizedError {
    case invalidURL
    case serverError(Int, String)
    case networkTimeout

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid Sherlocked backend URL."
        case .serverError(let code, let detail):
            return "Backend error (\(code)): \(detail)"
        case .networkTimeout:
            return "Analysis timed out; session remains active."
        }
    }
}

actor BackendAPIClient {
    private var baseURLString: String
    private let session: URLSession

    init(baseURLString: String? = nil, timeoutSeconds: TimeInterval = 35.0) {
        let savedOverride: String? = {
            if let s = UserDefaults.standard.string(forKey: "sherlocked_backend_url")?.trimmingCharacters(in: .whitespacesAndNewlines),
               !s.isEmpty {
                return s
            }
            return nil
        }()
        #if targetEnvironment(simulator)
        let defaultHost = "http://127.0.0.1:8787"
        let nonLoopbackInfoURL: String? = nil
        #else
        let defaultHost = "http://127.0.0.1:8787"
        let infoURL = Bundle.main.object(forInfoDictionaryKey: "SHERLOCKED_BACKEND_URL") as? String
        let nonLoopbackInfoURL: String? = {
            if let infoURL, !infoURL.contains("127.0.0.1"), !infoURL.contains("localhost"), !infoURL.isEmpty {
                return infoURL
            }
            return nil
        }()
        #endif
        let configured = baseURLString
            ?? ProcessInfo.processInfo.environment["SHERLOCKED_BACKEND_URL"]
            ?? savedOverride
            ?? nonLoopbackInfoURL
            ?? defaultHost
        self.baseURLString = configured.trimmingCharacters(in: CharacterSet(charactersIn: "/"))

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = timeoutSeconds
        config.timeoutIntervalForResource = timeoutSeconds + 5.0
        self.session = URLSession(configuration: config)
    }

    var baseURL: String { baseURLString }

    func updateBaseURL(_ newURLString: String) {
        let trimmed = newURLString
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !trimmed.isEmpty else { return }
        baseURLString = trimmed
        UserDefaults.standard.set(trimmed, forKey: "sherlocked_backend_url")
    }

    func analyzeTurn(_ requestPayload: SessionAnalyzeRequestDTO) async throws -> SessionAnalyzeResponseDTO {
        guard let url = URL(string: "\(baseURLString)/api/v1/session/analyze") else {
            throw BackendError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(requestPayload)

        do {
            let (data, response) = try await session.data(for: request)
            try Task.checkCancellation()
            guard let http = response as? HTTPURLResponse else {
                throw BackendError.serverError(0, "Non-HTTP response")
            }
            if !(200..<300).contains(http.statusCode) {
                let msg = String(data: data, encoding: .utf8) ?? "Unknown error"
                throw BackendError.serverError(http.statusCode, msg)
            }
            return try JSONDecoder().decode(SessionAnalyzeResponseDTO.self, from: data)
        } catch let urlErr as URLError where urlErr.code == .timedOut {
            throw BackendError.networkTimeout
        }
    }

    func sendManualCorrection(_ payload: ManualCorrectionRequestDTO) async throws {
        guard let url = URL(string: "\(baseURLString)/api/v1/session/manual-correction") else {
            throw BackendError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(payload)
        let (_, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw BackendError.serverError(http.statusCode, "Failed to save manual correction")
        }
    }

    func fetchFixtures() async throws -> [ReplayFixtureItemDTO] {
        guard let url = URL(string: "\(baseURLString)/api/v1/fixtures") else {
            throw BackendError.invalidURL
        }
        let (data, response) = try await session.data(from: url)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw BackendError.serverError(500, "Could not load fixtures")
        }
        let decoded = try JSONDecoder().decode(FixturesListResponseDTO.self, from: data)
        return decoded.fixtures
    }

    func replayFixture(
        fixtureID: String,
        faceMasked: Bool = false,
        includeFollowup: Bool = false
    ) async throws -> SessionAnalyzeResponseDTO {
        guard let url = URL(
            string: "\(baseURLString)/api/v1/fixtures/\(fixtureID)/replay?face_masked=\(faceMasked ? "true" : "false")&include_followup=\(includeFollowup ? "true" : "false")"
        ) else {
            throw BackendError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let msg = String(data: data, encoding: .utf8) ?? "Replay failed"
            throw BackendError.serverError((response as? HTTPURLResponse)?.statusCode ?? 500, msg)
        }
        return try JSONDecoder().decode(SessionAnalyzeResponseDTO.self, from: data)
    }

    func mediaURL(for fixtureID: String, masked: Bool = false) -> URL? {
        URL(string: "\(baseURLString)/api/v1/fixtures/\(fixtureID)/media/\(masked ? "masked" : "original")")
    }
}
