import AVFoundation
import CoreMedia
import Foundation
import MWDATCamera
import MWDATCore
import MWDATMockDevice
import UIKit
import VideoToolbox
import os

// MARK: - Thread-Safe HEVC Frame Decoder for Physical Ray-Ban Meta Streams

private final class HEVCFrameDecoder: @unchecked Sendable {
    private let lock = NSLock()
    private let ciContext = CIContext()
    private var decompressionSession: VTDecompressionSession?
    private var currentFormatDesc: CMFormatDescription?
    private var lastDecodedImage: UIImage?

    func decode(_ sampleBuffer: CMSampleBuffer) -> UIImage? {
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let ciImg = CIImage(cvPixelBuffer: pixelBuffer)
            if let cg = ciContext.createCGImage(ciImg, from: ciImg.extent) {
                let img = UIImage(cgImage: cg)
                lock.lock()
                lastDecodedImage = img
                lock.unlock()
                return img
            }
        }

        guard CMSampleBufferGetDataBuffer(sampleBuffer) != nil,
              let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer)
        else {
            lock.lock()
            defer { lock.unlock() }
            return lastDecodedImage
        }

        lock.lock()
        if let existing = decompressionSession,
           let oldDesc = currentFormatDesc,
           !CMFormatDescriptionEqual(oldDesc, otherFormatDescription: formatDesc) ||
           !VTDecompressionSessionCanAcceptFormatDescription(existing, formatDescription: formatDesc) {
            VTDecompressionSessionInvalidate(existing)
            decompressionSession = nil
            currentFormatDesc = nil
        }

        if decompressionSession == nil {
            let spec: [CFString: Any] = [
                kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: false
            ]
            let attrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
            ]
            var newSession: VTDecompressionSession?
            VTDecompressionSessionCreate(
                allocator: kCFAllocatorDefault,
                formatDescription: formatDesc,
                decoderSpecification: spec as CFDictionary,
                imageBufferAttributes: attrs as CFDictionary,
                outputCallback: nil,
                decompressionSessionOut: &newSession
            )
            decompressionSession = newSession
            currentFormatDesc = formatDesc
        }
        let activeSession = decompressionSession
        lock.unlock()

        guard let activeSession else { return nil }

        nonisolated(unsafe) var decodedBuffer: CVPixelBuffer?
        let status = VTDecompressionSessionDecodeFrame(
            activeSession,
            sampleBuffer: sampleBuffer,
            flags: [],
            infoFlagsOut: nil
        ) { decodeStatus, _, imageBuffer, _, _ in
            if decodeStatus == noErr, let imageBuffer {
                decodedBuffer = imageBuffer
            }
        }

        guard status == noErr, let pixelBuf = decodedBuffer else {
            lock.lock()
            defer { lock.unlock() }
            return lastDecodedImage
        }

        let rect = CGRect(
            x: 0,
            y: 0,
            width: CVPixelBufferGetWidth(pixelBuf),
            height: CVPixelBufferGetHeight(pixelBuf)
        )
        guard let cgImg = ciContext.createCGImage(CIImage(cvPixelBuffer: pixelBuf), from: rect) else {
            return nil
        }
        let uiImg = UIImage(cgImage: cgImg)
        lock.lock()
        lastDecodedImage = uiImg
        lock.unlock()
        return uiImg
    }
}

// MARK: - GlassesWearablesCoordinator

@MainActor
final class GlassesWearablesCoordinator: ObservableObject {
    private static let logger = Logger(subsystem: "com.tanmaysahu.sherlocked", category: "MWDAT")

    @Published private(set) var registrationState: RegistrationState
    @Published private(set) var connectedDeviceNames: [String] = []
    @Published private(set) var hasEligibleGlasses: Bool = false
    @Published private(set) var isDeviceLinkConnected: Bool = false
    @Published private(set) var sessionState: DeviceSessionState = .idle
    @Published private(set) var streamState: StreamState = .stopped
    @Published private(set) var latestPreviewFrame: UIImage?
    @Published private(set) var isMockModeEnabled: Bool = false
    @Published private(set) var voiceChannelActive: Bool = false
    @Published private(set) var isAwaitingPermissionRedirect: Bool = false
    @Published var lastWearablesError: String?

    var onVoiceLaunchReceived: (() -> Void)?
    var onVideoFrameDelivered: ((UIImage, Double) -> Void)?
    var onGlassesAudioDelivered: ((AVAudioPCMBuffer) -> Void)?
    var onGlassesDisconnectedMidSession: (() -> Void)?
    var onPermissionCallbackCompleted: (() -> Void)?

    private let wearables: WearablesInterface
    private let deviceSelector: AutoDeviceSelector
    private let frameDecoder = HEVCFrameDecoder()
    private var deviceSession: DeviceSession?
    private var cameraCapability: MWDATCamera.Camera?
    private var voiceStream: VoiceInvocationsStream?
    private var activeVoiceDevice: DeviceIdentifier?

    private var registrationTask: Task<Void, Never>?
    private var selectorTask: Task<Void, Never>?
    private var devicesTask: Task<Void, Never>?
    private var deviceDetailTokens: [AnyListenerToken] = []
    private let sessionTokens = ListenerTokenBag()
    private let streamTokens = ListenerTokenBag()
    private let voiceTokens = ListenerTokenBag()

    private var sessionStartMonotonic: TimeInterval = 0
    private var mockGlasses: MockGlasses?

    init(wearables: WearablesInterface = Wearables.shared) {
        self.wearables = wearables
        self.registrationState = wearables.registrationState
        self.deviceSelector = AutoDeviceSelector(wearables: wearables)
        startMonitoring()
    }

    private func startMonitoring() {
        registrationTask = Task { [weak self] in
            guard let self else { return }
            for await state in wearables.registrationStateStream() {
                self.registrationState = state
                if state == .registered {
                    self.lastWearablesError = nil
                    self.ensureVoiceInvocationsListening()
                }
            }
        }
        selectorTask = Task { [weak self] in
            guard let self else { return }
            for await activeID in deviceSelector.activeDeviceStream() {
                let wasEligible = self.hasEligibleGlasses
                self.hasEligibleGlasses = (activeID != nil) || self.isMockModeEnabled
                if let activeID {
                    self.ensureVoiceInvocationsListening(preferred: activeID)
                } else if wasEligible && !self.isAwaitingPermissionRedirect && self.sessionState != .idle && self.sessionState != .stopped {
                    self.onGlassesDisconnectedMidSession?()
                }
            }
        }
        devicesTask = Task { [weak self] in
            guard let self else { return }
            for await ids in wearables.devicesStream() {
                for tok in self.deviceDetailTokens {
                    await tok.cancel()
                }
                self.deviceDetailTokens.removeAll()
                self.refreshDeviceRoster(ids: ids)

                for id in ids {
                    if let dev = self.wearables.deviceForIdentifier(id) {
                        self.deviceDetailTokens.append(
                            dev.addLinkStateListener { [weak self] _ in
                                Task { @MainActor [weak self] in
                                    guard let self else { return }
                                    self.refreshDeviceRoster(ids: self.wearables.devices)
                                    self.ensureVoiceInvocationsListening()
                                }
                            }
                        )
                        self.deviceDetailTokens.append(
                            dev.addDeviceStateListener { [weak self] _ in
                                Task { @MainActor [weak self] in
                                    guard let self else { return }
                                    self.refreshDeviceRoster(ids: self.wearables.devices)
                                }
                            }
                        )
                    }
                }
            }
        }
    }

    private func refreshDeviceRoster(ids: [DeviceIdentifier]) {
        var names: [String] = []
        var anyConnected = false
        for id in ids {
            if let dev = wearables.deviceForIdentifier(id) {
                let rawName = dev.name.trimmingCharacters(in: .whitespacesAndNewlines)
                let fallback = dev.nameOrId()
                let label = !rawName.isEmpty ? rawName : (fallback.count <= 12 ? "RB Meta \(fallback)" : "Ray-Ban Meta")
                let linkTag = (dev.linkState == .connected) ? "Connected" : "Standby"
                names.append("\(label) (\(linkTag))")
                if dev.linkState == .connected {
                    anyConnected = true
                }
            }
        }
        connectedDeviceNames = names
        isDeviceLinkConnected = anyConnected
        if !ids.isEmpty && registrationState == .registered {
            hasEligibleGlasses = true
        }
    }

    private func preferredDeviceIdentifier() -> DeviceIdentifier? {
        let devices = wearables.devices.compactMap { wearables.deviceForIdentifier($0) }
        let ranked = devices.sorted { a, b in
            func rank(_ d: Device) -> Int {
                if d.linkState == .connected && d.donState == .donned { return 0 }
                if d.linkState == .connected { return 1 }
                return 2
            }
            return rank(a) < rank(b)
        }
        return ranked.first?.identifier ?? deviceSelector.activeDevice ?? wearables.devices.first
    }

    var registrationStateDescription: String {
        switch registrationState {
        case .registered:
            return "Linked with Meta AI"
        case .registering:
            return "Waiting for Meta AI approval..."
        case .available:
            return "Ready to Link"
        case .unavailable:
            return "Not Linked Yet"
        @unknown default:
            return String(describing: registrationState)
        }
    }

    func handleIncomingURL(_ url: URL) async {
        guard
            let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
            components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
        else {
            return
        }
        do {
            let handled = try await wearables.handleUrl(url)
            Self.logger.info("handleUrl succeeded: \(handled, privacy: .public)")
            isAwaitingPermissionRedirect = false
            lastWearablesError = nil
            onPermissionCallbackCompleted?()
        } catch let regErr as RegistrationError {
            isAwaitingPermissionRedirect = false
            lastWearablesError = regErr.description
        } catch {
            isAwaitingPermissionRedirect = false
            lastWearablesError = error.localizedDescription
        }
    }

    func startRegistration() async {
        if registrationState == .registered {
            lastWearablesError = nil
            refreshDeviceRoster(ids: wearables.devices)
            return
        }
        guard registrationState != .registering else { return }
        do {
            lastWearablesError = nil
            isAwaitingPermissionRedirect = true
            try await wearables.startRegistration()
        } catch let regErr as RegistrationError {
            isAwaitingPermissionRedirect = false
            if regErr == .alreadyRegistered {
                lastWearablesError = nil
            } else {
                lastWearablesError = regErr.description
            }
        } catch {
            isAwaitingPermissionRedirect = false
            lastWearablesError = error.localizedDescription
        }
    }

    /// Optional one-tap helper in Setup to open a brief DeviceSession and prompt Camera permission in Meta AI.
    func requestGlassesCameraPermissionInteractive() async {
        do {
            lastWearablesError = nil
            if registrationState != .registered {
                await startRegistration()
                return
            }
            // Must connect a DeviceSession first so DAT can read/write permissions off the glasses!
            let session: DeviceSession
            if let existing = deviceSession, existing.state == .started {
                session = existing
            } else {
                let created: DeviceSession
                if !isMockModeEnabled, let pref = preferredDeviceIdentifier() {
                    created = try wearables.createSession(deviceSelector: SpecificDeviceSelector(device: pref))
                } else {
                    created = try wearables.createSession(deviceSelector: deviceSelector)
                }
                deviceSession = created
                sessionState = .starting
                try created.start()
                let connected = await waitForSessionStarted(created, timeoutSeconds: 8.0)
                guard connected else {
                    lastWearablesError = "Make sure your Ray-Ban Meta glasses are unfolded, worn, and connected in Meta AI."
                    created.stop()
                    deviceSession = nil
                    sessionState = .idle
                    return
                }
                session = created
            }

            let current = try? await wearables.checkPermissionStatus(.camera)
            if current == .granted {
                lastWearablesError = nil
            } else {
                isAwaitingPermissionRedirect = true
                _ = try await wearables.requestPermission(.camera)
                isAwaitingPermissionRedirect = false
            }

            if streamState == .stopped {
                session.stop()
                deviceSession = nil
                sessionState = .idle
            }
        } catch {
            isAwaitingPermissionRedirect = false
            Self.logger.warning("Camera permission request note: \(error.localizedDescription, privacy: .public)")
            lastWearablesError = "Ensure glasses are unfolded & connected in Meta AI, then try again."
        }
    }

    func ensureVoiceInvocationsListening(preferred: DeviceIdentifier? = nil) {
        guard registrationState == .registered || isMockModeEnabled else { return }
        let targetDevice = preferred ?? preferredDeviceIdentifier()
        guard let targetDevice else { return }
        if !isMockModeEnabled,
           let dev = wearables.deviceForIdentifier(targetDevice),
           dev.linkState != .connected {
            return
        }
        if activeVoiceDevice == targetDevice && voiceStream != nil { return }

        Task {
            await voiceTokens.cancelAll()
            voiceStream?.stop()
            voiceStream = nil
            activeVoiceDevice = nil

            do {
                let stream = try VoiceInvocationsStream(wearables: wearables)
                voiceStream = stream
                stream.invocationsPublisher.listen { [weak self] invocation in
                    guard let launch = invocation as? LaunchApp else { return }
                    Task { @MainActor [weak self] in
                        _ = await launch.responseHandle.sendSuccess(actionOutput: nil)
                        self?.onVoiceLaunchReceived?()
                    }
                }.store(in: voiceTokens)

                stream.errorPublisher.listen { [weak self] err in
                    Task { @MainActor [weak self] in
                        // Log cleanly without showing scary UI errors when app is in local Developer Mode
                        Self.logger.debug("VoiceInvocationsStream closed: \(err.localizedDescription, privacy: .public)")
                        self?.voiceChannelActive = false
                        self?.activeVoiceDevice = nil
                    }
                }.store(in: voiceTokens)

                try stream.start(deviceIdentifier: targetDevice)
                activeVoiceDevice = targetDevice
                voiceChannelActive = true
            } catch {
                voiceChannelActive = false
            }
        }
    }

    private func waitForSessionStarted(_ session: DeviceSession, timeoutSeconds: Double) async -> Bool {
        if session.state == .started { return true }
        let states = session.stateStream()
        return await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                for await st in states {
                    if st == .started { return true }
                    if st == .stopped { return false }
                }
                return false
            }
            group.addTask {
                try? await Task.sleep(for: .milliseconds(Int(timeoutSeconds * 1000)))
                return session.state == .started
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
    }

    func startGlassesCapture(
        sessionStartClock: TimeInterval,
        enableSynchronizedGlassesAudio: Bool = false
    ) async throws {
        await stopGlassesCapture()
        sessionStartMonotonic = sessionStartClock
        lastWearablesError = nil

        // STEP 1: Create & start DeviceSession FIRST (never call checkPermissionStatus before DeviceSession connects!)
        let session: DeviceSession
        if !isMockModeEnabled, let pref = preferredDeviceIdentifier() {
            session = try wearables.createSession(deviceSelector: SpecificDeviceSelector(device: pref))
        } else {
            session = try wearables.createSession(deviceSelector: deviceSelector)
        }
        deviceSession = session

        session.statePublisher.listen { [weak self] newState in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.sessionState = newState
                if newState == .stopped && !self.isAwaitingPermissionRedirect {
                    self.onGlassesDisconnectedMidSession?()
                }
            }
        }.store(in: sessionTokens)

        session.errorPublisher.listen { [weak self] err in
            Task { @MainActor [weak self] in
                guard let self else { return }
                Self.logger.warning("DeviceSession notice: \(err.localizedDescription, privacy: .public)")
                if err == .datAppOnTheGlassesUpdateRequired {
                    self.lastWearablesError = "Meta glasses firmware/DAT app update required in Meta AI."
                }
            }
        }.store(in: sessionTokens)

        sessionState = .starting
        try session.start()

        let didStart = await waitForSessionStarted(session, timeoutSeconds: 8.5)
        guard didStart else {
            await stopGlassesCapture()
            throw NSError(
                domain: "SherlockedGlasses",
                code: 503,
                userInfo: [NSLocalizedDescriptionKey: "Glasses link still waking up"]
            )
        }

        // STEP 2: Now that DeviceSession is connected (.started), check/request Camera permission!
        if !isMockModeEnabled {
            let perm = try? await wearables.checkPermissionStatus(.camera)
            if perm != .granted {
                isAwaitingPermissionRedirect = true
                let granted = try await wearables.requestPermission(.camera)
                isAwaitingPermissionRedirect = false
                guard granted == .granted else {
                    throw NSError(
                        domain: "SherlockedGlasses",
                        code: 403,
                        userInfo: [NSLocalizedDescriptionKey: "Grant Camera permission in Meta AI"]
                    )
                }
            }
        }

        // STEP 3: Attach Camera & start streaming (.hvc1 low 24fps on real hardware for Bluetooth/Wi-Fi efficiency; .raw on MockDeviceKit)
        let codec: VideoCodec = isMockModeEnabled ? .raw : .hvc1
        let config: StreamConfiguration
        if enableSynchronizedGlassesAudio && !isMockModeEnabled {
            let micPerm = try? await wearables.checkPermissionStatus(.microphone)
            if micPerm == .granted {
                config = StreamConfiguration(
                    videoCodec: codec,
                    audioCodec: .pcm(sampleRate: .rate16000, numberOfChannels: 1),
                    resolution: .low,
                    frameRate: 24
                )
            } else {
                config = StreamConfiguration(videoCodec: codec, resolution: .low, frameRate: 24)
            }
        } else {
            config = StreamConfiguration(videoCodec: codec, resolution: .low, frameRate: 24)
        }

        guard let cam = try session.addCamera(config: config) else {
            throw NSError(
                domain: "SherlockedGlasses",
                code: 500,
                userInfo: [NSLocalizedDescriptionKey: "Camera capability busy"]
            )
        }
        cameraCapability = cam
        let stream = cam.stream

        stream.statePublisher.listen { [weak self] st in
            Task { @MainActor [weak self] in
                self?.streamState = st
            }
        }.store(in: streamTokens)

        stream.errorPublisher.listen { [weak self] err in
            Task { @MainActor [weak self] in
                Self.logger.warning("Stream notice: \(err.localizedDescription, privacy: .public)")
                if UIApplication.shared.applicationState == .active {
                    self?.lastWearablesError = nil
                }
            }
        }.store(in: streamTokens)

        let decoder = self.frameDecoder
        stream.videoFramePublisher.listen { [weak self] videoFrame in
            let decoded = videoFrame.makeUIImage() ?? decoder.decode(videoFrame.sampleBuffer)
            guard let img = decoded else { return }
            let now = CACurrentMediaTime()
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.latestPreviewFrame = img
                let relSec = max(0.0, now - self.sessionStartMonotonic)
                self.onVideoFrameDelivered?(img, relSec)
            }
        }.store(in: streamTokens)

        if enableSynchronizedGlassesAudio {
            stream.audioFramePublisher.listen { [weak self] audioFrame in
                let pcm = audioFrame.pcmBuffer
                Task { @MainActor [weak self] in
                    self?.onGlassesAudioDelivered?(pcm)
                }
            }.store(in: streamTokens)
        }

        streamState = .starting
        stream.start()
    }

    func stopGlassesCapture() async {
        await streamTokens.cancelAll()
        cameraCapability?.stop()
        cameraCapability = nil

        await sessionTokens.cancelAll()
        deviceSession?.stop()
        deviceSession = nil

        streamState = .stopped
        sessionState = .idle
        latestPreviewFrame = nil
    }

    // MARK: - MockDeviceKit Controls for Hardware-Free Development & Automated Tests

    func pairSimulatedRayBanMeta(feedVideoURL: URL? = nil) throws {
        MockDeviceKit.shared.enable()
        let glasses = try MockDeviceKit.shared.pairGlasses(model: .rayBanMeta)
        glasses.powerOn()
        glasses.unfold()
        glasses.don()
        if let feedVideoURL {
            glasses.services.camera.setCameraFeed(fileURL: feedVideoURL)
        }
        mockGlasses = glasses
        isMockModeEnabled = true
        hasEligibleGlasses = true
        lastWearablesError = nil
    }

    func simulateHeyMetaVoiceLaunch() {
        if let glasses = mockGlasses, glasses.services.voiceInvocation.hasConnectedApps {
            glasses.services.voiceInvocation.simulateLaunchAppAction()
        } else {
            onVoiceLaunchReceived?()
        }
    }

    func simulateGlassesFoldDisconnect() {
        mockGlasses?.doff()
        mockGlasses?.fold()
        mockGlasses?.powerOff()
    }

    func disableSimulatedGlasses() async {
        await stopGlassesCapture()
        await MockDeviceKit.shared.disable()
        mockGlasses = nil
        isMockModeEnabled = false
        hasEligibleGlasses = !wearables.devices.isEmpty && registrationState == .registered
    }
}
