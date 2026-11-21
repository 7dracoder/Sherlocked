import AVFoundation
import Foundation
import UIKit

final class PhoneCameraCaptureSource: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.sherlocked.phonecamera")
    private let ciContext = CIContext()
    private var sessionStartMonotonic: TimeInterval = 0
    private var frameIndex: Int = 0
    private var onFrameCallback: (@Sendable (UIImage, Double) -> Void)?

    var isRunning: Bool {
        captureSession.isRunning
    }

    func requestPermission() async -> Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status == .authorized { return true }
        if status == .notDetermined {
            return await AVCaptureDevice.requestAccess(for: .video)
        }
        return false
        #endif
    }

    func start(
        sessionStartClock: TimeInterval,
        preferFrontCamera: Bool = true,
        onFrame: @escaping @Sendable (UIImage, Double) -> Void
    ) async throws {
        stop()
        sessionStartMonotonic = sessionStartClock
        frameIndex = 0
        onFrameCallback = onFrame

        let granted = await requestPermission()
        guard granted else {
            throw NSError(
                domain: "SherlockedPhoneCamera",
                code: 403,
                userInfo: [NSLocalizedDescriptionKey: "iPhone camera permission denied."]
            )
        }

        let position: AVCaptureDevice.Position = preferFrontCamera ? .front : .back
        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
            ?? AVCaptureDevice.default(for: .video)

        guard let cameraDevice = device else {
            // On iOS Simulator without a hardware webcam, generate live timestamped diagnostic test frames
            // so the Simulator phone-fallback path exercises the full capture/ready-cue/stop lifecycle cleanly.
            startSimulatorSyntheticCamera(sessionStartClock: sessionStartClock, onFrame: onFrame)
            return
        }

        let input = try AVCaptureDeviceInput(device: cameraDevice)
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: sessionQueue)

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480
        for existing in captureSession.inputs { captureSession.removeInput(existing) }
        for existing in captureSession.outputs { captureSession.removeOutput(existing) }
        if captureSession.canAddInput(input) { captureSession.addInput(input) }
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }
        if let connection = output.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            if position == .front, connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }
        captureSession.commitConfiguration()

        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    private var simTimerTask: Task<Void, Never>?

    private func startSimulatorSyntheticCamera(
        sessionStartClock: TimeInterval,
        onFrame: @escaping @Sendable (UIImage, Double) -> Void
    ) {
        simTimerTask?.cancel()
        simTimerTask = Task {
            let renderer = UIGraphicsImageRenderer(size: CGSize(width: 320, height: 240))
            let img = renderer.image { ctx in
                UIColor(white: 0.55, alpha: 1.0).setFill()
                ctx.fill(CGRect(x: 0, y: 0, width: 320, height: 240))
            }
            while !Task.isCancelled {
                let rel = max(0.0, CACurrentMediaTime() - sessionStartClock)
                onFrame(img, rel)
                try? await Task.sleep(for: .milliseconds(83)) // ~12 fps sampled on Simulator
            }
        }
    }

    func stop() {
        simTimerTask?.cancel()
        simTimerTask = nil
        onFrameCallback = nil
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        frameIndex += 1
        // Throttle Vision sampling to ~8 fps on-device to preserve battery/thermals while tracking brief motion
        guard frameIndex % 3 == 1 else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImg = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImg)
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let now = pts.isFinite ? CACurrentMediaTime() : CACurrentMediaTime()
        let rel = max(0.0, now - sessionStartMonotonic)
        onFrameCallback?(uiImage, rel)
    }
}
