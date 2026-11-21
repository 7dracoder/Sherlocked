import Foundation
import UIKit
import Vision

final class OnDeviceFaceTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var frameCounter: Int = 0
    private var recentTimestamps: [Double] = []
    private var lastYawDeg: Double?
    private var lastTimestamp: Double?
    private var storedImagesByFrameID: [String: Data] = [:]

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        frameCounter = 0
        recentTimestamps.removeAll()
        lastYawDeg = nil
        lastTimestamp = nil
        storedImagesByFrameID.removeAll()
    }

    func imageData(for frameID: String) -> Data? {
        lock.lock()
        defer { lock.unlock() }
        return storedImagesByFrameID[frameID]
    }

    var effectiveFPS: Double {
        lock.lock()
        defer { lock.unlock() }
        guard recentTimestamps.count >= 2 else { return 24.0 }
        let span = (recentTimestamps.last ?? 0) - (recentTimestamps.first ?? 0)
        guard span > 0.01 else { return 24.0 }
        return Double(recentTimestamps.count - 1) / span
    }

    func processFrame(image: UIImage, sessionTimestampSec: Double) -> FrameSampleDTO {
        let currentIdx: Int
        let prevT: Double?
        let prevYaw: Double?
        lock.lock()
        frameCounter += 1
        currentIdx = frameCounter
        prevT = lastTimestamp
        prevYaw = lastYawDeg
        lastTimestamp = sessionTimestampSec
        recentTimestamps.append(sessionTimestampSec)
        if recentTimestamps.count > 30 {
            recentTimestamps.removeFirst(recentTimestamps.count - 30)
        }
        lock.unlock()

        let frameID = String(format: "frame_%05d", currentIdx)
        guard let cgImage = image.cgImage else {
            return FrameSampleDTO(
                frame_id: frameID,
                timestamp_sec: sessionTimestampSec,
                face_visible: false,
                usable: false,
                speaker_attributed: false,
                faces_detected: 0,
                quality_flags: ["frame_loss"]
            )
        }

        let lum = Self.estimateLuminance(cgImage: cgImage)
        var flags: [String] = []
        if lum < 42.0 {
            flags.append("low_light")
        }
        if let prevT, (sessionTimestampSec - prevT) > 0.45 {
            flags.append("frame_loss")
        }

        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        try? handler.perform([request])
        let observations = request.results ?? []

        if observations.isEmpty {
            return FrameSampleDTO(
                frame_id: frameID,
                timestamp_sec: sessionTimestampSec,
                face_visible: false,
                usable: false,
                speaker_attributed: false,
                faces_detected: 0,
                luminance: lum,
                quality_flags: flags
            )
        }

        if observations.count > 1 {
            flags.append("listener_in_frame")
        }

        let primary = observations.max(by: {
            ($0.boundingBox.width * $0.boundingBox.height) < ($1.boundingBox.width * $1.boundingBox.height)
        })!

        let yawRad = primary.yaw?.doubleValue ?? 0.0
        let yawDeg = yawRad * 180.0 / .pi
        var turnRate = 0.0
        if let prevYaw, let prevT {
            let dt = max(0.005, sessionTimestampSec - prevT)
            turnRate = abs(yawDeg - prevYaw) / dt
        }
        lock.lock()
        lastYawDeg = yawDeg
        lock.unlock()

        if turnRate > 65.0 || abs(yawDeg) > 48.0 {
            flags.append("fast_head_turn")
        }

        let metrics = Self.computeLandmarkChannels(face: primary)
        let cropJPEG = Self.cropFaceJPEG(cgImage: cgImage, boundingBox: primary.boundingBox)
        if let cropJPEG {
            lock.lock()
            if storedImagesByFrameID.count < 400 {
                storedImagesByFrameID[frameID] = cropJPEG
            }
            lock.unlock()
        }

        let usable = flags.isEmpty
        return FrameSampleDTO(
            frame_id: frameID,
            timestamp_sec: (sessionTimestampSec * 1000).rounded() / 1000,
            crop_phase: "sampled",
            jpeg_base64: (currentIdx % 6 == 0) ? cropJPEG?.base64EncodedString() : nil,
            face_visible: true,
            usable: usable,
            speaker_attributed: usable && observations.count == 1,
            faces_detected: observations.count,
            luminance: (lum * 100).rounded() / 100,
            head_yaw_deg: (yawDeg * 100).rounded() / 100,
            head_pitch_deg: 0.0,
            head_turn_deg_per_sec: (turnRate * 100).rounded() / 100,
            motion_score: metrics.motionScore,
            blendshapes: metrics.channels,
            quality_flags: flags
        )
    }

    private static func computeLandmarkChannels(
        face: VNFaceObservation
    ) -> (channels: [String: Double], motionScore: Double) {
        guard let lm = face.landmarks else {
            return ([:], 0.0)
        }
        var lipPress = 0.05
        var jawOpen = 0.05
        if let outerLips = lm.outerLips?.normalizedPoints, outerLips.count >= 6,
           let innerLips = lm.innerLips?.normalizedPoints, innerLips.count >= 4 {
            let xs = outerLips.map(\.x)
            let ys = outerLips.map(\.y)
            let width = max(0.01, (xs.max() ?? 0.2) - (xs.min() ?? 0.0))
            let outerH = max(0.0, (ys.max() ?? 0.1) - (ys.min() ?? 0.0))
            let innerYs = innerLips.map(\.y)
            let innerH = max(0.0, (innerYs.max() ?? 0.05) - (innerYs.min() ?? 0.0))
            let mar = innerH / width
            jawOpen = min(1.0, Double(mar * 2.6))
            let outerRatio = outerH / width
            lipPress = min(1.0, max(0.0, Double((0.22 - outerRatio) * 4.2)))
        }

        var eyeBlink = 0.05
        if let leftEye = lm.leftEye?.normalizedPoints, leftEye.count >= 4 {
            let xs = leftEye.map(\.x)
            let ys = leftEye.map(\.y)
            let ear = ((ys.max() ?? 0.1) - (ys.min() ?? 0.0)) / max(0.01, (xs.max() ?? 0.2) - (xs.min() ?? 0.0))
            eyeBlink = min(1.0, max(0.0, Double((0.28 - ear) * 4.5)))
        }

        var gazeShift = 0.04
        if let leftPupil = lm.leftPupil?.normalizedPoints.first,
           let leftEye = lm.leftEye?.normalizedPoints, !leftEye.isEmpty {
            let midX = leftEye.map(\.x).reduce(0, +) / CGFloat(leftEye.count)
            let spanX = max(0.01, (leftEye.map(\.x).max() ?? 0.2) - (leftEye.map(\.x).min() ?? 0.0))
            gazeShift = min(1.0, Double(abs(leftPupil.x - midX) / spanX * 2.8))
        }

        var browFurrow = 0.05
        if let leftBrow = lm.leftEyebrow?.normalizedPoints,
           let leftEye = lm.leftEye?.normalizedPoints,
           !leftBrow.isEmpty, !leftEye.isEmpty {
            let browY = leftBrow.map(\.y).reduce(0, +) / CGFloat(leftBrow.count)
            let eyeY = leftEye.map(\.y).reduce(0, +) / CGFloat(leftEye.count)
            let gap = Double(browY - eyeY)
            browFurrow = min(1.0, max(0.0, (0.11 - gap) * 6.0))
        }

        let channels: [String: Double] = [
            "lip_press": (lipPress * 1000).rounded() / 1000,
            "eye_blink": (eyeBlink * 1000).rounded() / 1000,
            "gaze_shift": (gazeShift * 1000).rounded() / 1000,
            "brow_furrow": (browFurrow * 1000).rounded() / 1000,
            "mouth_asymmetry": 0.02,
            "jaw_open": (jawOpen * 1000).rounded() / 1000,
        ]
        let motion = 0.34 * lipPress + 0.26 * browFurrow + 0.24 * gazeShift + 0.16 * eyeBlink
        return (channels, (motion * 1000).rounded() / 1000)
    }

    private static func cropFaceJPEG(cgImage: CGImage, boundingBox: CGRect) -> Data? {
        let w = CGFloat(cgImage.width)
        let h = CGFloat(cgImage.height)
        // Vision boundingBox origin is bottom-left normalized
        let rect = CGRect(
            x: max(0, (boundingBox.minX - 0.12) * w),
            y: max(0, (1.0 - boundingBox.maxY - 0.14) * h),
            width: min(w, (boundingBox.width + 0.24) * w),
            height: min(h, (boundingBox.height + 0.28) * h)
        )
        guard let cropped = cgImage.cropping(to: rect) else { return nil }
        return UIImage(cgImage: cropped).jpegData(compressionQuality: 0.78)
    }

    private static func estimateLuminance(cgImage: CGImage) -> Double {
        let width = 16
        let height = 16
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        guard let ctx = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return 128.0
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        var sum = 0.0
        for i in stride(from: 0, to: pixels.count, by: 4) {
            let r = Double(pixels[i])
            let g = Double(pixels[i + 1])
            let b = Double(pixels[i + 2])
            sum += 0.299 * r + 0.587 * g + 0.114 * b
        }
        return sum / Double(width * height)
    }
}
