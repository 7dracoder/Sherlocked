import MWDATCore
import MWDATMockDevice
import UIKit
import XCTest
@testable import Sherlocked

@MainActor
final class SherlockedSessionTests: XCTestCase {
    override func tearDown() async throws {
        await MockDeviceKit.shared.disable()
        try await super.tearDown()
    }

    func testPartialRecognitionReplacesWordsWithinOneClause() {
        let engine = AudioAndSpeechEngine()
        func snapshot(_ words: [String]) -> AudioAndSpeechEngine.SpeechSnapshot {
            .init(formattedString: words.joined(separator: " "), isFinal: false,
                  segments: words.enumerated().map { index, word in
                      .init(substring: word, timestamp: Double(index) * 0.2, duration: 0.2, confidence: 0)
                  })
        }
        engine.consumeSpeechSnapshot(snapshot(["I", "ski"]))
        engine.consumeSpeechSnapshot(snapshot(["I", "swim"]))
        XCTAssertEqual(engine.utterances.first?.text, "I swim")
        engine.consumeSpeechSnapshot(snapshot(["I", "swim", "daily"]))
        XCTAssertEqual(engine.utterances.first?.text, "I swim daily")
    }

    func testLiveTurnsYieldOneGameCueWithoutPrivateRoute() async throws {
        let coordinator = SessionCoordinator()
        await coordinator.startOrReturnToSession(forcePhoneFallback: true)
        defer { coordinator.stopSession() }
        let statements = ["Two truths and a lie.", "One I once lived in London.",
                          "Two I ran a marathon last June.", "Three I have climbed the tallest mountain."]
        for (index, words) in statements.enumerated() {
            coordinator.injectSimulatedMicrophoneSampleAndUtterance(words, startSec: Double(index * 3), endSec: Double(index * 3 + 2))
        }
        for _ in 0..<100 where coordinator.latestResponse?.initial_guess == nil {
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertNotNil(coordinator.latestResponse?.initial_guess)
        XCTAssertTrue(coordinator.onScreenCueBanner?.contains("The one starting:") == true)
    }

    func testReadyCueFiresOnlyAfterBothCameraAndMicFlow() async throws {
        UserDefaults.standard.set(true, forKey: "sherlocked_onboarding_completed")
        let coordinator = SessionCoordinator()

        await coordinator.startOrReturnToSession(trigger: .startButton, forcePhoneFallback: true)
        XCTAssertTrue(coordinator.isSessionActive)

        // Wait briefly for Simulator synthetic camera frame; before mic sample arrives, ready cue must remain false if mic hasn't fired
        if !coordinator.hasMicDeliveredFirstSample {
            XCTAssertFalse(coordinator.hasPlayedReadyCue, "Ready cue must not play before microphone delivers samples.")
        }

        // Deliver first camera frame + mic sample
        try await Task.sleep(for: .milliseconds(150))
        coordinator.injectSimulatedMicrophoneSampleAndUtterance(
            "Testing live audio readiness.",
            startSec: 0.2,
            endSec: 1.4
        )
        XCTAssertTrue(coordinator.hasCameraDeliveredFirstFrame)
        XCTAssertTrue(coordinator.hasMicDeliveredFirstSample)
        XCTAssertTrue(coordinator.hasPlayedReadyCue, "Ready cue must fire once both camera and microphone are flowing.")

        coordinator.stopSession()
        XCTAssertFalse(coordinator.isSessionActive, "Stop must leave capture inactive.")
    }

    func testRepeatedLaunchReturnsToActiveSessionWithoutDuplicateStart() async throws {
        UserDefaults.standard.set(true, forKey: "sherlocked_onboarding_completed")
        let coordinator = SessionCoordinator()

        await coordinator.startOrReturnToSession(trigger: .startButton, forcePhoneFallback: true)
        let originalSessionID = coordinator.sessionID
        XCTAssertEqual(coordinator.repeatedLaunchCount, 0)

        // Simulate repeated voice launch while session is active
        await coordinator.startOrReturnToSession(trigger: .voiceCommand)
        XCTAssertEqual(coordinator.sessionID, originalSessionID)
        XCTAssertEqual(coordinator.repeatedLaunchCount, 1)

        coordinator.stopSession()
        XCTAssertFalse(coordinator.isSessionActive)
    }

    func testMockRayBanMetaLifecycleAndDisconnectRecovery() async throws {
        UserDefaults.standard.set(true, forKey: "sherlocked_onboarding_completed")
        try? Wearables.configure()
        let coordinator = SessionCoordinator()

        try coordinator.glassesCoordinator.pairSimulatedRayBanMeta()
        XCTAssertTrue(coordinator.glassesCoordinator.hasEligibleGlasses)

        // Trigger simulated voice launch
        coordinator.glassesCoordinator.simulateHeyMetaVoiceLaunch()
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertTrue(coordinator.isSessionActive)

        // Simulate hinge fold disconnect mid-session
        coordinator.glassesCoordinator.simulateGlassesFoldDisconnect()
        try await Task.sleep(for: .milliseconds(250))

        coordinator.stopSession()
        XCTAssertFalse(coordinator.isSessionActive)
        await coordinator.glassesCoordinator.disableSimulatedGlasses()
    }

    func testOnDeviceFaceTrackerFlagsLowLightAndFrameLoss() {
        let tracker = OnDeviceFaceTracker()
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 120, height: 120))
        let darkImage = renderer.image { ctx in
            UIColor(white: 0.04, alpha: 1.0).setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: 120, height: 120))
        }

        let s1 = tracker.processFrame(image: darkImage, sessionTimestampSec: 0.0)
        XCTAssertFalse(s1.usable)
        XCTAssertTrue(s1.quality_flags.contains("low_light"))

        let s2 = tracker.processFrame(image: darkImage, sessionTimestampSec: 1.2)
        XCTAssertFalse(s2.usable)
        XCTAssertTrue(s2.quality_flags.contains("frame_loss"))
    }

    func testLiveBackendReplayAndMaskedComparison() async throws {
        UserDefaults.standard.set(true, forKey: "sherlocked_onboarding_completed")
        let coordinator = SessionCoordinator()
        await coordinator.loadFixtures()
        guard let gameFixture = coordinator.availableFixtures.first(where: { $0.fixture_id == "game_natural_lie2" }) else {
            XCTFail("Expected game_natural_lie2 fixture from live backend")
            return
        }

        await coordinator.runFixtureReplay(gameFixture)
        XCTAssertEqual(coordinator.detectedActivity, .game)
        XCTAssertEqual(coordinator.latestResponse?.initial_guess?.chosen_story_number, 2)
        XCTAssertEqual(coordinator.latestResponse?.should_speak_privately, false, "Replay calls must remain silent by default.")

        await coordinator.runFaceMaskedRerunForSelectedFixture()
        XCTAssertNotNil(coordinator.maskedComparisonResponse)
        XCTAssertEqual(coordinator.maskedComparisonResponse?.ModalityUsed, .audioPlusFaceMaskedVideo)
        XCTAssertEqual(coordinator.maskedComparisonResponse?.candidate_events.count, 0)
    }
}
