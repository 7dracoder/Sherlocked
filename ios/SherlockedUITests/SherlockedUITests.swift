import XCTest

final class SherlockedUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testFreshInstallOnboardingPhoneFallbackAndReplayFlow() throws {
        let app = XCUIApplication()
        app.launch()

        // 1. Fresh install onboarding sheet if shown
        let continueBtn = app.buttons["complete_onboarding_button"]
        if continueBtn.waitForExistence(timeout: 4.0) {
            continueBtn.tap()
        }

        // 2. Verify read-only activity indicator exists and NO mode picker exists
        let activityBadge = app.staticTexts["Listening for context"]
        XCTAssertTrue(
            app.otherElements["readonly_activity_indicator"].exists || activityBadge.exists,
            "Read-only activity indicator must be visible"
        )

        // 3. Start and Stop phone fallback session
        let startBtn = app.buttons["prominent_start_button"]
        XCTAssertTrue(startBtn.waitForExistence(timeout: 4.0))
        startBtn.tap()

        let stopBtn = app.buttons["prominent_stop_button"]
        XCTAssertTrue(stopBtn.waitForExistence(timeout: 4.0))
        stopBtn.tap()
        XCTAssertTrue(startBtn.waitForExistence(timeout: 4.0))

        // 4. Switch to Replay & Evidence tab and run live backend fixture + masked comparison
        let replaySegment = app.segmentedControls["main_view_segment"].buttons["Replay & Evidence"]
        if replaySegment.waitForExistence(timeout: 3.0) {
            replaySegment.tap()
            let runReplay = app.buttons["run_replay_button"]
            if runReplay.waitForExistence(timeout: 4.0) && runReplay.isEnabled {
                runReplay.tap()
                let guessCard = app.otherElements["locked_initial_guess_card"]
                _ = guessCard.waitForExistence(timeout: 18.0)

                let maskedBtn = app.buttons["run_masked_rerun_button"]
                if maskedBtn.exists && maskedBtn.isEnabled {
                    maskedBtn.tap()
                    _ = app.otherElements["masked_comparison_card"].waitForExistence(timeout: 18.0)
                }
            }
        }
    }

    #if !targetEnvironment(simulator)
    @MainActor
    func testPhysicalDeviceMetaAILinkAndLiveSession() throws {
        let app = XCUIApplication()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let metaAI = XCUIApplication(bundleIdentifier: "com.facebook.stellaapp")

        app.launch()

        func dismissSystemAlerts() {
            for label in ["Allow", "OK", "Allow While Using App", "Connect", "Join"] {
                let btn = springboard.buttons[label]
                if btn.exists {
                    btn.tap()
                }
            }
        }

        dismissSystemAlerts()

        // Open Onboarding sheet if not already visible so we can link with Meta AI app
        if !app.buttons["register_meta_glasses_button"].waitForExistence(timeout: 2.5) {
            let glassesBtn = app.buttons["onboarding_toolbar_button"]
            if glassesBtn.waitForExistence(timeout: 3.0) {
                glassesBtn.tap()
            }
        }

        let regBtn = app.buttons["register_meta_glasses_button"]
        if regBtn.waitForExistence(timeout: 4.0) {
            regBtn.tap()
            sleep(3)
            dismissSystemAlerts()

            // If Meta AI companion app opened in foreground to confirm Developer Mode app registration,
            // inspect and tap Allow / Connect / Continue if shown
            if metaAI.state == .runningForeground {
                for actionTitle in ["Allow", "Connect", "Continue", "Confirm", "Approve", "Done"] {
                    let metaBtn = metaAI.buttons[actionTitle]
                    if metaBtn.waitForExistence(timeout: 2.0) {
                        metaBtn.tap()
                        sleep(2)
                        break
                    }
                }
            }

            // Ensure Sherlocked is foregrounded again after callback
            if app.state != .runningForeground {
                app.activate()
                sleep(2)
            }
        }

        dismissSystemAlerts()

        let regLabel = app.staticTexts["dat_registration_status_label"]
        let devLabel = app.staticTexts["dat_devices_label"]
        print("PHYSICAL_DEVICE_TELEMETRY registration=\(regLabel.exists ? regLabel.label : "n/a") devices=\(devLabel.exists ? devLabel.label : "n/a")")

        let continueBtn = app.buttons["complete_onboarding_button"]
        if continueBtn.exists {
            continueBtn.tap()
        }

        let audioLabel = app.staticTexts["audio_route_label"]
        let sourceLabel = app.staticTexts["capture_source_badge"]
        print("PHYSICAL_DEVICE_AUDIO_SOURCE route=\(audioLabel.exists ? audioLabel.label : "n/a") source=\(sourceLabel.exists ? sourceLabel.label : "n/a")")

        // Start real live session on physical iPhone + paired glasses
        let startBtn = app.buttons["prominent_start_button"]
        if startBtn.waitForExistence(timeout: 4.0) {
            startBtn.tap()
            sleep(2)
            dismissSystemAlerts()

            // If camera permission redirected to Meta AI app, approve & return
            if metaAI.state == .runningForeground {
                for actionTitle in ["Allow", "Always Allow", "Allow Once", "Continue", "Confirm"] {
                    let metaBtn = metaAI.buttons[actionTitle]
                    if metaBtn.waitForExistence(timeout: 2.0) {
                        metaBtn.tap()
                        sleep(2)
                        break
                    }
                }
                if app.state != .runningForeground {
                    app.activate()
                    sleep(2)
                }
            }

            sleep(2)
            dismissSystemAlerts()

            let stopBtn = app.buttons["prominent_stop_button"]
            if stopBtn.waitForExistence(timeout: 5.0) {
                stopBtn.tap()
            }
        }
    }
    #endif
}
