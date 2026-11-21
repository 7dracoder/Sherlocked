import AVKit
import SwiftUI

struct MainSessionView: View {
    @ObservedObject var coordinator: SessionCoordinator
    @AppStorage(MicCapturePreference.phoneMicKey) private var preferPhoneMic = true
    @State private var selectedTab: Int = 0

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                topTelemetryHeader
                Picker("View", selection: $selectedTab) {
                    Text("Live Session").tag(0)
                    Text("Replay & Evidence").tag(1)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .accessibilityIdentifier("main_view_segment")

                if selectedTab == 0 {
                    liveSessionContent
                } else {
                    ReplayStudioView(coordinator: coordinator)
                }
            }
            .background(Color(red: 0.06, green: 0.07, blue: 0.10).ignoresSafeArea())
            .navigationTitle("Sherlocked")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        coordinator.showOnboarding = true
                    } label: {
                        Image(systemName: "eyeglasses")
                    }
                    .accessibilityIdentifier("onboarding_toolbar_button")
                }
                ToolbarItem(placement: .principal) {
                    HStack(spacing: 7) {
                        Image("LogoMark")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 24, height: 24)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        Text("Sherlocked")
                            .font(.headline.weight(.semibold))
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        coordinator.showMockGlassesSheet = true
                    } label: {
                        Image(systemName: "wrench.and.screwdriver")
                    }
                    .accessibilityIdentifier("mock_glasses_toolbar_button")
                }
            }
            .sheet(isPresented: $coordinator.showOnboarding) {
                OnboardingSheetView(coordinator: coordinator)
            }
            .sheet(isPresented: $coordinator.showManualCorrectionSheet) {
                ManualCorrectionSheet(coordinator: coordinator)
            }
            .sheet(isPresented: $coordinator.showMockGlassesSheet) {
                MockGlassesSheetView(coordinator: coordinator)
            }
            .task {
                await coordinator.loadFixtures()
            }
        }
    }

    // MARK: - Top Telemetry Header (Capture Source, Recording State, Audio Route, Read-Only Activity)

    private var topTelemetryHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Circle()
                    .fill(coordinator.isSessionActive ? Color.red : Color.gray)
                    .frame(width: 10, height: 10)
                Text(coordinator.isSessionActive ? "RECORDING LIVE" : "STANDBY")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(coordinator.isSessionActive ? .red : .secondary)
                    .accessibilityIdentifier("recording_state_label")

                Spacer()

                Text(coordinator.captureSource.displayLabel)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.09), in: Capsule())
                    .accessibilityIdentifier("capture_source_badge")
            }

            HStack(spacing: 8) {
                Label(coordinator.audioEngine.currentRoute.displayLabel, systemImage: coordinator.audioEngine.currentRoute.isPrivate ? "ear.badge.checkmark" : "eye.circle")
                    .font(.caption2)
                    .foregroundStyle(coordinator.audioEngine.currentRoute.isPrivate ? .green : .orange)
                    .accessibilityIdentifier("audio_route_label")
                PrivateAudioRoutePicker()
                    .frame(width: 36, height: 36)
                    .accessibilityLabel("Choose audio output")

                Spacer()

                // Read-only automatic activity indicator (NO mode picker!)
                Label(coordinator.detectedActivity.displayTitle, systemImage: coordinator.detectedActivity.systemIcon)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.cyan)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.cyan.opacity(0.14), in: Capsule())
                    .accessibilityIdentifier("readonly_activity_indicator")
            }

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Image(systemName: "mic.fill")
                        .foregroundStyle(coordinator.audioEngine.isMicrophoneFlowing ? Color.green : Color.orange)
                    Text(coordinator.audioEngine.micStatusLine)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                        .minimumScaleFactor(0.75)
                        .accessibilityIdentifier("mic_source_label")
                    Spacer(minLength: 4)
                    Toggle(isOn: Binding(
                        get: { preferPhoneMic },
                        set: { newValue in
                            preferPhoneMic = newValue
                            coordinator.setPreferPhoneMic(newValue)
                        }
                    )) {
                        Text("Phone mic")
                            .font(.caption2)
                    }
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .accessibilityLabel("Use iPhone mic for front speaker")
                    .accessibilityIdentifier("live_phone_mic_toggle")
                }
                ProgressView(value: coordinator.audioEngine.micLevel)
                    .tint(coordinator.audioEngine.micLevel > 0.05 ? .green : .gray)
                    .accessibilityIdentifier("mic_level_meter")
                HStack {
                    Label(coordinator.audioEngine.asrProviderLabel, systemImage: "waveform")
                        .font(.caption2)
                        .foregroundStyle(.mint)
                        .accessibilityIdentifier("asr_provider_label")
                    Spacer()
                    Text("Mode: \(preferPhoneMic ? "Phone mic" : "Glasses mic")")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(preferPhoneMic ? .cyan : .orange)
                        .accessibilityIdentifier("mic_mode_label")
                }
                Text(coordinator.faceStatusLine)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("face_status_label")
            }

            Text(coordinator.statusLabel)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .accessibilityIdentifier("session_status_label")
        }
        .padding(14)
        .background(Color.white.opacity(0.05))
    }

    // MARK: - Live Session Content

    private var liveSessionContent: some View {
        ScrollView {
            VStack(spacing: 14) {
                if coordinator.requiresForegroundUnlockPrompt {
                    bannerCard(
                        title: "Unlock & Foreground Required",
                        message: "iOS cannot start camera and microphone capture while locked or backgrounded. Unlock your iPhone and tap Start.",
                        color: .orange,
                        identifier: "unlock_foreground_banner"
                    )
                }

                if let onScreenCue = coordinator.onScreenCueBanner {
                    bannerCard(
                        title: coordinator.audioEngine.currentRoute.isPrivate ? "Private cue" : "Visual Cue (No Private Audio Route)",
                        message: onScreenCue,
                        color: .mint,
                        identifier: "onscreen_cue_banner"
                    )
                }

                if let recovery = coordinator.recoveryBannerMessage {
                    bannerCard(
                        title: "Session Notice",
                        message: recovery,
                        color: .yellow,
                        identifier: "recovery_notice_banner",
                        onDismiss: { coordinator.recoveryBannerMessage = nil }
                    )
                }

                cameraPreviewCard
                primaryControlsCard

                if let resp = coordinator.latestResponse {
                    EvidenceResultSummaryView(
                        response: resp,
                        onOpenManualCorrection: { coordinator.showManualCorrectionSheet = true },
                        onRequestFollowups: {
                            coordinator.scheduleTurnAnalysis(requestFollowups: true, isFollowupRound: false)
                        }
                    )
                }
            }
            .padding(16)
        }
    }

    private var cameraPreviewCard: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.white.opacity(0.06))
                .frame(height: 190)

            if let img = coordinator.latestPreviewImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFill()
                    .frame(height: 190)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
            } else {
                VStack(spacing: 8) {
                    Image(systemName: coordinator.glassesCoordinator.hasEligibleGlasses ? "eyeglasses" : "iphone.gen3")
                        .font(.system(size: 34))
                        .foregroundStyle(.secondary)
                    Text(
                        coordinator.glassesCoordinator.hasEligibleGlasses
                        ? "Say \"Hey Meta, start Sherlocked\" or tap Start"
                        : "Glasses unavailable · iPhone Camera + Mic fallback ready"
                    )
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 20)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            HStack(spacing: 10) {
                Text("Cam: \(coordinator.hasCameraDeliveredFirstFrame ? "Ready" : "Waiting")")
                Text("Mic: \(coordinator.hasMicDeliveredFirstSample ? "Ready" : "Waiting")")
                if coordinator.effectiveFPS > 0 {
                    Text(String(format: "%.1f fps", coordinator.effectiveFPS))
                }
                Text(coordinator.faceStatusLine)
                    .lineLimit(1)
            }
            .font(.caption2.monospaced())
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial, in: Capsule())
            .padding(10)
        }
    }

    private var primaryControlsCard: some View {
        VStack(spacing: 10) {
            if coordinator.isSessionActive {
                Button(role: .destructive) {
                    coordinator.stopSession()
                } label: {
                    Label("Stop Session", systemImage: "stop.circle.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .accessibilityIdentifier("prominent_stop_button")

                if coordinator.isAnalyzing {
                    Button("Cancel In-Flight Analysis") {
                        coordinator.cancelInFlightAnalysis()
                    }
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .accessibilityIdentifier("cancel_analysis_button")
                }
            } else {
                Button {
                    Task {
                        await coordinator.startOrReturnToSession(trigger: .startButton)
                    }
                } label: {
                    Label(
                        coordinator.glassesCoordinator.hasEligibleGlasses ? "Start with Ray-Ban Meta" : "Start Session (iPhone Fallback)",
                        systemImage: "play.fill"
                    )
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .tint(.cyan)
                .accessibilityIdentifier("prominent_start_button")
            }

            if coordinator.isSessionActive || !coordinator.audioEngine.livePartialTranscript.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Live Transcript · \(coordinator.audioEngine.micSourceLabel)")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.secondary)
                    Text(coordinator.audioEngine.inputName.isEmpty ? "Input route: waiting…" : "Input route: \(coordinator.audioEngine.inputName)")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                    Text(coordinator.audioEngine.livePartialTranscript.isEmpty ? "Waiting for speech… (speak into the phone)" : coordinator.audioEngine.livePartialTranscript)
                        .font(.footnote)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .accessibilityIdentifier("live_transcript_text")
                    if coordinator.isAnalyzing {
                        Text("Scoring stories…")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }
                    if let guess = coordinator.latestResponse?.initial_guess {
                        Text("Model answer: Story #\(guess.chosen_story_number) is the lie (\(Int((guess.confidence * 100).rounded()))%)")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.mint)
                            .accessibilityIdentifier("model_answer_label")
                    }
                }
                .padding(10)
                .background(Color.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private func bannerCard(
        title: String,
        message: String,
        color: Color,
        identifier: String,
        onDismiss: (() -> Void)? = nil
    ) -> some View {
        HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(color)
                Text(message)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white)
            }
            Spacer(minLength: 4)
            if let onDismiss {
                Button(action: onDismiss) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(color.opacity(0.16), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(color.opacity(0.45), lineWidth: 1))
        .accessibilityIdentifier(identifier)
    }
}

struct EvidenceResultSummaryView: View {
    let response: SessionAnalyzeResponseDTO
    var onSeekToFrame: ((String) -> Void)? = nil
    var onOpenManualCorrection: (() -> Void)? = nil
    var onRequestFollowups: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(response.activity.displayTitle)
                    .font(.headline)
                Spacer()
                Text(response.execution_mode == "live_gemini" ? "Live Gemini" : "Local Multimodal")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.purple.opacity(0.22), in: Capsule())
            }

            // Soft wearable flags (frame_loss, fast_head_turn, …) are normal — never alarm on them.
            // Only surface a hint when the server explicitly asked for a retry.
            if response.retry_requested, let reason = response.retry_reason, !reason.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(reason)
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                .accessibilityIdentifier("discarded_quality_flags_label")
            }

            switch response.activity {
            case .game:
                gameSection
            case .deal:
                dealSection
            case .practice:
                practiceSection
            case .unclear:
                Text("Ordinary or incomplete conversation — Sherlocked stays silent until an activity is established.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 16))
        .accessibilityIdentifier("evidence_summary_card")
    }

    private var gameSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(response.stories) { story in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Story #\(story.story_number)")
                            .font(.subheadline.weight(.bold))
                        Text(story.status.uppercased())
                            .font(.caption2)
                            .foregroundStyle(story.status == "complete" ? .green : .orange)
                        Spacer()
                        Text(String(format: "%.1fs–%.1fs", story.start_sec, story.end_sec))
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    Text(story.transcript)
                        .font(.footnote)
                    if !story.observations.isEmpty {
                        Text(story.observations.joined(separator: " · "))
                            .font(.caption2)
                            .foregroundStyle(.cyan)
                    }
                }
                .padding(10)
                .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
            }

            if let guess = response.initial_guess {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Label("Locked Initial Guess: #\(guess.chosen_story_number)", systemImage: "lock.fill")
                            .font(.subheadline.weight(.bold))
                            .foregroundStyle(.mint)
                        Spacer()
                        Text(String(format: "%.0f%% conf", guess.confidence * 100))
                            .font(.caption.monospaced())
                    }
                    Text("\"\(guess.spoken_whisper)\"")
                        .font(.footnote.italic())

                    if guess.exploratory_disclosure {
                        Text("Exploratory guess: camera evidence weighted conservatively pending broad human validation.")
                            .font(.caption2)
                            .foregroundStyle(.yellow)
                    }
                    if guess.sparse_visual_disclosure {
                        Text("Sparse visual evidence: speaker maintained a steady/straight face; choice relies primarily on speech & prosody.")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }

                    citedFramesRow(guess.cited_frame_ids)
                }
                .padding(10)
                .background(Color.mint.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityIdentifier("locked_initial_guess_card")
            }

            if let updated = response.updated_guess {
                VStack(alignment: .leading, spacing: 4) {
                    Label("Post-Follow-Up Updated Guess: #\(updated.chosen_story_number)", systemImage: "arrow.triangle.2.circlepath")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(.cyan)
                    Text("Initial locked guess remains #\(response.initial_guess?.chosen_story_number ?? 0).")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(10)
                .background(Color.cyan.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
            }

            if let reveal = response.revealed_lie_story_number {
                Text("Recorded Reveal: Story #\(reveal) (kept outside guesser inputs)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.green)
            }

            if !response.followup_questions.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Comparable Follow-Up Questions (1 per story)")
                        .font(.caption.weight(.bold))
                    ForEach(response.followup_questions, id: \.self) { q in
                        Text("• \(q)")
                            .font(.caption)
                    }
                }
            }

            HStack(spacing: 10) {
                if let onRequestFollowups, response.initial_guess != nil {
                    Button("Ask Follow-Ups") {
                        onRequestFollowups()
                    }
                    .buttonStyle(.bordered)
                    .font(.caption)
                }
                if let onOpenManualCorrection {
                    Button("Manual Story / Reveal Edit") {
                        onOpenManualCorrection()
                    }
                    .buttonStyle(.bordered)
                    .font(.caption)
                    .accessibilityIdentifier("manual_correction_button")
                }
            }
        }
    }

    private var dealSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let deal = response.deal_output {
                if deal.action == "whisper_question", let whisper = deal.spoken_whisper {
                    Label("Actionable Follow-Up", systemImage: "ear.and.waveform")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(.mint)
                    Text("\"\(whisper)\"")
                        .font(.body.weight(.semibold))
                    if let obs = deal.observable_evidence {
                        Text(obs)
                            .font(.caption)
                            .foregroundStyle(.cyan)
                    }
                    citedFramesRow(deal.cited_frame_ids)
                } else {
                    Label("Staying Silent", systemImage: "speaker.slash")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text(deal.silence_reason ?? "No usable facial observation coincided with a specific commitment.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var practiceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let prac = response.practice_output {
                if prac.exercise_complete {
                    Label("Post-Exercise Replay & Coaching", systemImage: "play.rectangle.on.rectangle")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(.cyan)
                    if let summary = prac.observable_moment_summary {
                        Text(summary)
                            .font(.footnote)
                    }
                    if let suggestion = prac.suggested_followup {
                        Text("Suggested adjustment: \(suggestion)")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.mint)
                    }
                    citedFramesRow(prac.cited_frame_ids)
                } else {
                    Text("Exercise in progress — holding feedback until rehearsal concludes.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func citedFramesRow(_ frameIDs: [String]) -> some View {
        if !frameIDs.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(frameIDs, id: \.self) { fid in
                        Button {
                            onSeekToFrame?(fid)
                        } label: {
                            Label(fid, systemImage: "film")
                                .font(.caption2.monospaced())
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.white.opacity(0.10), in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }
}

private struct PrivateAudioRoutePicker: UIViewRepresentable {
    func makeUIView(context: Context) -> AVRoutePickerView {
        let view = AVRoutePickerView()
        view.tintColor = .systemTeal
        return view
    }

    func updateUIView(_ view: AVRoutePickerView, context: Context) {}
}
