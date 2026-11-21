import AVKit
import SwiftUI

struct ReplayStudioView: View {
    @ObservedObject var coordinator: SessionCoordinator
    @State private var player: AVPlayer?
    @State private var includeFollowupInReplay: Bool = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Consented Replay Benchmark Fixtures")
                        .font(.headline)
                    Text("Explicitly identified replay clips. Replay calls run silently by default and link cited frame IDs directly to matching audio timestamps.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if coordinator.availableFixtures.isEmpty {
                    HStack {
                        Text("Backend fixtures not loaded yet.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Reload") {
                            Task { await coordinator.loadFixtures() }
                        }
                        .buttonStyle(.bordered)
                    }
                } else {
                    Menu {
                        ForEach(coordinator.availableFixtures) { item in
                            Button(item.title) {
                                coordinator.selectedReplayFixture = item
                            }
                        }
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(coordinator.selectedReplayFixture?.title ?? "Select Fixture")
                                    .font(.subheadline.weight(.semibold))
                                Text(coordinator.selectedReplayFixture?.description ?? "")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                            Spacer()
                            Image(systemName: "chevron.up.chevron.down")
                        }
                        .padding(12)
                        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                    }
                    .accessibilityIdentifier("fixture_picker_menu")
                }

                Toggle("Include Pre-Reveal Follow-Up Round", isOn: $includeFollowupInReplay)
                    .font(.caption)

                HStack(spacing: 10) {
                    Button {
                        guard let fixture = coordinator.selectedReplayFixture else { return }
                        Task {
                            await coordinator.runFixtureReplay(fixture, includeFollowup: includeFollowupInReplay)
                            refreshPlayer()
                        }
                    } label: {
                        Label("Run Replay", systemImage: "play.circle.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.cyan)
                    .disabled(coordinator.selectedReplayFixture == nil || coordinator.isAnalyzing)
                    .accessibilityIdentifier("run_replay_button")

                    Button {
                        Task {
                            await coordinator.runFaceMaskedRerunForSelectedFixture()
                            refreshPlayer()
                        }
                    } label: {
                        Label("Face-Masked Rerun", systemImage: "theatermask.and.paintbrush")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(coordinator.selectedReplayFixture == nil || coordinator.isAnalyzing)
                    .accessibilityIdentifier("run_masked_rerun_button")
                }

                if let player {
                    VideoPlayer(player: player)
                        .frame(height: 220)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                        .accessibilityIdentifier("replay_video_player")
                }

                if let resp = coordinator.latestResponse {
                    EvidenceResultSummaryView(
                        response: resp,
                        onSeekToFrame: { frameID in
                            seekPlayer(toFrameID: frameID, fps: resp.effective_fps)
                        },
                        onOpenManualCorrection: {
                            coordinator.showManualCorrectionSheet = true
                        }
                    )
                }

                if let masked = coordinator.maskedComparisonResponse {
                    VStack(alignment: .leading, spacing: 6) {
                        Label("Face-Masked Rerun Result (Audio Preserved)", systemImage: "checkmark.shield")
                            .font(.subheadline.weight(.bold))
                            .foregroundStyle(.orange)
                        Text("Modality: \(masked.ModalityUsed.rawValue) · Facial events used: \(masked.candidate_events.count)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if let g = masked.initial_guess {
                            Text("Masked Guess: Story #\(g.chosen_story_number) (\(Int(g.confidence * 100))% conf)")
                                .font(.footnote.weight(.semibold))
                        } else if let d = masked.deal_output {
                            Text("Masked Deal Action: \(d.action) (\(d.silence_reason ?? ""))")
                                .font(.footnote)
                        }
                    }
                    .padding(12)
                    .background(Color.orange.opacity(0.14), in: RoundedRectangle(cornerRadius: 14))
                    .accessibilityIdentifier("masked_comparison_card")
                }
            }
            .padding(16)
        }
    }

    private func refreshPlayer() {
        guard let url = coordinator.activeReplayVideoURL else { return }
        player = AVPlayer(url: url)
    }

    private func seekPlayer(toFrameID frameID: String, fps: Double) {
        // Parse frame_00142 -> index 142 -> timestamp = 142 / fps
        let digits = frameID.filter(\.isNumber)
        guard let idx = Int(digits), let player else { return }
        let safeFPS = fps > 1.0 ? fps : 24.0
        let sec = Double(idx) / safeFPS
        let target = CMTime(seconds: sec, preferredTimescale: 600)
        player.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
        player.play()
    }
}
