import SwiftUI

struct MockGlassesSheetView: View {
    @ObservedObject var coordinator: SessionCoordinator
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Ray-Ban Meta Hardware Simulation (MockDeviceKit)") {
                    Button("Pair & Don Simulated Ray-Ban Meta") {
                        try? coordinator.glassesCoordinator.pairSimulatedRayBanMeta()
                    }
                    .accessibilityIdentifier("pair_mock_glasses_button")

                    Button("Simulate \"Hey Meta, start Sherlocked\" (LaunchApp)") {
                        coordinator.glassesCoordinator.simulateHeyMetaVoiceLaunch()
                        dismiss()
                    }
                    .accessibilityIdentifier("simulate_voice_launch_button")

                    Button("Simulate Hinge Fold / Bluetooth Disconnect") {
                        coordinator.glassesCoordinator.simulateGlassesFoldDisconnect()
                        dismiss()
                    }
                    .accessibilityIdentifier("simulate_disconnect_button")

                    Toggle("Glasses PCM mic (DAT) for front speaker", isOn: $coordinator.useSynchronizedGlassesAudio)

                    if coordinator.glassesCoordinator.isMockModeEnabled {
                        Button("Unpair Simulated Glasses", role: .destructive) {
                            Task {
                                await coordinator.glassesCoordinator.disableSimulatedGlasses()
                            }
                        }
                    }
                }

                Section("Wearables Telemetry") {
                    LabeledContent("Eligible Glasses", value: coordinator.glassesCoordinator.hasEligibleGlasses ? "Yes" : "No")
                    LabeledContent("Voice Channel", value: coordinator.glassesCoordinator.voiceChannelActive ? "Listening" : "Inactive")
                    LabeledContent("Repeated Launches Handled", value: "\(coordinator.repeatedLaunchCount)")
                    if let err = coordinator.glassesCoordinator.lastWearablesError {
                        Text(err)
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
            }
            .navigationTitle("Glasses & Voice Test")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
