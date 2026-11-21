import SwiftUI

struct OnboardingSheetView: View {
    @ObservedObject var coordinator: SessionCoordinator
    @State private var permissionsGrantedSummary: String = ""
    @State private var backendURLDraft: String = ""
    @State private var backendSaveNote: String = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack(spacing: 12) {
                        Image("LogoMark")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 48, height: 48)
                            .clipShape(RoundedRectangle(cornerRadius: 11))
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Sherlocked Setup")
                                .font(.title3.weight(.bold))
                            Text("Connect Ray-Ban Meta glasses or use your iPhone camera and earbuds. Sherlocked infers the activity automatically from speech.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Label("1. Ray-Ban Meta Glasses (Developer Mode)", systemImage: "eyeglasses")
                            .font(.headline)
                        Text("Primary voice command: \"Hey Meta, start Sherlocked.\"\nUnfold & put on your glasses so Bluetooth audio & session link wake up.")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        HStack(spacing: 8) {
                            Button {
                                Task {
                                    await coordinator.glassesCoordinator.startRegistration()
                                }
                            } label: {
                                Label(
                                    coordinator.glassesCoordinator.registrationState == .registered ? "Linked with Meta AI" : "Link with Meta AI App",
                                    systemImage: coordinator.glassesCoordinator.registrationState == .registered ? "checkmark.seal.fill" : "link"
                                )
                                .font(.subheadline.weight(.semibold))
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(coordinator.glassesCoordinator.registrationState == .registered ? .green : .blue)
                            .accessibilityIdentifier("register_meta_glasses_button")
                        }

                        if coordinator.glassesCoordinator.registrationState == .registered {
                            Button {
                                Task {
                                    await coordinator.glassesCoordinator.requestGlassesCameraPermissionInteractive()
                                }
                            } label: {
                                Label("Authorize Glasses Camera in Meta AI", systemImage: "camera.viewfinder")
                                    .font(.caption.weight(.semibold))
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .tint(.cyan)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Meta AI Status: \(coordinator.glassesCoordinator.registrationStateDescription)")
                                .font(.caption2.monospaced())
                                .foregroundStyle(coordinator.glassesCoordinator.registrationState == .registered ? .mint : .secondary)
                                .accessibilityIdentifier("dat_registration_status_label")
                            Text("Glasses Roster: \(coordinator.glassesCoordinator.connectedDeviceNames.isEmpty ? "Searching..." : coordinator.glassesCoordinator.connectedDeviceNames.joined(separator: ", "))")
                                .font(.caption2.monospaced())
                                .accessibilityIdentifier("dat_devices_label")
                            if let err = coordinator.glassesCoordinator.lastWearablesError {
                                Text(err)
                                    .font(.caption2)
                                    .foregroundStyle(.orange)
                                    .accessibilityIdentifier("dat_error_label")
                            }
                        }
                    }
                    .padding(14)
                    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))

                    VStack(alignment: .leading, spacing: 10) {
                        Label("2. Camera, Mic & Speech Permissions", systemImage: "mic.and.signal.meter")
                            .font(.headline)
                        Text("Required for iPhone fallback capture and baseline audio alignment. Ready cue plays only once live frames and audio are flowing.")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Button {
                            Task {
                                let res = await coordinator.audioEngine.requestPermissions()
                                permissionsGrantedSummary = "Mic: \(res.mic ? "Granted" : "Denied") · Speech: \(res.speech ? "Granted" : "Denied")"
                            }
                        } label: {
                            Text("Grant iPhone Audio & Speech Access")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)

                        if !permissionsGrantedSummary.isEmpty {
                            Text(permissionsGrantedSummary)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.mint)
                        }
                    }
                    .padding(14)
                    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))

                    VStack(alignment: .leading, spacing: 10) {
                        Label("3. Server Backend Endpoint", systemImage: "network")
                            .font(.headline)
                        Text("All Gemini API credentials stay strictly server-side. Point your iPhone to your local LAN or HTTPS tunnel endpoint.")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        TextField("https://...", text: $backendURLDraft)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .font(.caption.monospaced())
                            .padding(10)
                            .background(Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                            .accessibilityIdentifier("backend_url_textfield")

                        Button {
                            Task {
                                await coordinator.updateBackendEndpoint(backendURLDraft)
                                backendSaveNote = "Connected (\(coordinator.availableFixtures.count) fixtures)"
                            }
                        } label: {
                            Text("Save & Test Backend URL")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("save_backend_url_button")

                        if !backendSaveNote.isEmpty {
                            Text(backendSaveNote)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.mint)
                        }
                    }
                    .padding(14)
                    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))

                    Button {
                        coordinator.completeOnboarding()
                    } label: {
                        Text("Continue to Sherlocked")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.cyan)
                    .accessibilityIdentifier("complete_onboarding_button")
                }
                .padding(20)
            }
            .navigationTitle("Welcome to Sherlocked")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                backendURLDraft = coordinator.configuredBackendURL
            }
        }
    }
}
