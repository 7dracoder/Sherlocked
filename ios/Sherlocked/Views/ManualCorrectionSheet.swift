import SwiftUI

struct ManualCorrectionSheet: View {
    @ObservedObject var coordinator: SessionCoordinator
    @Environment(\.dismiss) private var dismiss
    @State private var selectedRevealStory: Int = 2
    @State private var editableStories: [StorySpanDTO] = []

    var body: some View {
        NavigationStack {
            Form {
                Section("Final Reveal (Kept Outside Guesser Inputs)") {
                    Picker("Actual Lie Story", selection: $selectedRevealStory) {
                        Text("Story #1").tag(1)
                        Text("Story #2").tag(2)
                        Text("Story #3").tag(3)
                    }
                    .pickerStyle(.segmented)
                }

                if !editableStories.isEmpty {
                    Section("Manual Story Boundary Correction") {
                        ForEach($editableStories) { $story in
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Story #\(story.story_number)")
                                    .font(.caption.weight(.bold))
                                TextField("Story transcript", text: $story.transcript, axis: .vertical)
                                    .font(.footnote)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Manual Correction")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        Task {
                            await coordinator.submitManualReveal(
                                selectedRevealStory,
                                correctedStories: editableStories.isEmpty ? nil : editableStories
                            )
                            dismiss()
                        }
                    }
                    .accessibilityIdentifier("save_manual_correction_button")
                }
            }
            .onAppear {
                editableStories = coordinator.latestResponse?.stories ?? []
                if let rev = coordinator.latestResponse?.revealed_lie_story_number {
                    selectedRevealStory = rev
                }
            }
        }
    }
}
