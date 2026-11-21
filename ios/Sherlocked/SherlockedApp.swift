import MWDATCore
import SwiftUI

@main
struct SherlockedApp: App {
    @StateObject private var coordinator: SessionCoordinator

    init() {
        // Initialize Meta Wearables DAT SDK once at launch per official getting-started skill
        try? Wearables.configure()
        _coordinator = StateObject(wrappedValue: SessionCoordinator())
    }

    var body: some Scene {
        WindowGroup {
            MainSessionView(coordinator: coordinator)
                .preferredColorScheme(.dark)
                .onOpenURL { url in
                    print("[SHERLOCKED_OPEN_URL] incoming=\(url.absoluteString)")
                    Task {
                        // Handle Meta AI companion app registration & permission callbacks
                        await coordinator.glassesCoordinator.handleIncomingURL(url)
                        // Also support sherlocked://start voice/deep-link launch
                        if url.host == "start" {
                            await coordinator.startOrReturnToSession(trigger: .voiceCommand)
                        }
                    }
                }
        }
    }
}
