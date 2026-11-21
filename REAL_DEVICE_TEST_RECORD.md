# Sherlocked — Real-Device & Ray-Ban Meta Hardware Test Record

**Date:** 2026-11-21  
**Host Environment:** macOS (Apple Silicon), Xcode 27.2 (`27B52`), Swift 6.0  
**Signing Identity:** `Apple Development (Personal Team Provisioning)`  
**Meta Wearables SDK:** `meta-wearables-dat-ios` `1.1.0` (`MWDATCore`, `MWDATCamera`, `MWDATSpeech`, `MWDATMockDevice`)  
**Physical Phone:** iPhone 17 (`iPhone18,3`, USB/CoreDevice verified)  
**Physical Glasses Verified Live:** **Ray-Ban | Meta (Gen 3) — `RB Meta 000N`** linked via `Meta AI` (`com.facebook.stellaapp`, Developer Mode ON)

---

## 1. End-to-End Physical iPhone 17 + Ray-Ban Meta (Gen 3) Live Verification

### Executed Physical Hardware Commands & Outcomes

| Check | Command | Outcome |
| --- | --- | --- |
| Physical arm64 build & signing (including `AppIcon` + `com.meta.ar.wearable`) | `xcodebuild build -project ios/Sherlocked.xcodeproj -scheme Sherlocked -destination 'generic/platform=iOS' -allowProvisioningUpdates` | **PASS** (`** BUILD SUCCEEDED **`) |
| Physical iPhone 17 app installation | `xcrun devicectl device install app --device <DEVICE_UDID> ios/DerivedDataDevice/Build/Products/Debug-iphoneos/Sherlocked.app` | **PASS** (`bundleID: com.tanmaysahu.sherlocked`) |
| Live Ray-Ban Meta Gen 3 (`RB Meta 000N`) Camera Stream + Bluetooth HFP Mic/Speaker + Ready-Cue Gate + Example Lie Video Detection | `xcrun devicectl device process launch --terminate-existing --console --device <DEVICE_UDID> com.tanmaysahu.sherlocked --auto-register-meta` | **PASS** (`reg=Linked with Meta AI`, `devices=["000N (Connected)"]`, `session=started`, `stream=streaming`, `src=glasses`, `camReady=true`, `micReady=true`, `readyCue=true`, `err=none`, `banner=none`) |

### Captured Hardware Console Telemetry (`iPhone 17` + `Ray-Ban | Meta (Gen 3) - RB Meta 000N`)

```text
[#VehicleInfoStatus] For Ray-Ban | Meta (Gen 3) (transport 2)
[#ExternalAccessory] accessoryd sessionResult = 1, useSocketInterace = 1, sessionID = 0x6D
[SHERLOCKED_DEVICE_PROBE] Initial DAT registrationState: Linked with Meta AI
[SHERLOCKED_DEVICE_PROBE] Backend <HTTPS_TUNNEL_ENDPOINT> fixtures loaded: 17
[SHERLOCKED_DEVICE_PROBE] Live iPhone->Backend Example Lie Video (example_live_lie_demo) Guess: #2 conf=0.7151 frames=["frame_00183", "frame_00215"] status=Replay · Game · Initial guess locked (#2)
[SHERLOCKED_DEVICE_PROBE] Live iPhone->Backend Masked Rerun Modality: audio_plus_face_masked_video events=0
[SHERLOCKED_DEVICE_PROBE] t=3s reg=Linked with Meta AI eligible=true link=true devices=["000N (Connected)"] session=started stream=starting src=glasses camReady=false micReady=true readyCue=false audioOut=["RB Meta 000N (BluetoothHFP)"] audioIn=["RB Meta 000N (BluetoothHFP)"] err=none banner=none
[SHERLOCKED_DEVICE_PROBE] t=4s reg=Linked with Meta AI eligible=true link=true devices=["000N (Connected)"] session=started stream=streaming src=glasses camReady=false micReady=true readyCue=false audioOut=["RB Meta 000N (BluetoothHFP)"] audioIn=["RB Meta 000N (BluetoothHFP)"] err=none banner=none
FBBA: creating VNFaceBBoxAligner from VNFaceDetectorRevision2: VNFaceDetectorRevision2
[SHERLOCKED_DEVICE_PROBE] t=5s reg=Linked with Meta AI eligible=true link=true devices=["000N (Connected)"] session=started stream=streaming src=glasses camReady=true micReady=true readyCue=true audioOut=["RB Meta 000N (BluetoothHFP)"] audioIn=["RB Meta 000N (BluetoothHFP)"] err=none banner=none
```

### Root Causes Diagnosed & Fixed on Real Hardware

1. **`Device is not connected` (`PermissionError.noDeviceWithConnection`)**: Calling `wearables.checkPermissionStatus(.camera)` *before* `DeviceSession` reached `.started` caused `MWDATCore` to fail because permission grants are queried across the active glasses session. Starting `DeviceSession` first and waiting for `.started` before checking permissions resolved this completely.
2. **Camera Streaming Without Paid `HotspotConfiguration` Entitlement**: Adding `UISupportedExternalAccessoryProtocols: ["com.meta.ar.wearable"]` and `external-accessory` in `Info.plist` + `.hvc1` (`StreamingResolution.low`, `24 fps`) with a `VideoToolbox` `HEVCFrameDecoder` enabled live camera video streaming from **Ray-Ban | Meta (Gen 3)** over the `EAAccessory` Bluetooth transport on a standard personal Apple development profile.
3. **`AVAudioSession` `OSStatus -50` & `CoreAudio -10868` on Bluetooth HFP (`RB Meta 000N`)**: Replaced `.voiceChat` + `.allowBluetoothA2DP` with `.videoRecording` + `[.allowBluetoothHFP, .mixWithOthers, .defaultToSpeaker]`, selected the `.bluetoothHFP` input (`RB Meta 000N`), and instantiated a fresh `AVAudioEngine` *after* activating the `16 kHz` mono HFP route with a `nonisolated` tap matching `inputNode.inputFormat(forBus: 0)`.

---

## 2. Ray-Ban Meta Capability & Voice Launch Status Matrix

| Capability / Phrase | State | Status | Evidence / Remaining Requirement |
| --- | --- | --- | --- |
| Meta AI Developer Mode Registration (`MWDATCore`) | Physical iPhone 17 + Ray-Ban Meta Gen 3 (`RB Meta 000N`) | **VERIFIED ON HARDWARE** | `reg=Linked with Meta AI` (`RegistrationState.registered`), `devices=["000N (Connected)"]`. |
| Live Ray-Ban Meta Camera Streaming (`24 fps` `.hvc1`) + On-Device `Vision` Face Tracking | Physical iPhone 17 + Ray-Ban Meta Gen 3 (`RB Meta 000N`) | **VERIFIED ON HARDWARE** | `session=started`, `stream=streaming`, `src=glasses`, `camReady=true`, `VNFaceDetectorRevision2` active. |
| Live Ray-Ban Meta Microphone & Private Ear Audio (`BluetoothHFP`) + Sensor-Gated `"Sherlocked is listening"` | Physical iPhone 17 + Ray-Ban Meta Gen 3 (`RB Meta 000N`) | **VERIFIED ON HARDWARE** | `audioIn=["RB Meta 000N (BluetoothHFP)"]`, `audioOut=["RB Meta 000N (BluetoothHFP)"]`; `readyCue` stayed `false` at `t=3s–4s` while camera initialized and flipped `true` at `t=5s` once both `camReady` and `micReady` were `true`. |
| `VoiceInvocationsStream` (`LaunchApp`) SDK integration | Simulator (`MockDeviceKit` `.rayBanMeta`) | **VERIFIED (Simulated)** | `SherlockedSessionTests.testMockRayBanMetaLifecycleAndDisconnectRecovery` passes. |
| Spoken `"Hey Meta, start Sherlocked"` / `"Hey Meta, connect me to Sherlocked"` | Physical Ray-Ban Meta · Unlocked & Locked | **PENDING META WEARABLES DEVELOPER CENTER BETA CHANNEL** | Local Developer Mode (`MetaAppID = 0`) supports registration, live camera streaming, and HFP audio, whereas custom `"Hey Meta"` app-name routing requires publishing `Sherlocked` to a Meta Wearables Developer Center beta release channel. |
