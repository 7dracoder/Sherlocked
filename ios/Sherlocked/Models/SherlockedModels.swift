import Foundation

enum ActivityKind: String, Codable, CaseIterable, Sendable {
    case game = "two_truths_and_a_lie"
    case deal = "deal"
    case practice = "practice"
    case unclear = "unclear"

    var displayTitle: String {
        switch self {
        case .game: return "Two Truths & a Lie"
        case .deal: return "Deal Discussion"
        case .practice: return "Practice & Coaching"
        case .unclear: return "Listening for context"
        }
    }

    var systemIcon: String {
        switch self {
        case .game: return "sparkles.rectangle.stack"
        case .deal: return "handshake"
        case .practice: return "person.wave.2"
        case .unclear: return "waveform.badge.magnifyingglass"
        }
    }
}

enum CaptureSourceKind: String, Codable, Sendable {
    case glasses = "glasses"
    case iphone = "iphone"
    case replay = "replay"

    var displayLabel: String {
        switch self {
        case .glasses: return "Ray-Ban Meta Glasses"
        case .iphone: return "iPhone Camera + Mic"
        case .replay: return "Replay Session"
        }
    }
}

enum AudioRouteKind: String, Codable, Sendable {
    case glassesPrivate = "glasses_private"
    case earbudsPrivate = "earbuds_private"
    case speakerPublic = "speaker_public"
    case none = "none"

    var isPrivate: Bool {
        self == .glassesPrivate || self == .earbudsPrivate
    }

    var displayLabel: String {
        switch self {
        case .glassesPrivate: return "Private · Glasses Audio"
        case .earbudsPrivate: return "Private · Earbuds / Bluetooth"
        case .speakerPublic: return "Speaker · Visual Cue Only"
        case .none: return "No Audio · Visual Cue Only"
        }
    }
}

enum ModalityMode: String, Codable, Sendable {
    case audioPlusVisibleFace = "audio_plus_visible_face"
    case audioPlusFaceMaskedVideo = "audio_plus_face_masked_video"
    case audioOnly = "audio_only"
    case transcriptOnly = "transcript_only"
}

struct WordTimestampDTO: Codable, Hashable, Sendable {
    var word: String
    var start_sec: Double
    var end_sec: Double
    var confidence: Double = 0.95
    var speaker_id: String = "speaker_primary"
}

struct UtteranceSegmentDTO: Codable, Identifiable, Hashable, Sendable {
    var segment_id: String
    var speaker_id: String = "speaker_primary"
    var text: String
    var start_sec: Double
    var end_sec: Double
    var overlapping_speech: Bool = false
    var mean_pitch_hz: Double? = nil
    var pitch_std_hz: Double? = nil
    var rms_energy: Double? = nil
    var pause_before_sec: Double = 0.0
    var speech_rate_wps: Double? = nil
    var words: [WordTimestampDTO] = []

    var id: String { segment_id }
}

struct FrameSampleDTO: Codable, Identifiable, Sendable {
    var frame_id: String
    var timestamp_sec: Double
    var crop_phase: String = "sampled"
    var jpeg_base64: String? = nil
    var face_visible: Bool = true
    var usable: Bool = true
    var speaker_attributed: Bool = true
    var faces_detected: Int = 1
    var luminance: Double = 128.0
    var head_yaw_deg: Double = 0.0
    var head_pitch_deg: Double = 0.0
    var head_turn_deg_per_sec: Double = 0.0
    var eyewear_detected: Bool = false
    var eyewear_occludes_eyes: Bool = false
    var motion_score: Double = 0.0
    var blendshapes: [String: Double] = [:]
    var quality_flags: [String] = []

    var id: String { frame_id }
}

struct CandidateMotionIntervalDTO: Codable, Identifiable, Sendable {
    var event_id: String
    var start_sec: Double
    var end_sec: Double
    var apex_sec: Double
    var apex_frame_id: String
    var before_frame_id: String
    var during_frame_id: String
    var after_frame_id: String
    var observable_action: String
    var peak_delta: Double
    var coinciding_word: String?
    var usable: Bool

    var id: String { event_id }
}

struct StorySpanDTO: Codable, Identifiable, Sendable {
    var story_number: Int
    var speaker_id: String = "speaker_primary"
    var status: String = "complete"
    var start_sec: Double
    var end_sec: Double
    var transcript: String
    var cited_frame_ids: [String] = []
    var observations: [String] = []
    var sparse_visual_evidence: Bool = false

    var id: Int { story_number }
}

struct GameGuessRecordDTO: Codable, Sendable {
    var chosen_story_number: Int
    var spoken_whisper: String
    var probabilities: [String: Double]
    var confidence: Double
    var exploratory_disclosure: Bool
    var sparse_visual_disclosure: Bool
    var cited_frame_ids: [String]
    var observed_actions: [String]
    var comparison_rationale: String
    var locked_at_sec: Double
}

struct DealCueOutputDTO: Codable, Sendable {
    var action: String
    var identified_term: String?
    var spoken_whisper: String?
    var confidence: Double
    var cited_frame_ids: [String]
    var observable_evidence: String?
    var silence_reason: String?
}

struct PracticeFeedbackOutputDTO: Codable, Sendable {
    var exercise_complete: Bool
    var replay_start_sec: Double?
    var replay_end_sec: Double?
    var cited_frame_ids: [String]
    var observable_moment_summary: String?
    var suggested_followup: String?
    var confidence: Double
}

struct SessionAnalyzeRequestDTO: Codable, Sendable {
    var session_id: String
    var turn_id: String?
    var capture_source: CaptureSourceKind
    var audio_route: AudioRouteKind
    var modality_mode: ModalityMode = .audioPlusVisibleFace
    var is_replay: Bool = false
    var is_evaluation: Bool = false
    var request_followup_questions: Bool = false
    var is_followup_round: Bool = false
    var effective_fps: Double = 24.0
    var audio_video_drift_ms: Double = 0.0
    var utterances: [UtteranceSegmentDTO] = []
    var frames: [FrameSampleDTO] = []
}

struct SessionAnalyzeResponseDTO: Codable, Sendable {
    var result_id: String
    var session_id: String
    var activity_epoch: Int
    var activity: ActivityKind
    var status_label: String
    var execution_mode: String
    var models_used: [String: String]
    var already_delivered: Bool
    var should_speak_privately: Bool
    var present_cue_on_screen: Bool
    var spoken_cue_text: String?
    var retry_requested: Bool
    var retry_reason: String?
    var stories: [StorySpanDTO]
    var initial_guess: GameGuessRecordDTO?
    var followup_questions: [String]
    var updated_guess: GameGuessRecordDTO?
    var revealed_lie_story_number: Int?
    var deal_output: DealCueOutputDTO?
    var practice_output: PracticeFeedbackOutputDTO?
    var candidate_events: [CandidateMotionIntervalDTO]
    var discarded_quality_flags: [String]
    var effective_fps: Double
    var ModalityUsed: ModalityMode
}

struct ManualCorrectionRequestDTO: Codable, Sendable {
    var session_id: String
    var corrected_stories: [StorySpanDTO]?
    var revealed_lie_story_number: Int?
}

struct ReplayFixtureItemDTO: Codable, Identifiable, Sendable {
    var fixture_id: String
    var category: String
    var title: String
    var description: String
    var video_relpath: String
    var duration_sec: Double
    var fps: Int
    var straight_face_subtype: String?
    var explicitly_identified_fixture: Bool

    var id: String { fixture_id }
}

struct FixturesListResponseDTO: Codable, Sendable {
    var fixtures: [ReplayFixtureItemDTO]
    var is_synthetic_consented_fixture_set: Bool?
}
