"""
config.py
---------
Central configuration for the Gesture Control System.
All tunable parameters live here — edit this file to customize behaviour
without touching any logic modules.
"""

from dataclasses import dataclass, field
from typing import Tuple
import json
import os

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Webcam capture settings."""
    device_index: int = 0           # Camera device index (0 = default webcam)
    width: int = 1280               # Capture width in pixels
    height: int = 720               # Capture height in pixels
    fps_target: int = 30            # Target capture FPS


@dataclass
class TrackingConfig:
    """MediaPipe hand tracking parameters."""
    max_num_hands: int = 2                # Maximum hands to track
    model_complexity: int = 1             # 0=lite, 1=full — higher is slower but more accurate
    min_detection_confidence: float = 0.75
    min_tracking_confidence: float = 0.75
    dominant_hand: str = "Right"          # "Right" | "Left" | "Any"


@dataclass
class GestureConfig:
    """Gesture detection thresholds and cooldown values."""
    # ---- Pinch / click ----
    pinch_threshold: float = 0.055        # Normalised distance for pinch open→closed
    pinch_release_threshold: float = 0.08 # Normalised distance for pinch closed→open

    # ---- Fist ----
    fist_curl_threshold: float = 0.85     # Fraction of fingers curled to count as fist

    # ---- Scroll ----
    scroll_threshold: float = 0.06        # Two-finger pinch distance threshold
    scroll_sensitivity: float = 20.0      # Pixels scrolled per unit of hand movement

    # ---- Volume / Brightness ----
    vol_distance_min: float = 0.03        # Thumb-index distance that maps to 0 %
    vol_distance_max: float = 0.35        # Thumb-index distance that maps to 100 %

    # ---- General ----
    # Cooldown (seconds) between gesture state transitions to prevent jitter
    click_cooldown: float = 0.35
    scroll_cooldown: float = 0.05
    volume_cooldown: float = 0.05
    gesture_cooldown: float = 0.2         # Generic action cooldown

    # Number of consecutive frames a gesture must be stable before confirming
    stability_frames: int = 3


@dataclass
class CursorConfig:
    """Cursor movement and smoothing settings."""
    # Exponential moving average alpha for smoothing (0 < α ≤ 1, lower = smoother)
    smooth_alpha: float = 0.25
    # Additional Kalman-style velocity dampening
    velocity_dampening: float = 0.6
    # Fraction of frame edges to ignore (helps avoid edge jitter)
    margin: float = 0.10
    # Sensitive mode speed multiplier (index-finger-only mode)
    precision_speed: float = 0.4
    # Normal speed multiplier
    normal_speed: float = 1.0


@dataclass
class DisplayConfig:
    """On-screen display settings."""
    show_fps: bool = True
    show_gesture_name: bool = True
    show_landmarks: bool = True
    show_bounding_box: bool = True
    show_hand_label: bool = True
    # Landmark drawing colours (BGR)
    landmark_colour: Tuple[int, int, int] = (0, 255, 120)
    connection_colour: Tuple[int, int, int] = (255, 255, 255)
    bbox_colour: Tuple[int, int, int] = (0, 200, 255)
    text_colour: Tuple[int, int, int] = (255, 255, 255)
    fps_colour: Tuple[int, int, int] = (0, 255, 0)
    font_scale: float = 0.7
    thickness: int = 2
    # HUD background opacity (0.0–1.0)
    hud_alpha: float = 0.45


# ---------------------------------------------------------------------------
# Master config object
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Aggregate config passed throughout the application."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    # ---- Misc ----
    debug_mode: bool = False              # Extra console logging
    quit_key: str = "q"                   # Key to press to exit
    calibration_mode: bool = False        # Launch calibration wizard at startup
    user_gestures_file: str = "user_gestures.json"  # Custom gesture definitions


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "gesture_config.json")


def load_config(path: str = _CONFIG_FILE) -> AppConfig:
    """
    Load configuration from a JSON file.
    Falls back to defaults if the file does not exist or is malformed.
    """
    cfg = AppConfig()
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        # Shallow merge of each sub-config section
        for section_name, section_cls in [
            ("camera", CameraConfig),
            ("tracking", TrackingConfig),
            ("gesture", GestureConfig),
            ("cursor", CursorConfig),
            ("display", DisplayConfig),
        ]:
            if section_name in data:
                section_obj = getattr(cfg, section_name)
                for k, v in data[section_name].items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
        # Top-level flags
        for k in ("debug_mode", "quit_key", "calibration_mode", "user_gestures_file"):
            if k in data:
                setattr(cfg, k, data[k])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"[Config] Warning: could not parse config file ({exc}). Using defaults.")
    return cfg


def save_config(cfg: AppConfig, path: str = _CONFIG_FILE) -> None:
    """Persist the current AppConfig to JSON."""
    import dataclasses
    data = {
        "camera": dataclasses.asdict(cfg.camera),
        "tracking": dataclasses.asdict(cfg.tracking),
        "gesture": dataclasses.asdict(cfg.gesture),
        "cursor": dataclasses.asdict(cfg.cursor),
        "display": dataclasses.asdict(cfg.display),
        "debug_mode": cfg.debug_mode,
        "quit_key": cfg.quit_key,
        "calibration_mode": cfg.calibration_mode,
        "user_gestures_file": cfg.user_gestures_file,
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[Config] Saved to {path}")
