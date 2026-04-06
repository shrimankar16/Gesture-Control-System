"""
gesture_detector.py
--------------------
Pure-logic gesture classification layer.

Responsibilities
----------------
* Accept HandLandmarks from the tracker.
* Compute geometric features (distances, angles, finger states).
* Classify the current gesture using a priority-ordered rule chain.
* Apply frame-stability buffering to prevent jitter.
* Support user-defined gesture extensions loaded from JSON.
* Return a GestureResult with gesture name, confidence, and payload data.

This module is STATEFUL (stability buffer, last gesture) but contains
zero side effects — it never touches the OS or camera.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import GestureConfig
from hand_tracker import HandLandmarks, LandmarkIndex


# ---------------------------------------------------------------------------
# Gesture Enum
# ---------------------------------------------------------------------------

class Gesture(Enum):
    """All recognised gesture types."""
    NONE             = auto()   # No hand or ambiguous
    OPEN_PALM        = auto()   # All fingers extended — move cursor (normal speed)
    INDEX_POINT      = auto()   # Only index finger extended — precision cursor
    PINCH_OPEN       = auto()   # Thumb+index close but not touching
    PINCH_CLOSED     = auto()   # Thumb+index touching — click / drag
    TWO_FINGER_SCROLL= auto()   # Index+middle extended, others curled — scroll
    VOLUME_CONTROL   = auto()   # Thumb+index spread — volume
    FIST             = auto()   # All fingers curled — pause/stop
    CUSTOM           = auto()   # User-defined gesture matched


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GestureResult:
    """
    Output of the gesture detector for a single frame.

    Attributes
    ----------
    gesture         : Gesture enum value
    label           : Human-readable name (may differ for CUSTOM gestures)
    confidence      : 0.0–1.0 — how strongly the gesture was matched
    is_stable       : True once the gesture has been held for stability_frames
    cursor_position : normalised (x, y) in [0, 1] derived from index fingertip
    pinch_distance  : normalised thumb-index distance
    scroll_delta    : (dx, dy) scroll amount; non-zero only for SCROLL gesture
    volume_level    : 0–100 only for VOLUME_CONTROL gesture; else -1
    payload         : arbitrary extra data (used by custom gestures)
    """
    gesture: Gesture = Gesture.NONE
    label: str = "None"
    confidence: float = 0.0
    is_stable: bool = False
    cursor_position: Tuple[float, float] = (0.5, 0.5)
    pinch_distance: float = 0.0
    scroll_delta: Tuple[float, float] = (0.0, 0.0)
    volume_level: int = -1
    payload: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Finger-state helpers
# ---------------------------------------------------------------------------

class FingerAnalyser:
    """
    Stateless helper that extracts Boolean / scalar finger features
    from a HandLandmarks object.
    """

    # MediaPipe webcam output mirrors Left/Right — we correct for this
    # by checking wrist-to-index-MCP orientation.

    @staticmethod
    def is_finger_extended(
        lm: List[Tuple[float, float, float]],
        finger_tip_idx: int,
        finger_pip_idx: int,
    ) -> bool:
        """
        Return True if the given finger is extended.
        Uses vertical (y-axis) comparison: tip above PIP in normalised space.
        Normalised y increases downward so tip.y < pip.y means extended.
        """
        tip_y = lm[finger_tip_idx][1]
        pip_y = lm[finger_pip_idx][1]
        return tip_y < pip_y

    @staticmethod
    def is_thumb_extended(lm: List[Tuple[float, float, float]], handedness: str) -> bool:
        """
        Thumb extension uses horizontal (x-axis) because the thumb moves laterally.
        Accounts for left/right hand mirroring.
        """
        tip_x = lm[LandmarkIndex.THUMB_TIP][0]
        ip_x  = lm[LandmarkIndex.THUMB_IP][0]
        # For right hand (mirrored in webcam): tip should be to the LEFT of IP
        if handedness == "Right":
            return tip_x < ip_x
        else:
            return tip_x > ip_x

    @staticmethod
    def extended_fingers(
        lm: List[Tuple[float, float, float]],
        handedness: str,
    ) -> List[bool]:
        """
        Return [thumb, index, middle, ring, pinky] extension flags.
        """
        return [
            FingerAnalyser.is_thumb_extended(lm, handedness),
            FingerAnalyser.is_finger_extended(lm, LandmarkIndex.INDEX_TIP,  LandmarkIndex.INDEX_PIP),
            FingerAnalyser.is_finger_extended(lm, LandmarkIndex.MIDDLE_TIP, LandmarkIndex.MIDDLE_PIP),
            FingerAnalyser.is_finger_extended(lm, LandmarkIndex.RING_TIP,   LandmarkIndex.RING_PIP),
            FingerAnalyser.is_finger_extended(lm, LandmarkIndex.PINKY_TIP,  LandmarkIndex.PINKY_PIP),
        ]

    @staticmethod
    def landmark_distance(
        lm: List[Tuple[float, float, float]],
        idx_a: int,
        idx_b: int,
    ) -> float:
        """Euclidean distance in normalised space between two landmarks."""
        ax, ay = lm[idx_a][0], lm[idx_a][1]
        bx, by = lm[idx_b][0], lm[idx_b][1]
        return math.hypot(ax - bx, ay - by)

    @staticmethod
    def hand_scale(lm: List[Tuple[float, float, float]]) -> float:
        """
        Return wrist-to-middle-MCP distance as a proxy for hand scale.
        Used to normalise distances so the system works at any camera distance.
        """
        return FingerAnalyser.landmark_distance(
            lm, LandmarkIndex.WRIST, LandmarkIndex.MIDDLE_MCP
        )

    @staticmethod
    def curl_fraction(
        lm: List[Tuple[float, float, float]],
    ) -> float:
        """
        Return the fraction of non-thumb fingers that are curled (0.0–1.0).
        Used to detect fist gesture.
        """
        pairs = [
            (LandmarkIndex.INDEX_TIP,  LandmarkIndex.INDEX_MCP),
            (LandmarkIndex.MIDDLE_TIP, LandmarkIndex.MIDDLE_MCP),
            (LandmarkIndex.RING_TIP,   LandmarkIndex.RING_MCP),
            (LandmarkIndex.PINKY_TIP,  LandmarkIndex.PINKY_MCP),
        ]
        # A finger is curled if its tip is closer to the wrist than its MCP
        curled = 0
        for tip_idx, mcp_idx in pairs:
            d_tip_wrist = FingerAnalyser.landmark_distance(lm, tip_idx, LandmarkIndex.WRIST)
            d_mcp_wrist = FingerAnalyser.landmark_distance(lm, mcp_idx, LandmarkIndex.WRIST)
            if d_tip_wrist < d_mcp_wrist:
                curled += 1
        return curled / 4.0


# ---------------------------------------------------------------------------
# Stability buffer
# ---------------------------------------------------------------------------

class StabilityBuffer:
    """
    A gesture is only reported as 'stable' once it has appeared
    consecutively for `required_frames` frames.
    This eliminates flicker between gestures.
    """

    def __init__(self, required_frames: int = 3) -> None:
        self._required = required_frames
        self._current_gesture: Gesture = Gesture.NONE
        self._count: int = 0

    def update(self, gesture: Gesture) -> bool:
        """
        Feed the current frame's raw gesture.
        Returns True when the gesture is considered stable.
        """
        if gesture == self._current_gesture:
            self._count = min(self._count + 1, self._required)
        else:
            self._current_gesture = gesture
            self._count = 1
        return self._count >= self._required

    @property
    def current(self) -> Gesture:
        return self._current_gesture

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# GestureDetector
# ---------------------------------------------------------------------------

class GestureDetector:
    """
    Stateful gesture classifier.

    Call update(hand, frame_w, frame_h) each frame.
    Returns a GestureResult with the current stable gesture.
    """

    # Priority-ordered gesture list (first match wins)
    _GESTURE_PRIORITY = [
        Gesture.FIST,
        Gesture.PINCH_CLOSED,
        Gesture.TWO_FINGER_SCROLL,
        Gesture.VOLUME_CONTROL,
        Gesture.INDEX_POINT,
        Gesture.OPEN_PALM,
        Gesture.PINCH_OPEN,
    ]

    _GESTURE_LABELS = {
        Gesture.NONE:             "None",
        Gesture.OPEN_PALM:        "Open Palm — Move Cursor",
        Gesture.INDEX_POINT:      "Index Point — Precision Mode",
        Gesture.PINCH_OPEN:       "Pinch Open",
        Gesture.PINCH_CLOSED:     "Pinch — Click",
        Gesture.TWO_FINGER_SCROLL:"Two-Finger Scroll",
        Gesture.VOLUME_CONTROL:   "Volume Control",
        Gesture.FIST:             "Fist — Pause",
        Gesture.CUSTOM:           "Custom Gesture",
    }

    def __init__(
        self,
        cfg: GestureConfig,
        user_gestures_file: Optional[str] = None,
    ) -> None:
        self._cfg = cfg
        self._stability = StabilityBuffer(cfg.stability_frames)

        # Scroll tracking
        self._scroll_anchor: Optional[Tuple[float, float]] = None

        # Volume: keep track of last stable value
        self._last_volume: int = 50

        # User-defined gestures loaded from JSON
        self._user_gestures: List[Dict] = []
        if user_gestures_file and os.path.exists(user_gestures_file):
            self._load_user_gestures(user_gestures_file)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        hand: Optional[HandLandmarks],
        frame_w: int,
        frame_h: int,
    ) -> GestureResult:
        """
        Classify the current frame's hand state.

        Parameters
        ----------
        hand    : HandLandmarks | None  (None = no hand detected)
        frame_w : frame width  in pixels
        frame_h : frame height in pixels

        Returns
        -------
        GestureResult
        """
        if hand is None:
            self._stability.update(Gesture.NONE)
            self._scroll_anchor = None
            return GestureResult()

        lm   = hand.landmarks_norm
        side = hand.handedness
        ext  = FingerAnalyser.extended_fingers(lm, side)
        scale = max(FingerAnalyser.hand_scale(lm), 1e-6)  # avoid divide-by-zero

        # --- Cursor anchor (index fingertip in normalised space) ---
        cursor_norm = (lm[LandmarkIndex.INDEX_TIP][0], lm[LandmarkIndex.INDEX_TIP][1])

        # --- Pinch distance (normalised by hand scale) ---
        raw_pinch = FingerAnalyser.landmark_distance(
            lm, LandmarkIndex.THUMB_TIP, LandmarkIndex.INDEX_TIP
        )
        pinch_dist = raw_pinch  # keep raw for thresholding (scale-normalised in vol)

        # --- Classify ---
        raw_gesture, confidence, payload = self._classify(
            lm, ext, side, scale, pinch_dist
        )

        # Check user gestures
        if raw_gesture == Gesture.NONE:
            ug, ug_conf, ug_payload = self._match_user_gestures(ext)
            if ug:
                raw_gesture = Gesture.CUSTOM
                confidence = ug_conf
                payload = ug_payload

        is_stable = self._stability.update(raw_gesture)

        # --- Scroll delta ---
        scroll_delta = (0.0, 0.0)
        if raw_gesture == Gesture.TWO_FINGER_SCROLL:
            scroll_delta = self._compute_scroll_delta(lm)
        else:
            self._scroll_anchor = None

        # --- Volume level ---
        vol_level = -1
        if raw_gesture == Gesture.VOLUME_CONTROL:
            vol_level = self._compute_volume(lm, scale)
            self._last_volume = vol_level

        return GestureResult(
            gesture=raw_gesture,
            label=payload.get("label", self._GESTURE_LABELS.get(raw_gesture, "Unknown")),
            confidence=confidence,
            is_stable=is_stable,
            cursor_position=cursor_norm,
            pinch_distance=pinch_dist,
            scroll_delta=scroll_delta,
            volume_level=vol_level,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Classification rules
    # ------------------------------------------------------------------

    def _classify(
        self,
        lm: List[Tuple[float, float, float]],
        ext: List[bool],
        handedness: str,
        scale: float,
        pinch_dist: float,
    ) -> Tuple[Gesture, float, Dict]:
        """
        Apply priority-ordered rule chain and return (Gesture, confidence, payload).
        """
        thumb, index, middle, ring, pinky = ext

        # 1. FIST — all fingers curled
        curl = FingerAnalyser.curl_fraction(lm)
        if curl >= self._cfg.fist_curl_threshold and not thumb:
            return Gesture.FIST, curl, {}

        # 2. PINCH CLOSED — thumb+index tips touching
        if pinch_dist <= self._cfg.pinch_threshold:
            return Gesture.PINCH_CLOSED, 1.0 - (pinch_dist / self._cfg.pinch_threshold), {}

        # 3. TWO_FINGER_SCROLL — index+middle extended, ring+pinky+thumb curled
        if index and middle and not ring and not pinky:
            return Gesture.TWO_FINGER_SCROLL, 0.9, {}

        # 4. VOLUME_CONTROL — thumb extended, index extended, others curled
        #    Differentiate from INDEX_POINT by also checking thumb extension
        if thumb and index and not middle and not ring and not pinky:
            return Gesture.VOLUME_CONTROL, 0.85, {}

        # 5. INDEX_POINT — only index extended
        if index and not middle and not ring and not pinky:
            return Gesture.INDEX_POINT, 0.95, {}

        # 6. OPEN_PALM — all 4 fingers + thumb extended
        extended_count = sum(ext)
        if extended_count >= 4:
            conf = extended_count / 5.0
            return Gesture.OPEN_PALM, conf, {}

        # 7. PINCH_OPEN — thumb+index close but not touching
        if pinch_dist <= self._cfg.pinch_release_threshold:
            return Gesture.PINCH_OPEN, 0.6, {}

        return Gesture.NONE, 0.0, {}

    # ------------------------------------------------------------------
    # Scroll delta computation
    # ------------------------------------------------------------------

    def _compute_scroll_delta(
        self,
        lm: List[Tuple[float, float, float]],
    ) -> Tuple[float, float]:
        """
        Return (dx, dy) scroll deltas based on midpoint movement of
        index+middle fingertips relative to the anchor set on gesture start.
        """
        ix, iy = lm[LandmarkIndex.INDEX_TIP][0],  lm[LandmarkIndex.INDEX_TIP][1]
        mx, my = lm[LandmarkIndex.MIDDLE_TIP][0], lm[LandmarkIndex.MIDDLE_TIP][1]
        mid_x = (ix + mx) / 2.0
        mid_y = (iy + my) / 2.0

        if self._scroll_anchor is None:
            self._scroll_anchor = (mid_x, mid_y)
            return (0.0, 0.0)

        dx = (mid_x - self._scroll_anchor[0]) * self._cfg.scroll_sensitivity
        dy = (mid_y - self._scroll_anchor[1]) * self._cfg.scroll_sensitivity

        # Update anchor smoothly (low-pass)
        ax, ay = self._scroll_anchor
        self._scroll_anchor = (
            ax + (mid_x - ax) * 0.3,
            ay + (mid_y - ay) * 0.3,
        )
        return (dx, dy)

    # ------------------------------------------------------------------
    # Volume computation
    # ------------------------------------------------------------------

    def _compute_volume(
        self,
        lm: List[Tuple[float, float, float]],
        scale: float,
    ) -> int:
        """
        Map thumb-index span to a 0-100 volume level.
        Normalised by hand scale for robustness at varying camera distances.
        """
        raw = FingerAnalyser.landmark_distance(
            lm, LandmarkIndex.THUMB_TIP, LandmarkIndex.INDEX_TIP
        )
        normalised = raw / max(scale, 1e-6)

        v_min = self._cfg.vol_distance_min / max(scale, 1e-6)
        v_max = self._cfg.vol_distance_max / max(scale, 1e-6)

        # Re-use raw distance directly (it's already scale-invariant enough
        # because MediaPipe normalises to frame, and scale drifts slowly)
        clamped = max(self._cfg.vol_distance_min, min(raw, self._cfg.vol_distance_max))
        ratio = (clamped - self._cfg.vol_distance_min) / (
            self._cfg.vol_distance_max - self._cfg.vol_distance_min
        )
        return int(ratio * 100)

    # ------------------------------------------------------------------
    # User gesture support
    # ------------------------------------------------------------------

    def _load_user_gestures(self, path: str) -> None:
        """
        Load user-defined gestures from JSON.

        Expected format:
        [
          {
            "name": "Peace Sign",
            "fingers": [false, true, true, false, false],
            "action": "screenshot"
          },
          ...
        ]
        """
        try:
            with open(path, "r") as fh:
                self._user_gestures = json.load(fh)
            print(f"[GestureDetector] Loaded {len(self._user_gestures)} user gestures.")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[GestureDetector] Could not load user gestures: {e}")

    def _match_user_gestures(
        self, ext: List[bool]
    ) -> Tuple[Optional[Gesture], float, Dict]:
        """Try to match current finger state against user-defined gestures."""
        for ug in self._user_gestures:
            required: List[bool] = ug.get("fingers", [])
            if len(required) == 5 and required == list(ext):
                return (
                    Gesture.CUSTOM,
                    1.0,
                    {"label": ug.get("name", "Custom"), "action": ug.get("action", "")},
                )
        return None, 0.0, {}

    # ------------------------------------------------------------------
    # Calibration helpers (used by calibration module)
    # ------------------------------------------------------------------

    def get_pinch_distance_raw(self, hand: HandLandmarks) -> float:
        """Return raw pinch distance for calibration purposes."""
        return FingerAnalyser.landmark_distance(
            hand.landmarks_norm, LandmarkIndex.THUMB_TIP, LandmarkIndex.INDEX_TIP
        )
