"""
hand_tracker.py
---------------
Thin, high-performance wrapper around MediaPipe Hands.

Responsibilities
----------------
* Initialise and own the MediaPipe Hands solution.
* Accept raw BGR frames from OpenCV.
* Return normalised landmark data + handedness metadata.
* Expose drawing helpers so the OSD layer can render landmarks.
* Handle the RGB conversion internally (callers pass BGR frames directly).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import TrackingConfig

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HandLandmarks:
    """
    Processed data for a single detected hand.

    Attributes
    ----------
    landmarks_norm : list of (x, y, z)  — normalised [0,1] co-ordinates
    landmarks_px   : list of (x, y)     — pixel co-ordinates on the frame
    handedness     : "Left" | "Right"   — as reported by MediaPipe
    confidence     : float              — detection confidence score
    bounding_box   : (x_min, y_min, x_max, y_max) in pixel space
    """
    landmarks_norm: List[Tuple[float, float, float]]
    landmarks_px: List[Tuple[int, int]]
    handedness: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# MediaPipe landmark indices (handy constants)
# ---------------------------------------------------------------------------

class LandmarkIndex:
    """Named indices for MediaPipe's 21-point hand model."""
    WRIST = 0
    THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
    INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

    # Convenience groups
    FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    FINGER_PIPS = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


# ---------------------------------------------------------------------------
# HandTracker class
# ---------------------------------------------------------------------------

class HandTracker:
    """
    Wraps MediaPipe Hands and exposes a clean process() API.

    Usage
    -----
    tracker = HandTracker(config)
    for frame in camera_stream():
        results = tracker.process(frame)
        for hand in results:
            print(hand.handedness, hand.landmarks_norm)
    tracker.close()
    """

    def __init__(self, cfg: TrackingConfig) -> None:
        self._cfg = cfg
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

        # Frame dimensions — set on first call to process()
        self._frame_h: int = 0
        self._frame_w: int = 0

        # Cache last valid results for frames where detection drops out briefly
        self._last_results: List[HandLandmarks] = []
        self._frames_since_detection: int = 0
        self._MAX_CACHE_FRAMES = 5  # Drop cache after this many missed frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> List[HandLandmarks]:
        """
        Process a single BGR frame and return a list of HandLandmarks.

        Parameters
        ----------
        bgr_frame : numpy array (H, W, 3) in BGR format

        Returns
        -------
        List[HandLandmarks] — may be empty if no hands detected
        """
        self._frame_h, self._frame_w = bgr_frame.shape[:2]

        # Convert BGR → RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Mark as not writeable to avoid unnecessary copy inside MediaPipe
        rgb_frame.flags.writeable = False
        mp_result = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not mp_result.multi_hand_landmarks:
            self._frames_since_detection += 1
            if self._frames_since_detection > self._MAX_CACHE_FRAMES:
                self._last_results = []
            return self._last_results  # return stale cache or empty

        self._frames_since_detection = 0
        detected: List[HandLandmarks] = []

        for hand_lms, hand_info in zip(
            mp_result.multi_hand_landmarks,
            mp_result.multi_handedness,
        ):
            label = hand_info.classification[0].label        # "Left" | "Right"
            score = hand_info.classification[0].score

            # Skip non-dominant hand if configured
            if (
                self._cfg.dominant_hand != "Any"
                and label != self._cfg.dominant_hand
                and len(mp_result.multi_hand_landmarks) > 1
            ):
                continue

            norms, pixels = self._extract_landmarks(hand_lms)
            bbox = self._compute_bbox(pixels)

            detected.append(HandLandmarks(
                landmarks_norm=norms,
                landmarks_px=pixels,
                handedness=label,
                confidence=score,
                bounding_box=bbox,
            ))

        self._last_results = detected
        return detected

    def draw_landmarks(
        self,
        bgr_frame: np.ndarray,
        hand: HandLandmarks,
        landmark_colour: Tuple[int, int, int] = (0, 255, 120),
        connection_colour: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """
        Draw 21 landmarks and skeleton connections directly onto bgr_frame.
        This mutates the frame in-place for efficiency.
        """
        h, w = bgr_frame.shape[:2]

        # Draw connections first (under the dots)
        connections = self._mp_hands.HAND_CONNECTIONS
        for start_idx, end_idx in connections:
            x1, y1 = hand.landmarks_px[start_idx]
            x2, y2 = hand.landmarks_px[end_idx]
            cv2.line(bgr_frame, (x1, y1), (x2, y2), connection_colour, 2, cv2.LINE_AA)

        # Draw landmark dots
        for (px, py) in hand.landmarks_px:
            cv2.circle(bgr_frame, (px, py), 5, landmark_colour, -1, cv2.LINE_AA)
            cv2.circle(bgr_frame, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)  # outline

    def draw_bounding_box(
        self,
        bgr_frame: np.ndarray,
        hand: HandLandmarks,
        colour: Tuple[int, int, int] = (0, 200, 255),
        padding: int = 18,
    ) -> None:
        """Draw a padded bounding box around the detected hand."""
        x_min, y_min, x_max, y_max = hand.bounding_box
        h, w = bgr_frame.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        cv2.rectangle(bgr_frame, (x_min, y_min), (x_max, y_max), colour, 2, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_landmarks(
        self,
        mp_hand_landmarks,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
        """Convert raw MediaPipe landmark objects into plain Python tuples."""
        norms: List[Tuple[float, float, float]] = []
        pixels: List[Tuple[int, int]] = []

        for lm in mp_hand_landmarks.landmark:
            norms.append((lm.x, lm.y, lm.z))
            px = int(lm.x * self._frame_w)
            py = int(lm.y * self._frame_h)
            pixels.append((px, py))

        return norms, pixels

    def _compute_bbox(
        self, pixels: List[Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        """Compute axis-aligned bounding box from pixel landmark list."""
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        return (min(xs), min(ys), max(xs), max(ys))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of last processed frame."""
        return (self._frame_w, self._frame_h)
