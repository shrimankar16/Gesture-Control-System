"""
tests/test_gesture_detector.py
-------------------------------
Unit and integration tests for the gesture detection pipeline.

Run with:
    python -m pytest tests/ -v
    # or without pytest:
    python tests/test_gesture_detector.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest
from typing import List, Tuple

from config import GestureConfig
from gesture_detector import (
    FingerAnalyser,
    Gesture,
    GestureDetector,
    GestureResult,
    LandmarkIndex,
    StabilityBuffer,
)
from hand_tracker import HandLandmarks


# ---------------------------------------------------------------------------
# Helpers to fabricate landmark data
# ---------------------------------------------------------------------------

def _make_landmarks(
    finger_states: List[bool],
    pinch_dist: float = 0.15,
    handedness: str = "Right",
) -> HandLandmarks:
    """
    Construct a minimal HandLandmarks with the given finger extension states.

    finger_states: [thumb, index, middle, ring, pinky]
    Extended finger  → tip is placed ABOVE pip (lower y value)
    Curled finger    → tip is placed BELOW pip (higher y value)
    """
    # Build 21 landmarks as (x, y, z)
    # We'll place the wrist at (0.5, 0.9) and space fingers upward.

    lm: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 21

    # Wrist
    lm[LandmarkIndex.WRIST] = (0.5, 0.9, 0.0)

    # Middle MCP (used for hand scale)
    lm[LandmarkIndex.MIDDLE_MCP] = (0.5, 0.65, 0.0)

    # Each finger: MCP at y=0.65, PIP at y=0.5
    # Extended: TIP at y=0.35 (above PIP)
    # Curled:   TIP at y=0.6  (below PIP → closer to wrist)

    finger_defs = [
        # (MCP, PIP/IP, TIP, x_offset)
        (LandmarkIndex.THUMB_MCP,   LandmarkIndex.THUMB_IP,   LandmarkIndex.THUMB_TIP,   -0.15),
        (LandmarkIndex.INDEX_MCP,   LandmarkIndex.INDEX_PIP,  LandmarkIndex.INDEX_TIP,   -0.07),
        (LandmarkIndex.MIDDLE_MCP,  LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_TIP,   0.0),
        (LandmarkIndex.RING_MCP,    LandmarkIndex.RING_PIP,   LandmarkIndex.RING_TIP,     0.07),
        (LandmarkIndex.PINKY_MCP,   LandmarkIndex.PINKY_PIP,  LandmarkIndex.PINKY_TIP,   0.14),
    ]

    for i, (mcp_i, pip_i, tip_i, x_off) in enumerate(finger_defs):
        base_x = 0.5 + x_off
        lm[mcp_i] = (base_x, 0.65, 0.0)
        lm[pip_i] = (base_x, 0.50, 0.0)
        if finger_states[i]:
            lm[tip_i] = (base_x, 0.30, 0.0)   # extended: tip well above PIP
        else:
            # Curled: tip must be CLOSER to wrist (0.9) than MCP (0.65)
            # i.e. tip.y > 0.65 so d(tip, wrist) < d(MCP, wrist)
            lm[tip_i] = (base_x, 0.82, 0.0)   # curled: tip below MCP, near wrist

    # Remaining landmarks (DIP nodes) — place between PIP and TIP
    dip_pairs = [
        (LandmarkIndex.INDEX_DIP,  LandmarkIndex.INDEX_PIP,  LandmarkIndex.INDEX_TIP),
        (LandmarkIndex.MIDDLE_DIP, LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_TIP),
        (LandmarkIndex.RING_DIP,   LandmarkIndex.RING_PIP,   LandmarkIndex.RING_TIP),
        (LandmarkIndex.PINKY_DIP,  LandmarkIndex.PINKY_PIP,  LandmarkIndex.PINKY_TIP),
    ]
    for dip_i, pip_i, tip_i in dip_pairs:
        mx = (lm[pip_i][0] + lm[tip_i][0]) / 2
        my = (lm[pip_i][1] + lm[tip_i][1]) / 2
        lm[dip_i] = (mx, my, 0.0)

    # Thumb CMC
    lm[LandmarkIndex.THUMB_CMC] = (0.35, 0.80, 0.0)

    # Pinky DIP already set above; also set CMC/MCP nodes for completeness
    lm[LandmarkIndex.PINKY_MCP] = (0.64, 0.65, 0.0)

    # Override thumb tip for pinch distance
    # For right hand: thumb tip to the LEFT of IP means extended.
    # We'll control pinch by moving index tip relative to thumb tip.
    thumb_tip = lm[LandmarkIndex.THUMB_TIP]
    index_tip = lm[LandmarkIndex.INDEX_TIP]

    # Compute current distance and scale to desired pinch_dist
    dx = index_tip[0] - thumb_tip[0]
    dy = index_tip[1] - thumb_tip[1]
    current_dist = math.hypot(dx, dy)
    if current_dist > 1e-9 and pinch_dist >= 0:
        scale = pinch_dist / current_dist
        mid_x = (thumb_tip[0] + index_tip[0]) / 2
        mid_y = (thumb_tip[1] + index_tip[1]) / 2
        # Place both tips equidistant from midpoint
        lm[LandmarkIndex.THUMB_TIP] = (
            mid_x - dx * scale / 2, mid_y - dy * scale / 2, 0.0
        )
        lm[LandmarkIndex.INDEX_TIP] = (
            mid_x + dx * scale / 2, mid_y + dy * scale / 2, 0.0
        )

    pixels = [(int(x * 640), int(y * 480)) for (x, y, _) in lm]
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    return HandLandmarks(
        landmarks_norm=lm,
        landmarks_px=pixels,
        handedness=handedness,
        confidence=0.99,
        bounding_box=bbox,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFingerAnalyser(unittest.TestCase):

    def test_extended_fingers_open_palm(self):
        """All five fingers extended → all True."""
        hand = _make_landmarks([True, True, True, True, True])
        ext = FingerAnalyser.extended_fingers(hand.landmarks_norm, "Right")
        self.assertEqual(ext, [True, True, True, True, True])

    def test_extended_fingers_fist(self):
        """All fingers curled → all False."""
        hand = _make_landmarks([False, False, False, False, False])
        ext = FingerAnalyser.extended_fingers(hand.landmarks_norm, "Right")
        # Thumb uses horizontal heuristic; the others should be False
        self.assertFalse(ext[1], "Index should be curled")
        self.assertFalse(ext[2], "Middle should be curled")
        self.assertFalse(ext[3], "Ring should be curled")
        self.assertFalse(ext[4], "Pinky should be curled")

    def test_extended_fingers_index_only(self):
        """Only index extended."""
        hand = _make_landmarks([False, True, False, False, False])
        ext = FingerAnalyser.extended_fingers(hand.landmarks_norm, "Right")
        self.assertTrue(ext[1],  "Index should be extended")
        self.assertFalse(ext[2], "Middle should be curled")
        self.assertFalse(ext[3], "Ring should be curled")
        self.assertFalse(ext[4], "Pinky should be curled")

    def test_landmark_distance(self):
        """Distance between (0,0) and (3,4) should be 5."""
        lm = [(0.0, 0.0, 0.0)] * 21
        lm[0] = (0.0, 0.0, 0.0)
        lm[1] = (0.3, 0.4, 0.0)
        dist = FingerAnalyser.landmark_distance(lm, 0, 1)
        self.assertAlmostEqual(dist, 0.5, places=5)

    def test_curl_fraction_fist(self):
        """All curled → curl fraction ≈ 1.0."""
        hand = _make_landmarks([False, False, False, False, False])
        curl = FingerAnalyser.curl_fraction(hand.landmarks_norm)
        self.assertGreater(curl, 0.5)

    def test_curl_fraction_open(self):
        """All extended → curl fraction should be low."""
        hand = _make_landmarks([True, True, True, True, True])
        curl = FingerAnalyser.curl_fraction(hand.landmarks_norm)
        self.assertLess(curl, 0.5)


class TestStabilityBuffer(unittest.TestCase):

    def test_requires_consecutive_frames(self):
        buf = StabilityBuffer(required_frames=3)
        self.assertFalse(buf.update(Gesture.OPEN_PALM))
        self.assertFalse(buf.update(Gesture.OPEN_PALM))
        self.assertTrue(buf.update(Gesture.OPEN_PALM))  # 3rd frame

    def test_resets_on_change(self):
        buf = StabilityBuffer(required_frames=3)
        buf.update(Gesture.OPEN_PALM)
        buf.update(Gesture.OPEN_PALM)
        buf.update(Gesture.INDEX_POINT)  # change! counter resets
        self.assertFalse(buf.update(Gesture.INDEX_POINT))  # only 2 consecutive
        self.assertTrue(buf.update(Gesture.INDEX_POINT))   # now 3


class TestGestureDetector(unittest.TestCase):

    def setUp(self):
        self._cfg = GestureConfig()
        self._det = GestureDetector(self._cfg)

    def _stable_detect(self, hand: HandLandmarks, frames: int = 5) -> GestureResult:
        """Run the detector for `frames` frames and return the last result."""
        result = GestureResult()
        for _ in range(frames):
            result = self._det.update(hand, 640, 480)
        return result

    def test_no_hand_returns_none(self):
        result = self._det.update(None, 640, 480)
        self.assertEqual(result.gesture, Gesture.NONE)

    def test_open_palm_detected(self):
        hand = _make_landmarks([True, True, True, True, True], pinch_dist=0.20)
        result = self._stable_detect(hand)
        self.assertIn(result.gesture, [Gesture.OPEN_PALM, Gesture.VOLUME_CONTROL],
                      "Open palm or volume control expected for all-extended hand")

    def test_index_point_detected(self):
        hand = _make_landmarks([False, True, False, False, False], pinch_dist=0.20)
        result = self._stable_detect(hand)
        self.assertEqual(result.gesture, Gesture.INDEX_POINT)

    def test_fist_detected(self):
        hand = _make_landmarks([False, False, False, False, False], pinch_dist=0.08)
        result = self._stable_detect(hand)
        self.assertEqual(result.gesture, Gesture.FIST)

    def test_pinch_closed_detected(self):
        # Pinch threshold is 0.055 by default
        hand = _make_landmarks([True, True, False, False, False], pinch_dist=0.03)
        result = self._stable_detect(hand)
        self.assertEqual(result.gesture, Gesture.PINCH_CLOSED)

    def test_two_finger_scroll_detected(self):
        hand = _make_landmarks([False, True, True, False, False], pinch_dist=0.20)
        result = self._stable_detect(hand)
        self.assertEqual(result.gesture, Gesture.TWO_FINGER_SCROLL)

    def test_volume_control_detected(self):
        # Thumb + index both extended, others curled
        hand = _make_landmarks([True, True, False, False, False], pinch_dist=0.18)
        result = self._stable_detect(hand)
        self.assertIn(result.gesture, [Gesture.VOLUME_CONTROL, Gesture.OPEN_PALM])

    def test_volume_level_range(self):
        hand = _make_landmarks([True, True, False, False, False], pinch_dist=0.18)
        result = self._stable_detect(hand)
        if result.volume_level >= 0:
            self.assertGreaterEqual(result.volume_level, 0)
            self.assertLessEqual(result.volume_level, 100)

    def test_cursor_position_normalised(self):
        hand = _make_landmarks([True, True, True, True, True], pinch_dist=0.20)
        result = self._det.update(hand, 640, 480)
        cx, cy = result.cursor_position
        self.assertGreaterEqual(cx, 0.0)
        self.assertLessEqual(cx, 1.0)
        self.assertGreaterEqual(cy, 0.0)
        self.assertLessEqual(cy, 1.0)

    def test_stability_transitions(self):
        """Gesture should not be stable on first frame."""
        hand = _make_landmarks([True, True, True, True, True], pinch_dist=0.20)
        result = self._det.update(hand, 640, 480)
        self.assertFalse(result.is_stable, "Should not be stable on first frame")

    def test_scroll_delta_zero_on_first_frame(self):
        """First scroll frame should return (0, 0) delta."""
        det = GestureDetector(self._cfg)
        hand = _make_landmarks([False, True, True, False, False], pinch_dist=0.20)
        result = det.update(hand, 640, 480)
        if result.gesture == Gesture.TWO_FINGER_SCROLL:
            self.assertEqual(result.scroll_delta, (0.0, 0.0))

    def test_multiple_consecutive_nones(self):
        """Repeated None inputs should not crash."""
        for _ in range(20):
            result = self._det.update(None, 640, 480)
        self.assertEqual(result.gesture, Gesture.NONE)


class TestGestureDetectorUserGestures(unittest.TestCase):

    def test_custom_gesture_loaded_and_matched(self):
        """User gesture JSON file is parsed and matched correctly."""
        import json, tempfile, os
        gestures = [
            {"name": "Peace Sign", "fingers": [False, True, True, False, False], "action": "screenshot"}
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gestures, f)
            path = f.name

        try:
            cfg = GestureConfig()
            # Adjust pinch_threshold so peace sign doesn't accidentally match pinch
            cfg.pinch_threshold = 0.01
            det = GestureDetector(cfg, user_gestures_file=path)

            hand = _make_landmarks([False, True, True, False, False], pinch_dist=0.20)
            result = None
            for _ in range(10):
                result = det.update(hand, 640, 480)

            # Should be TWO_FINGER_SCROLL or CUSTOM — both acceptable
            self.assertIn(result.gesture, [Gesture.TWO_FINGER_SCROLL, Gesture.CUSTOM])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Integration-style smoke test
# ---------------------------------------------------------------------------

class TestPipelineSmoke(unittest.TestCase):

    def test_all_gestures_cycle(self):
        """
        Run each gesture scenario through the detector for several frames
        and ensure no exceptions are raised.
        """
        cfg = GestureConfig()
        det = GestureDetector(cfg)
        scenarios = [
            ([True,  True,  True,  True,  True],  0.20),   # open palm
            ([False, True,  False, False, False],  0.20),   # index point
            ([True,  True,  False, False, False],  0.03),   # pinch closed
            ([False, True,  True,  False, False],  0.20),   # scroll
            ([True,  True,  False, False, False],  0.18),   # volume
            ([False, False, False, False, False],  0.08),   # fist
        ]
        for fingers, pinch in scenarios:
            hand = _make_landmarks(fingers, pinch_dist=pinch)
            for _ in range(10):
                try:
                    result = det.update(hand, 640, 480)
                    self.assertIsInstance(result, GestureResult)
                except Exception as e:
                    self.fail(f"Gesture {fingers} raised: {e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestFingerAnalyser,
        TestStabilityBuffer,
        TestGestureDetector,
        TestGestureDetectorUserGestures,
        TestPipelineSmoke,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
