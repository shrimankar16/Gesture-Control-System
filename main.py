"""
main.py
-------
Application entry point and main loop orchestrator.

Responsibilities
----------------
* Parse CLI arguments.
* Load configuration.
* Optionally run the calibration wizard.
* Initialise all subsystems (tracker, detector, controller).
* Drive the camera capture loop at target FPS.
* Render the OSD (FPS, gesture name, landmarks, HUD).
* Handle graceful shutdown on quit key or window close.
* Optionally launch the Tkinter settings GUI.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from config import AppConfig, load_config, save_config
from controller import SystemController
from gesture_detector import Gesture, GestureDetector, GestureResult
from hand_tracker import HandLandmarks, HandTracker


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling-window FPS calculator."""

    def __init__(self, window: int = 30) -> None:
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record a frame timestamp and return current FPS."""
        now = time.monotonic()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# OSD renderer
# ---------------------------------------------------------------------------

class OSDRenderer:
    """Draws the heads-up display onto a frame."""

    # Gesture → accent colour (BGR)
    _GESTURE_COLOURS = {
        Gesture.NONE:              (100, 100, 100),
        Gesture.OPEN_PALM:         (0,   220, 100),
        Gesture.INDEX_POINT:       (0,   180, 255),
        Gesture.PINCH_OPEN:        (0,   220, 220),
        Gesture.PINCH_CLOSED:      (0,    80, 255),
        Gesture.TWO_FINGER_SCROLL: (255, 180,   0),
        Gesture.VOLUME_CONTROL:    (255,  80, 200),
        Gesture.FIST:              (0,    0,  220),
        Gesture.CUSTOM:            (180, 255,   0),
    }

    def __init__(self, cfg: AppConfig) -> None:
        self._dcfg = cfg.display
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_bold = cv2.FONT_HERSHEY_DUPLEX

    def draw(
        self,
        frame: np.ndarray,
        result: GestureResult,
        fps: float,
        hands: list[HandLandmarks],
        tracker: HandTracker,
        is_paused: bool,
        volume_level: int,
    ) -> np.ndarray:
        """Render all OSD elements onto frame (mutates in-place)."""
        h, w = frame.shape[:2]
        cfg = self._dcfg

        # --- Landmarks & bounding boxes ---
        for hand in hands:
            if cfg.show_landmarks:
                tracker.draw_landmarks(
                    frame, hand,
                    landmark_colour=cfg.landmark_colour,
                    connection_colour=cfg.connection_colour,
                )
            if cfg.show_bounding_box:
                tracker.draw_bounding_box(frame, hand, colour=cfg.bbox_colour)
            if cfg.show_hand_label:
                bx, by, _, _ = hand.bounding_box
                cv2.putText(
                    frame,
                    f"{hand.handedness} ({hand.confidence:.0%})",
                    (bx, max(0, by - 10)),
                    self._font, 0.55, cfg.text_colour, 1, cv2.LINE_AA,
                )

        # --- HUD panel (top-left semi-transparent bar) ---
        accent = self._GESTURE_COLOURS.get(result.gesture, (200, 200, 200))
        self._draw_hud(frame, result, fps, is_paused, volume_level, accent)

        # --- Volume bar (right side) ---
        if result.gesture == Gesture.VOLUME_CONTROL and result.volume_level >= 0:
            self._draw_volume_bar(frame, result.volume_level, accent)

        # --- PAUSED overlay ---
        if is_paused:
            self._draw_pause_overlay(frame)

        # --- Pinch indicator ---
        if result.gesture == Gesture.PINCH_CLOSED and result.is_stable:
            cx, cy = int(result.cursor_position[0] * w), int(result.cursor_position[1] * h)
            cv2.circle(frame, (cx, cy), 22, (0, 80, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 8, (0, 80, 255), -1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Private draw helpers
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        frame: np.ndarray,
        result: GestureResult,
        fps: float,
        is_paused: bool,
        volume_level: int,
        accent: tuple,
    ) -> None:
        h, w = frame.shape[:2]
        panel_h = 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, self._dcfg.hud_alpha, frame, 1 - self._dcfg.hud_alpha, 0, frame)

        # Left accent bar
        cv2.rectangle(frame, (0, 0), (6, panel_h), accent, -1)

        # FPS
        if self._dcfg.show_fps:
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (20, 30),
                self._font_bold, 0.75, self._dcfg.fps_colour, 1, cv2.LINE_AA,
            )

        # Gesture name
        if self._dcfg.show_gesture_name:
            stability_marker = "●" if result.is_stable else "○"
            label = result.label
            colour = accent if result.is_stable else (160, 160, 160)
            cv2.putText(
                frame, f"{stability_marker} {label}", (20, 62),
                self._font, 0.72, colour, 2, cv2.LINE_AA,
            )

        # Volume level display
        if volume_level >= 0:
            cv2.putText(
                frame, f"VOL: {volume_level}%", (w - 160, 30),
                self._font, 0.7, self._dcfg.text_colour, 1, cv2.LINE_AA,
            )

        # Confidence dot
        conf_x = w - 30
        conf_col = (0, int(result.confidence * 255), int((1 - result.confidence) * 255))
        cv2.circle(frame, (conf_x, 60), 10, conf_col, -1, cv2.LINE_AA)

    def _draw_volume_bar(
        self,
        frame: np.ndarray,
        level: int,
        accent: tuple,
    ) -> None:
        h, w = frame.shape[:2]
        bar_h = int(h * 0.5)
        bar_x = w - 40
        bar_y_top = h // 4
        bar_w = 18

        # Background
        cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + bar_w, bar_y_top + bar_h), (50, 50, 50), -1)

        # Fill
        fill = int(bar_h * (level / 100.0))
        fill_y = bar_y_top + bar_h - fill
        cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_y_top + bar_h), accent, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y_top), (bar_x + bar_w, bar_y_top + bar_h), (200, 200, 200), 1)

        # Label
        cv2.putText(
            frame, f"{level}%",
            (bar_x - 10, bar_y_top + bar_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def _draw_pause_overlay(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        text = "SYSTEM PAUSED  (fist to resume)"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
        tx = (w - ts[0]) // 2
        ty = h // 2
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Calibration wizard
# ---------------------------------------------------------------------------

class CalibrationWizard:
    """
    Interactive calibration that lets the user set:
    * Pinch closed threshold
    * Pinch open threshold
    * Volume min/max distances
    """

    def __init__(self, tracker: HandTracker, detector: GestureDetector) -> None:
        self._tracker = tracker
        self._detector = detector

    def run(self, cap: cv2.VideoCapture, cfg: AppConfig) -> AppConfig:
        """
        Block until calibration is complete.
        Returns updated config.
        """
        print("\n[Calibration] Starting calibration wizard...")
        steps = [
            ("PINCH CLOSED: Pinch thumb+index FULLY together. Press SPACE.", "closed"),
            ("PINCH OPEN:   Hold thumb+index slightly apart. Press SPACE.", "open"),
            ("VOLUME MIN:   Hold hand flat (palm up). Press SPACE.", "vol_min"),
            ("VOLUME MAX:   Spread thumb+index as wide as possible. Press SPACE.", "vol_max"),
        ]
        measured = {}

        for instruction, key in steps:
            print(f"\n  {instruction}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                hands = self._tracker.process(frame)
                info_text = instruction.split(":")[0]

                cv2.putText(frame, info_text, (30, 60), font, 0.8, (0, 255, 100), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press SPACE to capture", (30, 100), font, 0.65, (255, 255, 255), 1)

                if hands:
                    hand = hands[0]
                    dist = self._detector.get_pinch_distance_raw(hand)
                    self._tracker.draw_landmarks(frame, hand)
                    cv2.putText(frame, f"Distance: {dist:.4f}", (30, 140), font, 0.65, (0, 200, 255), 1)

                cv2.imshow("Gesture Control — Calibration", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(" ") and hands:
                    dist = self._detector.get_pinch_distance_raw(hands[0])
                    measured[key] = dist
                    print(f"  Captured: {dist:.4f}")
                    break
                elif k == ord("q"):
                    print("[Calibration] Cancelled.")
                    return cfg

        # Apply measured values
        cfg.gesture.pinch_threshold = measured.get("closed", cfg.gesture.pinch_threshold)
        cfg.gesture.pinch_release_threshold = measured.get("open", cfg.gesture.pinch_release_threshold)
        cfg.gesture.vol_distance_min = measured.get("vol_min", cfg.gesture.vol_distance_min)
        cfg.gesture.vol_distance_max = measured.get("vol_max", cfg.gesture.vol_distance_max)

        save_config(cfg)
        print("[Calibration] Complete! Config saved.\n")
        return cfg


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class GestureControlApp:
    """
    Top-level application class.
    Owns the camera loop and coordinates all subsystems.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._tracker = HandTracker(cfg.tracking)
        self._detector = GestureDetector(cfg.gesture, cfg.user_gestures_file)
        self._controller = SystemController(cfg)
        self._fps = FPSCounter()
        self._osd = OSDRenderer(cfg)
        self._running = False

    def run(self) -> None:
        """Main execution entry point."""
        cap = self._open_camera()
        if cap is None:
            return

        window_name = "Gesture Control System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self._cfg.camera.width, self._cfg.camera.height)

        # Optional calibration
        if self._cfg.calibration_mode:
            wizard = CalibrationWizard(self._tracker, self._detector)
            self._cfg = wizard.run(cap, self._cfg)
            # Re-init detector with updated config
            self._detector = GestureDetector(self._cfg.gesture, self._cfg.user_gestures_file)
            self._controller = SystemController(self._cfg)

        print("[GestureControl] Running — press 'q' to quit, 'c' for calibration, 'd' for debug.")

        self._running = True
        try:
            while self._running:
                self._frame_loop(cap, window_name)
        except KeyboardInterrupt:
            print("\n[GestureControl] Interrupted.")
        finally:
            self._shutdown(cap)

    # ------------------------------------------------------------------
    # Frame processing loop
    # ------------------------------------------------------------------

    def _frame_loop(self, cap: cv2.VideoCapture, window_name: str) -> None:
        ok, frame = cap.read()
        if not ok:
            print("[GestureControl] Frame capture failed — retrying...")
            time.sleep(0.05)
            return

        # Flip horizontally (mirror mode — more intuitive for the user)
        frame = cv2.flip(frame, 1)

        # Track hands
        hands = self._tracker.process(frame)
        dominant = hands[0] if hands else None

        # Detect gesture
        fw, fh = frame.shape[1], frame.shape[0]
        result = self._detector.update(dominant, fw, fh)

        # Control system
        self._controller.process(result)

        # Compute FPS
        fps = self._fps.tick()

        # Render OSD
        self._osd.draw(
            frame,
            result,
            fps,
            hands,
            self._tracker,
            self._controller.is_paused,
            self._controller.current_volume,
        )

        # Debug overlay
        if self._cfg.debug_mode:
            self._draw_debug(frame, result, hands)

        cv2.imshow(window_name, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord(self._cfg.quit_key):
            self._running = False
        elif key == ord("c"):
            self._run_calibration(cap)
        elif key == ord("d"):
            self._cfg.debug_mode = not self._cfg.debug_mode
            print(f"[Debug] Debug mode: {self._cfg.debug_mode}")
        elif key == ord("s"):
            save_config(self._cfg)
        elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            self._running = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Open and configure the webcam."""
        ccfg = self._cfg.camera
        cap = cv2.VideoCapture(ccfg.device_index, cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"[GestureControl] ERROR: Cannot open camera {ccfg.device_index}")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ccfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ccfg.height)
        cap.set(cv2.CAP_PROP_FPS,          ccfg.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # Minimise latency
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"[Camera] Opened: {actual_w}x{actual_h} @ {actual_fps} FPS")
        return cap

    def _run_calibration(self, cap: cv2.VideoCapture) -> None:
        """Run calibration mid-session."""
        wizard = CalibrationWizard(self._tracker, self._detector)
        self._cfg = wizard.run(cap, self._cfg)
        self._detector = GestureDetector(self._cfg.gesture, self._cfg.user_gestures_file)
        self._controller = SystemController(self._cfg)

    def _draw_debug(
        self,
        frame: np.ndarray,
        result: GestureResult,
        hands: list,
    ) -> None:
        """Extra debug info drawn at the bottom of the frame."""
        h, w = frame.shape[:2]
        y = h - 120
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            f"Gesture: {result.gesture.name}  Conf: {result.confidence:.2f}  Stable: {result.is_stable}",
            f"Cursor (norm): {result.cursor_position[0]:.3f}, {result.cursor_position[1]:.3f}",
            f"Pinch dist: {result.pinch_distance:.4f}  Scroll: {result.scroll_delta}",
            f"Volume: {result.volume_level}  Actions: {self._controller.action_counts}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, y + i * 22), font, 0.48, (200, 200, 50), 1, cv2.LINE_AA)

    def _shutdown(self, cap: cv2.VideoCapture) -> None:
        """Release all resources cleanly."""
        print("[GestureControl] Shutting down...")
        self._tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[GestureControl] Session stats: {self._controller.action_counts}")


# ---------------------------------------------------------------------------
# Settings GUI (Tkinter)
# ---------------------------------------------------------------------------

def launch_settings_gui(cfg: AppConfig) -> AppConfig:
    """
    Optional Tkinter settings panel.
    Returns (possibly modified) config.
    Opens in a separate window; the user can tweak sliders and save.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("[GUI] Tkinter not available.")
        return cfg

    root = tk.Tk()
    root.title("Gesture Control — Settings")
    root.resizable(False, False)

    # Style
    style = ttk.Style()
    style.theme_use("clam")

    frame = ttk.Frame(root, padding=20)
    frame.grid(row=0, column=0)

    ttk.Label(frame, text="Gesture Control Settings", font=("Helvetica", 14, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(0, 16))

    sliders = {}

    def add_slider(row: int, label: str, from_: float, to: float, init: float, step: float = 0.01):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        var = tk.DoubleVar(value=init)
        sl = ttk.Scale(frame, from_=from_, to=to, variable=var, orient="horizontal", length=300)
        sl.grid(row=row, column=1, padx=(10, 0))
        val_label = ttk.Label(frame, text=f"{init:.3f}", width=7)
        val_label.grid(row=row, column=2, padx=(6, 0))
        def update_label(*_):
            val_label.config(text=f"{var.get():.3f}")
        var.trace_add("write", update_label)
        sliders[label] = var

    add_slider(1, "Pinch Threshold",         0.01, 0.15, cfg.gesture.pinch_threshold)
    add_slider(2, "Pinch Release Threshold", 0.01, 0.20, cfg.gesture.pinch_release_threshold)
    add_slider(3, "Cursor Smooth Alpha",     0.05, 1.0,  cfg.cursor.smooth_alpha)
    add_slider(4, "Scroll Sensitivity",      1.0,  50.0, cfg.gesture.scroll_sensitivity, step=1.0)
    add_slider(5, "Cursor Margin",           0.0,  0.25, cfg.cursor.margin)
    add_slider(6, "Vol Distance Min",        0.01, 0.15, cfg.gesture.vol_distance_min)
    add_slider(7, "Vol Distance Max",        0.10, 0.60, cfg.gesture.vol_distance_max)

    def on_save():
        cfg.gesture.pinch_threshold         = sliders["Pinch Threshold"].get()
        cfg.gesture.pinch_release_threshold = sliders["Pinch Release Threshold"].get()
        cfg.cursor.smooth_alpha             = sliders["Cursor Smooth Alpha"].get()
        cfg.gesture.scroll_sensitivity      = sliders["Scroll Sensitivity"].get()
        cfg.cursor.margin                   = sliders["Cursor Margin"].get()
        cfg.gesture.vol_distance_min        = sliders["Vol Distance Min"].get()
        cfg.gesture.vol_distance_max        = sliders["Vol Distance Max"].get()
        save_config(cfg)
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=9, column=0, columnspan=3, pady=(20, 0))
    ttk.Button(btn_frame, text="Save & Launch", command=on_save).pack(side="left", padx=8)
    ttk.Button(btn_frame, text="Launch with defaults", command=on_cancel).pack(side="left", padx=8)

    root.mainloop()
    return cfg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gesture Control System — control your PC with hand gestures."
    )
    parser.add_argument("--config",      default=None,       help="Path to JSON config file")
    parser.add_argument("--calibrate",   action="store_true", help="Run calibration wizard at startup")
    parser.add_argument("--gui",         action="store_true", help="Show settings GUI before launching")
    parser.add_argument("--debug",       action="store_true", help="Enable debug overlay")
    parser.add_argument("--camera",      type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--no-flip",     action="store_true", help="Disable horizontal frame flip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(args.config) if args.config else load_config()

    # Apply CLI overrides
    if args.calibrate:
        cfg.calibration_mode = True
    if args.debug:
        cfg.debug_mode = True
    cfg.camera.device_index = args.camera

    # Optional GUI
    if args.gui:
        cfg = launch_settings_gui(cfg)

    print("=" * 60)
    print("  Gesture Control System")
    print("=" * 60)
    print(f"  Camera   : {cfg.camera.device_index}  ({cfg.camera.width}x{cfg.camera.height})")
    print(f"  Max hands: {cfg.tracking.max_num_hands}")
    print(f"  Dominant : {cfg.tracking.dominant_hand}")
    print(f"  Debug    : {cfg.debug_mode}")
    print("=" * 60)
    print("  Gestures:")
    print("    Open Palm         → Move cursor")
    print("    Index Finger      → Precision cursor mode")
    print("    Pinch (close)     → Click / drag")
    print("    Two Fingers       → Scroll")
    print("    Thumb + Index     → Volume control")
    print("    Fist              → Pause / resume")
    print("=" * 60)
    print("  Keys: Q=quit  C=calibrate  D=debug  S=save config")
    print("=" * 60)

    app = GestureControlApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
