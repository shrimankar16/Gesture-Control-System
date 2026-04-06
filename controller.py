"""
controller.py
-------------
System control layer — translates GestureResults into OS-level actions.

Responsibilities
----------------
* Smooth cursor movement with exponential moving average + velocity dampening.
* Left click, right click, double click via PyAutoGUI.
* System volume control (cross-platform: Windows/macOS/Linux).
* Scroll (vertical + horizontal).
* Cooldown management per action type to prevent jitter.
* Screen-space coordinate mapping with configurable margins.

This module is the ONLY place where OS interactions happen.
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyautogui

from config import AppConfig, CursorConfig, GestureConfig
from gesture_detector import Gesture, GestureResult

# Disable PyAutoGUI's fail-safe (corner detection) — we handle our own safety
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Disable inter-call delay for performance


# ---------------------------------------------------------------------------
# Cooldown tracker
# ---------------------------------------------------------------------------

class CooldownTimer:
    """Simple per-action cooldown tracker."""

    def __init__(self) -> None:
        self._timestamps: dict[str, float] = {}

    def is_ready(self, action: str, cooldown: float) -> bool:
        """Return True if enough time has passed since the last action."""
        now = time.monotonic()
        last = self._timestamps.get(action, 0.0)
        return (now - last) >= cooldown

    def trigger(self, action: str) -> None:
        """Record that the action just fired."""
        self._timestamps[action] = time.monotonic()

    def ready_and_trigger(self, action: str, cooldown: float) -> bool:
        """Atomically check readiness and record the trigger if ready."""
        if self.is_ready(action, cooldown):
            self.trigger(action)
            return True
        return False


# ---------------------------------------------------------------------------
# Smooth cursor
# ---------------------------------------------------------------------------

class SmoothCursor:
    """
    Converts normalised hand position to screen co-ordinates using:
    1. Margin clipping — ignore the outer `margin` fraction of the frame.
    2. Exponential moving average (EMA) for smooth movement.
    3. Velocity dampening to reduce overshoot.
    """

    def __init__(self, cfg: CursorConfig, screen_w: int, screen_h: int) -> None:
        self._cfg = cfg
        self._sw = screen_w
        self._sh = screen_h
        self._x: float = screen_w / 2
        self._y: float = screen_h / 2
        self._vx: float = 0.0
        self._vy: float = 0.0

    def update(
        self,
        norm_x: float,
        norm_y: float,
        speed_multiplier: float = 1.0,
    ) -> Tuple[int, int]:
        """
        Compute the new smoothed screen position.

        Parameters
        ----------
        norm_x, norm_y  : raw normalised position [0, 1]
        speed_multiplier: 1.0 = normal, 0.4 = precision, etc.

        Returns
        -------
        (screen_x, screen_y) — integer pixel co-ordinates
        """
        margin = self._cfg.margin

        # 1. Clip to active region (avoid edge jitter)
        clipped_x = max(margin, min(norm_x, 1.0 - margin))
        clipped_y = max(margin, min(norm_y, 1.0 - margin))

        # 2. Remap clipped range → [0, 1]
        remapped_x = (clipped_x - margin) / (1.0 - 2 * margin)
        remapped_y = (clipped_y - margin) / (1.0 - 2 * margin)

        # 3. Convert to screen pixels (mirror x axis — webcam is mirrored)
        target_x = (1.0 - remapped_x) * self._sw * speed_multiplier + self._x * (1.0 - speed_multiplier)
        target_y = remapped_y * self._sh

        # 4. EMA smoothing
        alpha = self._cfg.smooth_alpha
        new_x = self._x + alpha * (target_x - self._x)
        new_y = self._y + alpha * (target_y - self._y)

        # 5. Velocity dampening
        damp = self._cfg.velocity_dampening
        self._vx = damp * self._vx + (1 - damp) * (new_x - self._x)
        self._vy = damp * self._vy + (1 - damp) * (new_y - self._y)
        new_x += self._vx * 0.1
        new_y += self._vy * 0.1

        # 6. Clamp to screen bounds
        self._x = max(0.0, min(new_x, float(self._sw - 1)))
        self._y = max(0.0, min(new_y, float(self._sh - 1)))

        return int(self._x), int(self._y)

    @property
    def position(self) -> Tuple[int, int]:
        return int(self._x), int(self._y)


# ---------------------------------------------------------------------------
# Platform-specific volume/brightness helpers
# ---------------------------------------------------------------------------

class VolumeController:
    """Cross-platform volume control."""

    _platform = platform.system()  # "Windows" | "Darwin" | "Linux"

    @classmethod
    def set_volume(cls, level: int) -> None:
        """Set system volume to `level` (0–100)."""
        level = max(0, min(level, 100))
        try:
            if cls._platform == "Windows":
                cls._set_volume_windows(level)
            elif cls._platform == "Darwin":
                cls._set_volume_mac(level)
            else:
                cls._set_volume_linux(level)
        except Exception as e:
            print(f"[VolumeController] Error setting volume: {e}")

    @staticmethod
    def _set_volume_windows(level: int) -> None:
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        except ImportError:
            # Fallback: use nircmd if pycaw not installed
            subprocess.run(
                ["nircmd.exe", "setsysvolume", str(int(level / 100.0 * 65535))],
                capture_output=True,
            )

    @staticmethod
    def _set_volume_mac(level: int) -> None:
        subprocess.run(["osascript", "-e", f"set volume output volume {level}"],
                       capture_output=True)

    @staticmethod
    def _set_volume_linux(level: int) -> None:
        subprocess.run(
            ["amixer", "-q", "sset", "Master", f"{level}%"],
            capture_output=True,
        )


class BrightnessController:
    """Cross-platform brightness control (best-effort)."""

    _platform = platform.system()

    @classmethod
    def set_brightness(cls, level: int) -> None:
        """Set screen brightness to `level` (0–100). Best-effort."""
        level = max(0, min(level, 100))
        try:
            if cls._platform == "Windows":
                cls._set_brightness_windows(level)
            elif cls._platform == "Darwin":
                cls._set_brightness_mac(level)
            else:
                cls._set_brightness_linux(level)
        except Exception as e:
            print(f"[BrightnessController] Error setting brightness: {e}")

    @staticmethod
    def _set_brightness_windows(level: int) -> None:
        try:
            import screen_brightness_control as sbc
            sbc.set_brightness(level)
        except ImportError:
            pass  # Optional dependency

    @staticmethod
    def _set_brightness_mac(level: int) -> None:
        val = level / 100.0
        subprocess.run(
            ["osascript", "-e", f'tell application "System Events" to set brightness of first monitor to {val}'],
            capture_output=True,
        )

    @staticmethod
    def _set_brightness_linux(level: int) -> None:
        try:
            import screen_brightness_control as sbc
            sbc.set_brightness(level)
        except ImportError:
            pass  # Optional dependency


# ---------------------------------------------------------------------------
# SystemController — main class
# ---------------------------------------------------------------------------

class SystemController:
    """
    Receives GestureResult each frame and dispatches the appropriate
    OS-level actions.

    State machine:
    - Tracks pinch state (open/closed) for click detection
    - Tracks scroll state for continuous scrolling
    - Tracks volume state for continuous volume adjustment
    - Applies per-action cooldowns
    """

    def __init__(self, cfg: AppConfig) -> None:
        self._gcfg = cfg.gesture
        self._ccfg = cfg.cursor
        self._cooldown = CooldownTimer()

        # Screen dimensions
        self._sw, self._sh = pyautogui.size()
        self._cursor = SmoothCursor(cfg.cursor, self._sw, self._sh)

        # Click state machine
        self._pinch_was_closed: bool = False
        self._click_start_time: float = 0.0
        self._double_click_window: float = 0.4  # seconds

        # Volume state
        self._current_volume: int = -1  # -1 = unset, fetch on first use
        self._last_volume_level: int = 50

        # Pause state
        self._is_paused: bool = False
        self._pause_gesture_time: float = 0.0

        # Debug counter
        self._action_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def process(self, result: GestureResult) -> None:
        """
        Receive a GestureResult and perform system actions.
        Called once per frame from main.py.
        """
        gesture = result.gesture

        # FIST = pause all cursor actions (toggle)
        if gesture == Gesture.FIST and result.is_stable:
            self._handle_fist()
            return

        if self._is_paused:
            return  # All actions suppressed while paused

        # Route to handler
        if gesture in (Gesture.OPEN_PALM, Gesture.PINCH_OPEN):
            self._handle_cursor_move(result, precision=False)

        elif gesture == Gesture.INDEX_POINT:
            self._handle_cursor_move(result, precision=True)

        elif gesture == Gesture.PINCH_CLOSED:
            self._handle_pinch_closed(result)

        elif gesture == Gesture.TWO_FINGER_SCROLL:
            self._handle_scroll(result)

        elif gesture == Gesture.VOLUME_CONTROL:
            self._handle_volume(result)

        elif gesture == Gesture.CUSTOM:
            self._handle_custom(result)

        elif gesture == Gesture.NONE:
            # Reset pinch state when hand leaves
            self._pinch_was_closed = False

        # Release pinch if gesture transitions away from closed
        if gesture != Gesture.PINCH_CLOSED and self._pinch_was_closed:
            self._on_pinch_release()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_cursor_move(self, result: GestureResult, precision: bool) -> None:
        """Move the cursor based on finger tip position."""
        nx, ny = result.cursor_position
        speed = self._ccfg.precision_speed if precision else self._ccfg.normal_speed
        sx, sy = self._cursor.update(nx, ny, speed_multiplier=1.0)
        pyautogui.moveTo(sx, sy)

    def _handle_pinch_closed(self, result: GestureResult) -> None:
        """Handle click on pinch close."""
        if not result.is_stable:
            return

        # Also move cursor to current position
        nx, ny = result.cursor_position
        sx, sy = self._cursor.update(nx, ny)

        if not self._pinch_was_closed:
            # Pinch just closed — fire click
            if self._cooldown.ready_and_trigger("click", self._gcfg.click_cooldown):
                now = time.monotonic()
                # Double-click detection
                if (now - self._click_start_time) < self._double_click_window:
                    pyautogui.doubleClick(sx, sy)
                    self._log_action("double_click")
                else:
                    pyautogui.mouseDown(sx, sy, button="left")
                    self._log_action("mouse_down")
                self._click_start_time = now
                self._pinch_was_closed = True

    def _on_pinch_release(self) -> None:
        """Called when pinch transitions from closed to open."""
        if self._pinch_was_closed:
            sx, sy = self._cursor.position
            pyautogui.mouseUp(sx, sy, button="left")
            self._log_action("mouse_up")
            self._pinch_was_closed = False

    def _handle_scroll(self, result: GestureResult) -> None:
        """Scroll based on vertical hand movement."""
        if not result.is_stable:
            return
        if not self._cooldown.ready_and_trigger("scroll", self._gcfg.scroll_cooldown):
            return

        dx, dy = result.scroll_delta
        # PyAutoGUI scroll: positive = up, negative = down
        # Our dy: positive = hand moved down = scroll down
        if abs(dy) > 0.5:
            clicks = -int(dy / 3)  # scale to wheel clicks
            pyautogui.scroll(clicks)
            self._log_action("scroll")

    def _handle_volume(self, result: GestureResult) -> None:
        """Adjust system volume based on pinch spread."""
        if result.volume_level < 0:
            return
        if not self._cooldown.ready_and_trigger("volume", self._gcfg.volume_cooldown):
            return

        if result.volume_level != self._last_volume_level:
            VolumeController.set_volume(result.volume_level)
            self._last_volume_level = result.volume_level
            self._log_action("volume")

    def _handle_fist(self) -> None:
        """Toggle pause state on stable fist gesture."""
        if not self._cooldown.ready_and_trigger("fist", 0.8):
            return
        self._is_paused = not self._is_paused
        state = "PAUSED" if self._is_paused else "RESUMED"
        print(f"[Controller] System {state} (fist gesture)")
        self._log_action("fist_toggle")

    def _handle_custom(self, result: GestureResult) -> None:
        """Execute a user-defined custom gesture action."""
        action = result.payload.get("action", "")
        if not action:
            return
        if not self._cooldown.ready_and_trigger(f"custom_{action}", 1.0):
            return

        if action == "screenshot":
            pyautogui.hotkey("ctrl", "shift", "s")
        elif action == "copy":
            pyautogui.hotkey("ctrl", "c")
        elif action == "paste":
            pyautogui.hotkey("ctrl", "v")
        elif action == "undo":
            pyautogui.hotkey("ctrl", "z")
        elif action == "close_window":
            pyautogui.hotkey("alt", "F4")
        elif action == "next_tab":
            pyautogui.hotkey("ctrl", "tab")
        elif action == "prev_tab":
            pyautogui.hotkey("ctrl", "shift", "tab")
        else:
            print(f"[Controller] Unknown custom action: {action}")
        self._log_action(f"custom_{action}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_action(self, name: str) -> None:
        self._action_counts[name] = self._action_counts.get(name, 0) + 1

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def action_counts(self) -> dict:
        return dict(self._action_counts)

    @property
    def current_volume(self) -> int:
        return self._last_volume_level

    @property
    def screen_size(self) -> Tuple[int, int]:
        return (self._sw, self._sh)
