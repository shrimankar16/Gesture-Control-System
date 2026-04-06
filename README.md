# 🖐️ Gesture Control System

Control your PC entirely with hand gestures — real-time, production-ready, and fully customisable.

---

## ✨ Features

| Gesture | Action |
|---|---|
| Open Palm | Move cursor (normal speed) |
| Index Finger Point | Precision cursor mode (slower, accurate) |
| Pinch (close) | Left click / drag |
| Two Fingers (index + middle) | Scroll up/down |
| Thumb + Index spread | Volume control |
| Fist | Pause / resume all gesture control |
| Custom gestures | User-defined (screenshot, copy, tabs, etc.) |

- 🎯 **20+ FPS** real-time performance
- 🔄 **Smooth cursor** with EMA + velocity dampening
- 🛡️ **Anti-jitter** stability buffer (confirms gestures over N frames)
- ⚙️ **Per-action cooldowns** prevent accidental triggers
- 📏 **Hand-scale normalisation** works at any camera distance
- 🎛️ **Tkinter settings GUI** for live parameter tuning
- 🔧 **Calibration wizard** to personalise thresholds to your hand
- 📝 **JSON user gestures** — define custom finger patterns → actions
- 🪟 **Cross-platform** — Windows, macOS, Linux

---

## 🛠️ Setup

### Requirements

- **Python 3.10+**
- A webcam

### Install

```bash
# 1. Clone / download this repository
cd gesture_control

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# Windows users — also install volume control library:
pip install pycaw comtypes

# Optional brightness control:
pip install screen-brightness-control
```

### Launch

```bash
# Standard launch
python main.py

# With settings GUI (sliders for all thresholds)
python main.py --gui

# Run calibration wizard at startup
python main.py --calibrate

# Debug overlay (shows raw gesture values)
python main.py --debug

# Specify a different camera
python main.py --camera 1

# All options
python main.py --help
```

---

## 🕹️ How Gestures Work

### Gesture Detection Pipeline

```
Webcam Frame
     │
     ▼
HandTracker (MediaPipe)
  └─ 21 normalised landmarks per hand
     │
     ▼
GestureDetector
  ├─ FingerAnalyser: per-finger extension state + distances
  ├─ Priority-ordered rule chain (6 built-in + custom)
  ├─ StabilityBuffer: N-frame confirmation window
  └─ GestureResult: gesture enum + cursor pos + payload
     │
     ▼
SystemController
  ├─ SmoothCursor: EMA + velocity dampening → PyAutoGUI moveTo
  ├─ CooldownTimer: per-action throttling
  ├─ VolumeController: pycaw / osascript / amixer
  └─ PyAutoGUI: click, scroll, hotkeys
```

### Gesture Rules (in priority order)

| # | Gesture | Rule |
|---|---|---|
| 1 | Fist | ≥85 % of fingers curled + thumb curled |
| 2 | Pinch Closed | Thumb–index distance < `pinch_threshold` (0.055) |
| 3 | Two-Finger Scroll | Index + middle extended, ring + pinky curled |
| 4 | Volume Control | Thumb + index extended only (spread controls level) |
| 5 | Index Point | Only index extended |
| 6 | Open Palm | ≥4 fingers extended |
| 7 | Pinch Open | Thumb–index distance < `pinch_release_threshold` |

### Cursor Smoothing

The cursor uses a two-stage filter:

1. **Margin clipping** — the outer 10 % of the frame is dead-zone to avoid edge jitter.
2. **EMA (Exponential Moving Average)** — `new_pos = α × target + (1-α) × old_pos` where `α = 0.25`.
3. **Velocity dampening** — a residual velocity term reduces overshoot.

The hand's **index fingertip** position drives the cursor in all movement modes.

---

## ⚙️ Customisation Guide

### 1. Edit `gesture_config.json` (auto-created on first run)

After first launch, a `gesture_config.json` file is saved. Edit it directly or via `--gui`.

Key parameters:

```json
{
  "gesture": {
    "pinch_threshold": 0.055,
    "pinch_release_threshold": 0.08,
    "fist_curl_threshold": 0.85,
    "scroll_sensitivity": 20.0,
    "vol_distance_min": 0.03,
    "vol_distance_max": 0.35,
    "click_cooldown": 0.35,
    "stability_frames": 3
  },
  "cursor": {
    "smooth_alpha": 0.25,
    "margin": 0.10,
    "precision_speed": 0.4
  }
}
```

### 2. Add Custom Gestures — `user_gestures.json`

Define your own finger patterns → actions:

```json
[
  {
    "name": "Peace Sign",
    "fingers": [false, true, true, false, false],
    "action": "screenshot"
  },
  {
    "name": "Rock On",
    "fingers": [false, true, false, false, true],
    "action": "next_tab"
  }
]
```

**`fingers` array** = `[thumb, index, middle, ring, pinky]` — `true` = extended.

**Available actions:** `screenshot`, `copy`, `paste`, `undo`, `close_window`, `next_tab`, `prev_tab`.

To add new actions, edit the `_handle_custom()` method in `controller.py`.

### 3. Calibration Wizard

Run `python main.py --calibrate` or press **C** during a session. The wizard walks you through:

1. Pinch fully closed → sets `pinch_threshold`
2. Pinch slightly open → sets `pinch_release_threshold`
3. Hand flat → sets `vol_distance_min`
4. Thumb+index fully spread → sets `vol_distance_max`

Results are saved to `gesture_config.json` automatically.

### 4. Dominant Hand

Set `tracking.dominant_hand` to `"Left"`, `"Right"`, or `"Any"` in config or via the GUI.

### 5. Multi-hand

Set `tracking.max_num_hands: 2`. The system uses the first detected hand matching `dominant_hand`.
When `dominant_hand = "Any"`, the first detected hand is used.

---

## 🧪 Running Tests

```bash
# With pytest
pytest tests/ -v

# Without pytest
python tests/test_gesture_detector.py
```

Tests cover:
- `FingerAnalyser` — extension detection, distance, curl fraction
- `StabilityBuffer` — consecutive frame logic
- `GestureDetector` — all 6 gesture types, stability, custom gestures
- Integration smoke test — all gestures cycle without crashing

---

## 📂 Project Structure

```
gesture_control/
├── main.py                  # Entry point, camera loop, OSD, GUI
├── config.py                # All configuration dataclasses + JSON I/O
├── hand_tracker.py          # MediaPipe wrapper + drawing helpers
├── gesture_detector.py      # Gesture classification + stability filter
├── controller.py            # System control (cursor, click, volume, scroll)
├── requirements.txt
├── gesture_config.json      # Auto-generated config (editable)
├── user_gestures.json       # Custom gesture definitions
└── tests/
    └── test_gesture_detector.py
```

---

## 🔑 Keyboard Shortcuts (while running)

| Key | Action |
|---|---|
| `Q` | Quit |
| `C` | Open calibration wizard |
| `D` | Toggle debug overlay |
| `S` | Save current config to JSON |

---

## 🐛 Troubleshooting

**Camera not opening**
→ Try `--camera 1` or `--camera 2`. On Linux, check `/dev/video*` permissions.

**Gestures not detected / too sensitive**
→ Run `--calibrate` or open `--gui` and adjust thresholds. Ensure good, even lighting.

**Cursor jittery**
→ Lower `smooth_alpha` (e.g. 0.15) and/or increase `stability_frames` (e.g. 5).

**Volume control not working on Linux**
→ Ensure `alsa-utils` is installed: `sudo apt install alsa-utils`. Or install `screen-brightness-control` and `pulsectl`.

**False clicks**
→ Increase `pinch_threshold` (e.g. 0.04) or `click_cooldown` (e.g. 0.5).

---

## 📄 Licence

MIT — free to use, modify, and distribute.
