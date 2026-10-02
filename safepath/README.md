# SafePath — Real-Time Obstacle Assistant (MVP)

SafePath is a real-time assistive-technology prototype for visually impaired
users. It watches a camera feed, detects obstacles, estimates roughly how
far away and in what direction they are, scores how dangerous each one is
right now, and speaks **one short, actionable warning** about the single
most dangerous thing — instead of narrating every object it sees.

> **Safety disclaimer:** SafePath is an assistive aid, not a safety
> guarantee. Distance estimates are approximate. It can miss obstacles,
> misclassify them, or be wrong about distance. It must not be used as a
> replacement for a white cane, guide dog, training, or human judgment.
> This prototype cannot guarantee safe walking. Live guidance requires the
> camera and FastAPI/YOLO server to be connected; if either is unavailable,
> it reports that live detection is unavailable instead of using the demo
> scene. If live vision is unavailable, SafePath reports that state instead
> of substituting a canned scene.
> Tracking and motion labels are approximate and derived from consecutive
> camera frames. Monocular distance values
> are rough estimates rather than measured distances.

---

## Architecture

```
Camera frame
   │
   ▼
ObstacleDetector (YOLOv8n, pretrained)         core/detector.py
   │  exposes every class from the pretrained model
   ▼
IoUTracker (per live camera connection)         core/tracker.py
   │  stable IDs, short trajectories, rough motion labels
   ▼
DepthEstimator (geometric | MiDaS)             core/depth_estimator.py
   │  approximate distance per detection
   ▼
classify_position()                            core/position.py
   │  LEFT / CENTER / RIGHT + direction phrase
   ▼
RiskEngine.score + select_highest_risk()       core/risk_engine.py
   │  ONE most dangerous obstacle wins
   ▼
message_builder.build_warning()                core/message_builder.py
   │  short natural-language sentence
   ▼
AlertManager (debounced)                       core/alert_manager.py
   ├─► pyttsx3 TTS (background thread)
   └─► HapticFeedback (console sim / phone hook)   core/haptic.py

ui/overlay.py draws a dev-only debug window (boxes, risk colors, caption bar).
```

**Why this design:**

- **Modular pipeline** — each stage is a small class/function with a clear
  input/output contract (`core/types.py`). You can swap the detector for a
  fine-tuned model, or swap the depth backend, without touching anything
  else.
- **Two depth backends** — `geometric` (default) uses simple pinhole-camera
  math with a table of average object widths: fast, deterministic, no
  extra model download. `midas` is a real monocular depth network for
  better generalization, offered as an upgrade path; if it can't load
  (e.g. no internet), the app automatically falls back to `geometric` and
  logs a warning.
- **One warning per frame, debounced** — `select_highest_risk` always
  returns at most one obstacle. `AlertManager` further suppresses repeat
  announcements of the same message within a cooldown window, except
  CRITICAL risk, which always interrupts immediately.
- **Non-blocking TTS** — speech runs on a background thread with a
  single-slot queue, so the camera loop never stalls waiting for speech
  to finish.

---

## Folder structure

```
safepath/
├── main.py                  # entry point / real-time loop
├── config.py                 # all tunables in one place
├── requirements.txt
├── pytest.ini
├── README.md
├── core/
│   ├── types.py               # Detection, DepthEstimate, ScoredDetection, enums
│   ├── detector.py             # YOLOv8n wrapper
│   ├── tracker.py               # per-camera IDs, motion labels, trajectories
│   ├── depth_estimator.py       # geometric + MiDaS backends
│   ├── position.py               # LEFT/CENTER/RIGHT + direction phrasing
│   ├── risk_engine.py             # risk scoring + highest-risk selection
│   ├── message_builder.py          # builds the spoken sentence
│   ├── alert_manager.py             # debounced TTS + haptic dispatch
│   └── haptic.py                     # vibration abstraction
├── ui/
│   └── overlay.py             # dev-only debug window
├── utils/
│   ├── logger.py
│   └── fps_counter.py
└── tests/
    ├── test_position.py
    ├── test_risk_engine.py
    ├── test_message_builder.py
    ├── test_detector.py
    └── test_tracker.py
```

---

## Installation

Requires Python 3.9+.

```bash
cd safepath
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will auto-download `yolov8n.pt` (~6 MB) via ultralytics.

**macOS TTS note:** `pyttsx3` uses `nsss` on macOS and should work out of
the box. **Linux note:** `pyttsx3` needs `espeak`: `sudo apt install espeak`.

---

## Running it

From the `safepath/` folder:

```bash
# Default webcam, on-screen debug window, audio + haptic simulation on
python main.py

# Explicit webcam index
python main.py --source 0

# Run against a recorded video instead of a live camera
python main.py --source path/to/video.mp4

# Use the heavier MiDaS depth backend instead of the geometric one
python main.py --depth-backend midas

# Headless (no debug window) — closer to how it'd run on-device
python main.py --no-display

# Disable audio (useful while developing with the window open)
python main.py --no-audio
```

Press **`q`** in the video window to quit.

### Run the browser app

From the `safepath/` folder, install `requirements.txt` first, then run:

```bash
python run_server.py
```

Open `http://localhost:8000/` in a modern browser, allow camera access, and
select **START ASSISTANCE**. The backend serves both the frontend and YOLO API;
there is no separate frontend build step. Use the Navigation page to request
location permission and submit a destination. Browser camera/geolocation use
requires a secure context (localhost is treated as secure by modern browsers).
The public map, search, and routing services need internet access.

### Running the tests

The risk scoring, position classification, and message-building logic are
pure Python with no model dependency, so they're fully unit tested:

```bash
pip install pytest
pytest
```

---

## Calibrating the geometric depth estimator

The default depth backend assumes a rough focal length (`FOCAL_LENGTH_PX`
in `config.py`) that will be **inaccurate for your specific camera** until
calibrated. Quick calibration:

1. Place an object of known width (e.g. a person, width ≈ 0.45 m) at a
   known distance (e.g. exactly 2 m) from the camera.
2. Run SafePath and note the detected bounding-box pixel width for that
   object (visible in the debug overlay label, or add a print in
   `detector.py`).
3. Compute: `FOCAL_LENGTH_PX = (pixel_width * known_distance_m) / real_width_m`.
4. Put that value in `config.py`.

Without calibration, treat distance output as a relative "closer/farther"
signal rather than a precise measurement — which is also true even *after*
calibration, since it still relies on population-average object sizes.

---

## What's implemented (MVP scope)

- [x] Real-time camera feed (webcam or recorded video)
- [x] YOLOv8n pretrained object detection
- [x] All pretrained model classes exposed in live object results
- [x] Per-camera IoU tracking IDs, short trajectories, and cautious motion labels
- [x] Obstacle classes: person, bicycle, car, motorcycle, bus, truck,
      chair, backpack/handbag/suitcase, bottle, bench, potted plant, dog
- [x] Approximate monocular distance (geometric backend; optional MiDaS)
- [x] LEFT / CENTER / RIGHT position + "slightly/far left/right" phrasing
- [x] Risk scoring (category × distance × position × confidence)
- [x] Highest-risk-only obstacle selection (never narrates everything)
- [x] Debounced text-to-speech warnings (pyttsx3, offline, non-blocking)
- [x] Haptic feedback abstraction (console-simulated on desktop; ready for
      a real vibration API on-device)
- [x] Accessible debug UI: high-contrast overlay, large caption bar,
      persistent "approximate distance" disclaimer, risk color-coding

## Known limitations / not yet implemented

- **Stairs and poles are not detected.** Standard COCO-pretrained YOLO has
  no "stairs" or "pole" class. `config.TARGET_CLASSES` and the risk engine
  already have a `"structural"` category ready to receive them — this
  needs either a fine-tuned/custom-trained head or a different specialized
  model. Do not assume steps/curbs/poles are covered.
- **Distance is approximate**, more so with the default geometric backend
  (assumes average object sizes) than after per-camera calibration. MiDaS
  gives relative depth, converted to meters with a placeholder calibration
  constant that also needs per-camera tuning.
- **Motion is heuristic.** IDs use same-class bounding-box overlap, and
  movement uses changes in estimated depth and image position. Camera motion,
  depth noise, occlusion, and missed detections can produce incorrect labels.
- **No real phone deployment yet.** This runs as a desktop/webcam Python
  app. Porting to an actual phone (camera permissions, real vibration
  motor, background audio, battery/thermal behavior) is unbuilt.
- **Haptic feedback is simulated** (logged to console) since dev hardware
  has no vibration motor. `core/haptic.py` defines the interface a real
  on-device implementation should satisfy.
- **Single camera only** — no stereo depth, no LiDAR, no sensor fusion.

## Still needs testing

- Real-world accuracy of the geometric depth estimator after calibration,
  across different lighting and camera hardware.
- Behavior in low light / at night (YOLOv8n was trained on daylight-heavy
  data; expect degraded detection).
- TTS latency and audio-cooldown tuning with real users — 2 seconds may be
  too fast or too slow depending on walking speed and comfort.
- False-positive rate for `chair`/`bench`/`potted plant` classes, which
  tend to be over-triggered by YOLO on cluttered indoor scenes.
- Actual phone camera framerate/thermal behavior — this MVP was only
  profiled conceptually, not benchmarked on-device.

## Browser navigation and scene pages

The web app now includes live Objects, Path Assistant, Safety, History,
Accessibility, and Navigation views. Navigation uses the browser Geolocation
API, Leaflet/OpenStreetMap tiles, a user-submitted Nominatim search, and the
public FOSSGIS OSRM foot-routing endpoint. The app sends no map API key and
does not call Google Maps. Public services require internet access and may
rate-limit or be unavailable; the interface reports failures without
substituting sample coordinates or routes. Search is sent only when submitted.

The live History list is held in browser memory for the current page session.
English, Telugu, and Hindi are selectable for the main interface controls, live guidance, and camera-based question answers. Browser speech input uses the selected locale; spoken output requests a matching installed system voice when available. Browser speech recognition and localized speech voices vary by browser/device, and some object names may still fall back to the model's English class labels.
- Usability testing with actual visually impaired users on wording,
  timing, and haptic pattern meaning.
