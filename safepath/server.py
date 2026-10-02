"""
SafePath FastAPI Production Server & Real-Time Asynchronous Engine.

Features:
- WebSocket `/ws/live` for real-time video frame streaming at 30-60 FPS.
- REST APIs under `/api/*` for single-frame detection, scene description, risk calculations, voice Q&A, and healthchecks.
- Dynamic runtime configuration API (`GET` and `POST` `/api/config`).
- Mobile PWA Manifest support (`/manifest.json`).
- Zero-downtime lazy AI model loading & backpressure frame-dropping for edge/cloud deployment.
- Automatic Local Network IP discovery for seamless mobile/tablet connections over Wi-Fi.

Usage:
    python server.py                           # run on http://0.0.0.0:8000
    python run_server.py                       # universal runner
    python -m safepath.server                  # package runner
    python -m safepath.server --port 8080      # custom port
"""

import argparse
import asyncio
import base64
import io
import json
import os
import socket
import sys
import time
import urllib.parse
import urllib.request
import threading
from typing import Dict, List, Optional

# Ensure both current directory and parent directory are on sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
for _p in [_CURRENT_DIR, _PARENT_DIR]:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from safepath import config
    from safepath.core.depth_estimator import get_depth_estimator
    from safepath.core.detector import ObstacleDetector
    from safepath.core.message_builder import build_warning
    from safepath.core.position import classify_position, direction_phrase
    from safepath.core.risk_engine import build_scored_detection, score_detection, select_highest_risk, _risk_level_from_score
    from safepath.core.types import Detection, Position, RiskLevel, ScoredDetection
    from safepath.core.tracker import IoUTracker
    from safepath.utils.logger import get_logger
except ImportError:
    import config
    from core.depth_estimator import get_depth_estimator
    from core.detector import ObstacleDetector
    from core.message_builder import build_warning
    from core.position import classify_position, direction_phrase
    from core.risk_engine import build_scored_detection, score_detection, select_highest_risk, _risk_level_from_score
    from core.types import Detection, Position, RiskLevel, ScoredDetection
    from core.tracker import IoUTracker
    from utils.logger import get_logger

logger = get_logger(__name__)

# Server metadata
SERVER_START_TIME = time.time()
_candidate_web_dirs = [
    os.path.join(_CURRENT_DIR, "ui", "web"),
    os.path.join(_CURRENT_DIR, "web"),
    os.path.join(_PARENT_DIR, "ui", "web"),
    os.path.join(_PARENT_DIR, "web"),
    os.path.join(_CURRENT_DIR, "safepath", "ui", "web"),
    os.path.join(_CURRENT_DIR, "safepath", "web"),
]
WEB_DIR = next((p for p in _candidate_web_dirs if os.path.isdir(p)), os.path.join(_CURRENT_DIR, "ui", "web"))


# Global lazy-loaded AI models
detector_instance: Optional[ObstacleDetector] = None
depth_estimator_instance = None
active_ws_connections: int = 0
_nominatim_lock = threading.Lock()
_last_nominatim_request = 0.0


def get_local_ip() -> str:
    """Discovers the primary LAN IP address for mobile devices to connect to."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        # Doesn't need to be reachable, just initiates route lookup
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_models(depth_backend: str = config.DEPTH_BACKEND):
    global detector_instance, depth_estimator_instance
    if detector_instance is None:
        logger.info("Initializing YOLOv8n ObstacleDetector...")
        detector_instance = ObstacleDetector()
    if depth_estimator_instance is None:
        logger.info("Initializing DepthEstimator (backend=%s)...", depth_backend)
        depth_estimator_instance = get_depth_estimator(depth_backend)
    return detector_instance, depth_estimator_instance


# ---------------------------------------------------------------------------
# FastAPI Application & Middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SafePath AI Mobility Companion API",
    description="Real-time obstacle detection, monocular depth estimation, and spatial risk arbitration engine for the visually impaired.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class RiskScoreRequest(BaseModel):
    class_name: str = Field(description="Detected object class")
    category: str = Field(description="Safety category from a live detection")
    distance_m: Optional[float] = Field(description="Estimated distance in meters, if available")
    position: str = Field(description="Position zone: LEFT, CENTER, RIGHT")
    confidence: float = Field(ge=0.0, le=1.0, description="Detector confidence (0.0 to 1.0)")


class ConfigUpdateRequest(BaseModel):
    focal_length_px: Optional[float] = None
    center_zone_fraction: Optional[float] = None
    max_relevant_distance_m: Optional[float] = None
    alert_cooldown_seconds: Optional[float] = None
    confidence_threshold: Optional[float] = None
    depth_backend: Optional[str] = None


class VoiceQARequest(BaseModel):
    question: str = Field(description="User voice or text question")
    current_context: Optional[Dict] = Field(default=None, description="Optional active session context")


class SceneDescribeRequest(BaseModel):
    current_context: Optional[Dict] = Field(default=None, description="Active detections and spatial state")


class DetectionResponseItem(BaseModel):
    class_name: str
    category: str
    confidence: float
    bbox: List[int]
    distance_m: Optional[float]
    position: str
    direction_phrase: str
    risk_score: float
    risk_level: str


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Returns server health, active device, and telemetry status."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    uptime_sec = round(time.time() - SERVER_START_TIME, 1)
    local_ip = get_local_ip()

    return {
        "status": "healthy",
        "service": "SafePath AI Mobility Backend",
        "version": "1.1.0-Universal",
        "device": device,
        "yolo_model": config.YOLO_MODEL_PATH,
        "depth_backend": config.DEPTH_BACKEND,
        "target_classes_count": len(config.TARGET_CLASSES),
        "active_ws_clients": active_ws_connections,
        "local_ip": local_ip,
        "mobile_access_url": f"http://{local_ip}:8000/",
        "uptime_seconds": uptime_sec,
        "timestamp": time.time(),
    }


@app.get("/api/config")
async def get_system_config():
    """Returns system configuration and tunable risk weights."""
    return {
        "focal_length_px": config.FOCAL_LENGTH_PX,
        "center_zone_fraction": config.CENTER_ZONE_FRACTION,
        "max_relevant_distance_m": config.MAX_RELEVANT_DISTANCE_M,
        "alert_cooldown_seconds": config.ALERT_COOLDOWN_SECONDS,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "depth_backend": config.DEPTH_BACKEND,
        "target_classes": config.TARGET_CLASSES,
        "category_weights": config.CATEGORY_WEIGHT,
        "position_weights": config.POSITION_WEIGHT,
        "known_widths_m": config.KNOWN_OBJECT_WIDTH_M,
        "risk_thresholds": config.RISK_LEVEL_THRESHOLDS,
    }


@app.get("/api/geocode")
async def geocode_destination(q: str, lang: str = "en"):
    """Proxy one submitted query to Nominatim with an identifying app UA and rate limit."""
    language = lang if lang in {"en", "te", "hi"} else "en"

    def request_nominatim():
        global _last_nominatim_request
        with _nominatim_lock:
            delay = 1.0 - (time.monotonic() - _last_nominatim_request)
            if delay > 0:
                time.sleep(delay)
            _last_nominatim_request = time.monotonic()
        query = urllib.parse.urlencode({"format": "jsonv2", "limit": 5, "q": q, "accept-language": language})
        request = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/search?{query}",
            headers={"User-Agent": "SafePath-AI-Mobility-Companion/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return await asyncio.to_thread(request_nominatim)
    except Exception as exc:
        logger.warning("Nominatim lookup unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Destination search unavailable") from exc


@app.post("/api/config")
async def update_system_config(req: ConfigUpdateRequest):
    """Dynamically updates tunable runtime thresholds from mobile/web clients."""
    updated = {}
    if req.focal_length_px is not None and req.focal_length_px > 0:
        config.FOCAL_LENGTH_PX = float(req.focal_length_px)
        updated["focal_length_px"] = config.FOCAL_LENGTH_PX

    if req.center_zone_fraction is not None and 0.1 <= req.center_zone_fraction <= 0.8:
        config.CENTER_ZONE_FRACTION = float(req.center_zone_fraction)
        updated["center_zone_fraction"] = config.CENTER_ZONE_FRACTION

    if req.max_relevant_distance_m is not None and req.max_relevant_distance_m > 1.0:
        config.MAX_RELEVANT_DISTANCE_M = float(req.max_relevant_distance_m)
        updated["max_relevant_distance_m"] = config.MAX_RELEVANT_DISTANCE_M

    if req.alert_cooldown_seconds is not None and req.alert_cooldown_seconds >= 0.5:
        config.ALERT_COOLDOWN_SECONDS = float(req.alert_cooldown_seconds)
        updated["alert_cooldown_seconds"] = config.ALERT_COOLDOWN_SECONDS

    if req.confidence_threshold is not None and 0.1 <= req.confidence_threshold <= 0.95:
        config.CONFIDENCE_THRESHOLD = float(req.confidence_threshold)
        updated["confidence_threshold"] = config.CONFIDENCE_THRESHOLD

    if req.depth_backend is not None and req.depth_backend in ["geometric", "midas"]:
        global depth_estimator_instance
        config.DEPTH_BACKEND = req.depth_backend
        depth_estimator_instance = None  # Reload backend on next frame
        updated["depth_backend"] = config.DEPTH_BACKEND

    return {
        "status": "success",
        "message": f"Updated {len(updated)} configuration parameter(s).",
        "updated": updated,
    }


@app.post("/api/risk-score")
async def calculate_risk_score(req: RiskScoreRequest):
    """Calculates risk score for given obstacle parameters."""
    pos_enum = Position(req.position) if req.position in Position._value2member_map_ else Position.CENTER
    detection = Detection(
        class_name=req.class_name,
        category=req.category,
        confidence=req.confidence,
        bbox=(200, 100, 440, 300),
    )

    score = score_detection(detection, req.distance_m, pos_enum)
    level = _risk_level_from_score(score)

    scored = ScoredDetection(
        detection=detection,
        distance_m=req.distance_m,
        depth_source="api_calc",
        position=pos_enum,
        risk_score=score,
        risk_level=level,
    )

    phrase = build_warning(scored, config.FRAME_WIDTH)

    cat_w = config.CATEGORY_WEIGHT.get(req.category, 0.5)
    pos_w = config.POSITION_WEIGHT.get(pos_enum.value, 0.7)
    d_fac = max(0.0, 1.0 - ((req.distance_m or 3.0) / config.MAX_RELEVANT_DISTANCE_M))
    c_fac = 0.5 + 0.5 * req.confidence

    return {
        "class_name": req.class_name,
        "risk_score": score,
        "risk_level": level.value,
        "spoken_phrase": phrase,
        "factors": {
            "category_weight": cat_w,
            "position_weight": pos_w,
            "distance_factor": round(d_fac, 3),
            "confidence_factor": round(c_fac, 3),
        },
    }


@app.post("/api/detect")
async def detect_frame(file: UploadFile = File(...)):
    """Uploads a single camera snapshot (JPEG/PNG) and returns scored detections with bounding boxes."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding")

    detector, depth_estimator = get_models()
    frame_width = frame.shape[1]
    t0 = time.perf_counter()

    det_result = detector.detect_with_telemetry(frame)
    detections = det_result.detections
    t_detect = det_result.total_ms

    scored_list: List[ScoredDetection] = []
    response_items: List[DetectionResponseItem] = []

    for det in detections:
        depth = depth_estimator.estimate(frame, det)
        position = classify_position(det.bbox, frame_width)
        scored = build_scored_detection(det, depth.distance_m, depth.source, position)
        scored_list.append(scored)

        response_items.append(
            DetectionResponseItem(
                class_name=det.class_name,
                category=det.category,
                confidence=round(det.confidence, 3),
                bbox=list(det.bbox),
                distance_m=depth.distance_m,
                position=position.value,
                direction_phrase=direction_phrase(det.bbox, frame_width),
                risk_score=scored.risk_score,
                risk_level=scored.risk_level.value,
            )
        )

    winner = select_highest_risk(scored_list)
    winner_payload = None

    if winner and winner.risk_level != RiskLevel.LOW:
        winner_payload = {
            "class_name": winner.detection.class_name,
            "risk_score": winner.risk_score,
            "risk_level": winner.risk_level.value,
            "position": winner.position.value,
            "distance_m": winner.distance_m,
            "message": build_warning(winner, frame_width),
        }

    total_latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "count": len(response_items),
        "detections": response_items,
        "winner": winner_payload,
        "guidance": winner_payload["message"] if winner_payload else "No supported objects detected in this snapshot.",
        "telemetry": {
            "detector_ms": round(t_detect, 2),
            "preprocess_ms": det_result.preprocess_ms,
            "inference_ms": det_result.inference_ms,
            "postprocess_ms": det_result.postprocess_ms,
            "total_ms": round(total_latency_ms, 2),
            "device": det_result.device,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
        },
    }


@app.post("/api/describe-scene")
async def describe_scene(req: SceneDescribeRequest):
    """Synthesizes a comprehensive, accessible natural-language scene description."""
    ctx = req.current_context or {}
    detections = ctx.get("detections", [])
    lang = ctx.get("lang", "en")

    if not ctx.get("vision_available"):
        summaries = {
            "en": "Live vision unavailable. Start the camera and YOLO server to describe the current scene.",
            "hi": "लाइव कैमरा पहचान उपलब्ध नहीं है। दृश्य विवरण के लिए कैमरा और YOLO सर्वर शुरू करें।",
            "te": "లైవ్ విజన్ అందుబాటులో లేదు. దృశ్యాన్ని వివరించడానికి కెమెరా మరియు YOLO సర్వర్‌ను ప్రారంభించండి.",
        }
        return {
            "summary": summaries.get(lang, summaries["en"]),
            "detail": "Camera feed is offline or not sending frames.",
            "hazard_count": 0,
            "timestamp": time.time(),
        }
    if not detections:
        empty_summaries = {
            "en": "No supported objects detected in the latest camera frame. This does not prove that the path is clear.",
            "hi": "नवीनतम कैमरा फ्रेम में कोई समर्थित वस्तु नहीं मिली। इससे रास्ता सुरक्षित साबित नहीं होता।",
            "te": "తాజా కెమెరా ఫ్రేమ్‌లో మద్దతున్న వస్తువులు గుర్తించబడలేదు. ఇది దారి సురక్షితమని నిరూపించదు.",
        }
        return {
            "summary": empty_summaries.get(lang, empty_summaries["en"]),
            "detail": "0 detections in current frame.",
            "hazard_count": 0,
            "timestamp": time.time(),
        }

    left_items = [d for d in detections if d.get("position") == "LEFT"]
    center_items = [d for d in detections if d.get("position") == "CENTER"]
    right_items = [d for d in detections if d.get("position") == "RIGHT"]
    approaching_items = [d for d in detections if d.get("movement") == "approaching"]

    parts = []
    if center_items:
        center_desc = ", ".join([f"{d.get('class_name')} at ~{d.get('distance_m')}m ({d.get('movement', 'stationary')})" if d.get('distance_m') is not None else f"{d.get('class_name')} ({d.get('movement', 'stationary')})" for d in center_items])
        parts.append(f"In your forward path: {center_desc}.")
    else:
        parts.append("Forward walking path has no detected obstacles.")

    if left_items:
        left_desc = ", ".join([f"{d.get('class_name')} at ~{d.get('distance_m')}m" if d.get('distance_m') is not None else d.get('class_name') for d in left_items])
        parts.append(f"On your left: {left_desc}.")

    if right_items:
        right_desc = ", ".join([f"{d.get('class_name')} at ~{d.get('distance_m')}m" if d.get('distance_m') is not None else d.get('class_name') for d in right_items])
        parts.append(f"On your right: {right_desc}.")

    if approaching_items:
        app_names = ", ".join([d.get("class_name", "object") for d in approaching_items])
        parts.append(f"Approaching: {app_names}.")

    full_text = " ".join(parts)
    return {
        "summary": full_text,
        "detail": f"Detected {len(detections)} object(s) across field of view.",
        "hazard_count": len([d for d in detections if d.get("risk_level") in ("HIGH", "CRITICAL")]),
        "timestamp": time.time(),
    }


@app.post("/api/voice-qa")
async def conversational_voice_qa(req: VoiceQARequest):
    """Processes natural language spatial inquiries with contextual intelligence."""
    query = req.question.lower().replace("?", "").replace("!", "").replace(".", "").strip()
    ctx = req.current_context or {}
    lang = ctx.get("lang", "en")

    if not ctx.get("vision_available"):
        fallback_answers = {
            "en": "Live vision is unavailable. Start the camera and YOLO server to ask questions.",
            "hi": "लाइव कैमरा पहचान उपलब्ध नहीं है। कृपया कैमरा और YOLO सर्वर शुरू करें।",
            "te": "లైవ్ కెమెరా గుర్తింపు అందుబాటులో లేదు. ప్రశ్నలు అడగడానికి కెమెరా, YOLO సర్వర్‌ను ప్రారంభించండి.",
        }
        return {"question": req.question, "answer": fallback_answers.get(lang, fallback_answers["en"]), "timestamp": time.time()}

    detections = ctx.get("detections", [])
    ahead_items = [d for d in detections if d.get("position") == "CENTER"]
    left_items = [d for d in detections if d.get("position") == "LEFT"]
    right_items = [d for d in detections if d.get("position") == "RIGHT"]
    approaching = [d for d in detections if d.get("movement") == "approaching"]
    people = [d for d in detections if d.get("class_name") == "person"]

    describe = lambda items: "; ".join(
        f"{d.get('class_name', 'Object')} #{d.get('track_id', '?')}"
        + (f" at ~{d.get('distance_m')}m" if d.get('distance_m') is not None else "")
        + (f" ({d.get('movement')})" if d.get('movement') and d.get('movement') != 'unknown' else "")
        for d in items
    )

    if not detections:
        response_text = "No supported objects detected in the latest camera frame. This does not prove the path is completely clear."
    elif any(w in query for w in ["ahead", "front", "forward", "सामने", "आगे", "ముందు"]):
        response_text = f"Ahead: {describe(ahead_items)}" if ahead_items else "No objects detected in the forward corridor in the current frame."
    elif any(w in query for w in ["path", "clear", "walk", "safe", "रास्ता", "साफ", "దారి", "ఖాలీ"]):
        if ahead_items:
            response_text = f"The forward path appears occupied: {describe(ahead_items)}."
        else:
            response_text = "No major obstacle detected in the forward path corridor."
    elif any(w in query for w in ["left", "बाएँ", "बाएं", "बाईं", "ఎడమ"]):
        response_text = f"On your left: {describe(left_items)}" if left_items else "No objects detected on your left side."
    elif any(w in query for w in ["right", "दाएँ", "दाएं", "दाईं", "కుడి"]):
        response_text = f"On your right: {describe(right_items)}" if right_items else "No objects detected on your right side."
    elif any(w in query for w in ["approach", "coming", "moving", "आ रहा", "पास आ", "వస్తు", "కదులు"]):
        response_text = f"Approaching objects: {describe(approaching)}" if approaching else "No tracked objects are currently estimated to be approaching."
    elif any(w in query for w in ["people", "person", "human", "कितने लोग", "व्यक्ति", "మంది", "వ్యక్తు"]):
        response_text = f"{len(people)} {'person' if len(people) == 1 else 'people'} detected nearby: {describe(people)}" if people else "No people detected in the current camera frame."
    elif any(w in query for w in ["how many", "count", "कितनी", "ఎన్ని"]):
        response_text = f"{len(detections)} object(s) detected: {describe(detections)}."
    elif any(w in query for w in ["why", "warn", "reason", "चेतावनी", "హెచ్చరిక"]):
        reason = ctx.get("warning_reason") or ctx.get("last_warning")
        response_text = f"SafePath warning reason: {reason}" if reason else "There is no active warning at this time."
    elif any(w in query for w in ["describe", "scene", "around", "surroundings", "everything", "आसपास", "वर्णन", "పరిసరాలు"]):
        response_text = f"Surroundings: {len(detections)} object(s) detected. {describe(detections)}."
    elif any(w in query for w in ["distance", "how far", "दूरी", "దూరం"]):
        closest = ctx.get("closest_distance")
        response_text = f"The nearest tracked obstacle is at approximately {closest} meters." if closest is not None else "Distance is approximate for currently visible objects."
    else:
        response_text = f"Current camera view: {len(detections)} object(s) detected. {describe(detections)}."

    return {
        "question": req.question,
        "answer": response_text,
        "timestamp": time.time(),
    }


@app.get("/api/scenarios")
async def get_scenarios():
    """Compatibility endpoint: no prerecorded scene or fake alert is served."""
    return {"scenarios": [], "message": "No prerecorded detection scenarios are available."}


@app.get("/api/tests/run")
async def run_server_unit_tests():
    """Runs automated Python unit tests and returns pass/fail report."""
    results = []
    t_start = time.perf_counter()

    def add_test(name: str, test_func):
        t0 = time.perf_counter()
        try:
            test_func()
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            results.append({"name": name, "status": "PASSED", "duration_ms": elapsed})
        except Exception as e:
            results.append({"name": name, "status": "FAILED", "error": str(e)})

    # Test 1: Position Center
    add_test(
        "test_center_position_classification",
        lambda: (_ for _ in ()).throw(AssertionError("Not Center")) if classify_position((280, 100, 360, 300), 640) != Position.CENTER else None,
    )

    # Test 2: Position Left / Right
    add_test(
        "test_peripheral_position_classification",
        lambda: (_ for _ in ()).throw(AssertionError("Failed Left/Right")) if (
            classify_position((0, 100, 60, 300), 640) != Position.LEFT
            or classify_position((580, 100, 640, 300), 640) != Position.RIGHT
        ) else None,
    )

    # Test 3: Distance Boundaries
    add_test(
        "test_distance_factor_bounds",
        lambda: (_ for _ in ()).throw(AssertionError("Boundary clamp failed")) if (
            score_detection(Detection("car", "vehicle", 0.9, (0, 0, 1, 1)), 0.0, Position.CENTER) <= 0
        ) else None,
    )

    # Test 4: Single Winner
    add_test(
        "test_single_highest_risk_selection",
        lambda: (_ for _ in ()).throw(AssertionError("Winner arbitration failed")) if (
            select_highest_risk([
                build_scored_detection(Detection("person", "person", 0.8, (0, 0, 1, 1)), 4.0, "geom", Position.LEFT),
                build_scored_detection(Detection("car", "vehicle", 0.95, (0, 0, 1, 1)), 1.5, "geom", Position.CENTER),
            ]).detection.class_name != "car"
        ) else None,
    )

    total_time = round((time.perf_counter() - t_start) * 1000, 2)
    passed_count = sum(1 for r in results if r["status"] == "PASSED")

    return {
        "total_tests": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "total_time_ms": total_time,
        "results": results,
    }


# ---------------------------------------------------------------------------
# WebSocket Video Streaming Pipeline (/ws/live)
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """
    High-frequency bidirectional WebSocket stream.
    Receives camera JPEG frames -> executes YOLOv8 pipeline -> broadcasts spatial risk telemetry.
    Supports both Base64 JSON and binary JPEG frame payloads.
    """
    global active_ws_connections
    await websocket.accept()
    active_ws_connections += 1
    logger.info("WebSocket client connected. Active clients: %d", active_ws_connections)

    tracker = IoUTracker()

    try:
        detector, depth_estimator = get_models()
        frame_width = config.FRAME_WIDTH
        while True:
            message = await websocket.receive()
            raw_bytes = None

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "time": time.time()})
                        continue
                    if "data" in payload:
                        b64_data = payload["data"].split(",")[-1]
                        raw_bytes = base64.b64decode(b64_data)
                    else:
                        continue
                except Exception:
                    continue
            else:
                continue

            if not raw_bytes:
                continue

            t0 = time.perf_counter()
            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            frame_width = frame.shape[1]

            det_result = detector.detect_with_telemetry(frame)
            detections = det_result.detections
            t_detect = det_result.total_ms

            scored_list: List[ScoredDetection] = []
            detection_payloads = []

            t_depth0 = time.perf_counter()
            depths = [depth_estimator.estimate(frame, det) for det in detections]
            t_depth_ms = (time.perf_counter() - t_depth0) * 1000
            t_track0 = time.perf_counter()
            motion_by_index = tracker.update(detections, [depth.distance_m for depth in depths])
            t_tracking_ms = (time.perf_counter() - t_track0) * 1000

            t_safety0 = time.perf_counter()
            for detection_index, det in enumerate(detections):
                depth = depths[detection_index]
                position = classify_position(det.bbox, frame_width)
                scored = build_scored_detection(det, depth.distance_m, depth.source, position)
                scored_list.append(scored)

                detection_payloads.append({
                    "class_name": det.class_name,
                    "category": det.category,
                    "confidence": round(det.confidence, 3),
                    "bbox": list(det.bbox),
                    "distance_m": depth.distance_m,
                    "position": position.value,
                    "direction_phrase": direction_phrase(det.bbox, frame_width),
                    "risk_score": scored.risk_score,
                    "risk_level": scored.risk_level.value,
                    "track_id": det.track_id,
                    "movement": motion_by_index.get(detection_index, "unknown"),
                    "depth_source": depth.source,
                    "velocity_px_s": list(det.velocity or (0.0, 0.0)),
                    "trajectory": [list(point) for point in tracker._tracks[det.track_id].centers],
                })

            winner = select_highest_risk(scored_list)
            winner_payload = None

            if winner and winner.risk_level != RiskLevel.LOW:
                winner_payload = {
                    "class_name": winner.detection.class_name,
                    "risk_score": winner.risk_score,
                    "risk_level": winner.risk_level.value,
                    "position": winner.position.value,
                    "distance_m": winner.distance_m,
                    "message": build_warning(winner, frame_width),
                    "track_id": winner.detection.track_id,
                    "confidence": round(winner.detection.confidence, 3),
                    "bbox": list(winner.detection.bbox),
                    "movement": next((item["movement"] for item in detection_payloads if item["track_id"] == winner.detection.track_id), "unknown"),
                }

            forward_obstacles = [
                item for item in detection_payloads
                if item["position"] == "CENTER"
                and (item["distance_m"] is None or item["distance_m"] <= 4.0 or (item["movement"] == "approaching" and item["distance_m"] <= 6.0))
            ]
            left_obstacles = [
                item for item in detection_payloads
                if item["position"] == "LEFT"
                and (item["distance_m"] is None or item["distance_m"] <= 3.5)
            ]
            right_obstacles = [
                item for item in detection_payloads
                if item["position"] == "RIGHT"
                and (item["distance_m"] is None or item["distance_m"] <= 3.5)
            ]
            peripheral_hazards = [
                item for item in detection_payloads
                if item["position"] in ("LEFT", "RIGHT")
                and item["movement"] == "approaching"
                and (item["distance_m"] is not None and item["distance_m"] <= 3.5)
            ]

            if any(item["risk_level"] in ("HIGH", "CRITICAL") or (item["movement"] == "approaching" and item["distance_m"] is not None and item["distance_m"] <= 2.5) for item in forward_obstacles):
                safety_state = "DANGER"
            elif forward_obstacles or peripheral_hazards:
                safety_state = "CAUTION"
            else:
                safety_state = "SAFE"

            if forward_obstacles:
                if not left_obstacles and right_obstacles:
                    steering_recommendation = "steer_left"
                elif not right_obstacles and left_obstacles:
                    steering_recommendation = "steer_right"
                elif not left_obstacles and not right_obstacles:
                    steering_recommendation = "clear_sides"
                else:
                    steering_recommendation = "blocked"
            else:
                steering_recommendation = "clear_ahead"

            t_safety_ms = (time.perf_counter() - t_safety0) * 1000
            t_total = (time.perf_counter() - t0) * 1000

            await websocket.send_json({
                "type": "detection_result",
                "count": len(detection_payloads),
                "detections": detection_payloads,
                "winner": winner_payload,
                "safety_state": safety_state,
                "forward_obstacles": forward_obstacles,
                "left_obstacles": left_obstacles,
                "right_obstacles": right_obstacles,
                "steering_recommendation": steering_recommendation,
                "guidance": winner_payload["message"] if winner_payload else "No supported objects detected in the latest camera frame.",
                "telemetry": {
                    "detector_ms": round(t_detect, 1),
                    "preprocess_ms": det_result.preprocess_ms,
                    "inference_ms": det_result.inference_ms,
                    "postprocess_ms": det_result.postprocess_ms,
                    "depth_ms": round(t_depth_ms, 2),
                    "tracking_ms": round(t_tracking_ms, 2),
                    "safety_ms": round(t_safety_ms, 2),
                    "total_pipeline_ms": round(t_total, 1),
                    "fps": round(1000.0 / max(t_total, 1.0), 1),
                    "backend": config.DEPTH_BACKEND,
                    "device": det_result.device,
                },
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket streaming error: %s", str(e))
        try:
            await websocket.send_json({"type": "service_error", "service": "yolo", "message": "YOLO inference is unavailable. Check model weights and backend dependencies."})
        except Exception:
            pass
    finally:
        active_ws_connections = max(0, active_ws_connections - 1)


# ---------------------------------------------------------------------------
# Static Web Frontend & PWA Manifest Mounting
# ---------------------------------------------------------------------------

@app.get("/manifest.json")
async def get_pwa_manifest():
    """Serves PWA manifest for native installation on Android/iOS/iPadOS devices."""
    manifest_path = os.path.join(WEB_DIR, "manifest.json")
    if os.path.isfile(manifest_path):
        return FileResponse(manifest_path, media_type="application/manifest+json")
    return JSONResponse({
        "name": "SafePath AI Mobility Companion",
        "short_name": "SafePath",
        "description": "Real-time AI obstacle assistant & spatial navigation companion",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#050e1d",
        "theme_color": "#06b6d4",
        "icons": [
            {
                "src": "icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            }
        ]
    })


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static_web")


# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------

def run_server():
    parser = argparse.ArgumentParser(description="SafePath Production FastAPI Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Bind host (default: 0.0.0.0 for LAN & mobile access)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Bind port (default: 8000)")
    parser.add_argument("--depth-backend", default=config.DEPTH_BACKEND, choices=["geometric", "midas"])
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")
    args = parser.parse_args()

    config.DEPTH_BACKEND = args.depth_backend
    local_ip = get_local_ip()

    print("\n" + "=" * 74)
    print("  [SafePath] AI Mobility Companion -- Real-Time Multi-Device Server")
    print("=" * 74)
    print(f"  * Local Web App:      http://localhost:{args.port}/")
    print(f"  * Mobile & Tablet:    http://{local_ip}:{args.port}/")
    print(f"  * Swagger API Docs:   http://localhost:{args.port}/docs")
    print(f"  * WebSocket Stream:   ws://{local_ip}:{args.port}/ws/live")
    print(f"  * Depth Backend:      {args.depth_backend}")
    print(f"  * AI Acceleration:    {'CUDA GPU' if torch.cuda.is_available() else 'CPU Multi-Threading'}")
    print("=" * 74)
    print(f"  Tip: Open http://{local_ip}:{args.port}/ on your phone or tablet on same Wi-Fi!")
    print("=" * 74 + "\n")

    import uvicorn
    if args.reload:
        try:
            import safepath.server
            target = "safepath.server:app"
        except ImportError:
            target = "server:app"
        uvicorn.run(target, host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run_server()
