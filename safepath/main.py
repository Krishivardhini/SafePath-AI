"""
SafePath MVP entry point.

Usage:
    python main.py                          # default webcam
    python main.py --source 0                # explicit webcam index
    python main.py --source path/to/video.mp4
    python main.py --depth-backend midas
    python main.py --no-display               # audio/haptic only, no window
    python main.py --no-audio                  # visual/haptic only (debugging)

Press 'q' in the video window to quit.
"""

import argparse
import os
import sys

import cv2

# Allow running this file directly (`python main.py` from inside safepath/)
# as well as as a module (`python -m safepath.main` from the parent dir).
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from safepath import config
from safepath.core.detector import ObstacleDetector
from safepath.core.depth_estimator import get_depth_estimator
from safepath.core.position import classify_position
from safepath.core.risk_engine import build_scored_detection, select_highest_risk
from safepath.core.alert_manager import AlertManager
from safepath.ui.overlay import draw_overlay
from safepath.utils.fps_counter import FPSCounter
from safepath.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SafePath real-time obstacle assistant")
    parser.add_argument("--source", default=config.RuntimeFlags.source,
                         help="Camera index (e.g. 0) or path to a video file")
    parser.add_argument("--depth-backend", default=config.DEPTH_BACKEND,
                         choices=["geometric", "midas"])
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-haptic", action="store_true")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    # numeric strings mean "webcam index"; anything else is a file path.
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        logger.error("Could not open video source: %s", source)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def run():
    args = parse_args()

    logger.info("Starting SafePath | source=%s depth_backend=%s",
                args.source, args.depth_backend)

    detector = ObstacleDetector()
    depth_estimator = get_depth_estimator(args.depth_backend)
    alert_manager = AlertManager(
        enable_audio=not args.no_audio,
        enable_haptic=not args.no_haptic,
    )
    fps_counter = FPSCounter()

    cap = open_capture(args.source)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.info("End of stream / camera read failed. Exiting.")
                break

            frame_width = frame.shape[1]

            detections = detector.detect(frame)

            scored = []
            for det in detections:
                depth = depth_estimator.estimate(frame, det)
                position = classify_position(det.bbox, frame_width)
                scored.append(build_scored_detection(
                    det, depth.distance_m, depth.source, position
                ))

            top = select_highest_risk(scored)
            alert_manager.process(top, frame_width)

            fps = fps_counter.tick()

            if not args.no_display:
                annotated = draw_overlay(
                    frame, scored, top, alert_manager.last_message_spoken, fps
                )
                cv2.imshow("SafePath (dev view - not the end-user interface)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alert_manager.shutdown()
        logger.info("SafePath stopped.")


if __name__ == "__main__":
    run()
