"""
Decides WHEN to actually speak/vibrate (not just what to say), and runs
TTS on a background thread so it never blocks the camera loop.

Debounce rules:
  - A CRITICAL alert always interrupts and speaks immediately.
  - Otherwise, repeat alerts about the same risk level are suppressed
    until ALERT_COOLDOWN_SECONDS has passed, to avoid nagging the user
    every single frame.
  - A change in message content (different object/direction) is always
    allowed to speak once the cooldown for its risk level has elapsed.
"""

import queue
import threading
import time
from typing import Optional

from safepath import config
from safepath.core.haptic import HapticFeedback, ConsoleHapticSimulator
from safepath.core.types import RiskLevel, ScoredDetection
from safepath.core.message_builder import build_warning
from safepath.utils.logger import get_logger

logger = get_logger(__name__)


class AlertManager:
    def __init__(
        self,
        enable_audio: bool = True,
        enable_haptic: bool = True,
        haptic: Optional[HapticFeedback] = None,
        cooldown_seconds: float = config.ALERT_COOLDOWN_SECONDS,
    ):
        self._enable_audio = enable_audio
        self._enable_haptic = enable_haptic
        self._haptic = haptic or ConsoleHapticSimulator()
        self._cooldown_seconds = cooldown_seconds

        self._last_spoken_time = 0.0
        self._last_message = None
        self._last_risk_level = RiskLevel.LOW

        self._tts_queue: "queue.Queue[str]" = queue.Queue(maxsize=1)
        self._tts_thread = None
        if self._enable_audio:
            self._start_tts_worker()

        self.last_message_spoken = ""  # exposed for the UI overlay

    # -- TTS worker -----------------------------------------------------

    def _start_tts_worker(self):
        try:
            import pyttsx3  # local import: optional dependency
        except ImportError:
            logger.warning(
                "pyttsx3 not installed; running with audio disabled. "
                "Run: pip install pyttsx3"
            )
            self._enable_audio = False
            return

        def worker():
            engine = pyttsx3.init()
            engine.setProperty("rate", config.TTS_RATE_WPM)
            engine.setProperty("volume", config.TTS_VOLUME)
            while True:
                text = self._tts_queue.get()
                if text is None:  # sentinel -> shut down
                    break
                engine.say(text)
                engine.runAndWait()

        self._tts_thread = threading.Thread(target=worker, daemon=True)
        self._tts_thread.start()

    def _speak(self, text: str):
        if not self._enable_audio:
            return
        # Non-blocking: drop the previous queued message if the worker is
        # still busy speaking, so audio never lags behind reality.
        try:
            self._tts_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._tts_queue.put_nowait(text)
        except queue.Full:
            pass

    # -- Public API -------------------------------------------------------

    def process(self, top: Optional[ScoredDetection], frame_width: int) -> Optional[str]:
        """
        Call once per frame with the current highest-risk detection (or
        None if the frame is clear). Returns the message that was spoken
        this call, or None if nothing new was announced.
        """
        if top is None or top.risk_level == RiskLevel.LOW:
            self._last_risk_level = RiskLevel.LOW
            return None

        message = build_warning(top, frame_width)
        now = time.monotonic()

        is_new_critical = top.risk_level == RiskLevel.CRITICAL
        cooldown_elapsed = (now - self._last_spoken_time) >= self._cooldown_seconds
        content_changed = message != self._last_message

        should_speak = is_new_critical or (cooldown_elapsed and content_changed) or (
            cooldown_elapsed and self._last_risk_level != top.risk_level
        )

        if not should_speak:
            return None

        self._speak(message)
        if self._enable_haptic:
            self._haptic.trigger_for_risk(top.risk_level)

        self._last_spoken_time = now
        self._last_message = message
        self._last_risk_level = top.risk_level
        self.last_message_spoken = message
        return message

    def shutdown(self):
        if self._tts_thread and self._tts_thread.is_alive():
            try:
                self._tts_queue.put_nowait(None)
            except queue.Full:
                pass
