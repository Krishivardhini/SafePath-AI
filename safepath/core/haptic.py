"""
Haptic feedback abstraction.

A desktop/webcam dev environment has no vibration motor, so the default
implementation simulates it (console log). When SafePath is deployed to a
phone (e.g. wrapped in Kivy/BeeWare, or paired with a native app via a
local socket/BLE), swap in a real implementation of `HapticFeedback` --
nothing else in the codebase needs to change.
"""

from abc import ABC, abstractmethod

from safepath.core.types import RiskLevel
from safepath.utils.logger import get_logger

logger = get_logger(__name__)

# Vibration pattern names, not raw durations -- keeps the interface stable
# across very different underlying vibration APIs (Android, iOS, BLE wearables).
_PATTERN_BY_RISK = {
    RiskLevel.LOW: None,               # no haptic for low risk
    RiskLevel.MEDIUM: "SHORT_PULSE",
    RiskLevel.HIGH: "DOUBLE_PULSE",
    RiskLevel.CRITICAL: "URGENT_BUZZ",
}


class HapticFeedback(ABC):
    @abstractmethod
    def vibrate(self, pattern: str) -> None:
        ...

    def trigger_for_risk(self, risk_level: RiskLevel) -> None:
        pattern = _PATTERN_BY_RISK.get(risk_level)
        if pattern:
            self.vibrate(pattern)


class ConsoleHapticSimulator(HapticFeedback):
    """Default MVP implementation: logs what would have vibrated."""

    def vibrate(self, pattern: str) -> None:
        logger.info("[HAPTIC] pattern=%s", pattern)
