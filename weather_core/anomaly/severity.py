from enum import Enum
from typing import Dict, Any


class SeverityLevel(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class SeverityEngine:
    """
    SeverityEngine classifies extreme weather events into analytical severity levels.
    
    Disclaimer:
    These categories are internal analytical severity classifications for this prototype pipeline
    and do not represent official government weather warning categories (e.g. IMD Red/Orange alerts)
    unless an official government source is explicitly integrated.
    
    Default Threshold Mapping (Configurable):
    - NORMAL  : Z < 2.0
    - WATCH   : 2.0 <= Z < 3.0
    - MODERATE: 3.0 <= Z < 4.5
    - SEVERE  : Z >= 4.5
    """

    DEFAULT_THRESHOLDS = {
        "watch_z": 2.0,
        "moderate_z": 3.0,
        "severe_z": 4.5,
    }

    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def classify_z_score(self, z_score: float) -> SeverityLevel:
        abs_z = abs(z_score)
        if abs_z >= self.thresholds.get("severe_z", 4.5):
            return SeverityLevel.SEVERE
        elif abs_z >= self.thresholds.get("moderate_z", 3.0):
            return SeverityLevel.MODERATE
        elif abs_z >= self.thresholds.get("watch_z", 2.0):
            return SeverityLevel.WATCH
        return SeverityLevel.NORMAL
