from enum import Enum


class ExtremeEventType(str, Enum):
    EXTREME_PRECIPITATION = "extreme_precipitation"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    EXTREME_WIND = "extreme_wind"
    EXTREME_PRESSURE = "extreme_pressure"
    UNKNOWN = "unknown"

    @classmethod
    def from_variable(cls, var_name: str, z_score: float = 0.0) -> "ExtremeEventType":
        name_lower = var_name.lower()
        if "precip" in name_lower or name_lower in ("tp", "pr"):
            return cls.EXTREME_PRECIPITATION
        elif "temp" in name_lower or name_lower in ("t2m", "t"):
            return cls.EXTREME_HEAT if z_score >= 0 else cls.EXTREME_COLD
        elif "wind" in name_lower or name_lower in ("ws10", "si10"):
            return cls.EXTREME_WIND
        elif "press" in name_lower or name_lower in ("sp", "msl"):
            return cls.EXTREME_PRESSURE
        return cls.UNKNOWN
