from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class VariableThresholdConfig:
    """
    Documented threshold configuration for a specific weather variable.
    """
    variable_name: str
    z_score_threshold: float = 2.0
    percentile_threshold: float = 95.0
    absolute_threshold: Optional[float] = None
    units: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_name": self.variable_name,
            "z_score_threshold": self.z_score_threshold,
            "percentile_threshold": self.percentile_threshold,
            "absolute_threshold": self.absolute_threshold,
            "units": self.units,
        }


class ThresholdConfig:
    """
    Container for all variable threshold configurations in the anomaly engine.
    """

    DEFAULT_CONFIGS = {
        "total_precipitation": VariableThresholdConfig(
            variable_name="total_precipitation",
            z_score_threshold=2.5,
            percentile_threshold=98.0,
            absolute_threshold=10.0,  # mm/hr or mm total
            units="mm",
        ),
        "2m_temperature": VariableThresholdConfig(
            variable_name="2m_temperature",
            z_score_threshold=2.0,
            percentile_threshold=95.0,
            absolute_threshold=None,
            units="K",
        ),
        "10m_wind_speed": VariableThresholdConfig(
            variable_name="10m_wind_speed",
            z_score_threshold=2.0,
            percentile_threshold=95.0,
            absolute_threshold=15.0,  # m/s (~54 km/h)
            units="m/s",
        ),
    }

    def __init__(self, custom_configs: Optional[Dict[str, VariableThresholdConfig]] = None):
        self.configs = self.DEFAULT_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)

    def get(self, var_name: str) -> VariableThresholdConfig:
        return self.configs.get(
            var_name,
            VariableThresholdConfig(variable_name=var_name, z_score_threshold=2.0, percentile_threshold=95.0),
        )
