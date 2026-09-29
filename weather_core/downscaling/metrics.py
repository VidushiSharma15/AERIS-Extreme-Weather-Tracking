from dataclasses import dataclass, asdict
import numpy as np
from typing import Dict, Any


@dataclass
class AmplitudePreservationMetrics:
    """
    Dataclass capturing extreme amplitude preservation metrics between coarse and fine fields.
    """
    coarse_max: float
    fine_max: float
    coarse_min: float
    fine_min: float
    coarse_mean: float
    fine_mean: float
    coarse_p95: float
    fine_p95: float
    coarse_p99: float
    fine_p99: float
    peak_ratio: float
    peak_difference: float
    peak_preservation_score: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def calculate(cls, coarse_arr: np.ndarray, fine_arr: np.ndarray) -> "AmplitudePreservationMetrics":
        """
        Calculates extreme value and percentile preservation statistics.
        """
        c_vals = coarse_arr[~np.isnan(coarse_arr)]
        f_vals = fine_arr[~np.isnan(fine_arr)]

        c_max = float(np.max(c_vals)) if len(c_vals) > 0 else 0.0
        f_max = float(np.max(f_vals)) if len(f_vals) > 0 else 0.0
        c_min = float(np.min(c_vals)) if len(c_vals) > 0 else 0.0
        f_min = float(np.min(f_vals)) if len(f_vals) > 0 else 0.0

        c_mean = float(np.mean(c_vals)) if len(c_vals) > 0 else 0.0
        f_mean = float(np.mean(f_vals)) if len(f_vals) > 0 else 0.0

        c_p95 = float(np.percentile(c_vals, 95)) if len(c_vals) > 0 else 0.0
        f_p95 = float(np.percentile(f_vals, 95)) if len(f_vals) > 0 else 0.0

        c_p99 = float(np.percentile(c_vals, 99)) if len(c_vals) > 0 else 0.0
        f_p99 = float(np.percentile(f_vals, 99)) if len(f_vals) > 0 else 0.0

        # Avoid division by zero
        denom = c_max if abs(c_max) > 1e-6 else 1.0
        peak_ratio = f_max / denom
        peak_diff = f_max - c_max

        # Preservation score (1.0 is perfect preservation)
        peak_preservation_score = 1.0 - min(1.0, abs(peak_diff) / (abs(c_max) + 1e-6))

        return cls(
            coarse_max=c_max,
            fine_max=f_max,
            coarse_min=c_min,
            fine_min=f_min,
            coarse_mean=c_mean,
            fine_mean=f_mean,
            coarse_p95=c_p95,
            fine_p95=f_p95,
            coarse_p99=c_p99,
            fine_p99=f_p99,
            peak_ratio=peak_ratio,
            peak_difference=peak_diff,
            peak_preservation_score=peak_preservation_score,
        )
