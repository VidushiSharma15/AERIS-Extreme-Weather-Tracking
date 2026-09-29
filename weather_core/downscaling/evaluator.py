import numpy as np
from typing import Dict, Any, Optional
from .metrics import AmplitudePreservationMetrics


class DownscalingEvaluator:
    """
    Benchmark evaluation engine for comparing coarse fields, baseline downscaled fields,
    and high-resolution reference targets.
    """

    @classmethod
    def evaluate_downscaling_quality(
        self,
        coarse_arr: np.ndarray,
        fine_arr: np.ndarray,
        reference_arr: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates spatial refinement, amplitude preservation, and error metrics.
        """
        amplitude_metrics = AmplitudePreservationMetrics.calculate(coarse_arr, fine_arr).to_dict()

        # Resample coarse to match fine shape for point-by-point comparison if shapes differ
        if coarse_arr.shape != fine_arr.shape:
            # Resize coarse array to fine array length via 1D linear interpolation for global evaluation metrics
            c_flat_orig = coarse_arr.flatten()
            f_flat = fine_arr.flatten()
            x_coarse = np.linspace(0, 1, len(c_flat_orig))
            x_fine = np.linspace(0, 1, len(f_flat))
            c_flat = np.interp(x_fine, x_coarse, c_flat_orig)
        else:
            c_flat = coarse_arr.flatten()
            f_flat = fine_arr.flatten()

        valid_mask = ~np.isnan(c_flat) & ~np.isnan(f_flat)
        c_valid = c_flat[valid_mask]
        f_valid = f_flat[valid_mask]

        mae = float(np.mean(np.abs(f_valid - c_valid))) if len(c_valid) > 0 else 0.0
        rmse = float(np.sqrt(np.mean((f_valid - c_valid) ** 2))) if len(c_valid) > 0 else 0.0

        if len(c_valid) > 1 and np.std(c_valid) > 1e-6 and np.std(f_valid) > 1e-6:
            corr = float(np.corrcoef(c_valid, f_valid)[0, 1])
        else:
            corr = 1.0

        res = {
            "spatial_mae": mae,
            "spatial_rmse": rmse,
            "pearson_correlation": corr,
            "amplitude_preservation": amplitude_metrics,
        }

        # If reference high-res ground truth data is available
        if reference_arr is not None:
            r_flat_orig = reference_arr.flatten()
            if len(r_flat_orig) != len(f_flat):
                x_ref = np.linspace(0, 1, len(r_flat_orig))
                x_fine = np.linspace(0, 1, len(f_flat))
                ref_flat = np.interp(x_fine, x_ref, r_flat_orig)
            else:
                ref_flat = r_flat_orig

            valid_ref_mask = ~np.isnan(f_flat) & ~np.isnan(ref_flat)
            if np.sum(valid_ref_mask) > 0:
                ref_mae = float(np.mean(np.abs(f_flat[valid_ref_mask] - ref_flat[valid_ref_mask])))
                ref_rmse = float(np.sqrt(np.mean((f_flat[valid_ref_mask] - ref_flat[valid_ref_mask]) ** 2)))
            else:
                ref_mae = 0.0
                ref_rmse = 0.0
            res["reference_comparison"] = {
                "ref_mae": ref_mae,
                "ref_rmse": ref_rmse,
            }

        return res
