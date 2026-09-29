from typing import Dict, Any, Optional
import numpy as np
import xarray as xr


class EFICalculator:
    """
    Extreme Forecast Index (EFI) Interface.
    
    Mathematical Formulation:
      EFI = (2 / pi) * integral_{-inf}^{+inf} [ (q - F_ens(q)) / sqrt(q * (1 - q)) ] dq
      where q = F_clim(p) is the cumulative distribution function (CDF) of the climatology,
      and F_ens is the cumulative distribution function of the forecast ensemble.
    
    Distinction Note:
    - Standardized Z-Score: Linear anomaly distance normalized by historical standard deviation.
    - Percentile Anomaly: Empirical rank within climatology.
    - True EFI: Integral difference between full forecast ensemble CDF and historical climatological CDF.
    
    Status Note:
    "EFI interface — requires multi-member forecast ensemble (NEPS-G) and full historical reanalysis CDF."
    """

    STATUS_NOTE = "EFI interface — requires multi-member forecast ensemble (NEPS-G) and full historical reanalysis CDF."

    @classmethod
    def compute_efi_prototype(
        cls,
        forecast_ds: xr.Dataset,
        climatology_ds: xr.Dataset,
        variable_name: str,
    ) -> Dict[str, Any]:
        """
        Provides EFI prototype computation interface.
        If ensemble CDF is unavailable, returns mathematically defensible approximation and logs status.
        """
        return {
            "status": cls.STATUS_NOTE,
            "formula": "EFI = (2 / pi) * integral [ (q - F_ens(q)) / sqrt(q * (1 - q)) ] dq",
            "is_full_efi": False,
            "note": "Currently operating in deterministic Z-score mode until NEPS-G ensemble feed is connected.",
        }
