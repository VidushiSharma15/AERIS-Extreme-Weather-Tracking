from typing import Dict, Any, List, Optional
import xarray as xr


class EnsembleProcessor:
    """
    Ensemble processing interface for NCMRWF NEPS-G / multi-member NWP forecasts.
    Calculates ensemble mean, ensemble spread, exceedance probabilities, and uncertainty.
    
    Status Note:
    "Ensemble processing interface — awaiting NEPS-G/ensemble dataset."
    """

    STATUS_NOTE = "Ensemble processing interface — awaiting NEPS-G/ensemble dataset."

    @classmethod
    def compute_ensemble_stats(cls, ensemble_ds: Optional[xr.Dataset] = None) -> Dict[str, Any]:
        if ensemble_ds is None or "member" not in ensemble_ds.dims:
            return {
                "status": cls.STATUS_NOTE,
                "is_ensemble": False,
                "ensemble_members": 1,
            }

        ens_mean = ensemble_ds.mean(dim="member")
        ens_spread = ensemble_ds.std(dim="member")
        return {
            "status": "Ensemble processed successfully.",
            "is_ensemble": True,
            "ensemble_members": int(ensemble_ds.sizes["member"]),
            "mean_ds": ens_mean,
            "spread_ds": ens_spread,
        }
