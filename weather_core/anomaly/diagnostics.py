from typing import List
from .spatial_regions import ExtremeWeatherEvent


class AnomalyDiagnostics:
    """
    Generates human-readable visual diagnostic summaries of detected extreme weather anomalies.
    """

    @classmethod
    def print_event_summary(cls, events: List[ExtremeWeatherEvent]):
        print("=" * 75)
        print(f"EXTREME WEATHER ANOMALY DIAGNOSTIC REPORT ({len(events)} Event(s) Detected)")
        print("=" * 75)

        if not events:
            print("No extreme anomalies detected exceeding specified Z-score thresholds.")
            print("=" * 75)
            return

        for idx, evt in enumerate(events, start=1):
            print(f"\n--- [EVENT #{idx:02d}]: {evt.event_id} ({evt.severity} {evt.event_type.upper()}) ---")
            print(f"  Source Dataset   : {evt.source_dataset}")
            print(f"  Variable         : {evt.variable_name} ({evt.units})")
            print(f"  Timestamp        : {evt.timestamp}")
            print(f"  Peak Value       : {evt.max_intensity} {evt.units}")
            print(f"  Mean Intensity   : {evt.mean_intensity} {evt.units}")
            print(f"  Max Z-Score      : {evt.max_z_score:+.2f} std_dev")
            print(f"  Mean Z-Score     : {evt.mean_z_score:+.2f} std_dev")
            print(f"  Severity Level   : {evt.severity}")
            print(f"  Centroid Coord   : Lat {evt.centroid.latitude:.4f} N, Lon {evt.centroid.longitude:.4f} E")
            print(f"  Bounding Box     : [{evt.bounding_box.min_lat:.2f} N, {evt.bounding_box.max_lat:.2f} N, {evt.bounding_box.min_lon:.2f} E, {evt.bounding_box.max_lon:.2f} E]")
            print(f"  Affected Area    : {evt.area_km2:,.2f} km^2 ({evt.affected_grid_cells} grid cells)")
            print(f"  Provenance       : Method={evt.provenance_metadata.get('calculation_method')}, Threshold=Z>={evt.provenance_metadata.get('z_threshold')}")

        print("\n" + "=" * 75)
