import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from weather_core.tracking.geodesic import GeodesicMath
from weather_core.tracking.models import WeatherEventTrack, TrackPoint
from weather_core.tracking.matching import EventMatcher
from weather_core.tracking.trajectory import TrajectoryCalculator
from weather_core.tracking.extrapolation import BaselineTrajectoryExtrapolator
from weather_core.tracking.confidence import TrackConfidenceCalculator
from weather_core.tracking.geojson_track import GeoJSONTrackExporter
from weather_core.tracking.engine import SpatioTemporalTracker
from weather_core.anomaly.spatial_regions import ExtremeWeatherEvent, CentroidCoordinates, BoundingBox


class TestTracking(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_geodesic_math(self):
        # Distance between Mumbai (19.076°N, 72.877°E) and Delhi (28.613°N, 77.209°E) ~ 1150 km
        dist = GeodesicMath.haversine_distance_km(19.076, 72.877, 28.613, 77.209)
        self.assertTrue(1100.0 <= dist <= 1200.0)

        # Bearing from Mumbai to Delhi ~ 20-30° (NNE)
        bearing = GeodesicMath.calculate_bearing_degrees(19.076, 72.877, 28.613, 77.209)
        self.assertTrue(15.0 <= bearing <= 35.0)

        # Extrapolation
        lat2, lon2 = GeodesicMath.extrapolate_point(19.076, 72.877, 100.0, 0.0)  # 100km North
        self.assertGreater(lat2, 19.076)

    def test_event_matcher(self):
        evt1 = ExtremeWeatherEvent(
            event_id="E1",
            event_type="extreme_precipitation",
            severity="SEVERE",
            timestamp="2026-09-25T00:00:00",
            variable_name="tp",
            units="mm",
            centroid=CentroidCoordinates(20.0, 85.0),
            bounding_box=BoundingBox(19.0, 21.0, 84.0, 86.0),
            area_km2=1000.0,
            max_intensity=100.0,
            mean_intensity=50.0,
            max_z_score=3.5,
            mean_z_score=2.8,
            affected_grid_cells=10,
            source_dataset="TEST",
        )

        evt2 = ExtremeWeatherEvent(
            event_id="E2",
            event_type="extreme_precipitation",
            severity="SEVERE",
            timestamp="2026-09-25T01:00:00",
            variable_name="tp",
            units="mm",
            centroid=CentroidCoordinates(20.2, 85.3),  # ~40 km movement
            bounding_box=BoundingBox(19.2, 21.2, 84.3, 86.3),
            area_km2=1100.0,
            max_intensity=110.0,
            mean_intensity=55.0,
            max_z_score=3.8,
            mean_z_score=3.0,
            affected_grid_cells=11,
            source_dataset="TEST",
        )

        score = EventMatcher.calculate_match_score(evt1, evt2)
        self.assertGreater(score, 0.50)  # High match score for nearby evolving storm

    def test_spatiotemporal_tracker_continuation(self):
        # 3 Sequential Timestamps test fixture
        times = ["2026-09-25T00:00:00", "2026-09-25T01:00:00", "2026-09-25T02:00:00"]
        events_t0 = [
            ExtremeWeatherEvent(
                event_id="E_T0",
                event_type="extreme_precipitation",
                severity="SEVERE",
                timestamp=times[0],
                variable_name="tp",
                units="mm",
                centroid=CentroidCoordinates(20.0, 85.0),
                bounding_box=BoundingBox(19.0, 21.0, 84.0, 86.0),
                area_km2=1000.0,
                max_intensity=80.0,
                mean_intensity=40.0,
                max_z_score=3.0,
                mean_z_score=2.5,
                affected_grid_cells=10,
                source_dataset="FIXTURE",
            )
        ]
        events_t1 = [
            ExtremeWeatherEvent(
                event_id="E_T1",
                event_type="extreme_precipitation",
                severity="SEVERE",
                timestamp=times[1],
                variable_name="tp",
                units="mm",
                centroid=CentroidCoordinates(20.2, 85.4),
                bounding_box=BoundingBox(19.2, 21.2, 84.4, 86.4),
                area_km2=1200.0,
                max_intensity=110.0,
                mean_intensity=55.0,
                max_z_score=3.8,
                mean_z_score=3.0,
                affected_grid_cells=12,
                source_dataset="FIXTURE",
            )
        ]
        events_t2 = [
            ExtremeWeatherEvent(
                event_id="E_T2",
                event_type="extreme_precipitation",
                severity="SEVERE",
                timestamp=times[2],
                variable_name="tp",
                units="mm",
                centroid=CentroidCoordinates(20.5, 85.9),
                bounding_box=BoundingBox(19.5, 21.5, 84.9, 86.9),
                area_km2=1400.0,
                max_intensity=130.0,
                mean_intensity=60.0,
                max_z_score=4.2,
                mean_z_score=3.5,
                affected_grid_cells=14,
                source_dataset="FIXTURE",
            )
        ]

        tracker = SpatioTemporalTracker(match_threshold=0.25)
        tracks = tracker.track_events_over_time([events_t0, events_t1, events_t2])

        self.assertEqual(len(tracks), 1)  # Linked into single continuous track
        trk = tracks[0]
        self.assertEqual(len(trk.points), 3)
        self.assertGreater(trk.total_displacement_km, 50.0)
        self.assertGreater(trk.speed_kmh, 0.0)
        self.assertEqual(trk.intensity_change, 50.0)  # 130 - 80

    def test_baseline_extrapolator(self):
        pt1 = TrackPoint("2026-09-25T00:00:00", 20.0, 85.0, 80.0, 1000.0, "SEVERE", 3.0, "P1")
        pt2 = TrackPoint("2026-09-25T01:00:00", 20.2, 85.4, 110.0, 1200.0, "SEVERE", 3.8, "P2")
        trk = WeatherEventTrack(
            track_id="TRK_001",
            event_type="extreme_precipitation",
            start_time="2026-09-25T00:00:00",
            end_time="2026-09-25T01:00:00",
            points=[pt1, pt2],
        )
        TrajectoryCalculator.update_track_statistics(trk)

        extra = BaselineTrajectoryExtrapolator.extrapolate_short_term(trk, lead_hours_ahead=[1.0])
        self.assertEqual(len(extra), 1)
        self.assertEqual(extra[0]["method"], "BASELINE TRAJECTORY EXTRAPOLATION")
        self.assertGreater(extra[0]["extrapolated_latitude"], 20.2)

    def test_geojson_track_export(self):
        pt1 = TrackPoint("2026-09-25T00:00:00", 20.0, 85.0, 80.0, 1000.0, "SEVERE", 3.0, "P1")
        pt2 = TrackPoint("2026-09-25T01:00:00", 20.2, 85.4, 110.0, 1200.0, "SEVERE", 3.8, "P2")
        trk = WeatherEventTrack(
            track_id="TRK_001",
            event_type="extreme_precipitation",
            start_time="2026-09-25T00:00:00",
            end_time="2026-09-25T01:00:00",
            points=[pt1, pt2],
        )
        TrajectoryCalculator.update_track_statistics(trk)

        geojson = GeoJSONTrackExporter.tracks_to_geojson([trk])
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 2)  # LineString + Point
        types = [f["geometry"]["type"] for f in geojson["features"]]
        self.assertIn("LineString", types)
        self.assertIn("Point", types)


if __name__ == "__main__":
    unittest.main()
