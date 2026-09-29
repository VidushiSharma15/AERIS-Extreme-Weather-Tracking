from typing import Dict, Any, List, Optional
import xarray as xr

from weather_core.anomaly.spatial_regions import ExtremeWeatherEvent, CentroidCoordinates
from .models import WeatherEventTrack, TrackPoint
from .matching import EventMatcher
from .trajectory import TrajectoryCalculator
from .confidence import TrackConfidenceCalculator


class SpatioTemporalTracker:
    """
    Core SpatioTemporalTracker for SIH26078 AERIS.
    Tracks extreme weather anomalies across sequential forecast timestamps.
    """

    def __init__(self, match_threshold: float = 0.30, max_search_dist_km: float = 300.0):
        self.match_threshold = match_threshold
        self.max_search_dist_km = max_search_dist_km
        self.active_tracks: List[WeatherEventTrack] = []
        self.completed_tracks: List[WeatherEventTrack] = []
        self._track_counter = 1

    def _event_to_track_point(self, event: ExtremeWeatherEvent) -> TrackPoint:
        return TrackPoint(
            timestamp=event.timestamp,
            latitude=event.centroid.latitude,
            longitude=event.centroid.longitude,
            intensity=event.max_intensity,
            area_km2=event.area_km2,
            severity=event.severity,
            z_score=event.max_z_score,
            event_id=event.event_id,
            bounding_box=event.bounding_box,
        )

    def process_timestamp_events(self, events: List[ExtremeWeatherEvent]) -> List[WeatherEventTrack]:
        """
        Processes events detected at a specific timestamp T.
        Links events to active tracks or instantiates new tracks.
        """
        if not events:
            return self.active_tracks

        unmatched_events = events.copy()

        # Try matching unmatched events to active tracks
        for track in list(self.active_tracks):
            best_event: Optional[ExtremeWeatherEvent] = None
            best_score = 0.0

            if not track.points:
                continue
            last_pt = track.points[-1]

            # Construct event representation for track's last position
            track_last_evt = ExtremeWeatherEvent(
                event_id=last_pt.event_id,
                event_type=track.event_type,
                severity=last_pt.severity,
                timestamp=last_pt.timestamp,
                variable_name=track.event_type,
                units="",
                centroid=CentroidCoordinates(last_pt.latitude, last_pt.longitude),
                bounding_box=last_pt.bounding_box,
                area_km2=last_pt.area_km2,
                max_intensity=last_pt.intensity,
                mean_intensity=last_pt.intensity,
                max_z_score=last_pt.z_score,
                mean_z_score=last_pt.z_score,
                affected_grid_cells=1,
                source_dataset="",
            )

            for evt in unmatched_events:
                score = EventMatcher.calculate_match_score(
                    event1=track_last_evt,
                    event2=evt,
                    max_dist_km=self.max_search_dist_km,
                )
                if score > self.match_threshold and score > best_score:
                    best_score = score
                    best_event = evt

            if best_event is not None:
                # Append point to existing track
                pt = self._event_to_track_point(best_event)
                track.points.append(pt)
                TrajectoryCalculator.update_track_statistics(track)
                track.confidence = TrackConfidenceCalculator.calculate_confidence(track)
                unmatched_events.remove(best_event)

        # Create new tracks for remaining unmatched events
        for evt in unmatched_events:
            track_id = f"TRK_{evt.variable_name.upper()}_{self._track_counter:03d}"
            self._track_counter += 1

            pt = self._event_to_track_point(evt)
            new_track = WeatherEventTrack(
                track_id=track_id,
                event_type=evt.event_type,
                start_time=evt.timestamp,
                end_time=evt.timestamp,
                points=[pt],
                current_centroid=evt.centroid,
                initial_centroid=evt.centroid,
                confidence=0.50,
                provenance={
                    "detection_method": "SpatioTemporalTracker",
                    "initial_event_id": evt.event_id,
                },
            )
            TrajectoryCalculator.update_track_statistics(new_track)
            self.active_tracks.append(new_track)

        return self.active_tracks

    def track_events_over_time(
        self, events_by_timestamp: List[List[ExtremeWeatherEvent]]
    ) -> List[WeatherEventTrack]:
        """
        Executes tracking over a list of event sets for sequential timestamps.
        """
        for t_events in events_by_timestamp:
            self.process_timestamp_events(t_events)
        return self.active_tracks
