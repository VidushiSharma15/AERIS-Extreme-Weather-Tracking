export interface Centroid {
  latitude: number;
  longitude: number;
}

export interface BoundingBox {
  min_lat: number;
  max_lat: number;
  min_lon: number;
  max_lon: number;
}

export interface WeatherEvent {
  event_id: string;
  event_type: string;
  severity: 'WATCH' | 'MODERATE' | 'SEVERE' | string;
  timestamp: string;
  variable_name: string;
  units: string;
  centroid: Centroid;
  bounding_box: BoundingBox;
  area_km2: number;
  max_intensity: number;
  mean_intensity: number;
  max_z_score: number;
  mean_z_score: number;
  affected_grid_cells: number;
  source_dataset: string;
  provenance_metadata?: Record<string, any>;
}

export interface TrajectoryPoint {
  timestamp: string;
  latitude: number;
  longitude: number;
  intensity: number;
  area_km2: number;
  severity: string;
  z_score: number;
  event_id: string;
}

export interface WeatherTrack {
  track_id: string;
  event_type: string;
  severity: string;
  start_time: string;
  end_time: string;
  current_position: Centroid;
  direction_degrees: number;
  speed_kmh: number;
  intensity_change: number;
  area_change: number;
  confidence: number;
  trajectory: TrajectoryPoint[];
}

export interface Alert {
  alert_id: string;
  event_id: string;
  event_type: string;
  severity: string;
  latitude: number;
  longitude: number;
  timestamp: string;
  anomaly_score: number;
  affected_area_km2: number;
  confidence: number;
  source_dataset: string;
  disclaimer: string;
}

export interface DatasetMetadata {
  dataset_name: string;
  variable: string;
  units: string;
  source: string;
  geographical_bounds: BoundingBox;
  time_start: string;
  time_end: string;
  resolution_km: number;
  local_or_cloud: string;
  file_location: string;
  size_bytes: number;
  processing_version: string;
}

export interface GeoJSONFeature {
  type: 'Feature';
  id: string;
  geometry: {
    type: 'Polygon' | 'Point' | 'LineString';
    coordinates: any;
  };
  properties: Record<string, any>;
}

export interface GeoJSONFeatureCollection {
  type: 'FeatureCollection';
  metadata: Record<string, any>;
  features: GeoJSONFeature[];
}

export interface SystemHealth {
  status: string;
  service: string;
  version: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  if (!res.ok) {
    throw new Error(`API Request to ${endpoint} failed with status ${res.status}: ${res.statusText}`);
  }

  return res.json();
}

export async function checkHealth(): Promise<SystemHealth> {
  return fetchJson<SystemHealth>('/health');
}

export async function fetchEvents(): Promise<{ events: WeatherEvent[]; message?: string } | WeatherEvent[]> {
  const data = await fetchJson<any>('/api/v1/events');
  if (Array.isArray(data)) {
    return { events: data };
  }
  if (data && Array.isArray(data.events)) {
    return data;
  }
  return { events: [], message: 'No events returned' };
}

export async function fetchEventById(eventId: string): Promise<WeatherEvent> {
  return fetchJson<WeatherEvent>(`/api/v1/events/${eventId}`);
}

export async function fetchEventTrajectory(eventId: string): Promise<TrajectoryPoint[]> {
  return fetchJson<TrajectoryPoint[]>(`/api/v1/events/${eventId}/trajectory`);
}

export async function fetchAllEventsGeoJSON(): Promise<GeoJSONFeatureCollection> {
  return fetchJson<GeoJSONFeatureCollection>('/api/v1/events/geojson');
}

export async function fetchEventGeoJSON(eventId: string): Promise<GeoJSONFeatureCollection> {
  return fetchJson<GeoJSONFeatureCollection>(`/api/v1/events/${eventId}/geojson`);
}

export async function fetchTracks(): Promise<WeatherTrack[]> {
  return fetchJson<WeatherTrack[]>('/api/v1/tracks');
}

export async function fetchTracksGeoJSON(): Promise<GeoJSONFeatureCollection> {
  return fetchJson<GeoJSONFeatureCollection>('/api/v1/tracks/geojson');
}

export async function fetchAlerts(): Promise<Alert[]> {
  return fetchJson<Alert[]>('/api/v1/alerts');
}

export async function fetchDatasetMetadata(): Promise<DatasetMetadata> {
  return fetchJson<DatasetMetadata>('/api/v1/metadata');
}

export async function fetchDatasetStatus(): Promise<any> {
  return fetchJson<any>('/api/v1/data/status');
}

export async function fetchDemoManifest(): Promise<any> {
  return fetchJson<any>('/api/v1/demo/manifest');
}

export async function fetchDemoStatus(): Promise<any> {
  return fetchJson<any>('/api/v1/demo/status');
}

export async function fetchDownscalingStatus(): Promise<any> {
  return fetchJson<any>('/api/v1/downscaling/status');
}

export async function fetchDownscalingAmphan(): Promise<any> {
  return fetchJson<any>('/api/v1/downscaling/nepsg/amphan');
}

export async function fetchForecastField(variable: string = 'wind_speed', leadTime: number = 0, member: string = 'mean'): Promise<any> {
  return fetchJson<any>(`/api/v1/forecast/field?variable=${encodeURIComponent(variable)}&lead_time=${leadTime}&member=${encodeURIComponent(member)}`);
}

export async function fetchPopulationExposure(leadTime: number = 0): Promise<any> {
  return fetchJson<any>(`/api/v1/population/exposure?lead_time=${leadTime}`);
}

export async function fetchModelsStatus(): Promise<any> {
  return fetchJson<any>('/api/v1/models/status');
}

export async function fetchMultiModelForecast(): Promise<any> {
  return fetchJson<any>('/api/v1/forecast/multimodel');
}


