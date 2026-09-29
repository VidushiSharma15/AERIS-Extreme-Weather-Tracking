'use client';

import React, { useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Popup, Polyline, Rectangle, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { WeatherEvent, GeoJSONFeatureCollection, TrajectoryPoint } from '../lib/api';

// Fix Leaflet marker icon issue in Next.js
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface GISMapInnerProps {
  events: WeatherEvent[];
  geoJsonData: GeoJSONFeatureCollection | null;
  selectedEvent: WeatherEvent | null;
  trajectoryPoints: TrajectoryPoint[];
  forecastField?: any;
  activeVariable: string;
  activeLeadTime: number;
  activeEnsemble: string;
  showVectors: boolean;
  showFootprints: boolean;
  showEnsembleSpread: boolean;
  showPopulationExposure: boolean;
  show5kmOutput: boolean;
  populationExposure?: any;
  onSelectEvent: (event: WeatherEvent) => void;
}

// Map Controller component to fly to selected event centroid
function MapController({ selectedEvent }: { selectedEvent: WeatherEvent | null }) {
  const map = useMap();
  useEffect(() => {
    if (selectedEvent && selectedEvent.centroid) {
      map.flyTo([selectedEvent.centroid.latitude, selectedEvent.centroid.longitude], 7, {
        duration: 1.2,
      });
    }
  }, [selectedEvent, map]);
  return null;
}

export default function GISMapInner({
  events,
  geoJsonData,
  selectedEvent,
  trajectoryPoints,
  forecastField,
  activeVariable,
  activeLeadTime,
  activeEnsemble,
  showVectors,
  showFootprints,
  showEnsembleSpread,
  showPopulationExposure,
  show5kmOutput,
  populationExposure,
  onSelectEvent,
}: GISMapInnerProps) {
  const defaultCenter: [number, number] = [15.5, 87.5]; // Bay of Bengal Focus
  const defaultZoom = 5;

  const getSeverityColor = (severity?: string) => {
    switch (severity?.toUpperCase()) {
      case 'SUPER CYCLONIC STORM':
      case 'EXTREME':
      case 'SEVERE':
        return '#ef4444'; // Red
      case 'VERY SEVERE CYCLONIC STORM':
      case 'MODERATE':
        return '#f97316'; // Orange
      case 'WATCH':
        return '#eab308'; // Amber
      default:
        return '#00d2ff'; // Cyan
    }
  };

  const getFieldColor = (val: number, variable: string) => {
    if (variable === 'wind_speed' || variable === 'wind_anomaly') {
      if (val >= 33) return '#dc2626'; // Red (Severe Cyclone)
      if (val >= 25) return '#f97316'; // Orange (Very Severe)
      if (val >= 17) return '#eab308'; // Yellow (Gale)
      if (val >= 10) return '#06b6d4'; // Cyan
      return '#3b82f6'; // Blue
    } else if (variable === 'msl' || variable === 'msl_anomaly') {
      if (val <= 950) return '#991b1b'; // Dark Red
      if (val <= 970) return '#ef4444'; // Red
      if (val <= 985) return '#f97316'; // Orange
      if (val <= 1000) return '#3b82f6'; // Blue
      return '#1e3a8a'; // Dark Blue
    } else if (variable === 'tp') {
      if (val >= 200) return '#d946ef'; // Magenta
      if (val >= 100) return '#8b5cf6'; // Purple
      if (val >= 50) return '#3b82f6'; // Blue
      if (val >= 10) return '#06b6d4'; // Cyan
      return '#10b981'; // Emerald
    }
    return '#3b82f6';
  };

  const geoJsonStyle = (feature: any) => {
    const props = feature?.properties || {};
    const isFootprint = props.layer_type === 'event_footprint' || feature?.geometry?.type === 'Polygon';
    const color = getSeverityColor(props.severity);
    return {
      fillColor: color,
      weight: isFootprint ? 2 : 1,
      opacity: 0.9,
      color: color,
      fillOpacity: isFootprint ? 0.25 : 0.4,
      dashArray: isFootprint ? '4, 4' : undefined,
    };
  };

  const polylineCoords: [number, number][] = trajectoryPoints.map((pt) => [pt.latitude, pt.longitude]);

  // Construct grid cells for continuous spatial weather field visualization
  const gridCells: { bounds: [[number, number], [number, number]]; color: string; val: number }[] = [];
  const vectorArrows: { lat: number; lon: number; u: number; v: number; speed: number }[] = [];

  if (forecastField && forecastField.grid_values && forecastField.latitude && forecastField.longitude) {
    const lats = forecastField.latitude;
    const lons = forecastField.longitude;
    const grid = forecastField.grid_values;
    const u_grid = forecastField.u_component;
    const v_grid = forecastField.v_component;

    const latStep = Math.abs(lats[1] - lats[0]) / 2.0;
    const lonStep = Math.abs(lons[1] - lons[0]) / 2.0;

    // Subsample grid for responsive Leaflet performance (step by 2)
    for (let i = 0; i < lats.length; i += 2) {
      for (let j = 0; j < lons.length; j += 2) {
        const val = grid[i]?.[j];
        if (val !== undefined && val !== null) {
          const lat = lats[i];
          const lon = lons[j];
          gridCells.push({
            bounds: [
              [lat - latStep * 2, lon - lonStep * 2],
              [lat + latStep * 2, lon + lonStep * 2],
            ],
            color: getFieldColor(val, activeVariable),
            val: val,
          });

          if (showVectors && u_grid && v_grid && i % 4 === 0 && j % 4 === 0) {
            const u = u_grid[i][j];
            const v = v_grid[i][j];
            const spd = Math.sqrt(u * u + v * v);
            if (spd > 5.0) {
              vectorArrows.push({ lat, lon, u, v, speed: spd });
            }
          }
        }
      }
    }
  }

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden shadow-2xl border border-aeris-border bg-aeris-card">
      {/* Map Header Provenance Banner */}
      <div className="absolute top-3 left-14 z-[1000] bg-aeris-bg/95 backdrop-blur-md px-3.5 py-1.5 rounded-lg border border-aeris-border text-xs text-gray-200 font-mono shadow-lg flex items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span className="font-bold text-white uppercase">{activeVariable.replace('_', ' ')}</span>
          <span className="text-aeris-accent bg-aeris-accent/10 px-2 py-0.5 rounded font-bold border border-aeris-accent/30">
            +{activeLeadTime}h Lead
          </span>
        </div>
        <span className="text-gray-400 border-l border-aeris-border pl-3 text-[11px]">
          NCMRWF/TIGGE 0.5° Grid &bull; Member: <span className="text-amber-300 font-semibold">{activeEnsemble}</span>
        </span>
      </div>

      {/* Scientific Legend Overlay */}
      <div className="absolute bottom-6 right-6 z-[1000] bg-aeris-bg/95 backdrop-blur-md px-3.5 py-2.5 rounded-lg border border-aeris-border text-xs text-gray-200 shadow-xl space-y-1.5 min-w-[220px]">
        <div className="font-bold text-gray-300 text-[11px] border-b border-aeris-border pb-1 flex justify-between items-center">
          <span>LEGEND & UNITS</span>
          <span className="text-aeris-accent font-mono text-[10px]">
            {activeVariable.includes('msl') ? 'hPa' : activeVariable === 'tp' ? 'mm/h' : 'm/s'}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px] font-mono">
          {activeVariable.includes('msl') ? (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-red-800 inline-block"></span>
                <span>Intense (&lt;970 hPa)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-red-500 inline-block"></span>
                <span>Severe (970–985)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-orange-500 inline-block"></span>
                <span>Moderate (985–1000)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-blue-600 inline-block"></span>
                <span>Ambient (&gt;1000 hPa)</span>
              </div>
            </>
          ) : activeVariable === 'tp' ? (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-fuchsia-500 inline-block"></span>
                <span>Extreme (≥200 mm)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-purple-500 inline-block"></span>
                <span>Heavy (100–200)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-blue-500 inline-block"></span>
                <span>Moderate (50–100)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-emerald-500 inline-block"></span>
                <span>Light (&lt;50 mm)</span>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-red-600 inline-block"></span>
                <span>Severe (≥33 m/s)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-orange-500 inline-block"></span>
                <span>Very Severe (25–33)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-amber-400 inline-block"></span>
                <span>Gale (17–25 m/s)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-cyan-500 inline-block"></span>
                <span>Moderate (&lt;17 m/s)</span>
              </div>
            </>
          )}
        </div>
      </div>

      <MapContainer
        center={defaultCenter}
        zoom={defaultZoom}
        style={{ width: '100%', height: '100%' }}
        scrollWheelZoom={true}
      >
        <MapController selectedEvent={selectedEvent} />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Spatial Weather Field Grid Overlay */}
        {gridCells.map((cell, idx) => (
          <Rectangle
            key={`grid-${idx}`}
            bounds={cell.bounds}
            pathOptions={{
              fillColor: cell.color,
              fillOpacity: 0.35,
              stroke: false,
            }}
          />
        ))}

        {/* Wind Vector Arrows */}
        {showVectors &&
          vectorArrows.map((vec, idx) => {
            const endpoint: [number, number] = [
              vec.lat + (vec.v / 15.0) * 0.4,
              vec.lon + (vec.u / 15.0) * 0.4,
            ];
            return (
              <Polyline
                key={`vec-${idx}`}
                positions={[[vec.lat, vec.lon], endpoint]}
                pathOptions={{
                  color: '#ffffff',
                  weight: 1.5,
                  opacity: 0.8,
                }}
              />
            );
          })}

        {/* GeoJSON Anomaly Polygons & Footprints */}
        {showFootprints && geoJsonData && geoJsonData.features && geoJsonData.features.length > 0 && (
          <GeoJSON
            key={JSON.stringify(geoJsonData)}
            data={geoJsonData as any}
            style={geoJsonStyle}
            onEachFeature={(feature, layer) => {
              layer.on('click', () => {
                const matched = events.find((e) => e.event_id === feature.id || e.event_id === feature.properties?.event_id);
                if (matched) {
                  onSelectEvent(matched);
                }
              });
            }}
          />
        )}

        {/* Event Centroids */}
        {events.map((evt) => {
          const isSelected = selectedEvent?.event_id === evt.event_id;
          const color = getSeverityColor(evt.severity);
          return (
            <CircleMarker
              key={evt.event_id}
              center={[evt.centroid.latitude, evt.centroid.longitude]}
              radius={isSelected ? 12 : 8}
              pathOptions={{
                fillColor: color,
                color: isSelected ? '#ffffff' : color,
                weight: isSelected ? 3 : 1.5,
                fillOpacity: 0.9,
              }}
              eventHandlers={{
                click: () => onSelectEvent(evt),
              }}
            >
              <Popup className="aeris-popup">
                <div className="p-1 font-sans text-xs">
                  <div className="font-bold text-gray-900 border-b pb-1 mb-1">
                    {evt.event_type.toUpperCase()} ({evt.severity})
                  </div>
                  <div><strong>ID:</strong> {evt.event_id}</div>
                  <div><strong>Z-Score:</strong> +{evt.max_z_score.toFixed(2)} σ</div>
                  <div><strong>Area:</strong> {evt.area_km2.toLocaleString()} km²</div>
                  <div><strong>Time:</strong> {evt.timestamp}</div>
                  <button
                    onClick={() => onSelectEvent(evt)}
                    className="mt-2 w-full bg-blue-600 hover:bg-blue-700 text-white text-[10px] font-semibold py-1 rounded transition"
                  >
                    Inspect Event
                  </button>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        {/* Event Trajectory Line */}
        {polylineCoords.length > 0 && (
          <>
            <Polyline
              positions={polylineCoords}
              pathOptions={{
                color: '#00d2ff',
                weight: 4,
                dashArray: '6, 6',
                opacity: 0.95,
              }}
            />
            {trajectoryPoints.map((pt, idx) => (
              <CircleMarker
                key={`traj-${idx}`}
                center={[pt.latitude, pt.longitude]}
                radius={4}
                pathOptions={{
                  fillColor: '#00d2ff',
                  color: '#ffffff',
                  weight: 1.5,
                  fillOpacity: 1.0,
                }}
              >
                <Popup>
                  <div className="text-xs">
                    <div><strong>Track Step {idx + 1} (+{pt.event_id.slice(-3)})</strong></div>
                    <div>Lat: {pt.latitude.toFixed(2)}°, Lon: {pt.longitude.toFixed(2)}°</div>
                    <div>Min MSL: {pt.intensity.toFixed(1)} hPa</div>
                    <div>Time: {pt.timestamp}</div>
                  </div>
                </Popup>
              </CircleMarker>
            ))}
          </>
        )}
      </MapContainer>
    </div>
  );
}
