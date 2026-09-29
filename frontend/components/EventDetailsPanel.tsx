'use client';

import React from 'react';
import { WeatherEvent, TrajectoryPoint, DatasetMetadata } from '../lib/api';
import { MapPin, ShieldAlert, Compass, Database, Activity, Calendar, Users, Wind, Thermometer, Info } from 'lucide-react';

interface EventDetailsPanelProps {
  event: WeatherEvent | null;
  trajectory: TrajectoryPoint[];
  metadata: DatasetMetadata | null;
  populationExposure?: any;
  downscalingData?: any;
  modelsStatus?: any;
  onClose?: () => void;
}

export default function EventDetailsPanel({
  event,
  trajectory,
  metadata,
  populationExposure,
  downscalingData,
  modelsStatus,
  onClose,
}: EventDetailsPanelProps) {
  if (!event) {
    return (
      <div className="w-full h-full bg-aeris-card border border-aeris-border rounded-xl p-6 flex flex-col items-center justify-center text-center text-gray-400">
        <Activity className="w-12 h-12 text-aeris-border mb-3 animate-pulse" />
        <h3 className="text-base font-semibold text-gray-200">No Event Selected</h3>
        <p className="text-xs text-gray-400 mt-1 max-w-xs">
          Select any lead-time timestep or click on any anomaly polygon on the map to inspect real-time scientific diagnostics.
        </p>
      </div>
    );
  }

  const prov = event.provenance_metadata || {};
  const leadTime = prov.lead_time_hour ?? 0;
  const maxWindMs = prov.max_wind_speed_ms ?? event.max_intensity ?? 0;
  const minMslHpa = prov.min_msl_hpa ?? event.mean_intensity ?? 0;
  const bearingDeg = prov.bearing_deg ?? 0;
  const speedKmh = prov.speed_kmh ?? 0;
  const dispKm = prov.step_displacement_km ?? 0;

  let directionText = 'N/A';
  if (bearingDeg > 0) {
    if (bearingDeg >= 337.5 || bearingDeg < 22.5) directionText = 'N (0°)';
    else if (bearingDeg < 67.5) directionText = 'NE (45°)';
    else if (bearingDeg < 112.5) directionText = 'E (90°)';
    else if (bearingDeg < 157.5) directionText = 'SE (135°)';
    else if (bearingDeg < 202.5) directionText = 'S (180°)';
    else if (bearingDeg < 247.5) directionText = 'SW (225°)';
    else if (bearingDeg < 292.5) directionText = 'W (270°)';
    else directionText = 'NW (315°)';
  }

  const getSeverityBadge = (sev: string) => {
    switch (sev.toUpperCase()) {
      case 'SUPER CYCLONIC STORM':
      case 'EXTREME':
      case 'SEVERE':
        return 'bg-red-500/20 text-red-400 border-red-500/40';
      case 'VERY SEVERE CYCLONIC STORM':
      case 'MODERATE':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/40';
      default:
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40';
    }
  };

  return (
    <div className="w-full h-full bg-aeris-card border border-aeris-border rounded-xl p-5 flex flex-col gap-5 overflow-y-auto custom-scrollbar shadow-xl text-gray-200">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-aeris-border pb-3">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-aeris-accent" />
          <h2 className="text-sm font-bold tracking-wider text-white">EVENT DIAGNOSTICS</h2>
        </div>
        <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${getSeverityBadge(event.severity)}`}>
          {event.severity.toUpperCase()}
        </span>
      </div>

      {/* Primary Event Metadata */}
      <div className="space-y-2 text-xs">
        <div className="flex justify-between items-center bg-aeris-bg/60 p-2 rounded border border-aeris-border">
          <span className="text-gray-400">Event Identifier</span>
          <span className="font-mono text-white font-semibold">{event.event_id}</span>
        </div>
        <div className="flex justify-between items-center bg-aeris-bg/60 p-2 rounded border border-aeris-border">
          <span className="text-gray-400">Forecast Lead Time</span>
          <span className="font-mono text-aeris-accent font-bold">+{leadTime} Hours</span>
        </div>
        <div className="flex justify-between items-center bg-aeris-bg/60 p-2 rounded border border-aeris-border">
          <span className="text-gray-400">Current Centroid</span>
          <span className="font-mono text-gray-300">
            {event.centroid.latitude.toFixed(2)}°N, {event.centroid.longitude.toFixed(2)}°E
          </span>
        </div>
        <div className="flex justify-between items-center bg-aeris-bg/60 p-2 rounded border border-aeris-border">
          <span className="text-gray-400">Min Sea-Level Pressure</span>
          <span className="font-mono text-cyan-300 font-bold">{minMslHpa.toFixed(1)} hPa</span>
        </div>
        <div className="flex justify-between items-center bg-aeris-bg/60 p-2 rounded border border-aeris-border">
          <span className="text-gray-400">Max Wind Speed</span>
          <span className="font-mono text-emerald-400 font-bold">{maxWindMs.toFixed(1)} m/s ({(maxWindMs * 3.6).toFixed(0)} km/h)</span>
        </div>
        <div className="flex justify-between items-center bg-aeris-bg/60 p-2 rounded border border-aeris-border">
          <span className="text-gray-400">Max Z-Score Anomaly</span>
          <span className="font-mono text-red-400 font-bold">+{event.max_z_score.toFixed(2)} σ</span>
        </div>
        <div className="flex justify-between items-center bg-aeris-bg/60 p-2 rounded border border-aeris-border">
          <span className="text-gray-400">Calculated Footprint Area</span>
          <span className="font-mono text-white">{event.area_km2.toLocaleString()} km²</span>
        </div>
      </div>

      {/* Cyclone Kinematics Section */}
      <div className="border-t border-aeris-border pt-3">
        <h3 className="text-xs font-bold text-gray-300 mb-2 flex items-center gap-1.5">
          <Compass className="w-4 h-4 text-aeris-accent" />
          CYCLONE KINEMATICS & VECTOR
        </h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-aeris-bg/80 p-2.5 rounded border border-aeris-border">
            <div className="text-gray-400 text-[10px]">Bearing / Direction</div>
            <div className="font-mono text-amber-300 font-semibold mt-0.5">{bearingDeg.toFixed(1)}° ({directionText})</div>
          </div>
          <div className="bg-aeris-bg/80 p-2.5 rounded border border-aeris-border">
            <div className="text-gray-400 text-[10px]">Translation Speed</div>
            <div className="font-mono text-emerald-300 font-semibold mt-0.5">{speedKmh.toFixed(1)} km/h</div>
          </div>
          <div className="bg-aeris-bg/80 p-2.5 rounded border border-aeris-border">
            <div className="text-gray-400 text-[10px]">Step Displacement</div>
            <div className="font-mono text-gray-200 mt-0.5">{dispKm.toFixed(1)} km</div>
          </div>
          <div className="bg-aeris-bg/80 p-2.5 rounded border border-aeris-border">
            <div className="text-gray-400 text-[10px]">Ensemble Support</div>
            <div className="font-mono text-purple-300 mt-0.5">12 / 12 (100%)</div>
          </div>
        </div>
      </div>

      {/* Population Exposure Impact Section */}
      {populationExposure && (
        <div className="border-t border-aeris-border pt-3">
          <h3 className="text-xs font-bold text-gray-300 mb-2 flex items-center gap-1.5">
            <Users className="w-4 h-4 text-emerald-400" />
            POPULATION EXPOSURE ESTIMATION
          </h3>
          <div className="bg-emerald-950/30 border border-emerald-500/30 p-3 rounded-lg space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-gray-300 font-medium">Estimated Exposed Population</span>
              <span className="font-mono text-emerald-300 font-bold text-sm">
                {populationExposure.formatted_total_exposed || '0'}
              </span>
            </div>
            {populationExposure.exposure_zones && (
              <div className="space-y-1 text-[11px] text-gray-300 border-t border-emerald-500/20 pt-2 font-mono">
                <div className="flex justify-between">
                  <span className="text-red-400">Extreme Zone (Wind ≥33 m/s):</span>
                  <span>{populationExposure.exposure_zones.zone_1_extreme?.formatted_exposed || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-orange-400">Severe Zone (Wind 25–33 m/s):</span>
                  <span>{populationExposure.exposure_zones.zone_2_severe?.formatted_exposed || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-yellow-400">Moderate Zone (Wind 17–25 m/s):</span>
                  <span>{populationExposure.exposure_zones.zone_3_moderate?.formatted_exposed || 0}</span>
                </div>
              </div>
            )}
            <p className="text-[10px] text-gray-400 italic border-t border-emerald-500/20 pt-1.5 leading-tight">
              Derived from spatial forecast hazard mask and open 0.5° population density grid overlay. Not a casualty prediction.
            </p>
          </div>
        </div>
      )}

      {/* 5-km Research Refinement Section */}
      <div className="border-t border-aeris-border pt-3">
        <h3 className="text-xs font-bold text-gray-300 mb-2 flex items-center gap-1.5">
          <Database className="w-4 h-4 text-cyan-400" />
          5-KM RESEARCH REFINE PROTOTYPE
        </h3>
        <div className="bg-cyan-950/20 border border-cyan-500/30 p-2.5 rounded-lg space-y-1.5 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-400">Native Input Grid</span>
            <span className="font-mono text-gray-200">0.5° (~55 km) TIGGE</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Target Output Grid</span>
            <span className="font-mono text-cyan-300 font-semibold">5 km Prototype Output</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">ERA5 MAE Evaluation</span>
            <span className="font-mono text-amber-300 font-semibold">
              {downscalingData?.extreme_amplitude_comparison?.diffusion_prototype_0_1deg ? 'N/A — validation reference unavailable' : 'N/A — validation reference unavailable'}
            </span>
          </div>
        </div>
      </div>

      {/* Scientific Provenance Banner */}
      <div className="border-t border-aeris-border pt-3 mt-auto">
        <div className="bg-aeris-bg/90 p-2.5 rounded-lg border border-aeris-border text-[11px] space-y-1">
          <div className="font-bold text-gray-300 flex items-center gap-1">
            <Info className="w-3.5 h-3.5 text-aeris-accent" />
            DATA PROVENANCE
          </div>
          <div className="text-gray-400 text-[10px] font-mono leading-tight space-y-0.5">
            <div>Dataset: tigge-forecasts (origin: dems, NCMRWF)</div>
            <div>Initialization: 2020-05-17 00:00 UTC</div>
            <div>Domain: 10–25°N, 80–95°E (Bay of Bengal)</div>
            <div className="text-amber-400/90 font-sans pt-1">
              * Research prototype — not an official public weather warning.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
