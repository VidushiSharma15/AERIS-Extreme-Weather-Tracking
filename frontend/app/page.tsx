'use client';

import React, { useEffect, useState, useMemo } from 'react';
import {
  checkHealth,
  fetchEvents,
  fetchEventTrajectory,
  fetchAllEventsGeoJSON,
  fetchTracksGeoJSON,
  fetchDatasetMetadata,
  fetchDemoManifest,
  fetchForecastField,
  fetchPopulationExposure,
  fetchModelsStatus,
  fetchDownscalingAmphan,
  WeatherEvent,
  GeoJSONFeatureCollection,
  TrajectoryPoint,
  DatasetMetadata,
  SystemHealth,
} from '../lib/api';
import GISMap from '../components/GISMap';
import EventFilters from '../components/EventFilters';
import EventDetailsPanel from '../components/EventDetailsPanel';
import SystemStatusFooter from '../components/SystemStatusFooter';
import MethodologyModal from '../components/MethodologyModal';
import {
  Shield,
  RefreshCw,
  HelpCircle,
  AlertOctagon,
  CloudRain,
  Thermometer,
  Wind,
  CheckCircle2,
  Play,
  Pause,
  Layers,
  Users,
  Database,
  Info,
  Sliders,
} from 'lucide-react';

export default function DashboardPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [metadata, setMetadata] = useState<DatasetMetadata | null>(null);
  const [allEvents, setAllEvents] = useState<WeatherEvent[]>([]);
  const [geoJsonData, setGeoJsonData] = useState<GeoJSONFeatureCollection | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<WeatherEvent | null>(null);
  const [trajectoryPoints, setTrajectoryPoints] = useState<TrajectoryPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [apiError, setApiError] = useState<string | null>(null);
  const [emptyMessage, setEmptyMessage] = useState<string | null>(null);
  const [isMethodologyOpen, setIsMethodologyOpen] = useState<boolean>(false);

  // MOSDAC-Style Meteorological Controls State
  const [activeVariable, setActiveVariable] = useState<string>('wind_speed');
  const [activeLeadTime, setActiveLeadTime] = useState<number>(0);
  const [activeEnsemble, setActiveEnsemble] = useState<string>('mean');
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [unavailableNotice, setUnavailableNotice] = useState<string | null>(null);

  // Map Overlay Toggles
  const [showVectors, setShowVectors] = useState<boolean>(true);
  const [showFootprints, setShowFootprints] = useState<boolean>(true);
  const [showEnsembleSpread, setShowEnsembleSpread] = useState<boolean>(false);
  const [showPopulationExposure, setShowPopulationExposure] = useState<boolean>(true);
  const [show5kmOutput, setShow5kmOutput] = useState<boolean>(false);

  // Fetched Analytical & Raster State
  const [forecastField, setForecastField] = useState<any>(null);
  const [populationExposure, setPopulationExposure] = useState<any>(null);
  const [downscalingData, setDownscalingData] = useState<any>(null);
  const [modelsStatus, setModelsStatus] = useState<any>(null);

  const leadTimes = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72];

  const loadDashboardData = async () => {
    setLoading(true);
    setApiError(null);
    setEmptyMessage(null);

    try {
      // 1. Health check
      const h = await checkHealth().catch(() => null);
      setHealth(h);

      if (!h || h.status !== 'ok') {
        setApiError('Unable to connect to AERIS FastAPI backend at http://127.0.0.1:8000.');
        setLoading(false);
        return;
      }

      // 2. Fetch dataset metadata
      const meta = await fetchDatasetMetadata().catch(() => null);
      setMetadata(meta);

      // 3. Fetch events
      const eventsRes = await fetchEvents();
      let eventList: WeatherEvent[] = [];
      if (Array.isArray(eventsRes)) {
        eventList = eventsRes;
      } else if (eventsRes && Array.isArray(eventsRes.events)) {
        eventList = eventsRes.events;
        if (eventsRes.message && eventList.length === 0) {
          setEmptyMessage(eventsRes.message);
        }
      }

      setAllEvents(eventList);

      // Select event corresponding to current active lead time
      const matchedEvt = eventList.find((e) => (e.provenance_metadata?.lead_time_hour ?? 0) === activeLeadTime) || eventList[0];
      if (matchedEvt) {
        setSelectedEvent(matchedEvt);
        const traj = await fetchEventTrajectory(matchedEvt.event_id).catch(() => []);
        setTrajectoryPoints(traj);
      } else {
        setSelectedEvent(null);
        setTrajectoryPoints([]);
      }

      // 4. Fetch GeoJSON anomaly polygons
      const geojson = await fetchAllEventsGeoJSON().catch(() => null);
      setGeoJsonData(geojson);

      // 5. Fetch raster field for current lead time & variable
      if (activeVariable === 'temperature' || activeVariable === 'humidity') {
        setUnavailableNotice(`${activeVariable.toUpperCase()} — unavailable in current development dataset slice (requires temperature/humidity GRIB2 download).`);
        setForecastField(null);
      } else {
        setUnavailableNotice(null);
        const field = await fetchForecastField(activeVariable, activeLeadTime, activeEnsemble).catch(() => null);
        setForecastField(field);
      }

      // 6. Fetch population exposure for current lead time
      const popExp = await fetchPopulationExposure(activeLeadTime).catch(() => null);
      setPopulationExposure(popExp);

      // 7. Fetch downscaling and models status
      const down = await fetchDownscalingAmphan().catch(() => null);
      setDownscalingData(down);
      const mods = await fetchModelsStatus().catch(() => null);
      setModelsStatus(mods);

    } catch (err: any) {
      console.error('Error loading dashboard data:', err);
      setApiError(err.message || 'Error connecting to FastAPI backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, [activeLeadTime, activeVariable, activeEnsemble]);

  // Handle Play/Pause Lead Time animation
  useEffect(() => {
    let interval: any = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setActiveLeadTime((prev) => {
          const idx = leadTimes.indexOf(prev);
          const nextIdx = (idx + 1) % leadTimes.length;
          return leadTimes[nextIdx];
        });
      }, 2500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying]);

  // Handle Select Event
  const handleSelectEvent = async (event: WeatherEvent) => {
    setSelectedEvent(event);
    if (event.provenance_metadata?.lead_time_hour !== undefined) {
      setActiveLeadTime(event.provenance_metadata.lead_time_hour);
    }
    try {
      const traj = await fetchEventTrajectory(event.event_id);
      setTrajectoryPoints(traj);
    } catch {
      setTrajectoryPoints([]);
    }
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-aeris-bg text-gray-100">
      {/* Header */}
      <header className="w-full bg-aeris-card/95 border-b border-aeris-border px-6 py-2.5 flex flex-wrap items-center justify-between gap-4 shadow-xl z-20">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-aeris-accent/10 rounded-xl border border-aeris-accent/30 text-aeris-accent">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              AERIS
              <span className="text-xs font-normal text-aeris-accent bg-aeris-accent/10 px-2 py-0.5 rounded-full border border-aeris-accent/30 font-mono">
                SIH 26078 MVP
              </span>
            </h1>
            <p className="text-xs text-gray-400">
              AI Extreme Weather Intelligence System &bull; Spatio-Temporal Anomaly Tracking
            </p>
          </div>
        </div>

        {/* Top Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={async () => {
              setLoading(true);
              try {
                const manifest = await fetchDemoManifest();
                console.log('Loaded demo manifest:', manifest);
                await loadDashboardData();
              } catch (e: any) {
                console.error('Failed to load demo manifest', e);
              } finally {
                setLoading(false);
              }
            }}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs rounded-lg transition shadow-md disabled:opacity-50 border border-emerald-400/40"
          >
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-200" />
            <span>Load Amphan Demo</span>
          </button>

          <button
            onClick={() => setIsMethodologyOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-aeris-bg hover:bg-aeris-border border border-aeris-border text-xs text-gray-300 rounded-lg transition font-medium"
          >
            <HelpCircle className="w-4 h-4 text-aeris-accent" />
            <span>About / Methodology</span>
          </button>

          <button
            onClick={loadDashboardData}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-aeris-accent hover:bg-cyan-400 text-aeris-bg font-bold text-xs rounded-lg transition shadow-md disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh Data</span>
          </button>
        </div>
      </header>

      {/* MOSDAC-Style Meteorological Control Bar */}
      <div className="w-full bg-aeris-card border-b border-aeris-border px-6 py-2 flex flex-wrap items-center justify-between gap-4 text-xs z-10">
        {/* Variable Selector */}
        <div className="flex items-center gap-2">
          <span className="text-gray-400 font-semibold flex items-center gap-1">
            <Sliders className="w-3.5 h-3.5 text-aeris-accent" />
            Variable:
          </span>
          <div className="flex items-center bg-aeris-bg p-1 rounded-lg border border-aeris-border gap-1 font-mono">
            {[
              { id: 'wind_speed', label: 'Wind Speed (m/s)', disabled: false },
              { id: 'msl', label: 'Pressure (hPa)', disabled: false },
              { id: 'tp', label: 'Precipitation (mm)', disabled: false },
              { id: 'msl_anomaly', label: 'Pressure Anomaly', disabled: false },
              { id: 'wind_anomaly', label: 'Wind Anomaly', disabled: false },
              { id: 'temperature', label: 'Temp (Unavailable in dataset)', disabled: true },
              { id: 'humidity', label: 'Humidity (Unavailable in dataset)', disabled: true },
            ].map((v) => (
              <button
                key={v.id}
                disabled={v.disabled}
                onClick={() => !v.disabled && setActiveVariable(v.id)}
                className={`px-2 py-1 rounded text-[11px] transition ${
                  v.disabled
                    ? 'opacity-40 cursor-not-allowed text-gray-500 bg-aeris-bg'
                    : activeVariable === v.id
                    ? 'bg-aeris-accent text-aeris-bg font-bold shadow'
                    : 'text-gray-300 hover:text-white hover:bg-aeris-border'
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>

        {/* Ensemble Selector */}
        <div className="flex items-center gap-2 font-mono">
          <span className="text-gray-400 font-semibold">Forecast Mode:</span>
          <select
            value={activeEnsemble}
            onChange={(e) => setActiveEnsemble(e.target.value)}
            className="bg-aeris-bg border border-aeris-border text-white text-xs rounded-lg px-2.5 py-1 focus:outline-none focus:border-aeris-accent"
          >
            <option value="mean">NCMRWF/NEPS-G 12-member Ensemble Mean</option>
            <option value="control">NCMRWF Control Run (Member 0)</option>
            {Array.from({ length: 11 }, (_, i) => (
              <option key={`mem-${i+1}`} value={`${i+1}`}>Perturbed Member {i+1}</option>
            ))}
          </select>
        </div>

        {/* Layer Toggles */}
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showVectors}
              onChange={(e) => setShowVectors(e.target.checked)}
              className="accent-aeris-accent"
            />
            <span>Wind Vectors</span>
          </label>

          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showFootprints}
              onChange={(e) => setShowFootprints(e.target.checked)}
              className="accent-aeris-accent"
            />
            <span>Event Footprint</span>
          </label>

          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showPopulationExposure}
              onChange={(e) => setShowPopulationExposure(e.target.checked)}
              className="accent-emerald-400"
            />
            <span>Population Exposure</span>
          </label>

          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={show5kmOutput}
              onChange={(e) => setShow5kmOutput(e.target.checked)}
              className="accent-cyan-400"
            />
            <span>Forecast Grid: ~55 km | Downscaled Output: ~5 km</span>
          </label>
        </div>
      </div>

      {/* Forecast Time Slider Banner */}
      <div className="w-full bg-aeris-bg/90 border-b border-aeris-border px-6 py-2 flex items-center justify-between gap-4 z-10 shadow-inner">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-1.5 bg-aeris-accent hover:bg-cyan-400 text-aeris-bg rounded-lg transition shadow font-bold flex items-center gap-1"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span className="text-xs uppercase">{isPlaying ? 'Pause' : 'Play Track'}</span>
          </button>
          <div className="text-xs font-mono">
            <span className="text-gray-400">Active Forecast Lead Time: </span>
            <span className="text-aeris-accent font-bold text-sm">+{activeLeadTime}h</span>
          </div>
        </div>

        {/* Lead Time Slider */}
        <div className="flex-1 max-w-xl mx-4 flex items-center gap-2">
          <input
            type="range"
            min={0}
            max={72}
            step={6}
            value={activeLeadTime}
            onChange={(e) => setActiveLeadTime(Number(e.target.value))}
            className="w-full h-2 bg-aeris-card rounded-lg appearance-none cursor-pointer accent-aeris-accent border border-aeris-border"
          />
        </div>

        {/* Step Buttons */}
        <div className="flex items-center gap-1 font-mono text-[11px]">
          {leadTimes.map((lt) => (
            <button
              key={`lt-${lt}`}
              onClick={() => setActiveLeadTime(lt)}
              className={`px-2 py-0.5 rounded transition ${
                activeLeadTime === lt
                  ? 'bg-aeris-accent text-aeris-bg font-bold'
                  : 'bg-aeris-card hover:bg-aeris-border text-gray-300'
              }`}
            >
              +{lt}h
            </button>
          ))}
        </div>
      </div>

      {/* Unavailable Variable Warning Notice */}
      {unavailableNotice && (
        <div className="w-full bg-amber-950/40 border-b border-amber-500/40 px-6 py-1.5 flex items-center gap-2 text-xs text-amber-300 font-mono">
          <Info className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <span>{unavailableNotice}</span>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden p-4 gap-4 relative">
        {/* Left Map View */}
        <div className="flex-1 h-full relative">
          <GISMap
            events={allEvents}
            geoJsonData={geoJsonData}
            selectedEvent={selectedEvent}
            trajectoryPoints={trajectoryPoints}
            forecastField={forecastField}
            activeVariable={activeVariable}
            activeLeadTime={activeLeadTime}
            activeEnsemble={activeEnsemble}
            showVectors={showVectors}
            showFootprints={showFootprints}
            showEnsembleSpread={showEnsembleSpread}
            showPopulationExposure={showPopulationExposure}
            show5kmOutput={show5kmOutput}
            populationExposure={populationExposure}
            onSelectEvent={handleSelectEvent}
          />
        </div>

        {/* Right Event Diagnostics & Impact Summary Panel */}
        <div className="w-[380px] h-full flex-shrink-0">
          <EventDetailsPanel
            event={selectedEvent}
            trajectory={trajectoryPoints}
            metadata={metadata}
            populationExposure={populationExposure}
            downscalingData={downscalingData}
            modelsStatus={modelsStatus}
          />
        </div>
      </div>

      {/* Footer Status */}
      <SystemStatusFooter health={health} metadata={metadata} totalEvents={allEvents.length} />

      {/* Methodology Modal */}
      <MethodologyModal isOpen={isMethodologyOpen} onClose={() => setIsMethodologyOpen(false)} />
    </div>
  );
}
