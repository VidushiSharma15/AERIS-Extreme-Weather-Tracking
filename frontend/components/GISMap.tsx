'use client';

import dynamic from 'next/dynamic';
import React from 'react';
import { WeatherEvent, GeoJSONFeatureCollection, TrajectoryPoint } from '../lib/api';

const GISMapInner = dynamic(() => import('./GISMapInner'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full rounded-xl border border-aeris-border bg-aeris-card flex flex-col items-center justify-center p-6 text-gray-400">
      <div className="w-10 h-10 border-4 border-aeris-accent border-t-transparent rounded-full animate-spin mb-4"></div>
      <p className="font-mono text-sm">Initializing Meteorological GIS Map...</p>
    </div>
  ),
});

interface GISMapProps {
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

export default function GISMap(props: GISMapProps) {
  return <GISMapInner {...props} />;
}
